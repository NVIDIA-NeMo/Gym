# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pprint
from pathlib import Path
from typing import Any

import ray
import wandb.util
from nemo_rl.algorithms.grpo import (
    ColocatablePolicyInterface,
    EnvironmentInterface,
    GenerationInterface,
    Logger,
    MasterConfig,
    StatefulDataLoader,
    TokenizerType,
    _should_use_nemo_gym,
    grpo_train,
    refit_policy_generation,
    setup,
    validate,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    NemoGymConfig,
    setup_nemo_gym_config,
)
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from omegaconf import OmegaConf

from wandb import Table

# Increase the W&B single object size warning threshold from 100 KB to 10 MB.
wandb.util.VALUE_BYTES_LIMIT = 10_000_000


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run step-zero validation and exit before any optimizer update",
    )
    parser.add_argument("--validation-summary", type=Path)
    parser.add_argument("--min-validation-accuracy", type=float)
    parser.add_argument("--max-validation-accuracy", type=float)

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


def _json_scalars(value: Any) -> Any:
    """Retain JSON-safe metric values and omit W&B tables or tensors."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): converted
            for key, item in value.items()
            if (converted := _json_scalars(item)) is not None
        }
    if isinstance(value, (list, tuple)):
        converted = [_json_scalars(item) for item in value]
        return [item for item in converted if item is not None]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return None
    return None


def _write_validation_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.environ.get('SLURM_JOB_ID', 'local')}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


# These types are directly imported from grpo_train since if something about the architecture changes we want to immediately fail.
def collect_trajectories(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer: TokenizerType,
    val_task_to_env: dict[str, EnvironmentInterface],
    logger: Logger,
    master_config: MasterConfig,
) -> None:
    """Run trajectory collection."""
    # common config/state items
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    refit_policy_generation(policy, policy_generation, colocated_inference)

    log_filename = "trajectory_collection.jsonl"

    print("\n🔍 Running trajectory collection...", flush=True)
    generation_config = master_config["policy"]["generation"]
    for val_batch in val_dataloader:
        nemo_gym_rollout_result = run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=val_batch,
            tokenizer=tokenizer,
            task_to_env=val_task_to_env,
            max_seq_len=None,
            generation_config=generation_config,
            max_rollout_turns=None,
            greedy=False,
        )

        rows_to_log: list[str] = []
        for key, value in nemo_gym_rollout_result.rollout_metrics.items():
            if "full_result" not in key:
                continue

            value: Table
            data: list[list[str]] = value.data  # (n, 1)
            rows_to_log.extend(v[0] for v in data)

        logger.log_string_list_as_jsonl(rows_to_log, log_filename)

        # TODO: eventually as trajectory collection use cases exceed 4 hours, we can leverage the dataloader save functionality to resume
        # And also leverage the TimeoutChecker functionality as well

    policy_generation.finish_generation()


def main() -> None:
    """Main entry point."""
    # Parse arguments
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "grpo_workplace_assistant_nemotron_nano_v2_9b.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # NeMo-Gym specific config setup.
    setup_nemo_gym_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized.
    assert _should_use_nemo_gym(config)

    # NeMo-Gym environment needs to get dp_openai_server_base_urls from policy_generation, so we don't setup env here.
    print("\n▶ Setting up data...")
    train_dataset, val_dataset = setup_response_data(
        tokenizer, config["data"], env_configs=None
    )

    # Validation dataset config setup.
    if config["grpo"]["max_val_samples"] is not None:
        raise ValueError(
            """A non-null `grpo.max_val_samples` parameter is not supported.

Gym principle is that there is no hidden data pre or post processing from you. What you see is what you get.

The validation set you pass in will directly be used for validation with no additional preprocessing. If you want to have some number of repetitions, please include that in your dataset, via ``num_repeats``, in your dataset config and `ng_prepare_data` will prepare it accordingly."""
        )

    if val_dataset is not None:
        print(
            f"Setting `grpo.max_val_samples` and `grpo.val_batch_size` to the length of the validation dataset, which is {len(val_dataset)}"
        )
        config["grpo"]["max_val_samples"] = len(val_dataset)
        config["grpo"]["val_batch_size"] = config["grpo"]["max_val_samples"]

    # Print config
    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    is_trajectory_collection = (
        config["env"]["nemo_gym"].pop("is_trajectory_collection", False) or False
    )
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    # Blocking wait for NeMo-Gym to spin up
    ray.get(nemo_gym.health_check.remote())

    # Bind task_to_env and val_task_to_env for nemo_gym env
    # Hardcode here to match `run_async_nemo_gym_rollout`
    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    if args.validation_only:
        if args.validation_summary is None:
            raise ValueError("--validation-summary is required with --validation-only")
        print("🔍 Running validation-only gate before any optimizer update", flush=True)
        colocated_inference = master_config["policy"]["generation"]["colocated"][
            "enabled"
        ]
        refit_policy_generation(policy, policy_generation, colocated_inference)
        val_metrics, validation_timings = validate(
            policy_generation=policy_generation,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            val_task_to_env=val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, 0, prefix="validation")
        logger.log_metrics(
            validation_timings,
            0,
            prefix="timing/validation",
            step_finished=True,
        )
        accuracy = float(val_metrics["accuracy"])
        expected_rollouts = len(val_dataset)
        observed_rollouts = None
        val_log_path = Path(config["logger"]["log_dir"]) / "val_data_step0.jsonl"
        if val_log_path.is_file():
            observed_rollouts = sum(
                len(json.loads(line).get("rewards", []))
                for line in val_log_path.open()
                if line.strip()
            )
        passed = True
        if args.min_validation_accuracy is not None:
            passed = passed and accuracy >= args.min_validation_accuracy
        if args.max_validation_accuracy is not None:
            passed = passed and accuracy <= args.max_validation_accuracy
        summary = {
            "schema_version": 1,
            "validation_only": True,
            "optimizer_updates": 0,
            "accuracy": accuracy,
            "expected_rollouts": expected_rollouts,
            "observed_rollouts": observed_rollouts,
            "passed": passed,
            "minimum_accuracy": args.min_validation_accuracy,
            "maximum_accuracy": args.max_validation_accuracy,
            "metrics": _json_scalars(val_metrics),
            "timings": _json_scalars(validation_timings),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
        }
        _write_validation_summary(args.validation_summary, summary)
        if observed_rollouts != expected_rollouts:
            passed = False
            summary["passed"] = False
            _write_validation_summary(args.validation_summary, summary)
            raise RuntimeError(
                f"Validation scored {observed_rollouts} rollouts, expected "
                f"{expected_rollouts}"
            )
        if not passed:
            raise RuntimeError(
                f"Validation accuracy {accuracy:.6f} is outside the accepted interval "
                f"[{args.min_validation_accuracy}, {args.max_validation_accuracy}]"
            )
        print(f"✅ Validation-only gate passed at accuracy={accuracy:.6f}", flush=True)
        return

    if is_trajectory_collection:
        collect_trajectories(
            policy=policy,
            policy_generation=policy_generation,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            val_task_to_env=val_task_to_env,
            logger=logger,
            master_config=master_config,
        )
    # Check if async mode is enabled
    elif "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        # Async GRPO does not support dynamic sampling, reward scaling, or reward shaping (DAPO features)
        unsupported_features = [
            "use_dynamic_sampling",
            "reward_scaling",
            "reward_shaping",
        ]

        for feature in unsupported_features:
            if feature not in config["grpo"]:
                continue

            if feature == "use_dynamic_sampling":
                if config["grpo"][feature]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )
            else:
                if config["grpo"][feature]["enabled"]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )

        # Async GRPO does not support multiple dataloaders
        if config["data"]["use_multiple_dataloader"]:
            raise NotImplementedError(
                "use_multiple_dataloader is not supported with async GRPO"
            )

        from nemo_rl.algorithms.grpo import async_grpo_train

        print("🚀 Running async GRPO training")

        async_config = config["grpo"]["async_grpo"]
        # Run async GRPO training
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        print("🚀 Running synchronous GRPO training")

        # Run standard GRPO training
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )


if __name__ == "__main__":
    main()
