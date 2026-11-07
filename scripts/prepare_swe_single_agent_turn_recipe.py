# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare typed SWE tasksets and an RL-main recipe for SingleAgentTurnEnvironmentServer.

This only writes data, configuration and provenance. It never starts services or
training. Sugam's MR69 supplies reference settings; no pipeline checkout is needed.
The printed command uses the committed launcher and NeMo RL's existing ray.sub.
"""

import argparse
import hashlib
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nemo_gym.task_materialization import materialize_task


PIPELINE_REVISION = "36e2364f9f79e9071a4f8e76634f73d3c204e6f9"
RL_REVISION = "479735d76b3aa20e130d23b00c3e4bcd8b754613"
SANDBOX_DOMAIN = "k8s-cellrout-cellrout-101e0e7b17-b0ecd469dafdeefb.elb.us-east-2.amazonaws.com"
GYM_ROOT = Path(__file__).resolve().parents[1]
RECIPE = GYM_ROOT / "examples/training/swe_single_agent_turn.yaml"
HARNESSES = {
    "opencode": "opencode_agent",
    "claude_code": "claude_code_agent",
    "hermes": "hermes_agent",
    "pi": "pi_agent",
}
SOURCES = ("swe_rebench", "scale_swe", "swemer_v1", "swemer_v2", "swe_next")


def _load_config(path: Path) -> DictConfig:
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a mapping in {path}")
    parent = config.pop("defaults", None)
    if parent is None:
        return config
    if not isinstance(parent, str):
        raise ValueError(f"Expected a single-file defaults reference in {path}")
    return OmegaConf.merge(_load_config(path.parent / parent), config)


def _check_rl(rl_root: Path) -> None:
    revision = subprocess.check_output(["git", "-C", str(rl_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != RL_REVISION:
        raise ValueError(f"This recipe targets RL main at {RL_REVISION}, got {revision}")
    gym_root = rl_root / "3rdparty/Gym-workspace/Gym"
    if gym_root.resolve() != GYM_ROOT:
        raise ValueError(f"Check out this Gym tip at {gym_root}, so RL imports the same session implementation")


def _materialize_rows(source: Path, destination: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with source.open() as src, destination.open("x") as dst:
        for index, line in enumerate(src):
            row = json.loads(line)
            name = (row.get("agent_ref") or {}).get("name", "")
            taskset = name.removesuffix("_opencode_sandboxed_agent")
            if taskset not in SOURCES:
                raise ValueError(
                    f"{source}:{index + 1}: unsupported source {name!r}. No rows are filtered; "
                    "the default blend has five sources and DeNovo has no session adapter in this stack."
                )
            if not isinstance(row.get("responses_create_params"), dict):
                raise ValueError(f"{source}:{index + 1}: missing responses_create_params")
            task = materialize_task(row, taskset=taskset, task_index=index)
            flat = task["task_input"]
            # Use Gym's task-identity rules and canonical nested contract. RL
            # writes sampling into responses_create_params; Resources receives
            # task_data with the original image and verifier fields intact.
            params = flat.pop("responses_create_params")
            task["task_input"] = {"responses_create_params": params, "task_data": flat}
            dst.write(json.dumps(task) + "\n")
            counts[taskset] += 1
    if not counts:
        raise ValueError(f"Dataset is empty: {source}")
    return dict(counts)


def _compose(config: DictConfig, *, harness: str) -> DictConfig:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    gym = config.env.nemo_gym
    gym.config_paths = [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        f"environment_servers/single_agent_turn/configs/swe_{harness}.yaml",
        "resources_servers/denovo_swe/configs/denovo_swe.yaml",
        "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml",
    ]
    agent = HARNESSES[harness]
    settings = {"token_id_capture": True, "datasets": [], "num_workers": 1}
    if harness == "hermes":
        settings.update(max_tokens=196608, sandbox_runner_timeout_seconds=4800)
    else:
        settings.update(context_window=196608, max_output_tokens=196608, timeout=4800)
    gym[agent] = {"responses_api_agents": {agent: settings}}
    gym.environment_server_routes = {source: f"{source}_environment" for source in SOURCES}
    for source in SOURCES:
        gym[f"{source}_environment"] = {
            "environment_servers": {
                "single_agent_turn": {
                    "default_episode_timeout_seconds": 8400,
                    "cleanup_timeout_seconds": 180,
                    "resources_tool_transports": [],
                    "mask_incomplete_for_training": True,
                }
            }
        }
    gym.sandbox.opensandbox.connection.domain = SANDBOX_DOMAIN
    gym.sandbox.opensandbox.connection.protocol = "https"
    return config


def main() -> None:
    """Write tasksets and a reusable main-based recipe, without launching a job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-root", type=Path, required=True)
    parser.add_argument("--harness", choices=HARNESSES, default="opencode")
    parser.add_argument("--model-path", type=Path, required=True, help="The MR69 reference checkpoint's HF directory")
    parser.add_argument("--train-data", type=Path, required=True, help="MR69's mixed_3_7 shuffled JSONL")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A fresh directory on the cluster's shared filesystem"
    )
    args = parser.parse_args()
    rl_root, model, train = args.rl_root.resolve(), args.model_path.resolve(), args.train_data.resolve()
    _check_rl(rl_root)
    if not model.is_dir():
        raise ValueError(f"Model directory is missing: {model}")
    reference = OmegaConf.load(RECIPE)
    main_defaults = reference.pop("defaults")
    config = _compose(
        OmegaConf.merge(_load_config(rl_root / "examples/configs" / main_defaults), reference), harness=args.harness
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    counts = _materialize_rows(train, output / "train.jsonl")
    config.policy.model_name = str(model)
    config.data.train.data_path = str(output / "train.jsonl")
    config.data.validation.data_path = str(output / "train.jsonl")
    config.checkpointing.checkpoint_dir = str(output / "checkpoints")
    config.logger.log_dir = str(output / "logs")
    OmegaConf.save(config, output / "config.yaml", resolve=False)
    manifest = {
        "settings_reference": {"pipeline_mr": 69, "revision": PIPELINE_REVISION},
        "rl_revision": RL_REVISION,
        "gym_revision": subprocess.check_output(["git", "-C", str(GYM_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "harness": args.harness,
        "input": str(train),
        "counts": counts,
        "masking": "RL main masked_reward_policy=include; standard instance_config.mask_sample",
        "validation": "preparation only; no services, model rollouts or training run",
    }
    with train.open("rb") as stream:
        manifest["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    command = [
        "env",
        f"NEMO_RL_ROOT={rl_root}",
        f"CONFIG_PATH={output / 'config.yaml'}",
        f"MODEL_PATH={model}",
        "bash",
        str(GYM_ROOT / "scripts/launch_swe_single_agent_turn_training.sh"),
    ]
    print(f"Prepared {sum(counts.values())} taskset rows without changing task content. Training command:")
    print(shlex.join(command))


if __name__ == "__main__":
    main()
