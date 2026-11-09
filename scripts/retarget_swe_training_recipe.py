# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare new frozen SWE rows using an existing arm's model, runtime and training settings."""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path


sys.dont_write_bytecode = True

from omegaconf import OmegaConf
from scripts.prepare_swe_single_agent_turn_recipe import configure_gym, materialize_training_rows
from scripts.swe_rows_to_harbor.convert import convert

from nemo_gym.data.tasks.harbor.prepare import prepare_target


def retarget(
    *,
    prepared: Path,
    output: Path,
    task_rows: Path,
    dataset_manifest: Path,
    source_server: str,
    wandb_project: str,
    wandb_entity: str,
) -> dict:
    """Keep the existing arm's science while replacing its tasks and resources binding.

    Reads only the specified prepared inputs; no model/image inventories or network
    qualification runs. Harbor conversion uses the maintained task preparation path.
    """
    prepared, output = prepared.resolve(), output.resolve()
    original = json.loads((prepared / "provenance.json").read_text())
    source_bytes = task_rows.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if not rows:
        raise ValueError("The selected task rows are empty")
    harnesses, route = original["harnesses"], original["route"]
    config = OmegaConf.create(
        json.loads(
            json.dumps(OmegaConf.to_container(OmegaConf.load(prepared / "config.yaml"), resolve=False)).replace(
                str(prepared), str(output)
            )
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in original["files_sha256"]:
        if name in ("config.yaml", "source.jsonl", "train.jsonl", "source-mount.json"):
            continue
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((prepared / name).read_bytes().replace(str(prepared).encode(), str(output).encode()))
        copied.append(name)
    (output / "source.jsonl").write_bytes(source_bytes)
    (output / "dataset-manifest.json").write_bytes(dataset_manifest.read_bytes())
    harbor = None
    if route == "harbor":
        convert(
            output / "source.jsonl",
            output / "harbor",
            dataset_name=source_server,
            default_source=source_server,
            agent_timeout_sec=480,
            evaluation_timeout_sec=180,
        )
        harbor = prepare_target(
            str(output / "harbor"), output_root=output / "prepared", sandbox_provider="opensandbox"
        )
        if harbor.skipped or len(harbor.tasks) != len(rows):
            raise ValueError("Harbor conversion must preserve every selected task")
    gym = config.env.nemo_gym
    # Replace the prior task's servers, retaining the policy bridge and runtime settings.
    for name in list(gym):
        value = gym[name]
        if (
            name != "policy_model"
            and OmegaConf.is_dict(value)
            and any(kind in value for kind in ("resources_servers", "environment_servers", "responses_api_agents"))
        ):
            del gym[name]
    configure_gym(
        config,
        harnesses=harnesses,
        route=route,
        output=output,
        parser_cache=prepared,
        prepared=harbor,
        source_server=source_server,
    )
    train_rows = materialize_training_rows(
        rows, harnesses=harnesses, route=route, prepared=harbor, source_server=source_server
    )
    (output / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows))
    prompts = min(8, len(rows)) * len(harnesses)
    config.grpo.num_prompts_per_step = prompts
    config.grpo.max_num_epochs = math.ceil(config.grpo.max_num_steps * min(8, len(rows)) / len(rows))
    config.policy.train_global_batch_size = prompts * config.grpo.num_generations_per_prompt
    config.data.train.data_path = str(output / "train.jsonl")
    config.logger.wandb_enabled = True
    config.logger.wandb.update(
        project=wandb_project,
        entity=wandb_entity,
        mode="online",
        name=output.name,
        log_nemo_gym_full_result_tables=False,
    )
    config.logger.wandb.pop("id", None)
    config.logger.wandb.pop("resume", None)
    OmegaConf.save(config, output / "config.yaml", resolve=False)
    result = copy.deepcopy(original)
    # Prior task/parser identities belong to the retained parent, not the new dataset.
    for key in ("source_pool_sha256", "roster_sha256", "image_pins_sha256", "parser", "task_selection"):
        result.pop(key, None)
    result.update(
        reused_preparation=str(prepared),
        source_server=source_server,
        dataset_manifest="dataset-manifest.json",
        tasks=[{"instance_id": row["instance_id"]} for row in rows],
        steps=int(config.grpo.max_num_steps),
        episodes=int(config.grpo.max_num_steps * config.policy.train_global_batch_size),
        ordering="task-major, harness-minor, no shuffle; repeat complete epochs",
        qualification="New selected data; training and learning not yet measured",
    )
    names = ["source.jsonl", "train.jsonl", "config.yaml", "dataset-manifest.json", *copied]
    result["files_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}
    (output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return {
        "prepared": str(output),
        "route": route,
        "harnesses": harnesses,
        "tasks": len(rows),
        "materialized_rows": len(train_rows),
        "prompts_per_update": prompts,
        "epochs": int(config.grpo.max_num_epochs),
        "episodes": result["episodes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "task-rows", "dataset-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-server", choices=("swe_next", "swe_rebench"), required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", required=True)
    print(json.dumps(retarget(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
