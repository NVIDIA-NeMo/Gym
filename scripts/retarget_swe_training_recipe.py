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
from scripts.prepare_swe_single_agent_turn_recipe import (
    HARNESSES,
    configure_gym,
    configure_wandb_labels,
    harness_assets,
    materialize_training_rows,
    validate_harnesses,
)

from nemo_gym.data.tasks.harbor import prepare_target


def retarget(
    *,
    prepared: Path,
    output: Path,
    task_rows: Path,
    dataset_manifest: Path,
    source_server: str,
    wandb_project: str,
    wandb_entity: str,
    harness: str | None = None,
    harness_assets_from: Path | None = None,
    harbor_tasks: Path | None = None,
) -> dict:
    """Keep the existing arm's science while replacing its tasks and resources binding.

    Reads only the specified prepared inputs; no model/image inventories or network
    qualification runs. Harbor requires task folders supplied in their existing format.
    """
    prepared, output = prepared.resolve(), output.resolve()
    original = json.loads((prepared / "provenance.json").read_text())
    if source_server != "swe_next":
        raise ValueError("Only SWE-Next rows are supported by this recipe")
    source_bytes = task_rows.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if not rows:
        raise ValueError("The selected task rows are empty")
    harnesses, route = original["harnesses"], original["route"]
    if route == "harbor" and harbor_tasks is None:
        raise ValueError("Harbor retargeting requires --harbor-tasks with existing Harbor task folders")
    if route != "harbor" and harbor_tasks is not None:
        raise ValueError("--harbor-tasks applies only to the Harbor route")
    assets = {}
    if harness is not None:
        if len(harnesses) != 1:
            raise ValueError("Compose a single harness from a single-harness template to retain its concurrency")
        harnesses = [harness]
        validate_harnesses(harnesses, route=route)
        if harness == "codex":
            asset_root = (harness_assets_from or prepared) / "codex"
            assets = {
                f"codex/{name}": (asset_root / name).read_bytes()
                for name in ("models.json", "fallback.json", "provenance.json")
            }
        else:
            assets = harness_assets(harnesses, codex_catalog=None)
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
    for name, data in assets.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if name not in copied:
            copied.append(name)
    (output / "source.jsonl").write_bytes(source_bytes)
    (output / "dataset-manifest.json").write_bytes(dataset_manifest.read_bytes())
    harbor = None
    if route == "harbor":
        harbor = prepare_target(
            str(harbor_tasks.resolve()), output_root=output / "prepared", sandbox_provider="opensandbox"
        )
        selected = [row.get("harbor_task_id") for row in rows]
        if (
            harbor.skipped
            or any(not isinstance(task_id, str) or not task_id for task_id in selected)
            or len(set(selected)) != len(rows)
            or set(selected) != {task.task_id for task in harbor.tasks}
        ):
            raise ValueError("Each selected row must name one existing Harbor task through harbor_task_id")
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
        log_nemo_gym_full_result_tables=False,
    )
    configure_wandb_labels(
        config,
        model=original["model"].get("repo", config.policy.model_name),
        source_server=source_server,
        task_count=len(rows),
        harnesses=harnesses,
        route=route,
        output=output,
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
        harnesses=harnesses,
        source_server=source_server,
        dataset_manifest="dataset-manifest.json",
        tasks=[{"instance_id": row["instance_id"]} for row in rows],
        steps=int(config.grpo.max_num_steps),
        episodes=int(config.grpo.max_num_steps * config.policy.train_global_batch_size),
        ordering="task-major, harness-minor, no shuffle; repeat complete epochs",
        qualification="New selected data; training and learning not yet measured",
    )
    if harbor_tasks is not None:
        result["harbor_tasks"] = str(harbor_tasks.resolve())
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "output", "task-rows", "dataset-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--harbor-tasks", type=Path, help="Existing Harbor task directory; required for Harbor")
    parser.add_argument("--source-server", choices=("swe_next",), default="swe_next")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--harness", choices=HARNESSES, help="Compose one harness using a single-harness template")
    parser.add_argument(
        "--harness-assets-from", type=Path, help="Existing prepared directory containing codex/ assets"
    )
    print(json.dumps(retarget(**vars(parser.parse_args(argv)))))


if __name__ == "__main__":
    main()
