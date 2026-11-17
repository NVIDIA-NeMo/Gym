# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose Terminal-Bench episode routing over an existing NeMo RL recipe."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml

from nemo_gym.data import materialize_task
from nemo_gym.data.tasks.harbor import BuildOptions, prepare_target, taskset_mapping
from nemo_gym.data.tasks.harbor import materialize_task as materialize_harbor_task


HARNESS_CONFIGS = {
    "opencode_agent": "responses_api_agents/opencode_agent/configs/opencode_agent.yaml",
    "pi_agent": "responses_api_agents/pi_agent/configs/pi_agent.yaml",
}


def prepare(
    *,
    source: Path,
    validation_source: Path,
    tasks: Path,
    base: Path,
    output: Path,
    route: Literal["sat", "harbor"],
    harnesses: list[str],
    sandbox: str,
) -> Path:
    """Keep one source row per task; assign its harness before GRPO repeats it."""
    partitions = {
        name: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for name, path in (("train", source), ("validation", validation_source))
    }
    by_identity = {}
    for selected in partitions.values():
        if not selected or len({row["task_id"] for row in selected}) != len(selected):
            raise ValueError("Provide nonempty Terminal-Bench source partitions with unique task_id values")
        for row in selected:
            previous = by_identity.setdefault(row["task_id"], row)
            if previous != row:
                raise ValueError("The same task identity has different source content across partitions")
    rows = list(by_identity.values())
    if not base.is_file() or output.exists():
        raise ValueError("Provide an existing NeMo RL base recipe and a fresh output directory")
    if not harnesses or len(set(harnesses)) != len(harnesses) or set(harnesses) - HARNESS_CONFIGS.keys():
        raise ValueError("Choose opencode_agent, pi_agent, or both once each")
    tasks = tasks.resolve()
    for row in rows:
        folder = Path(row["task_folder"]).resolve()
        if not folder.is_relative_to(tasks) or folder.name != row["task_id"]:
            raise ValueError("Each source task_folder must identify its task under --tasks")
        if not (folder / "tests/test.sh").is_file():
            raise ValueError(f"Missing task verifier: {folder}")
    output.mkdir(parents=True)
    config_paths = [HARNESS_CONFIGS[name] for name in harnesses]
    config_paths.append(
        f"environment_servers/{'single_agent_turn' if route == 'sat' else 'harbor'}/configs/"
        f"{'single_agent_turn' if route == 'sat' else 'harbor'}.yaml"
    )
    graph = {"config_paths": config_paths, "environment_server_routes": {}}
    if route == "sat":
        config_paths.append("resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml")
        graph["single_agent_turn"] = None
        prepared = None
        by_id = {}
    else:
        graph["harbor_environment_server"] = None
        # Existing task images only. Source/config changes never trigger a build.
        prepared = prepare_target(
            str(tasks),
            output_root=output / "prepared",
            only=[row["task_id"] for row in rows],
            sandbox_provider=sandbox,
            build=BuildOptions(enabled=False),
        )
        if prepared.skipped or len(prepared.tasks) != len(rows):
            raise ValueError("Harbor preparation did not preserve every selected task; inspect prepared/skipped.json")
        by_id = {task.task_id: task for task in prepared.tasks}
    for harness in harnesses:
        taskset = f"terminal-bench:{harness}"
        instance = f"terminal_bench_{harness}_{route}"
        options = {"entrypoint": "app.py", "agent_server": {"type": "responses_api_agents", "name": harness}}
        if route == "sat":
            options["resources_server"] = {"type": "resources_servers", "name": "terminal_bench_2_1_resources_server"}
        else:
            options.update(
                sandbox_provider="sandbox", tasksets={taskset: taskset_mapping(prepared.tasks, prepared.folder)}
            )
        graph[instance] = {"environment_servers": {"single_agent_turn" if route == "sat" else "harbor": options}}
        graph["environment_server_routes"][taskset] = instance
    assignments = []
    for partition, selected in partitions.items():
        converted = []
        for index, row in enumerate(selected):
            harness = harnesses[index % len(harnesses)]
            taskset = f"terminal-bench:{harness}"
            if route == "sat":
                record = materialize_task(row, taskset=taskset)
            else:
                task = by_id[row["task_id"]]
                record = materialize_harbor_task(task, taskset).model_dump(mode="json")
                generated = record["task_input"]["responses_create_params"]
                original = row["responses_create_params"]
                original_instruction = [{"role": "user", "content": (task.path / "instruction.md").read_text()}]
                if (
                    original_instruction != original["input"]
                    or task.config.environment.docker_image != row["docker_image"]
                ):
                    raise ValueError(f"Harbor prompt/image differs from the source task: {row['task_id']}")
                params = copy.deepcopy(original)
                params["metadata"] = {**generated.get("metadata", {}), **params.get("metadata", {})}
                record["task_input"]["responses_create_params"] = params
            converted.append(record)
            assignments.append({"partition": partition, "task_id": row["task_id"], "harness": harness})
        (output / f"{partition}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted)
        )
    gym_path = output / "gym.yaml"
    gym_path.write_text(yaml.safe_dump(graph, sort_keys=False))
    # NeMo RL's maintained loader resolves defaults relative to this file; an
    # absolute base reference retains that recipe's own inheritance unchanged.
    overlay = {
        "defaults": str(base.resolve()),
        "data": {name: {"data_path": str(output / f"{name}.jsonl")} for name in partitions},
        "env": {
            "nemo_gym": {
                "config_paths": [
                    "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                    str(gym_path),
                ],
                "environment_server_routes": graph["environment_server_routes"],
            }
        },
    }
    config = output / "train.yaml"
    config.write_text(yaml.safe_dump(overlay, sort_keys=False))
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "base_recipe": str(base.resolve()),
                "validation_source_sha256": hashlib.sha256(validation_source.read_bytes()).hexdigest(),
                "route": route,
                "assignments": assignments,
                "rows": {name: len(selected) for name, selected in partitions.items()},
                "harnesses": harnesses,
                "validation": "Preparation only; training results are recorded separately",
            },
            indent=2,
        )
        + "\n"
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "validation-source", "tasks", "base", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--route", choices=("sat", "harbor"), required=True)
    parser.add_argument("--harnesses", nargs="+", choices=tuple(HARNESS_CONFIGS), required=True)
    parser.add_argument(
        "--sandbox", default="opensandbox", help="Provider used to prepare existing Harbor task images"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    print(prepare(**vars(args)))
