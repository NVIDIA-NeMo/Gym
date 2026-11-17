# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate task identity, split preservation and routing in the training example."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemo_gym.data.tasks.harbor.task import load_task
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnTaskInput


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "prepare_harness_routes", ROOT / "examples/training/prepare_harness_routes.py"
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def inputs(tmp_path):
    tasks = tmp_path / "tasks"
    originals = []
    loaded = []
    for index in range(4):
        folder = tasks / f"task-{index}"
        (folder / "tests").mkdir(parents=True)
        (folder / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
        instruction = f"Write {index} to /tmp/answer.\n"
        (folder / "instruction.md").write_text(instruction)
        (folder / "task.toml").write_text(
            'schema_version = "1.4"\n[environment]\ndocker_image = "ubuntu:24.04"\nworkdir = "/tmp"\n'
        )
        loaded.append(load_task(folder))
        originals.append(
            {
                "task_id": folder.name,
                "task_name": folder.name,
                "task_folder": str(folder),
                "docker_image": "ubuntu:24.04",
                "responses_create_params": {
                    "input": [{"role": "user", "content": instruction}],
                    "max_output_tokens": 128,
                },
            }
        )
    source = tmp_path / "train-source.jsonl"
    validation = tmp_path / "validation-source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in originals[:2]))
    validation.write_text("".join(json.dumps(row) + "\n" for row in originals[2:]))
    base = tmp_path / "researcher.yaml"
    base.write_text("policy:\n  model_name: researcher-policy\n  train_global_batch_size: 64\n")
    return tasks, originals, loaded, source, validation, base


@pytest.mark.parametrize("route", ["sat", "harbor"])
@pytest.mark.parametrize("harnesses", [["opencode_agent"], ["opencode_agent", "pi_agent"]])
def test_four_modes_preserve_tasks_splits_prompts_and_base(tmp_path, monkeypatch, route, harnesses):
    tasks, originals, loaded, source, validation, base = inputs(tmp_path)
    before = base.read_bytes()

    def prepare(target, **kwargs):
        assert not kwargs["build"].enabled
        assert set(kwargs["only"]) == {row["task_id"] for row in originals}
        return SimpleNamespace(tasks=loaded, folder=tasks, skipped={})

    monkeypatch.setattr(EXAMPLE, "prepare_target", prepare)
    output = tmp_path / "output"
    config = EXAMPLE.prepare(
        source=source,
        validation_source=validation,
        tasks=tasks,
        base=base,
        output=output,
        route=route,
        harnesses=harnesses,
        sandbox="opensandbox",
    )
    overlay = yaml.safe_load(config.read_text())
    graph = yaml.safe_load((output / "gym.yaml").read_text())
    assert base.read_bytes() == before and overlay["defaults"] == str(base)
    assert "policy" not in overlay and "grpo" not in overlay and "loss_fn" not in overlay
    for partition, expected in (("train", originals[:2]), ("validation", originals[2:])):
        records = [json.loads(line) for line in (output / f"{partition}.jsonl").read_text().splitlines()]
        assert [row["task_id"]["task_id"] for row in records] == [row["task_id"] for row in expected]
        for index, (record, original) in enumerate(zip(records, expected)):
            taskset = record["task_id"]["taskset"]
            assert taskset == "terminal-bench:" + harnesses[index % len(harnesses)]
            instance = graph[graph["environment_server_routes"][taskset]]
            options = instance["environment_servers"]["single_agent_turn" if route == "sat" else "harbor"]
            assert options["agent_server"]["name"] == harnesses[index % len(harnesses)]
            params = record["task_input"]["responses_create_params"]
            assert params["input"] == original["responses_create_params"]["input"]
            assert params["max_output_tokens"] == 128
            if route == "sat":
                parsed = SingleAgentTurnTaskInput.model_validate(record["task_input"])
                assert parsed.task_data["docker_image"] == original["docker_image"]
                assert parsed.task_data["task_folder"] == original["task_folder"]
            else:
                assert options["tasksets"][taskset]["tasks"][record["task_id"]["task_id"]]


def test_harbor_refuses_changed_instruction_instead_of_changing_the_task(tmp_path, monkeypatch):
    tasks, _, loaded, source, validation, base = inputs(tmp_path)
    monkeypatch.setattr(
        EXAMPLE, "prepare_target", lambda *a, **k: SimpleNamespace(tasks=loaded, folder=tasks, skipped={})
    )
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["responses_create_params"]["input"][0]["content"] = "Different instruction"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="prompt/image differs"):
        EXAMPLE.prepare(
            source=source,
            validation_source=validation,
            tasks=tasks,
            base=base,
            output=tmp_path / "output",
            route="harbor",
            harnesses=["opencode_agent"],
            sandbox="opensandbox",
        )
