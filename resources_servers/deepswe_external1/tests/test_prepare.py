# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter
from pytest import MonkeyPatch

import resources_servers.deepswe_external1.prepare_examples as module
from nemo_gym.config_types import DatasetConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.task_data import TaskDataValidator
from nemo_gym.train_data_utils import TrainDataProcessor
from resources_servers.deepswe_external1.task_data import TaskData
from resources_servers.deepswe_external1.task_store import PreparedTask


def test_public_preparation_keeps_original_prompt_and_separates_assets(
    task: PreparedTask, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    task.config.metadata["repository_url"] = "https://github.com/example/project"
    monkeypatch.setattr(module, "EXAMPLE_TASK_IDS", (task.definition.task_id,))
    monkeypatch.setattr(module, "ensure_source", lambda source, allow_download: source)

    class Store:
        def __init__(self, path: Path) -> None:
            assert path.name == "tasks"

        def get(self, task_id: str) -> PreparedTask:
            assert task_id == task.definition.task_id
            return task

    monkeypatch.setattr(module, "DeepSWETaskStore", Store)
    output = tmp_path / "public/example.jsonl"
    output.parent.mkdir()
    metrics_path = output.parent / "example_metrics.json"
    existing_metrics = '{"Number of examples": 1, "Number of turns": {"Average": 1.0}}\n'
    metrics_path.write_text(existing_metrics)
    rows = module.prepare_examples(
        source_dir=tmp_path / "source", tasks_dir=tmp_path / "cache/tasks", output_path=output, allow_download=False
    )
    assert json.loads(output.read_text()) == rows[0]
    assert rows[0]["responses_create_params"]["input"][0]["content"] == task.instruction
    assert rows[0]["public_source"]["revision"] == module.DEEPSWE_SOURCE_REVISION
    assert rows[0]["public_source"]["upstream_project"] == "https://github.com/example/project"
    assert metrics_path.read_text() == existing_metrics
    cache = tmp_path / "cache/tasks" / task.definition.task_id
    assert (cache / "solution/solve.sh").read_bytes() == task.asset_path("solution/solve.sh").read_bytes()
    assert set(rows[0]) == {
        "task_id",
        "image",
        "task_fingerprint",
        "responses_create_params",
        "verifier_metadata",
        "public_source",
    }
    validator = TaskDataValidator(
        server_name="deepswe_external1", adapter=TypeAdapter(TaskData), dataset_fpath=str(output)
    )
    validator.validate_row(0, rows[0])
    assert validator.report.clean and not validator.report.unknown_keys


def test_public_cli_and_row_schema(task: PreparedTask, tmp_path: Path, monkeypatch: MonkeyPatch, capsys) -> None:
    row = module.task_row(task)
    validated = TaskData.model_validate(row)
    assert validated.task_id == task.definition.task_id
    assert validated.task_fingerprint == task.definition.fingerprint()
    with pytest.raises(ValueError):
        TaskData.model_validate(row | {"task_fingerprint": "invalid"})
    calls = []

    def prepare(**kwargs):
        calls.append(kwargs)
        return [row]

    monkeypatch.setattr(module, "prepare_examples", prepare)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_examples",
            "--source-dir",
            str(tmp_path / "source"),
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--output",
            str(tmp_path / "example.jsonl"),
            "--no-download",
        ],
    )
    module.main()
    assert calls == [
        {
            "source_dir": tmp_path / "source",
            "tasks_dir": tmp_path / "tasks",
            "output_path": tmp_path / "example.jsonl",
            "allow_download": False,
        }
    ]
    assert "does not execute or validate" in capsys.readouterr().out


def test_committed_metrics_match_standard_dataset_statistics() -> None:
    config_path = module.PACKAGE_DIR / "configs/deepswe_external1.yaml"
    config = yaml.safe_load(config_path.read_text())
    settings = config["deepswe_external1_resources_server"]["resources_servers"]["deepswe_external1"]
    dataset = DatasetConfig.model_validate(settings["datasets"][0])
    absolute_dataset = dataset.model_copy(update={"jsonl_fpath": str(module.PACKAGE_DIR / "data/example.jsonl")})
    state = TrainDataProcessor()._validate_samples_and_aggregate_metrics_single_dataset(absolute_dataset)
    assert not state.offending_example_idxs
    expected = dataset.model_dump(mode="json", exclude={"agent"}) | state.metrics.aggregate().model_dump(
        mode="json", by_alias=True
    )
    actual = json.loads((module.PACKAGE_DIR / "data/example_metrics.json").read_text())
    assert actual == expected


def test_committed_rollouts_match_public_examples() -> None:
    data = Path(__file__).resolve().parents[1] / "data"
    examples = {row["task_id"]: row for row in map(json.loads, (data / "example.jsonl").read_text().splitlines())}
    rollouts = list(map(json.loads, (data / "example_rollouts.jsonl").read_text().splitlines()))
    assert len(examples) == len(rollouts) == 5
    assert {row["task_id"] for row in rollouts} == set(examples)
    for row in rollouts:
        for key, value in examples[row["task_id"]].items():
            assert row[key] == value
        NeMoGymResponse.model_validate(row["response"])
        assert row["validation_mode"] == "agent"
        assert row["evaluation_completed"] and row["opencode_finished"] and row["opencode_export_found"]
        assert not row.get("mask_sample") and not row.get("failure_kind")
        assert row["reward"] in (0.0, 1.0)
        assert row["rollout_provenance"]["num_repeats"] == 1
        assert any(item["type"] == "function_call" for item in row["response"]["output"])
