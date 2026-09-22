# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from pytest import MonkeyPatch

import resources_servers.deepswe_external1.prepare_examples as module
from nemo_gym.task_data import TaskDataValidator
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
    rows = module.prepare_examples(
        source_dir=tmp_path / "source", tasks_dir=tmp_path / "cache/tasks", output_path=output, allow_download=False
    )
    assert json.loads(output.read_text()) == rows[0]
    assert rows[0]["responses_create_params"]["input"][0]["content"] == task.instruction
    assert rows[0]["public_source"]["revision"] == module.DEEPSWE_SOURCE_REVISION
    assert rows[0]["public_source"]["upstream_project"] == "https://github.com/example/project"
    assert json.loads((output.parent / "example_metrics.json").read_text()) == {"Number of examples": 1}
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
