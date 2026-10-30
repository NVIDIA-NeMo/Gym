# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the music schema through Gym's dataset validation protocol."""

from pathlib import Path

import pytest

from nemo_gym.data.task_data import TaskDataValidator, load_task_data_schema, validate_jsonl_rows


SERVER_DIR = Path(__file__).resolve().parents[1]


def validator() -> TaskDataValidator:
    adapter = load_task_data_schema(SERVER_DIR)
    assert adapter is not None
    return TaskDataValidator(server_name="mimo_music", adapter=adapter, dataset_fpath="example.jsonl")


def test_committed_tasks_validate_cleanly() -> None:
    adapter = load_task_data_schema(SERVER_DIR)
    assert adapter is not None
    example = SERVER_DIR / "data" / "example.jsonl"
    report = validate_jsonl_rows("mimo_music", adapter, str(example), example.read_text().splitlines())
    assert report.rows == 5
    assert report.clean, report.summary()


@pytest.mark.parametrize("container", [None, "verifier_metadata", "task_data"])
def test_provenance_validates_in_supported_row_formats(container: str | None) -> None:
    fields = {"task_id": "music-1", "source": "synthetic-smoke"}
    row = {container: fields} if container else fields
    check = validator()
    check.validate_row(0, {"responses_create_params": {}, **row})
    assert check.report.clean, check.report.summary()


@pytest.mark.parametrize("metadata", [{}, {"task_id": None, "source": None}])
def test_provenance_is_optional(metadata: dict) -> None:
    check = validator()
    check.validate_row(0, {"responses_create_params": {}, "verifier_metadata": metadata})
    assert check.report.clean, check.report.summary()


@pytest.mark.parametrize("field", ["task_id", "source"])
def test_invalid_provenance_types_are_reported(field: str) -> None:
    check = validator()
    check.validate_row(0, {"verifier_metadata": {field: ["not-a-string"]}})
    assert check.report.error_rows == 1
    assert field in check.report.errors[0]


def test_unknown_metadata_is_preserved_and_reported() -> None:
    adapter = load_task_data_schema(SERVER_DIR)
    assert adapter is not None
    metadata = {"task_id": "music-1", "dataset_revision": "v2"}
    assert adapter.validate_python(metadata).model_dump()["dataset_revision"] == "v2"
    check = validator()
    check.validate_row(0, {"verifier_metadata": metadata})
    assert check.report.error_rows == 0
    assert check.report.unknown_keys == {"dataset_revision": 1}
    assert not check.report.clean
