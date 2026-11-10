# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_gym.data import TaskDataValidator, load_task_data_schema, validate_jsonl_rows
from resources_servers.visgym.task_data import TaskData


VISGYM_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_DATASETS = (
    "example.jsonl",
    "matchstick_rotation_easy_starter_scale64_t1024.jsonl",
    "maze_2d_easy_example.jsonl",
    "maze_2d_easy_smoke.jsonl",
    "patch_reassembly_easy_4x4_3patch_scale64_t1024.jsonl",
)


def test_schema_loads_without_visgym_runtime_dependencies() -> None:
    adapter = load_task_data_schema(VISGYM_ROOT)

    assert adapter is not None
    loaded_type = getattr(adapter, "_type", None)
    assert loaded_type is not None
    assert loaded_type.__name__ == "TaskData"
    assert set(loaded_type.model_fields) == set(TaskData.model_fields)


@pytest.mark.parametrize("dataset_name", SHIPPED_DATASETS)
def test_shipped_dataset_rows_validate_cleanly(dataset_name: str) -> None:
    dataset_path = VISGYM_ROOT / "data" / dataset_name
    adapter = load_task_data_schema(VISGYM_ROOT)

    assert adapter is not None
    report = validate_jsonl_rows(
        server_name="visgym",
        adapter=adapter,
        dataset_fpath=str(dataset_path),
        lines=dataset_path.read_text().splitlines(),
    )

    assert report.rows > 0
    assert report.clean, report.summary()


@pytest.mark.parametrize("missing_field", ("env_id", "seed"))
def test_wire_required_fields_remain_required(missing_field: str) -> None:
    row = {"env_id": "maze_2d/easy", "seed": 1234}
    row.pop(missing_field)

    with pytest.raises(ValidationError):
        TaskData.model_validate(row)


def test_all_optional_task_row_fields_validate() -> None:
    row = TaskData.model_validate(
        {
            "env_id": "maze_2d/easy",
            "env_kwargs": {"maze_width": 5, "maze_height": 5},
            "seed": 1234,
            "task_id": "maze_2d_easy_seed1234",
            "act_grammar_regex": r"^\('move', [0-3]\)$",
            "horizon_cap": 8,
            "task_metadata": {"reward_shaping": {"type": "distance_delta", "weight": 0.3}},
            "init_state": {"agent": [1, 1]},
            "seed_key": "seed",
            "prompt_kwargs": {"concise": True},
            "task_idx": 0,
        }
    )

    assert row.model_extra == {}
    assert row.horizon_cap == 8
    assert row.task_idx == 0


def test_horizon_cap_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TaskData.model_validate({"env_id": "maze_2d/easy", "seed": 1234, "horizon_cap": 0})


def test_undeclared_task_field_is_reported() -> None:
    validator = TaskDataValidator(
        server_name="visgym",
        adapter=TypeAdapter(TaskData),
        dataset_fpath="row.jsonl",
    )

    validator.validate_row(0, {"env_id": "maze_2d/easy", "seed": 1234, "env_kwarg": {}})

    assert validator.report.error_rows == 0
    assert validator.report.unknown_keys == {"env_kwarg": 1}
    assert not validator.report.clean
