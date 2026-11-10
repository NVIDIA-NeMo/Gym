# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused checks for Gym-V's dependency-light task-data contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.data import TaskDataValidator, load_task_data_schema, validate_jsonl_rows


SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_DIR.parents[1]


def _adapter():
    adapter = load_task_data_schema(SERVER_DIR)
    assert adapter is not None
    return adapter


@pytest.mark.parametrize(
    "example_path",
    [
        SERVER_DIR / "data" / "example.jsonl",
        REPO_ROOT / "environments" / "gym_v" / "data" / "example.jsonl",
    ],
)
def test_shipped_example_rows_validate_cleanly(example_path: Path) -> None:
    report = validate_jsonl_rows(
        "gym_v",
        _adapter(),
        str(example_path),
        example_path.read_text().splitlines(),
    )

    assert report.rows == 5
    assert report.clean, report.summary()


@pytest.mark.parametrize("missing", ["env_id", "seed"])
def test_wire_required_fields_remain_required(missing: str) -> None:
    row = {"env_id": "Games/FrozenLake-v0", "seed": 1234}
    row.pop(missing)

    with pytest.raises(ValidationError):
        _adapter().validate_python(row)


def test_optional_task_fields_and_horizon_constraint() -> None:
    parsed = _adapter().validate_python(
        {
            "env_id": "Games/FrozenLake-v0",
            "env_kwargs": {"size": 4},
            "seed": 1234,
            "task_id": "frozenlake-seed-1234",
            "horizon_cap": 30,
            "task_metadata": {"category": "Games"},
            "success_criterion": {
                "type": "terminal_step_reward",
                "comparison": "equal",
                "threshold": 1.0,
            },
            "task_idx": 7,
        }
    )

    assert parsed.horizon_cap == 30
    assert parsed.task_idx == 7
    assert parsed.success_criterion.type == "terminal_step_reward"
    with pytest.raises(ValidationError):
        _adapter().validate_python({"env_id": "Games/FrozenLake-v0", "seed": 1234, "horizon_cap": 0})
    with pytest.raises(ValidationError):
        _adapter().validate_python({"env_id": "Games/FrozenLake-v0", "seed": 1234, "task_idx": -1})


def test_unknown_task_key_is_reported() -> None:
    validator = TaskDataValidator(
        server_name="gym_v",
        adapter=_adapter(),
        dataset_fpath="example.jsonl",
    )
    validator.validate_row(0, {"env_id": "Games/FrozenLake-v0", "seed": 1234, "sead": 1234})

    assert validator.report.error_rows == 0
    assert validator.report.unknown_keys == {"sead": 1}
