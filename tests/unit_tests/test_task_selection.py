# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.task_selection import row_task_id, select_tasks


def _row(task_id: str, prompt: str = "solve it", **extra: object) -> dict:
    return {"task_id": task_id, "responses_create_params": {"input": [{"role": "user", "content": prompt}]}, **extra}


class TestRowTaskId:
    def test_flat_rows_use_task_id_problem_id_instance_id_in_order(self) -> None:
        assert row_task_id({"task_id": "t", "problem_id": "p"}) == "t"
        assert row_task_id({"problem_id": "p", "instance_id": "i"}) == "p"
        assert row_task_id({"instance_id": "i"}) == "i"
        assert row_task_id({"task_name": "x"}) is None

    def test_materialized_rows_use_their_task_id(self) -> None:
        row = {"task_id": {"taskset": "bench", "task_id": "a"}, "task_input": _row("other")}

        assert row_task_id(row) == "a"


class TestSelectTasks:
    def test_selects_first_row_of_each_task_in_input_order(self) -> None:
        rows = [_row("b"), _row("a"), _row("b"), _row("c")]

        assert select_tasks(rows, ["c", "b"]) == [0, 3]

    def test_repeats_differing_only_in_bookkeeping_are_one_task(self) -> None:
        rows = [_row("a", _ng_rollout_index=0), _row("a", _ng_rollout_index=1)]

        assert select_tasks(rows, ["a"]) == [0]

    def test_materialized_rows_are_selected_by_their_task_id(self) -> None:
        rows = [
            {"task_id": {"taskset": "bench", "task_id": "a"}, "task_input": _row("a"), "task_source": "env"},
            {"task_id": {"taskset": "bench", "task_id": "b"}, "task_input": _row("b"), "task_source": "env"},
        ]

        assert select_tasks(rows, ["b"]) == [1]

    def test_refuses_unknown_task_ids(self) -> None:
        with pytest.raises(ConfigError, match=r"No input rows for task ids \['z'\]"):
            select_tasks([_row("a")], ["z"])

    def test_refuses_rows_without_ids(self) -> None:
        with pytest.raises(ConfigError, match="have no task id"):
            select_tasks([_row("a"), {"responses_create_params": {"input": []}}], ["a"])

    def test_refuses_different_rows_with_the_same_id(self) -> None:
        with pytest.raises(ConfigError, match="share a task id"):
            select_tasks([_row("a", prompt="x"), _row("a", prompt="y")], ["a"])
