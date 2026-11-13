# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``task_ids`` and ``task_ids_fpath`` run only the named tasks of a dataset and record which ones ran."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.config.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.data.task_selection import task_ids_digest
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


# A prepared benchmark writes its repeat count as copies of each row: two copies of each of three tasks.
ROWS = [
    {"task_id": task_id, "responses_create_params": {"input": f"q-{task_id}"}}
    for task_id in ("a", "b", "c")
    for _ in range(2)
]


def run_config(tmp_path: Path, **selection) -> RolloutCollectionConfig:
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
    return RolloutCollectionConfig(
        agent_name="my_agent",
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "results" / "rollouts.jsonl"),
        num_repeats=1,
        **selection,
    )


def record(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "results" / "rollouts_task_selection.json").read_text())


def test_task_ids_run_every_copy_of_the_named_tasks_and_record_them(tmp_path: Path) -> None:
    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids=["c", "a"]))

    assert [row["task_id"] for row in rows] == ["a", "a", "c", "c"]
    assert record(tmp_path) == {
        "source": "task_ids",
        "tasks": 2,
        "rows": 4,
        "task_ids_sha256": task_ids_digest(["a", "c"]),
        "task_ids": ["a", "c"],
    }


def test_a_task_id_file_selects_like_the_inline_list_and_names_its_file(tmp_path: Path) -> None:
    ids = tmp_path / "subset_v1.txt"
    ids.write_text("b\n\n c \n")

    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids_fpath=str(ids)))

    assert [row["task_id"] for row in rows] == ["b", "b", "c", "c"]
    assert record(tmp_path)["source"] == str(ids)
    assert record(tmp_path)["task_ids_sha256"] == task_ids_digest(["b", "c"])


def test_a_numeric_inline_id_matches_an_id_stored_as_a_number(tmp_path: Path) -> None:
    # Hydra reads an unquoted `7` in `+task_ids=[7]` as a number.
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(json.dumps({"problem_id": n, "responses_create_params": {"input": "q"}}) + "\n" for n in (7, 8))
    )
    config = RolloutCollectionConfig(
        agent_name="my_agent",
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        task_ids=[7],
    )

    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    assert [row["problem_id"] for row in rows] == [7]


def test_an_id_the_dataset_does_not_hold_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="No input rows for task ids"):
        RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids=["a", "zz"]))


@pytest.mark.parametrize("content", ["", "\n  \n"])
def test_an_empty_task_id_file_stops_the_run(tmp_path: Path, content: str) -> None:
    ids = tmp_path / "subset.txt"
    ids.write_text(content)
    with pytest.raises(ConfigError, match="lists no task ids"):
        RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids_fpath=str(ids)))


def test_a_missing_task_id_file_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(ConfigPathNotFoundError, match="task_ids_fpath"):
        RolloutCollectionHelper._preprocess_rows_from_config(
            None, run_config(tmp_path, task_ids_fpath=str(tmp_path / "missing.txt"))
        )


def test_a_run_of_every_task_removes_a_selection_record_an_earlier_run_left(tmp_path: Path) -> None:
    RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids=["a"]))

    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path))

    assert len(rows) == len(ROWS)
    assert not (tmp_path / "results" / "rollouts_task_selection.json").exists()


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"task_ids": ["a"], "limit": 2}, "Set only one of task_ids, task_ids_fpath and limit"),
        ({"task_ids": ["a"], "task_ids_fpath": "ids.txt"}, "Set only one of task_ids, task_ids_fpath and limit"),
        ({"task_ids_fpath": "ids.txt", "limit": 2}, "Set only one of task_ids, task_ids_fpath and limit"),
        ({"task_ids": []}, "task_ids is empty"),
    ],
)
def test_one_selection_at_a_time(tmp_path: Path, selection: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        run_config(tmp_path, **selection)
