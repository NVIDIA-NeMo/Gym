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
from nemo_gym.rollouts.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _start_or_resume_store,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest


# A prepared benchmark writes its repeat count as copies of each row: two copies of each of three tasks.
ROWS = [
    {"task_id": task_id, "responses_create_params": {"input": f"q-{task_id}"}}
    for task_id in ("a", "b", "c")
    for _ in range(2)
]


# The servers the rows run on, as the run's global config names them.
SERVERS = {
    "my_agent": {"responses_api_agents": {"simple_agent": {"model_server": {"name": "policy_model"}}}},
    "policy_model": {"responses_api_models": {"vllm_model": {"model": "demo-model"}}},
}


def run_config(tmp_path: Path, *, resume: bool = False, **selection) -> RolloutCollectionConfig:
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in ROWS))
    return RolloutCollectionConfig(
        agent_name="my_agent",
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "results" / "rollouts.jsonl"),
        num_repeats=1,
        resume_from_cache=resume,
        **selection,
    )


def record_path(tmp_path: Path) -> Path:
    return tmp_path / "results" / "rollouts_task_selection.json"


def record(tmp_path: Path) -> dict:
    return json.loads(record_path(tmp_path).read_text())


def start(tmp_path: Path, config: RolloutCollectionConfig) -> None:
    """Start or resume the run as the collector does: its inputs prepared, checked and recorded by the store."""

    def prepare_inputs(selection):
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config, selection)
        return rows, RunManifest.create(Path(config.input_jsonl_fpath), rows, config.model_dump(), SERVERS)

    with _start_or_resume_store(Path(config.output_jsonl_fpath), config, prepare_inputs):
        pass


def test_task_ids_run_every_copy_of_the_named_tasks(tmp_path: Path) -> None:
    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids=["c", "a"]))

    assert [row["task_id"] for row in rows] == ["a", "a", "c", "c"]


def test_a_fresh_subset_run_records_its_selection(tmp_path: Path) -> None:
    start(tmp_path, run_config(tmp_path, task_ids=["c", "a"]))

    assert record(tmp_path) == {
        "source": "task_ids",
        "tasks": 2,
        "rows": 4,
        "task_ids_sha256": task_ids_digest(["a", "c"]),
        "task_ids": ["a", "c"],
    }


def test_a_task_id_file_selects_like_the_inline_list_and_names_its_file(tmp_path: Path) -> None:
    ids = tmp_path / "subset_v1.txt"
    ids.write_text("# subset v1\nb\n\n c \n  # retired: a\n")
    config = run_config(tmp_path, task_ids_fpath=str(ids))

    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
    start(tmp_path, config)

    assert [row["task_id"] for row in rows] == ["b", "b", "c", "c"]
    assert record(tmp_path)["source"] == str(ids)
    assert record(tmp_path)["task_ids_sha256"] == task_ids_digest(["b", "c"])


def test_computing_the_rows_writes_nothing(tmp_path: Path) -> None:
    # Read paths (the run summary, a resume check) compute the expected rows too: they must not touch the record.
    record_path(tmp_path).parent.mkdir(parents=True)
    record_path(tmp_path).write_text('{"saved": true}')

    RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path, task_ids=["b"]))
    RolloutCollectionHelper._preprocess_rows_from_config(None, run_config(tmp_path))

    assert record(tmp_path) == {"saved": True}


@pytest.mark.parametrize("resumed", [{"task_ids": ["b"]}, {}])
def test_a_refused_resume_keeps_the_saved_record(tmp_path: Path, resumed: dict) -> None:
    start(tmp_path, run_config(tmp_path, task_ids=["a"]))
    saved = record(tmp_path)

    with pytest.raises(ConfigError, match="semantic settings differ"):
        start(tmp_path, run_config(tmp_path, resume=True, **resumed))

    assert record(tmp_path) == saved


def test_an_accepted_resume_keeps_the_saved_record(tmp_path: Path) -> None:
    ids = tmp_path / "subset.txt"
    ids.write_text("a\n")
    start(tmp_path, run_config(tmp_path, task_ids_fpath=str(ids)))
    saved = record_path(tmp_path).read_bytes()

    start(tmp_path, run_config(tmp_path, resume=True, task_ids_fpath=str(ids)))

    assert record_path(tmp_path).read_bytes() == saved


def test_a_resume_after_a_crash_before_the_record_write_records_the_selection(tmp_path: Path) -> None:
    config = run_config(tmp_path, task_ids=["a"])
    start(tmp_path, config)
    written = record_path(tmp_path).read_bytes()
    record_path(tmp_path).unlink()

    start(tmp_path, run_config(tmp_path, resume=True, task_ids=["a"]))

    assert record_path(tmp_path).read_bytes() == written


def test_a_resume_without_a_selection_writes_no_record(tmp_path: Path) -> None:
    start(tmp_path, run_config(tmp_path))

    start(tmp_path, run_config(tmp_path, resume=True))

    assert not record_path(tmp_path).exists()


def test_inline_ids_hydra_read_as_numbers_or_booleans_are_ids(tmp_path: Path) -> None:
    # Hydra reads an unquoted `7` in `+task_ids=[7]` as an int and `true` as a bool.
    assert run_config(tmp_path, task_ids=[7, True, "x"]).task_ids == ["7", "true", "x"]


def test_a_numeric_inline_id_matches_an_id_stored_as_a_number(tmp_path: Path) -> None:
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


def test_a_fresh_run_of_every_task_removes_a_record_an_earlier_run_left(tmp_path: Path) -> None:
    start(tmp_path, run_config(tmp_path, task_ids=["a"]))

    start(tmp_path, run_config(tmp_path))

    assert not record_path(tmp_path).exists()


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"task_ids": ["a"], "limit": 2}, "Set only one of task_ids, task_ids_fpath and limit"),
        ({"task_ids": ["a"], "task_ids_fpath": "ids.txt"}, "Set only one of task_ids, task_ids_fpath and limit"),
        ({"task_ids_fpath": "ids.txt", "limit": 2}, "Set only one of task_ids, task_ids_fpath and limit"),
        ({"task_ids": []}, "task_ids is empty"),
        ({"task_ids_fpath": ""}, "task_ids_fpath is empty"),
        ({"task_ids": [1e3]}, "was read as a number"),
    ],
)
def test_one_selection_at_a_time(tmp_path: Path, selection: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        run_config(tmp_path, **selection)
