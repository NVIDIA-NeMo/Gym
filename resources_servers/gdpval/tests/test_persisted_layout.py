# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from nemo_gym.config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME
from nemo_gym.core import EpisodeId, maybe_rollout_id_from_run_body
from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper
from resources_servers.gdpval.multistage_orchestrator import build_stage_rows, index_rows_by_task
from resources_servers.gdpval.persisted_layout import repeat_dir, repeat_key_from_rollout_id


EXAMPLE_DATA = Path(__file__).parents[1] / "data" / "example.jsonl"


def test_the_repeat_of_a_gdpval_rollout_is_its_rollout_index(tmp_path: Path) -> None:
    """The episode a session sees names its repeat, for every row shape GDPVal dispatches."""
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(EXAMPLE_DATA),
        output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
        agent_name="gdpval_stirrup_agent",
        num_repeats=3,
    )
    rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
    rows_by_task = index_rows_by_task(rows)
    # A later stage offsets the task index; an in-process retry or resume adds an attempt.
    stage_rows = build_stage_rows(rows_by_task, {task_id: "reference" for task_id in rows_by_task}, stage_index=1)
    dispatched = [*rows, *stage_rows, *({**row, ATTEMPT_INDEX_KEY_NAME: 2} for row in stage_rows)]

    assert {row[ROLLOUT_INDEX_KEY_NAME] for row in rows} == {0, 1, 2}
    for row in dispatched:
        episode_id = EpisodeId.from_capture_key(maybe_rollout_id_from_run_body(row))
        assert repeat_key_from_rollout_id(episode_id.rollout_id) == str(row[ROLLOUT_INDEX_KEY_NAME])


@pytest.mark.parametrize(
    "rollout_id",
    [
        "explicit",
        "task-x",
        "batch-7",
        "x-3",
        "0-x",
        "0-1-2",
        "1-",
        "0",
        "12",
        "job-job_01m6pvh1afeeaacts1mfkr3jtw-sample-1",
    ],
)
def test_a_rollout_id_that_is_not_two_integers_keys_its_own_repeat_directory(rollout_id: str, tmp_path: Path) -> None:
    """An id another orchestrator assigns names its rollout's directory whole, apart from Gym's repeats."""
    assert repeat_key_from_rollout_id(rollout_id) == f"id_{rollout_id}"
    directory = repeat_dir(tmp_path, "task-1", repeat_key_from_rollout_id(EpisodeId(rollout_id=rollout_id).rollout_id))
    assert directory == tmp_path / "task_task-1" / f"repeat_id_{rollout_id}"


def test_gym_rollout_ids_keep_their_repeat_directories(tmp_path: Path) -> None:
    assert repeat_dir(tmp_path, "task-1", repeat_key_from_rollout_id("7-0")).name == "repeat_0"
    assert repeat_dir(tmp_path, "task-1", repeat_key_from_rollout_id("0-12")).name == "repeat_12"


def test_an_external_numeric_id_and_a_gym_id_with_that_rollout_index_get_different_directories() -> None:
    assert repeat_key_from_rollout_id("0") != repeat_key_from_rollout_id("7-0")
    assert repeat_key_from_rollout_id("id_0") != repeat_key_from_rollout_id("0")
