# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A strict run that was rejected as incomplete recomputes no metrics after its re-judge pass until it is complete."""

from types import SimpleNamespace

import pytest

from nemo_gym.cli import eval as eval_cli


@pytest.fixture
def pass_settles_one_row(monkeypatch):
    aggregations = []

    async def rejudge(run_config, output):
        return 1

    class Helper:
        async def run_from_config(self, config):
            aggregations.append(config)

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr("nemo_gym.rollouts.rollout_collection.RolloutAggregationHelper", Helper)
    monkeypatch.setattr(
        "nemo_gym.rollouts.rollout_collection.RolloutAggregationConfig",
        SimpleNamespace(model_validate=lambda values: values),
    )
    return aggregations


@pytest.mark.parametrize(
    "complete_only, run_is_complete, recomputed",
    [
        pytest.param(True, False, 0, id="rejected-and-still-incomplete"),
        pytest.param(True, True, 1, id="rejected-and-completed-by-the-pass"),
        pytest.param(False, False, 1, id="not-a-rejection"),
    ],
)
def test_metrics_are_recomputed_only_for_a_run_that_may_publish_them(
    tmp_path, monkeypatch, pass_settles_one_row, complete_only, run_is_complete, recomputed
) -> None:
    reads = []
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: reads.append(output) or run_is_complete)

    returned = eval_cli._rejudge_saved_judge_failures(
        {},
        tmp_path / "rollouts.jsonl",
        SimpleNamespace(disable_aggregation=False),
        complete_only=complete_only,
    )

    assert len(pass_settles_one_row) == recomputed
    # Completeness is read once, after the pass, and returned; a non-rejection never reads it.
    assert len(reads) == (1 if complete_only else 0)
    assert returned is (complete_only and run_is_complete)


@pytest.mark.parametrize(
    "complete_only, run_is_complete, recomputed",
    [
        pytest.param(True, False, 0, id="rejected-and-still-incomplete"),
        pytest.param(True, True, 1, id="rejected-and-completed-before-the-pass-raised"),
        pytest.param(False, True, 0, id="not-a-rejection-keeps-log-and-return"),
    ],
)
def test_a_pass_that_raises_still_lets_a_completed_strict_run_publish_its_metrics(
    tmp_path, monkeypatch, pass_settles_one_row, complete_only, run_is_complete, recomputed
) -> None:
    async def rejudge(run_config, output):
        raise OSError("disk full")

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: run_is_complete)

    eval_cli._rejudge_saved_judge_failures(
        {},
        tmp_path / "rollouts.jsonl",
        SimpleNamespace(disable_aggregation=False),
        complete_only=complete_only,
    )

    assert len(pass_settles_one_row) == recomputed


def test_disabled_aggregation_is_still_respected_when_the_pass_raised(
    tmp_path, monkeypatch, pass_settles_one_row
) -> None:
    async def rejudge(run_config, output):
        raise OSError("disk full")

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", rejudge)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda output: True)

    eval_cli._rejudge_saved_judge_failures(
        {}, tmp_path / "rollouts.jsonl", SimpleNamespace(disable_aggregation=True), complete_only=True
    )

    assert pass_settles_one_row == []
