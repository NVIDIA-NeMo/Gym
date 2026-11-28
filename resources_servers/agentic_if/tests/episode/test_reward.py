# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reward of a verify result under each reward mode: the task verdict, or the task verdict times the IF score."""

from dataclasses import replace

import pytest

from resources_servers.agentic_if.episode.outcome import Verdict
from resources_servers.agentic_if.episode.reward import IF_ERROR, IF_NOT_APPLICABLE, reward_verdict
from resources_servers.agentic_if.if_grading.contracts import CheckResult, IFResult


def task(reward: float, **failure: object) -> Verdict:
    return Verdict(reward=reward, mask_sample=False, evaluation_completed=True, **failure)


def graded(*statuses: str) -> IFResult:
    """An IF result as the grading engine aggregates its checks' statuses."""
    checks = [
        CheckResult(id=f"check-{index}", status=status, explanation=status) for index, status in enumerate(statuses)
    ]
    applicable = [status for status in statuses if status != "not_applicable"]
    valid = bool(applicable) and "error" not in applicable
    return IFResult(
        checks=checks,
        if_all=int(all(status == "pass" for status in applicable)) if valid else None,
        if_frac=round(applicable.count("pass") / len(applicable), 3) if valid else None,
    )


def rewarded(
    verdict: Verdict,
    if_result: IFResult,
    *,
    mode: str = "task_and_if",
    if_score: str = "if_all",
    cut_run_reward: str = "as_is",
    outcome_tracked: bool = True,
) -> Verdict:
    return reward_verdict(
        verdict,
        if_result,
        mode=mode,
        if_score=if_score,
        cut_run_reward=cut_run_reward,
        outcome_tracked=outcome_tracked,
    )


@pytest.mark.parametrize(
    "verdict, if_result",
    [
        (task(1.0), graded("fail")),
        (task(0.0), graded("not_applicable")),
        (task(1.0), graded("error")),
        (Verdict(reward=1.0, mask_sample=True, evaluation_completed=True, failure_kind="agent_run_error"), graded()),
    ],
)
@pytest.mark.parametrize("outcome_tracked", [True, False])
def test_the_task_mode_keeps_the_task_verdict_whatever_the_if_result(verdict, if_result, outcome_tracked):
    assert rewarded(verdict, if_result, mode="task", outcome_tracked=outcome_tracked) is verdict


@pytest.mark.parametrize(
    "task_reward, statuses, if_score, reward",
    [
        (1.0, ["pass"], "if_all", 1.0),
        (1.0, ["pass", "fail"], "if_all", 0.0),
        (0.0, ["pass"], "if_all", 0.0),
        (1.0, ["pass", "fail"], "if_frac", 0.5),
        (0.0, ["pass", "fail"], "if_frac", 0.0),
        # A check that did not apply to the run is left out of the IF score.
        (1.0, ["pass", "not_applicable"], "if_all", 1.0),
    ],
)
def test_a_tracked_outcome_earns_the_task_verdict_times_the_if_score(task_reward, statuses, if_score, reward):
    result = rewarded(task(task_reward), graded(*statuses), if_score=if_score)
    assert result == task(reward)


@pytest.mark.parametrize(
    "task_reward, statuses, if_score, reward",
    [
        (0.0, ["pass"], "if_all", 1.0),
        (1.0, ["fail"], "if_all", 0.0),
        (0.0, ["pass", "fail"], "if_frac", 0.5),
    ],
)
def test_an_untracked_outcome_earns_the_if_score_alone(task_reward, statuses, if_score, reward):
    result = rewarded(task(task_reward), graded(*statuses), if_score=if_score, outcome_tracked=False)
    assert result == task(reward)


@pytest.mark.parametrize("if_score", ["if_all", "if_frac"])
@pytest.mark.parametrize("outcome_tracked", [True, False])
@pytest.mark.parametrize("task_reward", [1.0, 0.0])
def test_a_run_without_an_if_score_is_masked_whatever_its_task_verdict(task_reward, outcome_tracked, if_score):
    for statuses, kind, reason in (
        (["not_applicable", "not_applicable"], IF_NOT_APPLICABLE, "None of the row's 2 instruction checks applied"),
        ([], IF_NOT_APPLICABLE, "The row has no instruction checks"),
        (["pass", "error"], IF_ERROR, "Instruction following could not be graded: check-1: error"),
    ):
        result = rewarded(task(task_reward), graded(*statuses), if_score=if_score, outcome_tracked=outcome_tracked)
        assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, True, kind)
        assert result.failure_reason.startswith(reason)
        # The task verdict was measured: the mask is a training decision, not a missing measurement.
        assert result.evaluation_completed is True


def test_a_masked_task_verdict_stays_masked_with_its_own_failure():
    verdict = replace(task(1.0, failure_kind="model_request_error", failure_reason="HTTP 400"), mask_sample=True)
    for if_result in (graded("pass"), graded("error")):
        assert rewarded(verdict, if_result) == replace(verdict, reward=0.0)


CUT_RUNS = [
    task(1.0, failure_kind="agent_timeout", failure_reason="OpenCode stopped at its harness timeout"),
    task(1.0, failure_kind="model_context_limit", failure_reason="prompt too long"),
]


@pytest.mark.parametrize("cut", CUT_RUNS)
def test_a_cut_run_is_scored_as_it_stands_by_default(cut):
    assert rewarded(cut, graded("pass"), mode="task") is cut
    assert rewarded(cut, graded("pass")) == cut
    assert rewarded(cut, graded("fail")) == replace(cut, reward=0.0)
    # An interrupted tail usually leaves the evidence incomplete, so its checks errored: no IF score, masked.
    assert rewarded(cut, graded("error")).failure_kind == IF_ERROR


@pytest.mark.parametrize("mode", ["task", "task_and_if"])
@pytest.mark.parametrize("cut", CUT_RUNS)
def test_cut_run_reward_zero_gives_a_cut_run_nothing_under_either_mode(cut, mode):
    for if_result in (graded("pass"), graded("error")):
        # Unmasked, IF score or not: running out of time or context trains as a failure, and the kind says which.
        assert rewarded(cut, if_result, mode=mode, cut_run_reward="zero") == replace(cut, reward=0.0)


@pytest.mark.parametrize("mode", ["task", "task_and_if"])
def test_cut_run_reward_zero_leaves_runs_that_ended_by_themselves_and_masked_runs_alone(mode):
    finished = task(1.0)
    assert rewarded(finished, graded("pass"), mode=mode, cut_run_reward="zero") == finished
    # A masked sample is no measurement, whatever its kind: the cut-run rule only applies to scored runs.
    masked = Verdict(reward=1.0, mask_sample=True, evaluation_completed=True, failure_kind="agent_timeout")
    expected = masked if mode == "task" else replace(masked, reward=0.0)
    assert rewarded(masked, graded("pass"), mode=mode, cut_run_reward="zero") == expected


def test_an_error_reason_is_bounded():
    if_result = graded("error")
    if_result.checks[0].explanation = "x" * 5000
    assert len(rewarded(task(1.0), if_result).failure_reason) == 2000
