# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reward a verify result carries, under the resources server's reward mode.

The default mode, ``task``, keeps the task backend's verdict that ``classify`` reports: the benchmark's reward.
``task_and_if`` is for training on instruction following: the task verdict times the run's IF score, so a run earns
reward only when it does the task and follows the instructions. Under either mode, ``cut_run_reward: zero`` gives a
run that a timeout or a full context window cut short reward 0.0 instead of scoring it as it stands.
"""

from dataclasses import replace
from typing import Literal

from nemo_gym.core import AGENT_TIMEOUT, MODEL_CONTEXT_LIMIT
from resources_servers.agentic_if.episode.outcome import Verdict
from resources_servers.agentic_if.if_grading.contracts import IFResult


RewardMode = Literal["task", "task_and_if"]
IFScore = Literal["if_all", "if_frac"]
CutRunReward = Literal["as_is", "zero"]

# Masks of task_and_if, when the run has no IF score to combine: no check applied to it, or a check could not be
# graded (the `if/not_applicable` and `if/error` counts of the metrics).
IF_NOT_APPLICABLE = "agentic_if:if_not_applicable"
IF_ERROR = "agentic_if:if_error"
# The failure kinds of a scored run that OpenCode's harness timeout or a full context window cut short (classify).
CUT_RUN_KINDS = frozenset({AGENT_TIMEOUT, MODEL_CONTEXT_LIMIT})


def reward_verdict(
    verdict: Verdict,
    if_result: IFResult,
    *,
    mode: RewardMode,
    if_score: IFScore,
    cut_run_reward: CutRunReward,
    outcome_tracked: bool,
) -> Verdict:
    """The verdict with the reward of ``mode``; ``verdict`` is the task verdict ``classify`` reported.

    With ``cut_run_reward: zero``, a scored run that a timeout or a full context window cut short (``agent_timeout``,
    ``model_context_limit``) earns 0.0 under either mode, unmasked and with its failure fields. Otherwise ``task``
    returns the verdict unchanged, and ``task_and_if`` (``if_score`` names the IF score: ``if_all`` or ``if_frac``)
    applies the first rule that fits:

    1. A masked verdict stays masked, with its own failure fields; its reward is 0.0.
    2. A run without an IF score is masked: ``agentic_if:if_not_applicable`` when no check applied to it,
       ``agentic_if:if_error`` when a check could not be graded. Its reward is 0.0, and it is masked whatever the
       task verdict, so that which runs are masked never depends on their task result.
    3. A row sealed with ``outcome_tracked: false`` earns its IF score alone.
    4. Any other run earns the task verdict times its IF score.
    """
    if cut_run_reward == "zero" and not verdict.mask_sample and verdict.failure_kind in CUT_RUN_KINDS:
        return replace(verdict, reward=0.0)
    if mode == "task":
        return verdict
    if verdict.mask_sample:
        return replace(verdict, reward=0.0)
    score = if_result.if_all if if_score == "if_all" else if_result.if_frac
    if score is None:
        errors = [check for check in if_result.checks if check.status == "error"]
        if errors:
            kind = IF_ERROR
            reason = "Instruction following could not be graded: " + "; ".join(
                f"{check.id}: {check.explanation}" for check in errors
            )
        else:
            kind = IF_NOT_APPLICABLE
            reason = (
                f"None of the row's {len(if_result.checks)} instruction checks applied to this run"
                if if_result.checks
                else "The row has no instruction checks"
            )
        return replace(verdict, reward=0.0, mask_sample=True, failure_kind=kind, failure_reason=reason[:2000])
    if not outcome_tracked:
        return replace(verdict, reward=float(score))
    return replace(verdict, reward=verdict.reward * score)
