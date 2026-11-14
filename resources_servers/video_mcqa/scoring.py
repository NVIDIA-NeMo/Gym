# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Question-level accuracy with explicit parser policy and failure accounting."""

import re
from collections import Counter
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal

from resources_servers.video_mcqa.task import LABELS, Setting, VerifierMetadata


AnswerPolicy = Literal["boxed_only_v1", "boxed_or_letter_v1"]
Failure = Literal["execution_failure", "infrastructure_failure"]
Outcome = Literal[
    "correct",
    "incorrect",
    "no_answer",
    "format_error",
    "ambiguous_answer",
    "execution_failure",
    "infrastructure_failure",
]
_POLICIES = ("boxed_only_v1", "boxed_or_letter_v1")
_OUTCOMES = (
    "correct",
    "incorrect",
    "no_answer",
    "format_error",
    "ambiguous_answer",
    "execution_failure",
    "infrastructure_failure",
)
_BOXED = re.compile(r"\\boxed\s*\{\s*(?:([A-Z])|\\(?:text|mathrm)\s*\{\s*([A-Z])\s*\})\s*\}")
# Boxed letters count up to H, as for the eight-option benchmarks, or up to the question's last option.
_EIGHT_OPTIONS = 8
_THOUGHT = re.compile(r"<(think|thinking)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# Response metadata key the OpenCode video agent sets to "true" when its run failed. Gym grades such a run on an
# emptied copy while the saved row keeps the original answer, so replaying that row reads the failure from here.
EXECUTION_FAILURE_METADATA_KEY = "video_execution_failure"


@dataclass(frozen=True, slots=True)
class QuestionScore:
    """One attempt; infrastructure failures are explicitly inconclusive."""

    task_id: str
    dataset_revision: str
    setting: Setting
    answer_policy: AnswerPolicy
    outcome: Outcome
    predicted_answer: str | None
    reward: float | None
    evaluation_completed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a nonempty string")
        if not isinstance(self.dataset_revision, str) or re.fullmatch(r"[0-9a-f]{40}", self.dataset_revision) is None:
            raise ValueError("dataset_revision must be a fixed source commit")
        if self.setting not in ("single_turn", "agentic") or self.answer_policy not in _POLICIES:
            raise ValueError("Unknown setting or answer policy")
        if self.outcome not in _OUTCOMES:
            raise ValueError("Unknown outcome")
        if self.predicted_answer is not None and self.predicted_answer not in tuple(LABELS):
            raise ValueError("predicted_answer must be one uppercase choice label")
        completed = self.outcome != "infrastructure_failure"
        expected_reward = (1.0 if self.outcome == "correct" else 0.0) if completed else None
        if (
            isinstance(self.reward, bool)
            or self.evaluation_completed is not completed
            or self.reward != expected_reward
        ):
            raise ValueError("Outcome, reward and completion status disagree")
        if (self.outcome in ("correct", "incorrect")) != (self.predicted_answer is not None):
            raise ValueError("Only a valid parsed answer may be marked correct or incorrect")


@dataclass(frozen=True, slots=True)
class AccuracySummary:
    """Fixed-cohort accuracy, provisional while any row is missing/inconclusive.

    This is not the benchmark's nonlinear/group score or a certification claim.
    Different settings and extraction policies must be aggregated separately.
    """

    setting: Setting
    dataset_revision: str
    answer_policy: AnswerPolicy
    expected: int
    returned: int
    completed: int
    correct: int
    missing: int
    avg_accuracy: float
    coverage: float
    is_complete: bool
    outcome_counts: dict[str, int]


def _final_text(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("final_text must be a string or None")
    text = _THOUGHT.sub("", text)
    if re.search(r"<(?:think|thinking)\b", text, re.IGNORECASE):
        return ""  # An unfinished reasoning block is not a final answer.
    return re.split(r"</(?:think|thinking)\s*>", text, flags=re.IGNORECASE)[-1].strip()


def grade_answer(
    final_text: str | None,
    metadata: VerifierMetadata,
    *,
    setting: Setting,
    answer_policy: AnswerPolicy = "boxed_only_v1",
    failure: Failure | None = None,
) -> QuestionScore:
    """Grade FINAL assistant text only, never a transcript or tool responses.

    ``boxed_or_letter_v1`` additionally accepts an exact standalone uppercase
    letter. It does not search prose for a convenient letter or invoke a judge.
    A setup/transport failure is inconclusive; a completed execution failure is
    zero even if its partial text happens to contain the correct answer.
    """
    if setting not in ("single_turn", "agentic") or answer_policy not in _POLICIES:
        raise ValueError("Unknown setting or answer policy")
    if failure not in (None, "execution_failure", "infrastructure_failure"):
        raise ValueError("Unknown failure classification")
    predicted = None
    if failure is not None:
        outcome = failure
    else:
        text = _final_text(final_text)
        counted = set(LABELS[: max(_EIGHT_OPTIONS, metadata.option_count)])
        letters = {plain or wrapped for plain, wrapped in _BOXED.findall(text)} & counted
        if not text:
            outcome = "no_answer"
        elif len(letters) > 1:
            outcome = "ambiguous_answer"
        else:
            if letters:
                predicted = next(iter(letters))
            elif answer_policy == "boxed_or_letter_v1" and text in tuple(LABELS):
                predicted = text
            if predicted not in tuple(LABELS[: metadata.option_count]):
                predicted = None
                outcome = "format_error"
            else:
                outcome = "correct" if predicted == metadata.expected_answer else "incorrect"
    completed = outcome != "infrastructure_failure"
    reward = (1.0 if outcome == "correct" else 0.0) if completed else None
    return QuestionScore(
        metadata.task_id, metadata.dataset_revision, setting, answer_policy, outcome, predicted, reward, completed
    )


def aggregate_scores(
    scores: Iterable[QuestionScore],
    *,
    expected_task_ids: Collection[str],
    dataset_revision: str,
    setting: Setting,
    answer_policy: AnswerPolicy = "boxed_only_v1",
) -> AccuracySummary:
    """Account for every frozen ID, rejecting duplicate or mixed-protocol rows.

    Missing and infrastructure-failed tasks remain in the accuracy denominator;
    coverage and is_complete expose that the resulting number is provisional.
    """
    if setting not in ("single_turn", "agentic") or answer_policy not in _POLICIES:
        raise ValueError("Unknown setting or answer policy")
    if not isinstance(dataset_revision, str) or re.fullmatch(r"[0-9a-f]{40}", dataset_revision) is None:
        raise ValueError("dataset_revision must be a fixed source commit")
    if isinstance(expected_task_ids, (str, bytes)):
        raise ValueError("Expected a collection of task IDs, not a string")
    expected = set(expected_task_ids)
    if not expected or len(expected) != len(expected_task_ids):
        raise ValueError("Expected cohort must be nonempty and contain unique IDs")
    if any(not isinstance(identity, str) or not identity.strip() for identity in expected):
        raise ValueError("Expected IDs must be nonempty strings")
    seen = set()
    counts = Counter({outcome: 0 for outcome in _OUTCOMES})
    for score in scores:
        if (score.setting, score.answer_policy, score.dataset_revision) != (setting, answer_policy, dataset_revision):
            raise ValueError("Do not mix settings, extraction policies or source revisions in one metric")
        if score.task_id not in expected or score.task_id in seen:
            raise ValueError("Unexpected or duplicate task result")
        seen.add(score.task_id)
        counts[score.outcome] += 1
    completed = len(seen) - counts["infrastructure_failure"]
    missing = len(expected) - len(seen)
    return AccuracySummary(
        setting=setting,
        dataset_revision=dataset_revision,
        answer_policy=answer_policy,
        expected=len(expected),
        returned=len(seen),
        completed=completed,
        correct=counts["correct"],
        missing=missing,
        avg_accuracy=counts["correct"] / len(expected),
        coverage=completed / len(expected),
        is_complete=completed == len(expected),
        outcome_counts=dict(counts),
    )
