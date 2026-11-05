# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grading results and failure kinds. The contract types live in task_data.py and are re-exported here."""

from typing import Literal

from pydantic import Field, StrictBool

# Contract types live in task_data.py, which Gym's data tooling loads without server dependencies.
from resources_servers.partner_if.task_data import Dimension, StrictModel, Text
from resources_servers.partner_if.task_data import GradingContract as GradingContract
from resources_servers.partner_if.task_data import GradingItem as GradingItem
from resources_servers.partner_if.task_data import VerifierSpec as VerifierSpec


# Namespaced failure kinds (see nemo_gym.core.failure_kinds). NO_ANSWER and TRUNCATED are the
# model's own failures and score zero; the others mask the rollout.
NO_ANSWER = "partner_if:no_answer"
TRUNCATED = "partner_if:truncated"
UNSCORABLE = "partner_if:unscorable"
UNSIGNED_CONTRACT = "partner_if:unsigned_contract"
EVIDENCE_TOO_LARGE = "partner_if:evidence_too_large"


class EvidenceQuote(StrictModel):
    source: Text
    quote: Text


class RawJudgment(StrictModel):
    id: Text
    passed: StrictBool | None
    reason: Text
    evidence: Text
    quotes: list[EvidenceQuote] = Field(default_factory=list)
    # The failure kind behind a verdict that no check or judge gave, such as NO_ANSWER.
    failure_kind: str | None = Field(default=None, exclude_if=lambda value: value is None)


class ItemResult(RawJudgment):
    raw_passed: StrictBool | None
    verifier: Text
    dimension: Dimension
    blocked_by: list[str] = Field(default_factory=list)


class DimensionScore(StrictModel):
    points_earned: int | None
    points_possible: int = Field(gt=0)
    partial_score: float | None
    binary_score: int | None


class Evaluation(StrictModel):
    # Availability of the IF headline; completion availability lives in dimensions.
    status: Literal["scored", "unscorable"]
    items: list[ItemResult]
    prerequisites: list[ItemResult] = Field(default_factory=list)
    dimensions: dict[Dimension, DimensionScore]
    reward: float | None
