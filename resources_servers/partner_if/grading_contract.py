# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-independent scoring contract and internal evidence/error results."""

from typing import Literal

from pydantic import Field, StrictBool

# Task schemas must be self-contained for Gym's dependency-light data tooling.
# Re-export existing names so execution adapters retain their public import paths.
from resources_servers.partner_if.task_data import Dimension, StrictModel, Text
from resources_servers.partner_if.task_data import GradingContract as GradingContract
from resources_servers.partner_if.task_data import GradingItem as GradingItem
from resources_servers.partner_if.task_data import VerifierSpec as VerifierSpec


class EvidenceQuote(StrictModel):
    source: Text
    quote: Text


class RawJudgment(StrictModel):
    id: Text
    passed: StrictBool | None
    reason: Text
    evidence: Text
    quotes: list[EvidenceQuote] = Field(default_factory=list)


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
