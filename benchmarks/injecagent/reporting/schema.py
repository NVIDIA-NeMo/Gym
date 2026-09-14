# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed contracts shared by the package builder, BLADE exporter, and model-card writer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


OutcomeClass = Literal[
    "policy_scored",
    "invalid_model_output",
    "infrastructure_failure",
    "judge_missing",
    "pending_annotation",
]
MetricKind = Literal["primary", "component", "diagnostic", "operational"]
Direction = Literal["lower_is_better", "higher_is_better", "neutral"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricValue(_Strict):
    # A metric key, optionally followed by "/<slice path>"; slice values are free text (category names
    # may contain spaces, parentheses, or slashes) but must not contain newlines.
    id: str = Field(pattern=r"^[a-z0-9_]+(/[^\n]+)?$")
    name: str
    value: float | None
    numerator: int | None = None
    denominator: int | None = None
    unit: Literal["rate", "count", "tokens", "seconds"] = "rate"
    direction: Direction
    kind: MetricKind
    slice: dict[str, str] | None = None
    definition: str


class OutcomeCounts(_Strict):
    expected_tasks: int
    expected_rollouts: int
    scored_rollouts: int
    policy_scored: int
    invalid_model_output: int
    infrastructure_failure: int
    judge_missing: int
    pending_annotation: int
    missing_task_ids: list[str] = Field(default_factory=list)
    duplicate_task_ids: list[str] = Field(default_factory=list)


class RolloutSummary(_Strict):
    task_id: str
    rollout_id: str
    receipt_id: str
    slices: dict[str, str]
    outcome_class: OutcomeClass
    reward: float | None
    facts: dict[str, Any]
    prompt_excerpt: str
    response_excerpt: str


class AnchorFact(_Strict):
    id: str
    category: Literal[
        "outcome", "slice", "behavior", "invalid", "infrastructure", "calibration", "coverage", "example"
    ]
    fact: str
    evidence: list[str] = Field(min_length=1)
    excerpt: str | None = None


class CalibrationSummary(_Strict):
    method: str
    cases: int
    agreement: int
    disagreements: int
    notes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ReferenceComparison(_Strict):
    label: str
    value: str
    source: str
    comparability: str


class NormalizedRun(_Strict):
    schema_version: int = 1
    benchmark: dict[str, Any]
    run: dict[str, Any]
    outcomes: OutcomeCounts
    metrics: list[MetricValue]
    rollouts: list[RolloutSummary]
    anchor_facts: list[AnchorFact]
    calibration: CalibrationSummary | None = None
    limitations: list[str]
    reference_comparisons: list[ReferenceComparison] = Field(default_factory=list)
    reward_is_semantic: bool
    reward_semantics: str

    def metric(self, metric_id: str) -> MetricValue:
        for metric in self.metrics:
            if metric.id == metric_id:
                return metric
        raise KeyError(metric_id)


# --------------------------------------------------------------------------- model-card document


class GroundedClaim(_Strict):
    text: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=1, description="Anchor-fact ids that support this sentence.")


class BehavioralExample(_Strict):
    anchor_id: str
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)


class ModelCardDocument(_Strict):
    """What the report-writing model is allowed to produce. Every number and example must trace back."""

    purpose: str = Field(min_length=1, max_length=400, description="One sentence: what the benchmark measures.")
    what_the_run_says: list[GroundedClaim] = Field(min_length=2, max_length=3)
    examples: list[BehavioralExample] = Field(max_length=4, description="One entry per required example anchor id.")
    calibration_and_reference: list[GroundedClaim] = Field(min_length=1, max_length=3)
    limitations: list[GroundedClaim] = Field(min_length=2, max_length=5)
    interpretation: str = Field(min_length=1, max_length=1400, description="Two to four complete sentences.")
