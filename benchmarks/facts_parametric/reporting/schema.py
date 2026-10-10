# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed contracts shared by the run-package builder, the BLADE exporter, and the model-card writer.

A run is first normalized into ``NormalizedRun``: every metric carries its denominator, direction, and role; every
rollout carries its outcome class, receipt ids, and component facts; every anchor fact cites task/rollout/receipt
ids. Everything downstream (BLADE files, deterministic report sections, the grounded prose) reads only this
structure, so the numbers in the package cannot drift from the reconciled run.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


OutcomeClass = Literal[
    "scored",  # the official verifier/judge produced a score for this rollout
    "judge_failed",  # the judge/verifier call failed (transport, parse exhaustion); no score
    "simulation_failed",  # a simulator/helper model failed before the target could be scored
    "infrastructure_failed",  # agent/request/model-server failure; no score
    "missing",  # expected but never observed
]
MetricKind = Literal["primary", "component", "slice", "diagnostic", "operational"]
Direction = Literal["higher_is_better", "lower_is_better", "neutral"]
MetricUnit = Literal["rate", "score", "percent", "count", "tokens", "seconds"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricValue(_Strict):
    id: str = Field(pattern=r"^[a-z0-9_]+(/[^\n]+)?$")
    name: str
    value: float | None
    numerator: float | None = None
    denominator: int | None = None
    unit: MetricUnit = "rate"
    direction: Direction
    kind: MetricKind
    slice: dict[str, str] | None = None
    definition: str
    ci95: tuple[float, float] | None = None


class OutcomeCounts(_Strict):
    expected_tasks: int
    expected_rollouts: int
    scored_rollouts: int
    judge_failed: int
    simulation_failed: int
    infrastructure_failed: int
    missing_rollouts: int
    duplicate_rollouts: int
    replaced_attempts: int = Field(
        description="Sidecar attempts superseded by a later scored attempt of the same rollout."
    )
    missing_task_ids: list[str] = Field(default_factory=list)
    duplicate_task_ids: list[str] = Field(default_factory=list)


class RolloutSummary(_Strict):
    task_id: str
    rollout_id: str
    receipt_ids: list[str] = Field(default_factory=list)
    slices: dict[str, str] = Field(default_factory=dict)
    outcome_class: OutcomeClass
    reward: float | None
    components: dict[str, Any] = Field(default_factory=dict)
    prompt_excerpt: str = ""
    response_excerpt: str = ""
    judge_excerpt: str = ""


class AnchorFact(_Strict):
    id: str
    category: Literal[
        "outcome", "slice", "behavior", "invalid", "infrastructure", "calibration", "coverage", "example"
    ]
    fact: str
    evidence: list[str] = Field(
        min_length=1, description="Task ids, rollout ids, receipt ids, or calibration case ids."
    )
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


class BladeMapping(_Strict):
    d1_metrics: dict[str, str] = Field(description="BLADE D1 metric name -> normalized metric id or 'N/A: reason'.")
    d2_anchor_categories: list[str]
    d3_notes: list[str] = Field(default_factory=list)
    not_applicable: dict[str, str] = Field(default_factory=dict, description="BLADE concept -> why it does not apply.")


class NormalizedRun(_Strict):
    schema_version: int = 2
    benchmark: dict[str, Any]
    run: dict[str, Any]
    outcomes: OutcomeCounts
    metrics: list[MetricValue]
    rollouts: list[RolloutSummary]
    anchor_facts: list[AnchorFact]
    calibration: CalibrationSummary | None = None
    limitations: list[str]
    reference_comparisons: list[ReferenceComparison] = Field(default_factory=list)
    blade: BladeMapping
    reward_semantics: str

    def metric(self, metric_id: str) -> MetricValue:
        for metric in self.metrics:
            if metric.id == metric_id:
                return metric
        raise KeyError(metric_id)

    def primary_metric(self) -> MetricValue:
        for metric in self.metrics:
            if metric.kind == "primary":
                return metric
        raise KeyError("no primary metric")


# --------------------------------------------------------------------------- model-card document (writer output)


class GroundedClaim(_Strict):
    text: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(min_length=1, description="Anchor-fact ids that support this sentence.")


class BehavioralExample(_Strict):
    anchor_id: str
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)


class ModelCardDocument(_Strict):
    """What the report-writing model may produce. Every number and example must trace back to the run."""

    purpose: str = Field(min_length=1, max_length=400, description="One sentence: what the benchmark measures.")
    what_the_run_says: list[GroundedClaim] = Field(min_length=2, max_length=3)
    examples: list[BehavioralExample] = Field(max_length=4)
    calibration_and_reference: list[GroundedClaim] = Field(min_length=1, max_length=3)
    limitations: list[GroundedClaim] = Field(min_length=2, max_length=5)
    interpretation: str = Field(min_length=1, max_length=1400, description="Two to four complete sentences.")
