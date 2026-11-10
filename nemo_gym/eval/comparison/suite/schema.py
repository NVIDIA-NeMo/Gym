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
"""Manifest, configuration, and report contracts for benchmark-suite comparisons."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.config.config_types import BaseNeMoGymCLIConfig
from nemo_gym.eval.comparison.schema import ReportFormat


class _SuiteModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SuiteComparisonConfig(BaseNeMoGymCLIConfig):
    """Compare baseline and candidate runs across a suite of models and benchmarks.

    Run paths in the suite YAML resolve relative to that file. Reports give
    ``delta = candidate - baseline`` on a 0-100 normalized scale with unpaired Welch intervals.
    """

    suite_fpath: str = Field(description="Suite YAML naming baseline and candidate runs per model and benchmark.")
    confidence_level: float = Field(
        default=0.90, gt=0, lt=1, allow_inf_nan=False, description="Two-sided confidence level for delta intervals."
    )
    output_dirpath: str | None = Field(
        default=None, description="Report directory; defaults to the suite YAML's directory."
    )
    report_format: ReportFormat = Field(default="both", description="Reports to write: md, json, or both.")


class SuiteMetricDefinition(_SuiteModel):
    """A selected repeat metric and an optional complete theoretical scale override."""

    name: str = Field(min_length=1)
    low: float | None = None
    high: float | None = None
    higher_is_better: bool | None = None

    @model_validator(mode="after")
    def validate_complete_override(self) -> "SuiteMetricDefinition":
        given = (self.low is not None, self.high is not None, self.higher_is_better is not None)
        if any(given) and not all(given):
            raise ValueError("Metric scale overrides require low, high, and higher_is_better together.")
        return self


class SuiteBenchmarkDefinition(_SuiteModel):
    """Metrics averaged within each repeat of one benchmark."""

    metrics: list[SuiteMetricDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_metrics(self) -> "SuiteBenchmarkDefinition":
        names = [metric.name for metric in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("A benchmark cannot select the same metric more than once.")
        return self


class SuiteRunSource(_SuiteModel):
    """One run with companion artifacts discovered from its rollout path."""

    rollouts: Path
    agent: str | None = None
    environment_server: str | None = None


class SuiteRunPair(_SuiteModel):
    """Nonempty source collections, accepting a single mapping as shorthand in YAML."""

    baseline: list[SuiteRunSource] = Field(min_length=1)
    candidate: list[SuiteRunSource] = Field(min_length=1)

    @field_validator("baseline", "candidate", mode="before")
    @classmethod
    def normalize_sources(cls, value: object) -> object:
        """Choose the source shape before validation so errors describe only that shape."""
        return [value] if isinstance(value, (Mapping, SuiteRunSource)) else value


class SuiteManifest(_SuiteModel):
    """Explicit benchmark definitions and a possibly sparse model-by-benchmark matrix."""

    schema_version: Literal[1] = 1
    benchmarks: dict[str, SuiteBenchmarkDefinition] = Field(min_length=1)
    models: dict[str, dict[str, SuiteRunPair]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cells(self) -> "SuiteManifest":
        for model, benchmarks in self.models.items():
            if not model.strip() or not benchmarks:
                raise ValueError("Each model needs a name and at least one benchmark comparison.")
            unknown = set(benchmarks) - self.benchmarks.keys()
            if unknown:
                raise ValueError(f"Model '{model}' refers to undefined benchmarks: {', '.join(sorted(unknown))}.")
        return self


class SuiteSampleSummary(_SuiteModel):
    count: int
    mean: float
    sample_variance: float


CoverageStatus = Literal["verified", "unverified", "incomplete"]


class SuiteCoverage(_SuiteModel):
    """Recorded and planned task-coverage evidence for one selected agent."""

    status: CoverageStatus
    recorded_task_count: int
    planned_task_count: int | None = None
    notes: list[str] = Field(default_factory=list)


class SuiteFailureCounts(_SuiteModel):
    """Saved selected failure outcomes for the entire rollout file."""

    total: int = Field(strict=True, ge=0)
    final: int = Field(strict=True, ge=0)
    not_final: int = Field(strict=True, ge=0)
    by_class: dict[str, Annotated[int, Field(strict=True, ge=0)]]
    counted_as_zero: int | None = Field(default=None, strict=True, ge=0)
    final_zeros: int | None = Field(default=None, strict=True, ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> "SuiteFailureCounts":
        if self.final + self.not_final != self.total:
            raise ValueError("Final and not-final failure counts must sum to the total.")
        if sum(self.by_class.values()) != self.total:
            raise ValueError("Failure class counts must sum to the total.")
        if self.counted_as_zero is not None and self.counted_as_zero > self.total:
            raise ValueError("Failures counted as zero must not exceed the total failure count.")
        if self.final_zeros is not None and self.final_zeros > self.final:
            raise ValueError("Final policy zeros must not exceed the final failure count.")
        if (
            self.counted_as_zero is not None
            and self.final_zeros is not None
            and self.counted_as_zero < self.final_zeros
        ):
            raise ValueError("Failures counted as zero must include all final policy zeros.")
        return self


class SuiteFailureSummary(_SuiteModel):
    """Precomputed accounting, without rescanning or certifying the source artifacts."""

    source: Path
    scope: Literal["rollout_file"] = "rollout_file"
    counts: SuiteFailureCounts | None = None
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "SuiteFailureSummary":
        if (self.counts is None) == (self.unavailable_reason is None):
            raise ValueError("Provide exactly one of failure counts and an unavailable reason.")
        return self


class SuiteRunSummary(_SuiteModel):
    """One source or pooled repeats, with full per-source provenance under ``sources``.

    For pooled repeats, path and selector fields identify the first source for
    compatibility. Local repeat identities and saved failure accounting live under ``sources``.
    """

    sources: list["SuiteRunSummary"] = Field(default_factory=list)
    rollouts: Path
    aggregate_metrics: Path
    materialized_inputs: Path | None = None
    agent: str
    environment_server: str | None = None
    repeat_count: int | None
    repeat_indices: list[int] = Field(default_factory=list)
    repeat_scores: list[float] = Field(default_factory=list)
    unscored_repeat_indices: list[int] = Field(default_factory=list)
    summary: SuiteSampleSummary | None = None
    coverage: SuiteCoverage
    failures: SuiteFailureSummary | None = None


def summary_sources(summary: SuiteRunSummary) -> tuple[SuiteRunSummary, ...]:
    """Return constituent provenance, preserving a standalone source's report contract."""
    return tuple(summary.sources) if summary.sources else (summary,)


class SuiteMetricResult(_SuiteModel):
    name: str
    low: float
    high: float
    higher_is_better: bool
    definition_source: str
    baseline_raw: list[float] = Field(default_factory=list)
    candidate_raw: list[float] = Field(default_factory=list)
    baseline_scaled: list[float] = Field(default_factory=list)
    candidate_scaled: list[float] = Field(default_factory=list)


class SuiteVarianceContribution(_SuiteModel):
    weight: float
    baseline: float
    candidate: float


class SuiteStatistics(_SuiteModel):
    """Completed numerical results; presentation only formats these fields."""

    baseline_mean: float
    candidate_mean: float
    delta: float
    variance: float
    standard_error: float
    degrees_of_freedom: float | None
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    contributions: list[SuiteVarianceContribution]
    interval_unavailable_reason: str | None = None


class SuiteCellResult(_SuiteModel):
    model: str
    benchmark: str
    eligible: bool
    exclusion_reason: str | None = None
    baseline: SuiteRunSummary
    candidate: SuiteRunSummary
    metrics: list[SuiteMetricResult]
    statistics: SuiteStatistics | None = None
    model_weight: float = 0.0
    overall_weight: float = 0.0


class SuiteAggregateResult(_SuiteModel):
    name: str
    cell_ids: list[str]
    weights: list[float]
    statistics: SuiteStatistics | None


class SuiteComparisonResult(_SuiteModel):
    """Separate schema for suite_compare_report.json; run reports remain unchanged."""

    schema_version: Literal["1"] = "1"
    generated_at: str
    nemo_gym_version: str
    command: str
    suite_fpath: Path
    confidence_level: float
    cells: list[SuiteCellResult]
    per_model: list[SuiteAggregateResult]
    overall: SuiteAggregateResult
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(
        default_factory=lambda: [
            "Deltas are candidate minus baseline on the normalized score scale; positive deltas favor the candidate.",
            "Benchmarks form a fixed suite; uncertainty describes independent repeat collections.",
            "Baseline and candidate repeats are unpaired and independent within and across cells.",
            "Pooled source runs must use independent repeats, the same tasks, and comparable evaluation settings.",
            "Metrics are equally averaged within each repeat before estimating benchmark variance.",
            "Eligible benchmarks have equal model weights; eligible model-benchmark cells have equal overall weights.",
        ]
    )
