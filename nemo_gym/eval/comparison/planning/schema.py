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
"""Strict direct-input manifest and completed report contracts for repeat planning."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from nemo_gym.config.config_types import BaseNeMoGymCLIConfig
from nemo_gym.eval.comparison.schema import ReportFormat
from nemo_gym.eval.comparison.statistics.repeat_planning import RepeatAllocation, SimulationResult


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, strict=True)]
StandardDeviation = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class _PlanningModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepeatPlanningCLIConfig(BaseNeMoGymCLIConfig):
    """Plan fixed repeat counts from a YAML of normalized repeat standard deviations."""

    repeat_plan_fpath: str = Field(description="Direct-input repeat-planning YAML (--plan).")
    output_dirpath: str | None = Field(default=None, description="Report directory; defaults to the plan directory.")
    report_format: ReportFormat = Field(default="both", description="Reports to write: md, json, or both.")


class RepeatPlanningBaseline(_PlanningModel):
    """Whether baseline repeats are existing measurements or a planned investment."""

    mode: Literal["optimize", "existing"] = "optimize"


class RepeatPlanningMetric(_PlanningModel):
    """Repeat SDs in normalized score points for one selected benchmark metric."""

    name: Name
    baseline_std_points: StandardDeviation
    candidate_std_points: StandardDeviation | None = None


class RepeatPlanningCell(_PlanningModel):
    """One model and benchmark, with one shared repeat count for its selected metrics.

    Multi-metric cells require cell-level SDs of the normalized, equally averaged
    complete-repeat scores. Their individual metric SDs are descriptive inputs.
    An omitted candidate composite SD defaults to the baseline composite SD.
    """

    model: Name
    benchmark: Name
    baseline_std_points: StandardDeviation | None = None
    candidate_std_points: StandardDeviation | None = None
    metrics: list[RepeatPlanningMetric] | None = Field(default=None, min_length=1)
    repeat_step: int = Field(default=1, strict=True, ge=1)
    repeat_cost: float = Field(default=1.0, strict=True, gt=0, allow_inf_nan=False)
    existing_baseline_repeats: int | None = Field(default=None, strict=True, ge=2)

    @model_validator(mode="after")
    def validate_metrics(self) -> "RepeatPlanningCell":
        if self.metrics is not None:
            names = [metric.name for metric in self.metrics]
            if len(names) != len(set(names)):
                raise ValueError("Metric names must be unique within a benchmark cell.")
            if len(self.metrics) > 1:
                if self.baseline_std_points is None:
                    raise ValueError(
                        "Multiple metrics require cell-level baseline_std_points: supply the SD across complete-repeat "
                        "scores after normalizing and averaging the selected metrics within each repeat. "
                        "Individual metric SDs cannot determine the SD of their average. "
                        "Omit cell-level candidate_std_points to use that baseline composite SD."
                    )
            elif self.baseline_std_points is not None or self.candidate_std_points is not None:
                raise ValueError("For one metric, use either scalar standard deviations or metrics, not both.")
        elif self.baseline_std_points is None:
            raise ValueError("Provide baseline_std_points or a nonempty metrics list.")
        return self

    def metric_definitions(self) -> tuple[RepeatPlanningMetric, ...]:
        """Return the selected metrics, giving scalar inputs the common name ``score``."""
        if self.metrics is not None:
            return tuple(self.metrics)
        return (
            RepeatPlanningMetric(
                name="score",
                baseline_std_points=self.baseline_std_points,
                candidate_std_points=self.candidate_std_points,
            ),
        )


class RepeatPlanningManifest(_PlanningModel):
    """One fixed aggregate planned from direct normalized SD inputs.

    ``benchmark`` selects one model/benchmark cell, ``model`` selects cells of a
    single model, and ``suite`` selects all declared cells. Every cell has equal
    aggregate weight, and its selected metrics have equal within-cell weights.
    """

    schema_version: Literal[1]
    aggregation_level: Literal["benchmark", "model", "suite"]
    baseline: RepeatPlanningBaseline = Field(default_factory=RepeatPlanningBaseline)
    candidate_experiments: int = Field(default=1, strict=True, ge=1)
    margin_points: float = Field(default=1.0, strict=True, gt=0, allow_inf_nan=False)
    target_power: float = Field(default=0.8, strict=True, gt=0, lt=1, allow_inf_nan=False)
    alpha: float = Field(default=0.05, strict=True, gt=0, lt=0.5, allow_inf_nan=False)
    expected_delta_points: float = Field(default=0.0, strict=True, allow_inf_nan=False)
    max_repeats: int = Field(default=20000, strict=True, ge=2)
    validation_trials: int = Field(default=10000, strict=True, ge=0)
    seed: int = Field(default=20261109, strict=True, ge=0)
    cells: list[RepeatPlanningCell] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @model_validator(mode="after")
    def validate_cells(self) -> "RepeatPlanningManifest":
        identities = [(cell.model, cell.benchmark) for cell in self.cells]
        if len(identities) != len(set(identities)):
            raise ValueError("Each model/benchmark cell must be unique.")
        if self.aggregation_level == "benchmark" and len(self.cells) != 1:
            raise ValueError("aggregation_level benchmark requires exactly one model/benchmark cell.")
        if self.aggregation_level == "model" and len({cell.model for cell in self.cells}) != 1:
            raise ValueError("aggregation_level model requires every cell to have the same model.")
        metric_names: dict[str, tuple[str, ...]] = {}
        for cell in self.cells:
            names = tuple(metric.name for metric in cell.metric_definitions())
            previous = metric_names.setdefault(cell.benchmark, names)
            if names != previous:
                raise ValueError(
                    f"Benchmark '{cell.benchmark}' must select the same metric names and order for all models."
                )
            if self.baseline.mode == "existing" and cell.existing_baseline_repeats is None:
                raise ValueError(f"Cell '{cell.model}/{cell.benchmark}' requires existing_baseline_repeats.")
            if self.baseline.mode == "optimize" and cell.existing_baseline_repeats is not None:
                raise ValueError("existing_baseline_repeats is only allowed when baseline.mode is existing.")
        return self


class RepeatPlanningMetricResult(_PlanningModel):
    name: str
    weight: float
    baseline_std_points: float
    candidate_std_points: float
    candidate_std_defaulted: bool


class RepeatPlanningCellResult(_PlanningModel):
    model: str
    benchmark: str
    weight: float
    metrics: list[RepeatPlanningMetricResult]
    baseline_std_points: float
    candidate_std_points: float
    repeat_step: int
    repeat_cost: float
    existing_baseline_repeats: int
    baseline_repeats: int
    new_baseline_repeats: int
    candidate_repeats_per_experiment: int
    candidate_repeats_all_experiments: int
    new_repeat_cost: float


class RepeatPlanningTotals(_PlanningModel):
    existing_baseline_passes: int
    new_baseline_passes: int
    candidate_passes_per_experiment: int
    candidate_passes_all_experiments: int
    total_additional_passes: int
    total_cost: float


class RepeatPlanningResult(_PlanningModel):
    """Completed numerical results and resolved inputs for JSON and Markdown reports."""

    schema_version: Literal["1"] = "1"
    generated_at: str
    nemo_gym_version: str
    command: str
    plan_fpath: Path
    settings: RepeatPlanningManifest
    cells: list[RepeatPlanningCellResult]
    allocation: RepeatAllocation
    simulation: SimulationResult | None
    totals: RepeatPlanningTotals
