# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The config and report envelope every statistical test builds on."""

from typing import ClassVar, List, Literal, Optional

from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from nemo_gym.config_types import RunSelectionConfig


ReportFormat = Literal["md", "json", "both"]
MAX_CANDIDATES = 1
STATS_SUBDIR_NAME = "statistical_tests"
DEFAULT_STAT_TEST = "paired-t-test"


class StatTestConfig(RunSelectionConfig):
    MAX_CANDIDATES: ClassVar[int] = MAX_CANDIDATES

    test: str = Field(default=DEFAULT_STAT_TEST, description="Which statistical test to run.")
    output_dirpath: Optional[str] = Field(
        default=None, description=f"Report directory. Defaults to `<candidate run dir>/{STATS_SUBDIR_NAME}/`."
    )
    report_format: ReportFormat = Field(default="both", description="Artifacts to write: `md`, `json`, or `both`.")
    alpha: float = Field(default=0.05, description="Significance level.")

    @model_validator(mode="after")
    def _check_alpha(self) -> "StatTestConfig":
        if not (0 < self.alpha < 1):
            raise ValueError(f"--alpha must be between 0 and 1, exclusive (got {self.alpha}).")
        return self

    def filename_parts(self) -> List[str]:
        return []


class StatTestResult(BaseModel):
    """One metric's outcome. A test subclasses this with its own statistics and renders them in `line()`.

    That method is the only test-specific part of a report: `nemo_gym.statistical_tests.common` renders
    and summarises everything around it for any test.
    """

    metric: str

    def line(self) -> str:
        raise NotImplementedError


class StatTestReport(BaseModel):
    schema_version: Literal["1"] = "1"
    generated_at: str
    nemo_gym_version: str
    command: str
    test: str
    baseline_rollouts_jsonl_fpath: str
    baseline_aggregate_metrics_fpath: str
    candidate_rollouts_jsonl_fpath: str
    candidate_aggregate_metrics_fpath: str
    baseline_agent: str
    candidate_agent: str
    baseline_task_count: int
    candidate_task_count: int
    results: List[SerializeAsAny[StatTestResult]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
