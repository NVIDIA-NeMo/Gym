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
"""Config and result schema for `gym eval test`.

Deliberately independent of `nemo_gym.comparison.schema` -- this package is its own report
identity (see `nemo_gym/statistical_tests/__init__.py`), not an extension of `compare`'s.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from nemo_gym.config_types import BaseNeMoGymCLIConfig


ReportFormat = Literal["md", "json", "both"]

# v0 tests one candidate against one baseline, matching `gym eval compare`'s own v0 restriction.
MAX_CANDIDATES = 1

# Directory `gym eval test` (and `gym eval compare`'s stats step) write into by default.
STATS_SUBDIR_NAME = "statistical_tests"


class PairedTestConfig(BaseNeMoGymCLIConfig):
    """Paired statistical test of a baseline run against a candidate run.

    Reads only each run's `<stem>_aggregate_metrics.json`, exactly like `gym eval compare`. By
    default every key metric with per-task pairing data is tested; pass `--metric` to restrict to
    a subset.

    Examples:

    ```bash
    # Repeatability: is the difference between two runs of the same config noise, or real?
    gym eval test \
        --baseline outputs/run_a/rollouts.jsonl \
        --candidates outputs/run_b/rollouts.jsonl

    # FP4 quantization: is the candidate not meaningfully worse than a 1pp margin?
    gym eval test \
        --baseline outputs/bf16/rollouts.jsonl \
        --candidates outputs/nvfp4/rollouts.jsonl \
        --metric reward --margin 0.01
    ```
    """

    baseline_rollouts_jsonl_fpath: str = Field(
        description="Baseline run's rollouts JSONL, as passed to `gym eval run --output`. Used to derive "
        "`<stem>_aggregate_metrics.json`; the JSONL itself is not read."
    )
    candidate_rollouts_jsonl_fpaths: List[str] = Field(
        min_length=1,
        description="Candidate run's rollouts JSONL path (comma-separated; one candidate is supported today).",
    )
    baseline_aggregate_metrics_fpath: Optional[str] = Field(
        default=None,
        description="Override for the baseline's aggregate-metrics JSON. Defaults to the "
        "`<stem>_aggregate_metrics.json` sibling of baseline_rollouts_jsonl_fpath.",
    )
    candidate_aggregate_metrics_fpaths: Optional[List[str]] = Field(
        default=None,
        description="Override for the candidate's aggregate-metrics JSON. When set, must have the same "
        "length as candidate_rollouts_jsonl_fpaths.",
    )

    agent_name: Optional[str] = Field(
        default=None,
        description="Agent to compare on both sides. When unset, compares the agent present in both runs.",
    )
    baseline_agent_name: Optional[str] = Field(
        default=None,
        description="Agent to read from the baseline's metrics. Takes precedence over agent_name.",
    )
    candidate_agent_names: Optional[List[str]] = Field(
        default=None,
        description="Agent to read from the candidate's metrics. Takes precedence over agent_name.",
    )

    output_dirpath: Optional[str] = Field(
        default=None,
        description=f"Directory to write the report into. Given explicitly, used literally. Left unset, "
        f"defaults to `<candidate run's directory>/{STATS_SUBDIR_NAME}/` (auto-created).",
    )
    report_format: ReportFormat = Field(
        default="both",
        description="Which report artifacts to write: `md`, `json`, or `both`.",
    )

    metric: Optional[List[str]] = Field(
        default=None,
        description="Metric(s) to test (the bare field name, e.g. `reward`, not `mean/reward`). Unset "
        "tests every key metric with per-task pairing data available.",
    )
    margin: Optional[float] = Field(
        default=None,
        description="Non-inferiority margin `δ` (e.g. 0.01 for 1pp). When set, runs a one-sided test of "
        "H0: candidate is worse than baseline by at least δ. Unset runs a two-sided test of H0: no "
        "difference at all.",
    )
    alpha: float = Field(default=0.05, description="Significance level.")

    @model_validator(mode="after")
    def _check_candidate_parallel_lists(self) -> "PairedTestConfig":
        num_candidates = len(self.candidate_rollouts_jsonl_fpaths)
        if num_candidates > MAX_CANDIDATES:
            raise ValueError(
                f"{num_candidates} candidates were given, but testing more than {MAX_CANDIDATES} candidate "
                "is not supported yet. Give a single candidate run."
            )
        for field_name, value in (
            ("candidate_agent_names", self.candidate_agent_names),
            ("candidate_aggregate_metrics_fpaths", self.candidate_aggregate_metrics_fpaths),
        ):
            if value is not None and len(value) != num_candidates:
                raise ValueError(
                    f"{field_name} has {len(value)} entries but {num_candidates} candidate run(s) were given. "
                    "Give one entry per candidate, in the same order."
                )
        return self

    @model_validator(mode="after")
    def _check_margin_and_alpha(self) -> "PairedTestConfig":
        if self.margin is not None and self.margin <= 0:
            raise ValueError(f"--margin must be a positive number (got {self.margin}).")
        if not (0 < self.alpha < 1):
            raise ValueError(f"--alpha must be between 0 and 1, exclusive (got {self.alpha}).")
        return self


class PairedTestResult(BaseModel):
    """One metric's paired test outcome. Raw statistical output only -- no PASS/WARN/FAIL verdict.

    `mean_diff` is `candidate - baseline`, in the metric's own units (not standardized), so it
    reads consistently next to `gym eval compare`'s delta column.
    """

    metric: str
    margin: Optional[float] = None
    alpha: float
    n_pairs: int
    mean_diff: Optional[float] = None
    se: Optional[float] = None
    p_value: Optional[float] = None
    # Two-sided (margin=None): True means the runs differ. Margin set: True means the candidate is
    # NOT meaningfully worse than the margin allows -- the opposite reading depending on framing.
    significant: Optional[bool] = None
    minimum_detectable_effect: Optional[float] = None
    note: Optional[str] = None


class PairedTestReport(BaseModel):
    """The machine-readable artifact written under `statistical_tests/`."""

    schema_version: Literal["1"] = "1"
    generated_at: str
    nemo_gym_version: str
    command: str
    baseline_rollouts_jsonl_fpath: str
    baseline_aggregate_metrics_fpath: str
    candidate_rollouts_jsonl_fpath: str
    candidate_aggregate_metrics_fpath: str
    baseline_agent: str
    candidate_agent: str
    baseline_task_count: int
    candidate_task_count: int
    results: List[PairedTestResult] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
