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
"""Config and result schema for `gym eval compare`.

Every candidate-varying field is a list, positionally parallel to `ComparisonResult.candidates`.
v0 only ever compares one candidate, but keeping the shape list-valued means lifting that
restriction later is a behavior change rather than a schema change.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from nemo_gym.config_types import BaseNeMoGymCLIConfig


ReportFormat = Literal["md", "json", "both"]

# v0 compares a single candidate. The plumbing is list-shaped throughout so raising this is a
# one-line change plus renderer work, not a schema migration.
MAX_CANDIDATES = 1


class CompareConfig(BaseNeMoGymCLIConfig):
    """Compare a baseline eval run against a candidate run.

    Reads only each run's `<stem>_aggregate_metrics.json`, derived from the rollouts JSONL path you
    pass, and writes `compare_report.md` and/or `compare_report.json`. The rollouts JSONL itself is
    never opened -- it is the run's identity and the handle the sibling path is derived from.

    Examples:

    ```bash
    gym eval compare \
        --baseline outputs/run_a/rollouts.jsonl \
        --candidates outputs/run_b/rollouts.jsonl
    ```

    To point at metrics files that do not follow the `<stem>_aggregate_metrics.json` convention,
    pass `--baseline-agg-metrics` and `--candidates-agg-metrics`.
    """

    baseline_rollouts_jsonl_fpath: str = Field(
        description="Baseline run's rollouts JSONL, as passed to `gym eval run --output`. Used to derive "
        "`<stem>_aggregate_metrics.json`; the JSONL itself is not read."
    )
    candidate_rollouts_jsonl_fpaths: List[str] = Field(
        min_length=1,
        description="Candidate runs' rollouts JSONL paths (comma-separated via --candidates).",
    )
    baseline_aggregate_metrics_fpath: Optional[str] = Field(
        default=None,
        description="Override for the baseline's aggregate-metrics JSON. Defaults to the "
        "`<stem>_aggregate_metrics.json` sibling of baseline_rollouts_jsonl_fpath.",
    )
    candidate_aggregate_metrics_fpaths: Optional[List[str]] = Field(
        default=None,
        description="Overrides for the candidates' aggregate-metrics JSON files. When set, must have the "
        "same length as candidate_rollouts_jsonl_fpaths.",
    )

    agent_name: Optional[str] = Field(
        default=None,
        description="Agent to compare on every side. When unset, compares every agent name present in all "
        "runs. Overridden per side by baseline_agent_name / candidate_agent_names.",
    )
    baseline_agent_name: Optional[str] = Field(
        default=None,
        description="Agent to read from the baseline's metrics. Takes precedence over agent_name.",
    )
    candidate_agent_names: Optional[List[str]] = Field(
        default=None,
        description="Agent to read from each candidate's metrics, by position. When set, must have the same "
        "length as candidate_rollouts_jsonl_fpaths. Takes precedence over agent_name.",
    )

    output_dirpath: Optional[str] = Field(
        default=None,
        description="Directory to write the comparison report into. Defaults to the candidate rollouts "
        "file's parent directory. Created if absent; existing report files are overwritten.",
    )
    report_format: ReportFormat = Field(
        default="both",
        description="Which report artifacts to write: `md`, `json`, or `both`.",
    )

    @model_validator(mode="after")
    def _check_candidate_parallel_lists(self) -> "CompareConfig":
        num_candidates = len(self.candidate_rollouts_jsonl_fpaths)
        if num_candidates > MAX_CANDIDATES:
            raise ValueError(
                f"{num_candidates} candidates were given, but comparing more than {MAX_CANDIDATES} candidate "
                "is not supported yet. Pass a single path to --candidates."
            )
        # Name the flag, not just the config key: neither is derivable from the other
        # (`--candidates-agg-metrics` vs `candidate_aggregate_metrics_fpaths`).
        for field_name, value, flag in (
            ("candidate_agent_names", self.candidate_agent_names, "--candidate-agents"),
            (
                "candidate_aggregate_metrics_fpaths",
                self.candidate_aggregate_metrics_fpaths,
                "--candidates-agg-metrics",
            ),
        ):
            if value is not None and len(value) != num_candidates:
                raise ValueError(
                    f"{field_name} has {len(value)} entries but {num_candidates} candidate run(s) were given. "
                    f"Pass {flag} once per candidate, in the same order."
                )
        return self


class MetricValue(BaseModel):
    """One side's reading of a metric, plus whatever uncertainty the run recorded for it."""

    value: float
    # `ci_{low,high}_95_across_repeats/<metric>`: only written for `mean/*` metrics on runs with >= 2
    # repeats, so these are None for pass@k families and single-repeat runs.
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    se_across_repeats: Optional[float] = None
    mean_across_repeats: Optional[float] = None
    # `<metric>/std_err_across_runs` from compute_pass_majority_metrics: carried through for the
    # statistics layer, deliberately not turned into an interval here.
    std_err_across_runs: Optional[float] = None


class CandidateMetricValue(MetricValue):
    """A candidate's reading, plus its movement from the baseline."""

    delta: Optional[float] = None
    delta_pct: Optional[float] = None


class MetricRow(BaseModel):
    metric: str
    is_key_metric: bool
    # Which sides carried this metric: "baseline" and/or "candidate[<i>]".
    present_in: List[str]
    baseline: Optional[MetricValue] = None
    candidates: List[Optional[CandidateMetricValue]] = Field(default_factory=list)


class TaskFlip(BaseModel):
    task_index: int
    baseline_score: float
    candidate_score: float
    delta: float
    direction: Literal["pass_to_fail", "fail_to_pass", "changed"]
    # Per-repeat rewards from `rollout_infos`, purely illustrative -- counts never depend on them.
    baseline_rewards: Optional[List[float]] = None
    candidate_rewards: Optional[List[float]] = None


class FlipSummary(BaseModel):
    candidate_index: int
    mode: Literal["binary", "continuous", "unavailable"]
    reason: Optional[str] = None
    field: str = "reward"
    common_task_count: int = 0
    baseline_only_task_count: int = 0
    candidate_only_task_count: int = 0
    # None in continuous mode, where there is no pass/fail notion to flip.
    pass_to_fail_count: Optional[int] = None
    fail_to_pass_count: Optional[int] = None
    tied_count: Optional[int] = None
    unchanged_count: Optional[int] = None
    net: Optional[int] = None
    flips: List[TaskFlip] = Field(default_factory=list)


class AgentComparison(BaseModel):
    baseline_agent: str
    candidate_agents: List[str]
    baseline_task_count: int
    candidate_task_counts: List[int]
    baseline_repeat_count: Optional[int] = None
    candidate_repeat_counts: List[Optional[int]] = Field(default_factory=list)
    metrics: List[MetricRow] = Field(default_factory=list)
    flips: List[FlipSummary] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RunRef(BaseModel):
    role: Literal["baseline", "candidate"]
    label: str
    rollouts_jsonl_fpath: str
    aggregate_metrics_fpath: str
    agents: List[str] = Field(default_factory=list)
    num_repeats: Optional[int] = None


class ComparisonResult(BaseModel):
    """The machine-readable artifact written as `compare_report.json`."""

    schema_version: Literal["1"] = "1"
    generated_at: str
    nemo_gym_version: str
    command: str
    baseline: RunRef
    candidates: List[RunRef]
    comparisons: List[AgentComparison] = Field(default_factory=list)
    skipped_agents: Dict[str, List[str]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
