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

Configuration lists follow source order, including agent selectors and aggregate overrides.
Pooled candidate sources produce one result side; candidate-varying result fields follow
`ComparisonResult.candidates`. Comparison currently supports one candidate side.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from nemo_gym.config.config_types import BaseNeMoGymCLIConfig


ReportFormat = Literal["md", "json", "both"]

# v0 compares a single candidate. The plumbing is list-shaped throughout so raising this is a
# one-line change plus renderer work, not a schema migration.
MAX_CANDIDATES = 1

# Flips shown per direction in the markdown table, and the cap `diff.py` uses to decide which
# flips' per-repeat reward lists are worth keeping in the JSON. Shared so the two can't drift.
MAX_FLIPS_SHOWN = 10


class ComparisonConfig(BaseNeMoGymCLIConfig):
    """Compare baseline and candidate eval runs, optionally pooling independent collections.

    Reads only each run's `<stem>_aggregate_metrics.json`, derived from the rollouts JSONL path you
    pass, and writes `compare_report.md` and/or `compare_report.json`. The rollouts JSONL itself is
    never opened -- it is the run's identity and the handle from which the sibling path is derived. When
    a legacy aggregate lacks repeat statistics, they are reconstructed in memory from its embedded
    rollout summaries for this comparison.

    Examples:

    ```bash
    gym eval compare \
        --baseline outputs/run_a/rollouts.jsonl \
        --candidates outputs/run_b/rollouts.jsonl

    gym eval compare \
        --baseline-runs outputs/b1/rollouts.jsonl,outputs/b2/rollouts.jsonl \
        --candidate-runs outputs/c1/rollouts.jsonl,outputs/c2/rollouts.jsonl
    ```

    To point at metrics files that do not follow the `<stem>_aggregate_metrics.json` convention,
    use --baseline-agg-metrics, --baseline-runs-agg-metrics and --candidates-agg-metrics
    in the corresponding source order. Each pooled repeat is an equally weighted observation.
    """

    baseline_rollouts_jsonl_fpath: Optional[str] = Field(
        default=None,
        description="Baseline run's rollouts JSONL, as passed to `gym eval run --output`. Used to derive "
        "`<stem>_aggregate_metrics.json`; the JSONL itself is not read.",
    )
    candidate_rollouts_jsonl_fpaths: List[str] = Field(
        default_factory=list,
        description="Candidate runs' rollouts JSONL paths (comma-separated via --candidates).",
    )
    baseline_rollouts_jsonl_fpaths: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        description="Independent baseline runs to pool (--baseline-runs). Mutually exclusive with --baseline.",
    )
    pooled_candidate_rollouts_jsonl_fpaths: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        description="Independent candidate runs to pool as one candidate (--candidate-runs). "
        "Mutually exclusive with --candidates.",
    )
    baseline_aggregate_metrics_fpath: Optional[str] = Field(
        default=None,
        description="Override for the baseline's aggregate-metrics JSON. Defaults to the "
        "`<stem>_aggregate_metrics.json` sibling of baseline_rollouts_jsonl_fpath.",
    )
    baseline_aggregate_metrics_fpaths: Optional[List[str]] = Field(
        default=None,
        description="Aggregate-metrics overrides, one per --baseline-runs source in the same order.",
    )
    candidate_aggregate_metrics_fpaths: Optional[List[str]] = Field(
        default=None,
        description="Overrides for the candidates' aggregate-metrics JSON files. When set, must have the "
        "same length and order as --candidates or --candidate-runs.",
    )

    agent_name: Optional[str] = Field(
        default=None,
        description="Agent to compare on every side. When unset, compares every agent name present in all "
        "runs. Overridden by baseline_agent_name, baseline_agent_names or candidate_agent_names.",
    )
    baseline_agent_name: Optional[str] = Field(
        default=None,
        description="Agent to read from every baseline source. Takes precedence over agent_name.",
    )
    baseline_agent_names: Optional[List[str]] = Field(
        default=None,
        description="Agent names, one per --baseline-runs source. Overrides baseline_agent_name and agent_name.",
    )
    candidate_agent_names: Optional[List[str]] = Field(
        default=None,
        description="Agent to read from each candidate's metrics, by position. When set, must have the same "
        "length and order as --candidates or --candidate-runs. Takes precedence over agent_name.",
    )

    output_dirpath: Optional[str] = Field(
        default=None,
        description="Directory to write the comparison report into. Defaults to the last candidate source's "
        "directory. Created if absent; existing report files are overwritten.",
    )
    report_format: ReportFormat = Field(
        default="both",
        description="Which report artifacts to write: `md`, `json`, or `both`.",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        description="Confidence level for candidate-minus-baseline intervals, as a fraction between 0 and 1. "
        "Side intervals stay at 95% and are recomputed for pooled sides.",
    )

    @model_validator(mode="after")
    def _check_candidate_parallel_lists(self) -> "ComparisonConfig":
        if (self.baseline_rollouts_jsonl_fpath is not None) == (self.baseline_rollouts_jsonl_fpaths is not None):
            raise ValueError("Give exactly one of --baseline or --baseline-runs.")
        if bool(self.candidate_rollouts_jsonl_fpaths) == (self.pooled_candidate_rollouts_jsonl_fpaths is not None):
            raise ValueError("Give exactly one of --candidates or --candidate-runs.")
        if self.baseline_rollouts_jsonl_fpaths is None:
            if self.baseline_aggregate_metrics_fpaths is not None or self.baseline_agent_names is not None:
                raise ValueError("baseline_aggregate_metrics_fpaths and baseline_agent_names require --baseline-runs.")
        else:
            if self.baseline_aggregate_metrics_fpath is not None:
                raise ValueError("Use baseline_aggregate_metrics_fpaths with --baseline-runs.")
            for field_name in ("baseline_aggregate_metrics_fpaths", "baseline_agent_names"):
                value = getattr(self, field_name)
                if value is not None and len(value) != len(self.baseline_rollouts_jsonl_fpaths):
                    raise ValueError(
                        f"{field_name} must have one entry per --baseline-runs source, in the same order."
                    )
        num_candidates = len(self.pooled_candidate_rollouts_jsonl_fpaths or self.candidate_rollouts_jsonl_fpaths)
        if self.pooled_candidate_rollouts_jsonl_fpaths is None and num_candidates > MAX_CANDIDATES:
            raise ValueError(
                f"{num_candidates} candidates were given, but comparing more than {MAX_CANDIDATES} candidate "
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


class MetricValue(BaseModel):
    """One side's reading of a metric, plus whatever uncertainty the run recorded for it."""

    value: float
    # `ci_{low,high}_95_across_repeats/<metric>`: written for eligible point metrics,
    # including benchmark-defined metrics, on runs with >= 2 repeat values.
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
    # Two-sided Welch t-interval for candidate minus baseline, computed from each run's
    # repeat-level values. Only available when both runs have `mean_across_repeats/<metric>`.
    delta_ci_low: Optional[float] = None
    delta_ci_high: Optional[float] = None


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


class RunFile(BaseModel):
    """Parsed aggregates indexed by agent; pooled paths represent the first of authoritative `sources`."""

    model_config = ConfigDict(frozen=True)

    role: Literal["baseline", "candidate"]
    index: int = 0
    rollouts_jsonl_fpath: Path
    aggregate_metrics_fpath: Path
    # The whole parsed payload. Typed `Any` so pydantic does not walk a multi-megabyte file, and
    # excluded so the report references the run rather than embedding its input.
    entries_by_agent: Dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    sources: List["RunSource"] = Field(default_factory=list)

    @computed_field
    @property
    def label(self) -> str:
        """Short display name for the run: the directory its rollouts live in."""
        if self.sources:
            return f"pooled ({len(self.sources)} sources)"
        return self.rollouts_jsonl_fpath.parent.name or self.rollouts_jsonl_fpath.stem

    @computed_field
    @property
    def agent_names(self) -> List[str]:
        return sorted(self.entries_by_agent)


class RunSource(BaseModel):
    """An input artifact contributing independent repeats to a pooled side."""

    model_config = ConfigDict(frozen=True)

    rollouts_jsonl_fpath: Path
    aggregate_metrics_fpath: Path
    selected_agents: List[str] = Field(default_factory=list)
    repeat_counts_by_agent: Dict[str, Optional[int]] = Field(
        default_factory=dict, description="Recorded repeat rows, including accounting-only repeats."
    )
    scored_repeat_counts_by_agent: Dict[str, int] = Field(default_factory=dict)
    task_counts_by_agent: Dict[str, int] = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    """The machine-readable artifact written as `compare_report.json`."""

    schema_version: Literal["1"] = "1"
    generated_at: str
    nemo_gym_version: str
    command: str
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        description="Confidence level of delta intervals. Side intervals stay at 95%, recomputed for pooled sides.",
    )
    baseline: RunFile
    candidates: List[RunFile]
    comparisons: List[AgentComparison] = Field(default_factory=list)
    skipped_agents: Dict[str, List[str]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
