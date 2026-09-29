# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `paired-t-test` statistical test (`--test paired-t-test`, default): a paired-difference t-test."""

from math import isfinite
from typing import List, Literal, Optional, Tuple

from pydantic import Field, model_validator
from scipy import stats

from nemo_gym.comparison.loading import LoadedRun
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import MEAN_PREFIX, TASK_INDEX_KEY_NAME
from nemo_gym.statistical_tests.common import fmt, fmt_bool, fmt_p, load_run_pair, sanitize_filename_part
from nemo_gym.statistical_tests.schema import StatTestConfig, StatTestReport, StatTestResult


Alternative = Literal["two-sided", "candidate-lower", "candidate-higher"]

# How this test names itself in its own report, as prose rather than the `--test paired-t-test` slug.
LABEL = "paired t-test"

# Power the reported minimum detectable effect is quoted at.
MDE_POWER = 0.8

# Differences this small are float64 rounding noise rather than signal.
EPS = 1e-12
# Below this share of the larger run's tasks, the two runs overlap too little to conclude anything.
MIN_PAIRED_COVERAGE = 0.5


class PairedTTestConfig(StatTestConfig):
    test: Literal["paired-t-test"] = "paired-t-test"
    metric: Optional[List[str]] = Field(default=None, description="Metric(s) to test, e.g. `reward`.")
    margin: Optional[List[float]] = Field(
        default=None,
        description="Smallest difference(s) that count as a real change, e.g. 0.01 for 1pp (default 0). One "
        "value for every metric, or one per --metric in the same order. Ignored when --alternative is two-sided.",
    )
    alternative: Alternative = Field(
        default="two-sided",
        description="`two-sided`: did anything change at all. `candidate-lower`: the candidate metric dropped "
        "by more than the margin. `candidate-higher`: it rose by more than the margin.",
    )

    @model_validator(mode="after")
    def _check_margin(self) -> "PairedTTestConfig":
        if self.metric:
            self.metric = list(dict.fromkeys(self.metric))
        if not self.margin:
            return self
        if any(not isfinite(m) or m < 0 for m in self.margin):
            raise ValueError(f"--margin must be non-negative and finite (got {self.margin}).")
        if 1 < len(self.margin) != len(self.metric or []):
            raise ValueError("--margin must be either a single value or one per --metric.")
        return self

    def filename_parts(self) -> List[str]:
        parts = [self.alternative]
        if self.margin:
            parts.append("margin-" + "+".join(f"{m:g}" for m in self.margin))
        if self.metric:
            parts.insert(0, "metric-" + "+".join(sanitize_filename_part(m) for m in self.metric))
        return parts


class PairedTTestResult(StatTestResult):
    margin: float = 0.0
    alternative: Alternative = "two-sided"
    alpha: float
    n_pairs: int
    mean_diff: Optional[float] = None
    se: Optional[float] = None
    p_value: Optional[float] = None
    significant: Optional[bool] = None
    # Smallest true effect this sample could detect at `MDE_POWER`, in the metric's own units. Read it
    # alongside a non-significant result: it says whether the test could have found anything.
    minimum_detectable_effect: Optional[float] = None
    note: Optional[str] = None

    def line(self) -> str:
        """Labelled with the test that produced it: several tests' results can end up in one place
        (`compare_report.md`, or one `statistical_tests/` directory).
        """
        if self.p_value is None:
            return f"{self.metric} ({LABEL}): {self.note}"
        return (
            f"{self.metric} ({LABEL}): n={self.n_pairs} mean_diff={fmt(self.mean_diff)} se={fmt(self.se)} "
            f"p={fmt_p(self.p_value)} significant={fmt_bool(self.significant)} "
            f"mde@{MDE_POWER:.0%}-power={fmt(self.minimum_detectable_effect)}"
        )


class PairedTTestReport(StatTestReport):
    results: List[PairedTTestResult] = Field(default_factory=list)


def _groups_by_task(run: LoadedRun) -> dict:
    return {group[TASK_INDEX_KEY_NAME]: group for group in run.group_level_metrics if TASK_INDEX_KEY_NAME in group}


def paired_task_deltas(baseline: LoadedRun, candidate: LoadedRun, metric: str) -> Optional[List[float]]:
    key = f"{MEAN_PREFIX}{metric}"
    baseline_groups, candidate_groups = _groups_by_task(baseline), _groups_by_task(candidate)
    deltas: List[float] = []
    for task_index in sorted(set(baseline_groups) & set(candidate_groups)):
        b, c = baseline_groups[task_index].get(key), candidate_groups[task_index].get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            deltas.append(float(c) - float(b))
    return deltas or None


def resolve_metrics(baseline: LoadedRun, candidate: LoadedRun, requested: Optional[List[str]]) -> Tuple[List, List]:
    if requested:
        return list(dict.fromkeys(requested)), []

    resolved: List[str] = []
    skipped: List[str] = []
    for name in sorted(set(baseline.key_metrics) | set(candidate.key_metrics)):
        if not name.startswith(MEAN_PREFIX):
            skipped.append(name)
            continue
        metric = name[len(MEAN_PREFIX) :]
        (resolved if paired_task_deltas(baseline, candidate, metric) else skipped).append(metric)
    return resolved, skipped


def run_metric(
    baseline: LoadedRun, candidate: LoadedRun, *, metric: str, margin: float, alpha: float, alternative: Alternative
):
    def result(**kw) -> PairedTTestResult:
        return PairedTTestResult(metric=metric, margin=margin, alternative=alternative, alpha=alpha, **kw)

    deltas = paired_task_deltas(baseline, candidate, metric)
    if not deltas:
        return result(n_pairs=0, note=f"no per-task `mean/{metric}` value on both sides for any common task.")

    n, n_tasks = len(deltas), max(baseline.num_tasks, candidate.num_tasks)
    mean_diff = sum(deltas) / n
    if n < MIN_PAIRED_COVERAGE * n_tasks:
        note = f"only {n} of {n_tasks} tasks paired: too little overlap to draw a conclusion."
        return result(n_pairs=n, mean_diff=mean_diff, note=note)
    if n < 2:
        return result(n_pairs=n, mean_diff=mean_diff, note="only 1 paired task: cannot estimate a standard error.")

    # `alternative` names H1 -- the claim a significant result demonstrates, never the one it assumes:
    #   candidate-higher -> H0: mu_d == +margin vs H1: mu_d > +margin, tested in the right tail.
    #   candidate-lower  -> H0: mu_d == -margin vs H1: mu_d < -margin, tested in the left tail.
    # `sign` carries that direction: it places the boundary at sign*margin and picks H1's tail.
    sign = 1 if alternative == "candidate-higher" else -1
    se = (sum((d - mean_diff) ** 2 for d in deltas) / (n - 1)) ** 0.5 / n**0.5
    if se < EPS:
        # No spread left to test against, so H1 either holds outright or it does not -- decided with the
        # same tolerance that opened this branch, so rounding noise cannot read as a certain difference.
        significant = abs(mean_diff) > EPS if alternative == "two-sided" else sign * mean_diff > margin + EPS
        p_value = 0.0 if significant else 1.0
        note = "every paired delta was identical (zero variance)."
        return result(n_pairs=n, mean_diff=mean_diff, se=0.0, p_value=p_value, significant=significant, note=note)

    df = n - 1
    if alternative == "two-sided":
        t_stat = mean_diff / se
        p_value = 2 * stats.t.sf(abs(t_stat), df)
    else:
        # How many standard errors the observed difference sits from H1's boundary, then the area
        # beyond it in H1's tail. `sf(-t) == cdf(t)` by symmetry, so one `sf` serves both directions.
        t_stat = (mean_diff - sign * margin) / se
        p_value = stats.t.sf(sign * t_stat, df)

    # The distance from the boundary an effect must clear to be called significant, plus the distance
    # the estimate must then travel to land there `MDE_POWER` of the time -- both in standard errors.
    tail_alpha = alpha / 2 if alternative == "two-sided" else alpha
    mde = (stats.t.ppf(1 - tail_alpha, df) + stats.t.ppf(MDE_POWER, df)) * se
    return result(
        n_pairs=n,
        mean_diff=mean_diff,
        se=se,
        p_value=float(p_value),
        significant=p_value < alpha,
        minimum_detectable_effect=float(mde),
    )


def build_report(config: PairedTTestConfig, command: str) -> PairedTTestReport:
    pair = load_run_pair(config)

    notes: List[str] = []
    if config.metric:
        metrics = config.metric
        for metric in metrics:
            if not paired_task_deltas(pair.baseline, pair.candidate, metric):
                raise ConfigError(f"--metric '{metric}' has no per-task `mean/{metric}` value on both sides.")
    else:
        metrics, skipped = resolve_metrics(pair.baseline, pair.candidate, None)
        if not metrics:
            raise ConfigError("No key metric has per-task pairing data to test. Pass --metric explicitly.")
        if skipped:
            notes.append(f"Skipped {len(skipped)} key metric(s) with no per-task pairing data: {', '.join(skipped)}.")

    margins = config.margin or [0.0]
    results = [
        run_metric(
            pair.baseline, pair.candidate, metric=m, margin=g, alpha=config.alpha, alternative=config.alternative
        )
        for m, g in zip(metrics, margins * len(metrics) if len(margins) == 1 else margins)
    ]
    identity = pair.report_identity(config, command)
    n_tasks = max(pair.baseline.num_tasks, pair.candidate.num_tasks)
    unpaired = [f"{r.metric} ({r.n_pairs} of {n_tasks})" for r in results if r.n_pairs < n_tasks]
    if unpaired:
        identity["warnings"].append(f"Not every task paired across the two runs: {', '.join(unpaired)}.")
    return PairedTTestReport(**identity, notes=notes, results=results)
