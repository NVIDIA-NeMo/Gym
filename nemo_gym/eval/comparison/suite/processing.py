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
"""Validate coverage and assemble statistical inputs from stored suite artifacts."""

from dataclasses import asdict, replace

from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME
from nemo_gym.eval.comparison.statistics.unpaired_welch_fixed_suite import (
    BenchmarkSamples,
    MetricScale,
    SampleSummary,
    aggregate_welch,
    normalize_and_average_metrics_per_repeat,
    normalize_score,
    summarize_sample,
)
from nemo_gym.eval.comparison.suite.coverage import check_collection_coverage, recorded_task_sets, run_coverage
from nemo_gym.eval.comparison.suite.loading import LoadedSuiteRun, _repeat_count
from nemo_gym.eval.comparison.suite.pooling import pool_suite_summaries
from nemo_gym.eval.comparison.suite.schema import (
    SuiteAggregateResult,
    SuiteCellResult,
    SuiteCoverage,
    SuiteMetricResult,
    SuiteRunSummary,
    SuiteSampleSummary,
    SuiteStatistics,
)
from nemo_gym.metrics.metrics_config import COMPLETION_TOKEN_METRIC_NAMES


def _select_metric_repeats(run: LoadedSuiteRun, metrics: list[SuiteMetricResult]) -> LoadedSuiteRun:
    """Omit token-only rows lacking selected estimates; score rows need not have sample_count."""
    selected_names = {metric.name for metric in metrics}
    token_names = set(COMPLETION_TOKEN_METRIC_NAMES)
    omitted = set()
    for row in run.repeat_rows:
        measurement_names = row.keys() - {AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME}
        if measurement_names and measurement_names <= token_names and not selected_names <= row.keys():
            omitted.add(int(row[ROLLOUT_INDEX_KEY_NAME]))
    if not omitted:
        return run
    rows = tuple(row for row in run.repeat_rows if row[ROLLOUT_INDEX_KEY_NAME] not in omitted)
    recorded, _, complete_identities = recorded_task_sets(run)
    # A valid single-repeat run also has only token summaries. Its scored task records
    # distinguish that case from repeats with no usable measurements.
    unscored = omitted - recorded.keys() if complete_identities else omitted
    return replace(
        run,
        repeat_rows=rows,
        repeat_count=_repeat_count(
            rows=rows, groups=run.groups, agent_metrics=run.agent_metrics, path=run.aggregate_metrics, agent=run.agent
        ),
        unscored_repeat_indices=tuple(sorted(unscored)),
    )


def _metric_values(run: LoadedSuiteRun, metrics: list[SuiteMetricResult]) -> list[list[float]]:
    values = []
    for row in run.repeat_rows:
        repeat_values = []
        for metric in metrics:
            value = row.get(metric.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(
                    f"'{run.aggregate_metrics}' agent '{run.agent}' repeat {row[ROLLOUT_INDEX_KEY_NAME]} "
                    f"requires numeric metric '{metric.name}', got {value!r}. Regenerate aggregate metrics "
                    "if this metric was not stored per repeat."
                )
            repeat_values.append(float(value))
        values.append(repeat_values)
    return values


def _run_summary(run: LoadedSuiteRun, coverage: SuiteCoverage, scores: tuple[float, ...]) -> SuiteRunSummary:
    summary = SuiteSampleSummary(**asdict(summarize_sample(scores))) if len(scores) >= 2 else None
    return SuiteRunSummary(
        rollouts=run.rollouts,
        aggregate_metrics=run.aggregate_metrics,
        materialized_inputs=run.materialized_inputs,
        agent=run.agent,
        environment_server=run.environment_server,
        repeat_count=run.repeat_count,
        repeat_indices=[int(row[ROLLOUT_INDEX_KEY_NAME]) for row in run.repeat_rows],
        repeat_scores=list(scores),
        unscored_repeat_indices=list(run.unscored_repeat_indices),
        summary=summary,
        coverage=coverage,
    )


def _samples(cell: SuiteCellResult) -> BenchmarkSamples:
    if cell.baseline.summary is None or cell.candidate.summary is None:
        raise ValueError("An eligible comparison needs both sample summaries.")
    return BenchmarkSamples(
        baseline=SampleSummary(**cell.baseline.summary.model_dump()),
        candidate=SampleSummary(**cell.candidate.summary.model_dump()),
    )


def _statistics(cells: list[SuiteCellResult], *, confidence_level: float) -> SuiteStatistics:
    result = SuiteStatistics(
        **asdict(aggregate_welch([_samples(cell) for cell in cells], confidence_level=confidence_level))
    )
    if any(run.coverage.status == "incomplete" for cell in cells for run in (cell.baseline, cell.candidate)):
        result.ci_low = result.ci_high = None
        result.interval_unavailable_reason = "Known incomplete task coverage in an included benchmark."
    elif result.ci_low is None:
        result.interval_unavailable_reason = "Total repeat variance is zero."
    return result


def _collection_summary(
    runs: tuple[LoadedSuiteRun, ...],
    coverages: list[SuiteCoverage],
    scores: tuple[float, ...],
    collection_notes: list[str],
) -> SuiteRunSummary:
    summaries = []
    offset = 0
    for run, coverage in zip(runs, coverages, strict=True):
        count = len(run.repeat_rows)
        summaries.append(_run_summary(run, coverage, scores[offset : offset + count]))
        offset += count
    return pool_suite_summaries(summaries, runs=runs, collection_notes=collection_notes)


def compare_suite_cell(
    *,
    model: str,
    benchmark: str,
    baseline: LoadedSuiteRun | tuple[LoadedSuiteRun, ...],
    candidate: LoadedSuiteRun | tuple[LoadedSuiteRun, ...],
    metrics: list[SuiteMetricResult],
    confidence_level: float,
) -> SuiteCellResult:
    """Compare concatenated independent repeat samples, retaining source-level evidence."""
    baseline_runs = baseline if isinstance(baseline, tuple) else (baseline,)
    candidate_runs = candidate if isinstance(candidate, tuple) else (candidate,)
    if not baseline_runs or not candidate_runs:
        raise ConfigError("Baseline and candidate source collections must not be empty.")
    baseline_runs = tuple(_select_metric_repeats(run, metrics) for run in baseline_runs)
    candidate_runs = tuple(_select_metric_repeats(run, metrics) for run in candidate_runs)
    baseline_coverage = [run_coverage(run) for run in baseline_runs]
    candidate_coverage = [run_coverage(run) for run in candidate_runs]
    collection_notes = check_collection_coverage(
        baseline=baseline_runs,
        candidate=candidate_runs,
        baseline_coverage=baseline_coverage,
        candidate_coverage=candidate_coverage,
    )
    baseline_count = sum(run.repeat_count for run in baseline_runs)
    candidate_count = sum(run.repeat_count for run in candidate_runs)
    if baseline_count < 2 or candidate_count < 2:
        return SuiteCellResult(
            model=model,
            benchmark=benchmark,
            eligible=False,
            exclusion_reason=(
                "Excluded: at least two repeats are required on each side "
                f"(baseline={baseline_count}, candidate={candidate_count})."
            ),
            baseline=_collection_summary(baseline_runs, baseline_coverage, (), collection_notes),
            candidate=_collection_summary(candidate_runs, candidate_coverage, (), collection_notes),
            metrics=metrics,
        )
    for run in (*baseline_runs, *candidate_runs):
        if run.repeat_count and not run.repeat_rows:
            raise ConfigError(
                f"'{run.aggregate_metrics}' agent '{run.agent}' has no stored repeat rows. "
                "Pooling requires stored metrics for every scored repeat; regenerate aggregate metrics "
                "with `gym eval aggregate`. Suite comparison does not reconstruct repeat metrics."
            )
    scales = [MetricScale(metric.low, metric.high, metric.higher_is_better) for metric in metrics]
    baseline_values = [row for run in baseline_runs for row in _metric_values(run, metrics)]
    candidate_values = [row for run in candidate_runs for row in _metric_values(run, metrics)]
    try:
        baseline_scores = normalize_and_average_metrics_per_repeat(baseline_values, scales)
        candidate_scores = normalize_and_average_metrics_per_repeat(candidate_values, scales)
        for index, (metric, scale) in enumerate(zip(metrics, scales, strict=True)):
            metric.baseline_raw = [row[index] for row in baseline_values]
            metric.candidate_raw = [row[index] for row in candidate_values]
            metric.baseline_scaled = [normalize_score(value, scale) for value in metric.baseline_raw]
            metric.candidate_scaled = [normalize_score(value, scale) for value in metric.candidate_raw]
    except ValueError as error:
        raise ConfigError(f"Cannot normalize '{model}/{benchmark}' repeat metrics: {error}") from error
    cell = SuiteCellResult(
        model=model,
        benchmark=benchmark,
        eligible=True,
        metrics=metrics,
        baseline=_collection_summary(baseline_runs, baseline_coverage, baseline_scores, collection_notes),
        candidate=_collection_summary(candidate_runs, candidate_coverage, candidate_scores, collection_notes),
    )
    cell.statistics = _statistics([cell], confidence_level=confidence_level)
    return cell


def aggregate_suite_cells(
    name: str,
    cells: list[SuiteCellResult],
    *,
    confidence_level: float,
) -> SuiteAggregateResult:
    """Aggregate eligible cells equally, retaining declared exclusions in the cell report."""
    eligible = [cell for cell in cells if cell.eligible]
    return SuiteAggregateResult(
        name=name,
        cell_ids=[f"{cell.model}/{cell.benchmark}" for cell in eligible],
        weights=[1 / len(eligible)] * len(eligible) if eligible else [],
        statistics=_statistics(eligible, confidence_level=confidence_level) if eligible else None,
    )
