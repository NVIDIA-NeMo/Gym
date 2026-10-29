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
"""Presentation of completed suite results; no loading or statistical calculations."""

from pathlib import Path

import orjson

from nemo_gym.comparison.schema import ReportFormat
from nemo_gym.comparison.suite.schema import SuiteComparisonResult, SuiteStatistics
from nemo_gym.config_types import ConfigError


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:#.4g}"


def _text(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").replace("`", "\\`")


def _interval(statistics: SuiteStatistics | None) -> str:
    if statistics is None or statistics.ci_low is None or statistics.ci_high is None:
        return "unavailable"
    return f"[{_number(statistics.ci_low)}, {_number(statistics.ci_high)}]"


def render_suite_summary(result: SuiteComparisonResult) -> str:
    """Render per-model and overall results for the terminal."""
    lines = [
        f"Benchmark suite comparison ({100 * result.confidence_level:g}% confidence; normalized score points)",
        "Delta = candidate - baseline; positive deltas favor the candidate.",
    ]
    for aggregate in [*result.per_model, result.overall]:
        statistics = aggregate.statistics
        if statistics is None:
            lines.append(f"{aggregate.name}: no eligible benchmark comparisons")
            continue
        lines.append(
            f"{aggregate.name}: candidate={_number(statistics.candidate_mean)} "
            f"baseline={_number(statistics.baseline_mean)} delta={_number(statistics.delta)} "
            f"CI={_interval(statistics)} SE={_number(statistics.standard_error)} "
            f"df={_number(statistics.degrees_of_freedom)}"
        )
        if statistics.interval_unavailable_reason:
            lines.append(f"  Interval unavailable: {statistics.interval_unavailable_reason}")
    lines.extend(f"Warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


def render_suite_markdown(result: SuiteComparisonResult) -> str:
    """Render suite statistics, effective membership, metric scales, and source evidence."""
    confidence = f"{100 * result.confidence_level:g}%"
    lines = [
        "# gym eval compare-suite",
        "",
        f"Generated: {_text(result.generated_at)} | Gym: {_text(result.nemo_gym_version)}",
        "",
        f"Suite: `{_text(result.suite_fpath)}`",
        "",
        "Scores use a 0–100 scale. Delta = candidate − baseline; positive deltas favor the candidate.",
        "",
        "## Aggregate results",
        "",
        f"| Aggregate | Cells | Candidate | Baseline | Delta | {confidence} CI for delta | SE | Effective df |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for aggregate in [*result.per_model, result.overall]:
        statistics = aggregate.statistics
        if statistics is None:
            lines.append(f"| {_text(aggregate.name)} | 0 | — | — | — | unavailable | — | — |")
            continue
        lines.append(
            f"| {_text(aggregate.name)} | {len(aggregate.cell_ids)} | {_number(statistics.candidate_mean)} "
            f"| {_number(statistics.baseline_mean)} | {_number(statistics.delta)} | {_interval(statistics)} "
            f"| {_number(statistics.standard_error)} | {_number(statistics.degrees_of_freedom)} |"
        )
    lines.extend(
        [
            "",
            "## Requested and effective suite",
            "",
            f"| Model | Benchmark | Status | Candidate repeats | Baseline repeats | Delta | {confidence} CI for delta | Model weight | Overall weight |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for cell in result.cells:
        status = "included" if cell.eligible else f"excluded: {cell.exclusion_reason}"
        lines.append(
            f"| {_text(cell.model)} | {_text(cell.benchmark)} | {_text(status)} "
            f"| {cell.candidate.repeat_count} | {cell.baseline.repeat_count} "
            f"| {_number(cell.statistics.delta if cell.statistics else None)} | {_interval(cell.statistics)} "
            f"| {_number(cell.model_weight)} | {_number(cell.overall_weight)} |"
        )
    lines.extend(
        [
            "",
            "## Metric normalization",
            "",
            "Each benchmark shares its metric scales across all models.",
            "Each benchmark averages its selected metrics within each repeat before variance estimation.",
            "",
            "| Benchmark | Metric | Theoretical low | Theoretical high | Higher is better | Definition source |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    rendered_benchmarks: set[str] = set()
    for cell in result.cells:
        if cell.benchmark in rendered_benchmarks:
            continue
        rendered_benchmarks.add(cell.benchmark)
        for metric in cell.metrics:
            lines.append(
                f"| {_text(cell.benchmark)} | {_text(metric.name)} "
                f"| {_number(metric.low)} | {_number(metric.high)} | {metric.higher_is_better} "
                f"| {_text(metric.definition_source)} |"
            )
    lines.extend(["", "## Source runs and coverage", ""])
    for cell in result.cells:
        lines.append(f"### {_text(cell.model)} / {_text(cell.benchmark)}")
        lines.append("")
        for role, run in (("Candidate", cell.candidate), ("Baseline", cell.baseline)):
            lines.append(f"- {role}: `{_text(run.aggregate_metrics)}`; agent `{_text(run.agent)}`.")
            lines.append(
                f"  Coverage: {run.coverage.status}; recorded tasks: {run.coverage.recorded_task_count}; "
                f"planned tasks: {run.coverage.planned_task_count if run.coverage.planned_task_count is not None else 'unverified'}."
            )
            if run.materialized_inputs:
                lines.append(f"  Materialized inputs: `{_text(run.materialized_inputs)}`.")
            for note in run.coverage.notes:
                lines.append(f"  {_text(note)}")
        if cell.statistics and cell.statistics.interval_unavailable_reason:
            lines.append(f"Interval unavailable: {_text(cell.statistics.interval_unavailable_reason)}")
        lines.append("")
    lines.extend(["## Assumptions", ""])
    lines.extend(f"- {_text(assumption)}" for assumption in result.assumptions)
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {_text(warning)}" for warning in result.warnings)
    lines.extend(["", "## Invocation", "", "```bash", result.command, "```", ""])
    return "\n".join(lines)


def write_suite_reports(result: SuiteComparisonResult, output_dir: Path, report_format: ReportFormat) -> list[Path]:
    """Write separate suite artifacts without overwriting single-run comparison reports."""
    sources = {result.suite_fpath.resolve()}
    for cell in result.cells:
        for run in (cell.baseline, cell.candidate):
            sources.update((run.rollouts.resolve(), run.aggregate_metrics.resolve()))
            if run.materialized_inputs:
                sources.add(run.materialized_inputs.resolve())
    names = [
        name
        for extension, name in (("md", "suite_compare_report.md"), ("json", "suite_compare_report.json"))
        if report_format in (extension, "both")
    ]
    for name in names:
        if (output_dir / name).resolve() in sources:
            raise ConfigError(
                f"Suite report '{output_dir / name}' would overwrite a source artifact; choose --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if report_format in ("md", "both"):
        path = output_dir / "suite_compare_report.md"
        path.write_text(render_suite_markdown(result), encoding="utf-8")
        written.append(path)
    if report_format in ("json", "both"):
        path = output_dir / "suite_compare_report.json"
        path.write_bytes(orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2) + b"\n")
        written.append(path)
    return written
