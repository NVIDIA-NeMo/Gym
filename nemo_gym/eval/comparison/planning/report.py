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
"""Render completed repeat-planning results without statistical calculations."""

from pathlib import Path

import orjson

from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.planning.schema import RepeatPlanningResult
from nemo_gym.eval.comparison.schema import ReportFormat


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:#.4g}"


def _text(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").replace("`", "\\`")


def render_repeat_planning_summary(result: RepeatPlanningResult) -> str:
    """Render the fixed allocation and validation result for the terminal."""
    totals = result.totals
    lines = [
        f"Repeat planning ({result.settings.aggregation_level}; {result.settings.candidate_experiments} candidate experiments)",
        f"Existing baseline passes: {totals.existing_baseline_passes}; new baseline passes: {totals.new_baseline_passes}; "
        f"candidate passes per experiment: {totals.candidate_passes_per_experiment}",
        f"Total new passes: {totals.total_additional_passes}; cost: {_number(totals.total_cost)}; "
        f"allocation method: {result.allocation.method}",
        f"Approximate planning power: {_number(result.allocation.statistics.power)}",
    ]
    if result.simulation is not None:
        lines.append(
            f"Simulated Welch decision power: {_number(result.simulation.power.estimate)} "
            f"(Wilson 95% CI [{_number(result.simulation.power.ci_low)}, {_number(result.simulation.power.ci_high)}])"
        )
    else:
        lines.append("Simulation validation: disabled (validation_trials=0).")
    return "\n".join(lines)


def render_repeat_planning_markdown(result: RepeatPlanningResult) -> str:
    """Render settings, metric weights, allocation, and completed validation statistics."""
    settings = result.settings
    statistics = result.allocation.statistics
    lines = [
        "# gym eval plan-repeats",
        "",
        f"Generated: {_text(result.generated_at)} | Gym: {_text(result.nemo_gym_version)}",
        "",
        f"Plan: `{_text(result.plan_fpath)}`",
        "",
        "## Settings",
        "",
        f"- Aggregation level: {settings.aggregation_level}; cells: {len(result.cells)}; equal model/benchmark cell weights.",
        f"- Baseline mode: {settings.baseline.mode}; candidate experiments: {settings.candidate_experiments}.",
        f"- Margin: {_number(settings.margin_points)} score points; expected candidate-minus-baseline delta: "
        f"{_number(settings.expected_delta_points)} score points.",
        f"- Target power: {_number(settings.target_power)}; alpha: {_number(settings.alpha)}.",
        f"- Maximum repeats per arm and cell: {settings.max_repeats}; allocation method: {_text(result.allocation.method)}.",
        f"- Validation trials: {settings.validation_trials}; seed: {settings.seed}.",
        "",
        "SD inputs use normalized 0–100 scores with higher values indicating better performance.",
        "Success means the one-sided Welch lower bound exceeds the negative margin. "
        "Power describes one candidate comparison.",
        "Counts are fixed before collecting the planned runs. Validation does not adjust them.",
        "Repeat costs share one user-chosen unit and apply once per complete benchmark repeat.",
        "",
        "## Metric inputs",
        "",
        "| Model | Benchmark | Metric | Metric weight | Baseline SD | Candidate SD |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for cell in result.cells:
        for metric in cell.metrics:
            lines.append(
                f"| {_text(cell.model)} | {_text(cell.benchmark)} | {_text(metric.name)} | {_number(metric.weight)} "
                f"| {_number(metric.baseline_std_points)} | {_number(metric.candidate_std_points)} |"
            )
    if any(
        cell.baseline_std_points is not None and cell.candidate_std_points is None for cell in settings.cells
    ) or any(metric.candidate_std_defaulted for cell in result.cells for metric in cell.metrics):
        lines.extend(["", "Omitted candidate SDs default to the corresponding baseline SDs."])
    lines.extend(
        [
            "",
            "## Benchmark composites",
            "",
            "Multi-metric cells use the supplied SD of the normalized, equally averaged complete-repeat scores. "
            "Their individual metric SDs above are descriptive inputs; they do not determine the composite SD.",
            "",
            "| Model | Benchmark | Cell weight | Baseline SD | Candidate SD |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cell in result.cells:
        lines.append(
            f"| {_text(cell.model)} | {_text(cell.benchmark)} | {_number(cell.weight)} "
            f"| {_number(cell.baseline_std_points)} | {_number(cell.candidate_std_points)} |"
        )
    lines.extend(
        [
            "",
            "## Fixed repeat allocation",
            "",
            "| Model | Benchmark | Existing baseline | New baseline | Candidate per experiment |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cell in result.cells:
        lines.append(
            f"| {_text(cell.model)} | {_text(cell.benchmark)} | {cell.existing_baseline_repeats} "
            f"| {cell.new_baseline_repeats} | {cell.candidate_repeats_per_experiment} |"
        )
    lines.extend(
        [
            "",
            "## Costs",
            "",
            "| Model | Benchmark | Repeat step | Cost per repeat | New cost |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for cell in result.cells:
        lines.append(
            f"| {_text(cell.model)} | {_text(cell.benchmark)} | {cell.repeat_step} "
            f"| {_number(cell.repeat_cost)} | {_number(cell.new_repeat_cost)} |"
        )
    totals = result.totals
    lines.extend(
        [
            "",
            f"Existing baseline passes: **{totals.existing_baseline_passes}**. "
            f"New baseline passes: **{totals.new_baseline_passes}**. "
            f"Candidate passes: **{totals.candidate_passes_per_experiment} per experiment**, "
            f"**{totals.candidate_passes_all_experiments} across all experiments**.",
            "",
            f"Total new passes: **{totals.total_additional_passes}**. Total new cost: **{_number(totals.total_cost)}**. "
            "Existing baseline measurements are excluded from new cost.",
            "",
            "## Planning statistics",
            "",
            "| Approximate planning power | Variance | Standard error | Effective df | Critical value | Half-width |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {_number(statistics.power)} | {_number(statistics.variance)} | {_number(statistics.standard_error)} "
            f"| {_number(statistics.degrees_of_freedom)} | {_number(statistics.critical_value)} "
            f"| {_number(statistics.half_width)} |",
            "",
            "Approximate planning power uses the supplied SDs. Simulated power below evaluates actual Welch decisions; "
            "it can differ from the planning approximation.",
            "",
            "## Simulation validation",
            "",
        ]
    )
    if result.simulation is None:
        lines.append("Simulation validation is disabled (validation_trials=0).")
    else:
        lines.extend(
            [
                "| Measurement | Successes | Trials | Estimate | Wilson 95% CI |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for label, rate in (
            ("Welch decision power", result.simulation.power),
            ("Boundary false-positive rate", result.simulation.false_positive),
            ("One-sided lower-bound coverage", result.simulation.coverage),
        ):
            lines.append(
                f"| {label} | {rate.successes} | {rate.trials} | {_number(rate.estimate)} "
                f"| [{_number(rate.ci_low)}, {_number(rate.ci_high)}] |"
            )
        lines.extend(
            [
                "",
                "Lower-bound coverage has nominal level 1 − alpha.",
                "The diagnostic redraws sample means and sample variances in both arms, including with existing "
                "baseline counts. It describes hypothetical normal experiments, not conditional power given "
                "already collected baseline observations.",
            ]
        )
    lines.extend(["", "## Invocation", "", "```bash", result.command, "```", ""])
    return "\n".join(lines)


def write_repeat_planning_reports(
    result: RepeatPlanningResult, output_dir: Path, report_format: ReportFormat
) -> list[Path]:
    """Write separate planning artifacts, preserving the input plan even through aliases."""
    names = [
        name
        for extension, name in (("md", "repeat_planning_report.md"), ("json", "repeat_planning_report.json"))
        if report_format in (extension, "both")
    ]
    for name in names:
        target = output_dir / name
        if target.resolve() == result.plan_fpath.resolve() or (target.exists() and target.samefile(result.plan_fpath)):
            raise ConfigError(f"Planning report '{target}' would overwrite the input plan; choose --output-dir.")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if report_format in ("md", "both"):
        path = output_dir / "repeat_planning_report.md"
        path.write_text(render_repeat_planning_markdown(result), encoding="utf-8")
        written.append(path)
    if report_format in ("json", "both"):
        path = output_dir / "repeat_planning_report.json"
        path.write_bytes(orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2) + b"\n")
        written.append(path)
    return written
