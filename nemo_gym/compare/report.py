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
"""Markdown rendering of a `ComparisonResult`.

A pure function of the result object, so what the statistics layer eventually adds to the schema
is the only thing that changes what this prints.
"""

from typing import List, Optional, Sequence

from nemo_gym.compare.schema import (
    AgentComparison,
    ComparisonResult,
    FlipSummary,
    MetricRow,
    MetricValue,
    RunRef,
)


MISSING = "—"
# Flips shown per direction in the markdown; the JSON always carries the full list.
MAX_FLIPS_SHOWN = 10

CI_FOOTNOTE = (
    "CI = 95% t-interval of the per-repeat mean across repeats, read verbatim from "
    "`ci_{low,high}_95_across_repeats/<metric>` in `*_aggregate_metrics.json`. "
    f"`{MISSING}` means no interval was written: the metric is not a `mean/*` field, the run has fewer "
    "than 2 repeats, or the run predates repeat-level metrics. Units differ by family — `pass@*` and "
    "`majority@*` are on 0–100, `mean/*` are in the verifier's own units."
)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return MISSING
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def _fmt_signed(value: Optional[float]) -> str:
    if value is None:
        return MISSING
    return f"{value:+.4f}" if abs(value) < 10 else f"{value:+.2f}"


def _fmt_ci(value: Optional[MetricValue]) -> str:
    if value is None or value.ci_low is None or value.ci_high is None:
        return MISSING
    return f"[{_fmt(value.ci_low)}, {_fmt(value.ci_high)}]"


def _fmt_delta_cell(candidate) -> str:
    if candidate is None or candidate.delta is None:
        return MISSING
    if candidate.delta_pct is None:
        return f"{_fmt_signed(candidate.delta)} (n/a)"
    return f"{_fmt_signed(candidate.delta)} ({candidate.delta_pct:+.1f}%)"


def _fmt_rewards(rewards: Optional[Sequence[float]]) -> str:
    if not rewards:
        return MISSING
    return ", ".join(f"{reward:g}" for reward in rewards)


def _table(header: Sequence[str], alignments: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(alignments) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _candidate_labels(result: ComparisonResult) -> List[str]:
    if len(result.candidates) == 1:
        return ["Candidate"]
    return [f"Candidate {index + 1}" for index in range(len(result.candidates))]


def _run_summary_table(result: ComparisonResult, comparison: AgentComparison) -> List[str]:
    labels = _candidate_labels(result)

    def row(name: str, baseline_cell: str, candidate_cells: Sequence[str]) -> List[str]:
        return [name, baseline_cell, *candidate_cells]

    runs: Sequence[RunRef] = result.candidates
    rows = [
        row("Run", f"`{result.baseline.label}`", [f"`{run.label}`" for run in runs]),
        row(
            "Rollouts",
            f"`{result.baseline.rollouts_jsonl_fpath}`",
            [f"`{run.rollouts_jsonl_fpath}`" for run in runs],
        ),
        row(
            "Aggregate metrics",
            f"`{result.baseline.aggregate_metrics_fpath}`",
            [f"`{run.aggregate_metrics_fpath}`" for run in runs],
        ),
        row(
            "Agent",
            f"`{comparison.baseline_agent}`",
            [f"`{agent}`" for agent in comparison.candidate_agents],
        ),
        row("Tasks", str(comparison.baseline_task_count), [str(count) for count in comparison.candidate_task_counts]),
        row(
            "Repeats",
            str(comparison.baseline_repeat_count if comparison.baseline_repeat_count is not None else "unknown"),
            [str(count if count is not None else "unknown") for count in comparison.candidate_repeat_counts],
        ),
    ]
    header = ["", "Baseline", *labels]
    alignments = ["---"] * len(header)
    return _table(header, alignments, rows)


def _metric_table(rows: Sequence[MetricRow], candidate_labels: Sequence[str]) -> List[str]:
    header = ["Metric", "Drop (cand − base)"]
    alignments = ["---", "---:"]
    header += ["Baseline", "Baseline 95% CI"]
    alignments += ["---:", "---"]
    for label in candidate_labels:
        header += [label, f"{label} 95% CI"]
        alignments += ["---:", "---"]

    table_rows = []
    for row in rows:
        cells = [f"`{row.metric}`", " / ".join(_fmt_delta_cell(candidate) for candidate in row.candidates)]
        cells += [_fmt(row.baseline.value if row.baseline else None), _fmt_ci(row.baseline)]
        for candidate in row.candidates:
            cells += [_fmt(candidate.value if candidate else None), _fmt_ci(candidate)]
        table_rows.append(cells)
    return _table(header, alignments, table_rows)


def _flip_section(summary: FlipSummary, label: str) -> List[str]:
    lines = [f"### Sample flips{label}", ""]
    if summary.mode == "unavailable":
        lines += [f"Not available: {summary.reason}", ""]
        return lines

    if summary.mode == "binary":
        lines += [
            f"Field: `{summary.field}` (task mean over its repeats). A task passes when its mean is above 0.5.",
            "",
            f"**{summary.common_task_count} common tasks · {summary.pass_to_fail_count} pass→fail · "
            f"{summary.fail_to_pass_count} fail→pass · net {summary.net:+d} · "
            f"{summary.unchanged_count} unchanged · {summary.tied_count} tied**",
            "",
        ]
    else:
        lines += [
            f"Rewards are not binary, so this lists the largest per-task changes in `{summary.field}` rather "
            "than pass/fail flips.",
            "",
            f"**{summary.common_task_count} common tasks · {len(summary.flips)} changed · "
            f"{summary.unchanged_count} unchanged**",
            "",
        ]

    shown = _select_shown_flips(summary)
    if shown:
        lines += _table(
            ["Task", "Direction", "Baseline", "Candidate", "Δ", "Baseline per-repeat", "Candidate per-repeat"],
            ["---:", "---", "---:", "---:", "---:", "---", "---"],
            [
                [
                    str(flip.task_index),
                    flip.direction.replace("_to_", "→"),
                    _fmt(flip.baseline_score),
                    _fmt(flip.candidate_score),
                    _fmt_signed(flip.delta),
                    _fmt_rewards(flip.baseline_rewards),
                    _fmt_rewards(flip.candidate_rewards),
                ]
                for flip in shown
            ],
        )
        remaining = len(summary.flips) - len(shown)
        if remaining > 0:
            lines += ["", f"… and {remaining} more (full list in `compare_report.json`)."]
        lines.append("")

    lines += [
        "Tasks are joined on `_ng_task_index`. `*_aggregate_metrics.json` carries no task identity, so flips "
        "are reported by index only — this assumes both runs used the same dataset, split, `--limit` and "
        "ordering.",
        "",
    ]
    return lines


def _select_shown_flips(summary: FlipSummary) -> List:
    """At most `MAX_FLIPS_SHOWN` per direction, keeping the ordering `build_flip_summary` chose."""
    shown = []
    per_direction: dict = {}
    for flip in summary.flips:
        count = per_direction.get(flip.direction, 0)
        if count >= MAX_FLIPS_SHOWN:
            continue
        per_direction[flip.direction] = count + 1
        shown.append(flip)
    return shown


def render_markdown(result: ComparisonResult) -> str:
    candidate_labels = _candidate_labels(result)
    lines: List[str] = ["# gym compare", ""]

    for comparison in result.comparisons:
        if len(result.comparisons) > 1:
            lines += [f"## Agent: `{comparison.baseline_agent}`", ""]
        lines += _run_summary_table(result, comparison)
        lines += [
            "",
            f"Generated {result.generated_at} by nemo-gym {result.nemo_gym_version}.",
            "",
            f"Command: `{result.command}`",
            "",
        ]

        key_rows = [row for row in comparison.metrics if row.is_key_metric]
        other_rows = [row for row in comparison.metrics if not row.is_key_metric]

        lines += ["### Key metrics", ""]
        if key_rows:
            lines += _metric_table(key_rows, candidate_labels)
        else:
            lines += ["No key metrics were recorded for this agent."]
        lines += ["", CI_FOOTNOTE, ""]

        if other_rows:
            lines += [
                "<details>",
                f"<summary>All other metrics ({len(other_rows)} rows)</summary>",
                "",
                *_metric_table(other_rows, candidate_labels),
                "",
                "</details>",
                "",
            ]

        for index, summary in enumerate(comparison.flips):
            label = f" — {candidate_labels[index]}" if len(comparison.flips) > 1 else ""
            lines += _flip_section(summary, label)

        one_sided = [row for row in comparison.metrics if len(row.present_in) < 1 + len(result.candidates)]
        if one_sided:
            lines += [
                "### Metrics present in only one run",
                "",
                "Metric names embed the repeat count (`pass@1[avg-of-3]` vs `pass@1[avg-of-5]`), so runs "
                "collected with different `--num-repeats` will not line these up.",
                "",
                *(
                    f"- `{row.metric}` — {', '.join(row.present_in) or 'no numeric value on either side'}"
                    for row in one_sided
                ),
                "",
            ]

        if comparison.notes:
            lines += ["### Notes", "", *(f"- {note}" for note in comparison.notes), ""]

    if result.warnings:
        lines += ["## Warnings", "", *(f"- {warning}" for warning in result.warnings), ""]

    lines += [
        "---",
        f"Generated by `gym compare` (nemo-gym {result.nemo_gym_version}). "
        f"Machine-readable output: `compare_report.json` (schema_version {result.schema_version}).",
        "",
    ]
    return "\n".join(lines)
