# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLADE-style exports derived deterministically from a ``NormalizedRun``.

- ``metrics.json``: benchmark-native metrics sidecar (the BLADE ``pass_at_1`` slot is filled only when the official
  metric is a binary pass rate; otherwise it is ``null`` with the reason).
- ``d1-metrics.json``: workflow funnel and outcome accounting (expected -> materialized -> answered -> judged -> scored).
- ``d2-anchor-facts.json``: anchor facts, each citing task/rollout/receipt ids.
- ``d3-shallow-baseline.md``: the script-only tables (negative control for report judging).
- ``BLADE-report.md``: the deterministic analysis report (tables, slices, examples, calibration, limitations).

These files are mostly single-turn or paired model/judge evaluations, so there is no pass@k, tool-call funnel,
or task root-cause taxonomy; those BLADE concepts are marked not applicable rather than reported as zero.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import MetricValue, NormalizedRun


def _fmt(metric: MetricValue) -> str:
    if metric.value is None:
        return "n/a"
    if metric.unit == "rate":
        return f"{metric.value:.1%}"
    if metric.unit == "percent":
        return f"{metric.value:.1f}%"
    if metric.unit == "score":
        return f"{metric.value:.2f}"
    if metric.unit == "count":
        return f"{int(round(metric.value))}"
    return f"{metric.value:.1f}"


def _den(metric: MetricValue) -> str:
    if metric.denominator is None:
        return "-"
    if metric.numerator is None:
        return f"n={metric.denominator}"
    numerator = int(metric.numerator) if float(metric.numerator).is_integer() else round(metric.numerator, 2)
    return f"{numerator} / {metric.denominator}"


def blade_metrics(run: NormalizedRun, generated_at: str) -> dict[str, Any]:
    primary = run.primary_metric()
    payload: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": run.benchmark["id"],
        "benchmark_display_name": run.benchmark["display_name"],
        "model_name": run.run["model"],
        "run_id": run.run["run_id"],
        "generated_at": generated_at,
        "total_tasks": run.outcomes.expected_tasks,
        "total_rollouts": run.outcomes.expected_rollouts,
        "scored_rollouts": run.outcomes.scored_rollouts,
        "primary_metric": primary.model_dump(mode="json"),
        "pass_at_1": None,
        "pass_at_1_note": run.blade.d1_metrics.get("pass_at_1", "N/A"),
        "metrics": {metric.id: metric.model_dump(mode="json") for metric in run.metrics},
        "outcomes": run.outcomes.model_dump(mode="json"),
        "reward_semantics": run.reward_semantics,
    }
    if run.blade.d1_metrics.get("pass_at_1", "").startswith("metric:"):
        payload["pass_at_1"] = run.metric(run.blade.d1_metrics["pass_at_1"].split(":", 1)[1]).value
    return payload


def d1_metrics(run: NormalizedRun, generated_at: str) -> dict[str, Any]:
    outcomes = run.outcomes
    answered = sum(1 for rollout in run.rollouts if rollout.outcome_class in ("scored", "judge_failed"))
    funnel = {
        "expected_rollouts": outcomes.expected_rollouts,
        "materialized_rollouts": outcomes.expected_rollouts - outcomes.missing_rollouts,
        "target_model_responded": answered,
        "official_judge_completed": outcomes.scored_rollouts,
        "scored": outcomes.scored_rollouts,
    }
    return {
        "schema_version": 2,
        "benchmark": run.benchmark["id"],
        "model_name": run.run["model"],
        "run_id": run.run["run_id"],
        "generated_at": generated_at,
        "total_tasks": outcomes.expected_tasks,
        "total_rollouts": outcomes.expected_rollouts,
        "scored_rollouts": outcomes.scored_rollouts,
        "coverage": outcomes.scored_rollouts / outcomes.expected_rollouts if outcomes.expected_rollouts else None,
        "workflow_funnel": funnel,
        "outcome_counts": outcomes.model_dump(mode="json"),
        "metric_map": run.blade.d1_metrics,
        "not_applicable": run.blade.not_applicable,
        "slices": {
            metric.id: {
                "value": metric.value,
                "numerator": metric.numerator,
                "denominator": metric.denominator,
                "slice": metric.slice,
            }
            for metric in run.metrics
            if metric.slice
        },
    }


def d2_anchor_facts(run: NormalizedRun) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "benchmark": run.benchmark["id"],
        "model_name": run.run["model"],
        "anchor_facts": [
            {
                "id": fact.id,
                "category": fact.category,
                "fact": fact.fact,
                "evidence": fact.evidence,
                "excerpt": fact.excerpt,
                "difficulty": "N/A",
                "root_cause": "N/A",
            }
            for fact in run.anchor_facts
        ],
        "notes": run.blade.d3_notes,
    }


def _metrics_table(metrics: list[MetricValue]) -> list[str]:
    lines = ["| Metric | Value | Numerator / denominator | Direction | Role |", "|---|---:|---:|---|---|"]
    for metric in metrics:
        lines.append(
            f"| {metric.name} | {_fmt(metric)} | {_den(metric)} | {metric.direction.replace('_', ' ')} | {metric.kind} |"
        )
    return lines


def _slice_table(run: NormalizedRun) -> list[str]:
    sliced = [metric for metric in run.metrics if metric.slice]
    if not sliced:
        return []
    lines = ["| Slice | Metric | Value | Numerator / denominator |", "|---|---|---:|---:|"]
    for metric in sliced:
        label = ", ".join(f"{key}={value}" for key, value in metric.slice.items())
        lines.append(f"| {label} | {metric.name} | {_fmt(metric)} | {_den(metric)} |")
    return lines


def _outcome_table(run: NormalizedRun) -> list[str]:
    outcomes = run.outcomes
    rows = [
        ("Expected tasks", outcomes.expected_tasks),
        ("Expected rollouts", outcomes.expected_rollouts),
        ("Scored by the official judge/verifier", outcomes.scored_rollouts),
        ("Judge failed (no score)", outcomes.judge_failed),
        ("Simulator failed (no score)", outcomes.simulation_failed),
        ("Infrastructure failed (no score)", outcomes.infrastructure_failed),
        ("Missing", outcomes.missing_rollouts),
        ("Duplicate rollouts", outcomes.duplicate_rollouts),
        ("Superseded sidecar attempts", outcomes.replaced_attempts),
    ]
    return ["| Accounting | Count |", "|---|---:|"] + [f"| {label} | {value} |" for label, value in rows]


def shallow_baseline(run: NormalizedRun) -> str:
    lines = [f"# Shallow metrics baseline: {run.run['model']} on {run.benchmark['display_name']}", ""]
    lines += (
        ["## Aggregate metrics", ""]
        + _metrics_table([m for m in run.metrics if m.kind in ("primary", "component")])
        + [""]
    )
    lines += ["## Outcome accounting", ""] + _outcome_table(run) + [""]
    slices = _slice_table(run)
    if slices:
        lines += ["## Slices", ""] + slices + [""]
    return "\n".join(lines)


def blade_report(run: NormalizedRun) -> str:
    primary = run.primary_metric()
    lines = [
        f"# BLADE analysis report: {run.run['model']} on {run.benchmark['display_name']}",
        "",
        f"Run `{run.run['run_id']}` · {run.benchmark['protocol']} · finished {run.run.get('finished_at', 'n/a')}",
        "",
        "## Executive summary",
        "",
        f"{primary.name}: {_fmt(primary)} ({_den(primary)}; {primary.direction.replace('_', ' ')}). "
        f"{run.outcomes.scored_rollouts} of {run.outcomes.expected_rollouts} expected rollouts were scored by the official "
        f"judge/verifier. {run.reward_semantics}",
        "",
        "## Aggregate results",
        "",
    ]
    lines += _metrics_table([m for m in run.metrics if m.kind in ("primary", "component")]) + [""]
    diagnostics = [m for m in run.metrics if m.kind in ("diagnostic", "operational")]
    if diagnostics:
        lines += ["## Judge validity, coverage, and operational diagnostics", ""] + _metrics_table(diagnostics) + [""]
    lines += ["## Outcome accounting", ""] + _outcome_table(run) + [""]
    slices = _slice_table(run)
    if slices:
        lines += ["## Slices", ""] + slices + [""]
    lines += ["## Anchor facts", ""]
    for fact in run.anchor_facts:
        lines.append(f"- **{fact.id}** ({fact.category}): {fact.fact} Evidence: {', '.join(fact.evidence)}.")
    lines.append("")
    examples = [fact for fact in run.anchor_facts if fact.category == "example"]
    if examples:
        lines += ["## Trace-linked examples", ""]
        for fact in examples:
            lines.append(f"### {fact.id}")
            lines.append("")
            lines.append(fact.fact)
            if fact.excerpt:
                lines.append("")
                lines.append("> " + fact.excerpt.replace("\n", "\n> "))
            lines.append("")
    if run.calibration:
        cal = run.calibration
        lines += [
            "## Calibration against the upstream implementation",
            "",
            f"Method: {cal.method}. Cases: {cal.cases}; agreement: {cal.agreement}; disagreements: {cal.disagreements}.",
            "",
        ]
        lines += [f"- {note}" for note in cal.notes] + [""]
    if run.reference_comparisons:
        lines += [
            "## Reference comparisons",
            "",
            "| Reference | Value | Source | Comparability |",
            "|---|---|---|---|",
        ]
        lines += [f"| {r.label} | {r.value} | {r.source} | {r.comparability} |" for r in run.reference_comparisons] + [
            ""
        ]
    lines += ["## Limitations", ""] + [f"- {item}" for item in run.limitations] + [""]
    lines += ["## BLADE mapping", ""]
    lines += [f"- D1 `{key}`: {value}" for key, value in run.blade.d1_metrics.items()]
    lines += [f"- Not applicable: `{key}`: {value}" for key, value in run.blade.not_applicable.items()]
    return "\n".join(lines) + "\n"


def write_blade_files(run: NormalizedRun, blade_dir: Path) -> None:
    blade_dir.mkdir(parents=True, exist_ok=True)
    generated_at = run.run.get("package_generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, payload in (
        ("metrics.json", blade_metrics(run, generated_at)),
        ("d1-metrics.json", d1_metrics(run, generated_at)),
        ("d2-anchor-facts.json", d2_anchor_facts(run)),
    ):
        (blade_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (blade_dir / "d3-shallow-baseline.md").write_text(shallow_baseline(run), encoding="utf-8")
    (blade_dir / "BLADE-report.md").write_text(blade_report(run), encoding="utf-8")
