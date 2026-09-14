# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLADE exports for a k=1 safety benchmark run: D1 deterministic metrics, D2 evidence-anchored
facts, D3 shallow baseline, and a deterministic BLADE report. Nothing here is written by a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import MetricValue, NormalizedRun


def _fmt(metric: MetricValue) -> str:
    if metric.value is None:
        return "n/a"
    if metric.unit == "rate":
        return f"{metric.value:.1%}"
    if isinstance(metric.value, float) and not metric.value.is_integer():
        return f"{metric.value:.2f}"
    return f"{int(metric.value)}"


def _denominator(metric: MetricValue) -> str:
    if metric.numerator is None or metric.denominator is None:
        return "-"
    return f"{metric.numerator} / {metric.denominator}"


def d1_metrics(run: NormalizedRun) -> dict[str, Any]:
    outcomes = run.outcomes
    d1: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": run.benchmark["id"],
        "protocol": run.benchmark["protocol"],
        "model_name": run.run["model"],
        "harness": run.run["harness"],
        "expected_k": 1,
        "expected_tasks": outcomes.expected_tasks,
        "expected_rollouts": outcomes.expected_rollouts,
        "scored_rollouts": outcomes.scored_rollouts,
        "coverage": outcomes.scored_rollouts / outcomes.expected_rollouts if outcomes.expected_rollouts else None,
        "outcome_classes": {
            "policy_scored": outcomes.policy_scored,
            "invalid_model_output": outcomes.invalid_model_output,
            "infrastructure_failure": outcomes.infrastructure_failure,
            "judge_missing": outcomes.judge_missing,
            "pending_annotation": outcomes.pending_annotation,
        },
        "reward_is_semantic": run.reward_is_semantic,
        "reward_semantics": run.reward_semantics,
        "metrics": {
            metric.id: {
                "name": metric.name,
                "value": metric.value,
                "numerator": metric.numerator,
                "denominator": metric.denominator,
                "unit": metric.unit,
                "direction": metric.direction,
                "kind": metric.kind,
                "slice": metric.slice,
            }
            for metric in run.metrics
        },
        "not_applicable": [
            "pass@k for k>1 (single rollout per task)",
            "tool-call funnel stages beyond what the verifier records",
            "root-cause taxonomy labels (no repeated trials to contrast)",
        ],
    }
    if run.reward_is_semantic:
        scored = [
            rollout
            for rollout in run.rollouts
            if rollout.outcome_class == "policy_scored" and rollout.reward is not None
        ]
        d1["pass_at_1"] = sum(rollout.reward for rollout in scored) / len(scored) if scored else None
        d1["pass_at_1_denominator"] = len(scored)
    return d1


def d2_anchor_facts(run: NormalizedRun) -> dict[str, Any]:
    return {
        "benchmark": run.benchmark["id"],
        "model_name": run.run["model"],
        "anchor_facts": [fact.model_dump(mode="json") for fact in run.anchor_facts],
    }


def d3_shallow_baseline(run: NormalizedRun) -> str:
    lines = [
        "# Shallow Metrics Baseline",
        "",
        "| Metric | Value | Numerator / denominator | Direction |",
        "|---|---:|---:|---|",
    ]
    for metric in run.metrics:
        if metric.kind in {"primary", "component"} and metric.slice is None:
            lines.append(f"| {metric.name} | {_fmt(metric)} | {_denominator(metric)} | {metric.direction} |")
    outcomes = run.outcomes
    lines += [
        "",
        "## Outcome accounting",
        "",
        "| Class | Count |",
        "|---|---:|",
        f"| Expected rollouts | {outcomes.expected_rollouts} |",
        f"| Scored rollouts | {outcomes.scored_rollouts} |",
        f"| Policy-scored | {outcomes.policy_scored} |",
        f"| Invalid model output | {outcomes.invalid_model_output} |",
        f"| Infrastructure failure | {outcomes.infrastructure_failure} |",
        f"| Judge missing | {outcomes.judge_missing} |",
        f"| Pending annotation | {outcomes.pending_annotation} |",
        "",
    ]
    return "\n".join(lines)


def blade_report(run: NormalizedRun, generated_at: str) -> str:
    outcomes = run.outcomes
    primary = [metric for metric in run.metrics if metric.kind == "primary"]
    components = [metric for metric in run.metrics if metric.kind == "component"]
    slices = [metric for metric in run.metrics if metric.slice]
    lines = [
        f"# {run.benchmark['display_name']} BLADE Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"Deterministic analysis of `{run.run['model']}` on {run.benchmark['display_name']} "
        f"({run.benchmark['protocol']}), k=1, {outcomes.scored_rollouts} of {outcomes.expected_rollouts} rollouts scored.",
    ]
    for metric in primary:
        lines.append(f"- {metric.name}: {_fmt(metric)} ({_denominator(metric)}; {metric.direction.replace('_', ' ')})")
    lines += [
        "",
        "## Artifact Inventory",
        "",
        f"- Rollouts collected by NeMo Gym `gym eval run` (harness `{run.run['harness']}`, Gym revision `{run.run['gym_revision']}`)",
        f"- Verifier: {run.run.get('verifier', {}).get('summary', 'see manifest')}",
        f"- Dataset: {run.benchmark['dataset']['name']} ({run.benchmark['dataset']['count']} rows, sha256 `{run.benchmark['dataset']['sha256']}`)",
        f"- Upstream: {run.benchmark['upstream']['repository']} @ `{run.benchmark['upstream']['revision']}`",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Value | Numerator / denominator | Direction | Kind |",
        "|---|---:|---:|---|---|",
    ]
    for metric in primary + components:
        lines.append(
            f"| {metric.name} | {_fmt(metric)} | {_denominator(metric)} | {metric.direction} | {metric.kind} |"
        )
    lines += [
        "",
        "## Outcome Accounting (replaces a tool-call funnel; k=1)",
        "",
        "| Class | Count |",
        "|---|---:|",
        f"| Expected rollouts | {outcomes.expected_rollouts} |",
        f"| Scored rollouts | {outcomes.scored_rollouts} |",
        f"| Policy-scored | {outcomes.policy_scored} |",
        f"| Invalid model output | {outcomes.invalid_model_output} |",
        f"| Infrastructure failure (sidecar, excluded from denominators) | {outcomes.infrastructure_failure} |",
        f"| Judge missing | {outcomes.judge_missing} |",
        f"| Pending annotation | {outcomes.pending_annotation} |",
    ]
    if outcomes.missing_task_ids or outcomes.duplicate_task_ids:
        lines.append(f"| Missing task ids | {len(outcomes.missing_task_ids)} |")
        lines.append(f"| Duplicate task ids | {len(outcomes.duplicate_task_ids)} |")
    lines += ["", "## Slice Results", ""]
    if slices:
        lines += ["| Slice | Metric | Value | Numerator / denominator |", "|---|---|---:|---:|"]
        for metric in slices:
            slice_label = ", ".join(f"{key}={value}" for key, value in metric.slice.items())
            lines.append(f"| {slice_label} | {metric.name} | {_fmt(metric)} | {_denominator(metric)} |")
    else:
        lines.append("No slice metrics were declared.")
    lines += ["", "## Evidence-Anchored Facts", ""]
    for fact in run.anchor_facts:
        lines.append(f"- **{fact.id}** ({fact.category}): {fact.fact} Evidence: {', '.join(fact.evidence)}.")
    lines += ["", "## Calibration", ""]
    if run.calibration:
        lines.append(
            f"{run.calibration.method}: {run.calibration.agreement} of {run.calibration.cases} cases agree; "
            f"{run.calibration.disagreements} disagreement(s)."
        )
        lines += [f"- {note}" for note in run.calibration.notes]
    else:
        lines.append("No calibration record was attached.")
    lines += ["", "## Reference Comparisons", ""]
    if run.reference_comparisons:
        lines += ["| Reference | Value | Source | Comparability |", "|---|---:|---|---|"]
        for reference in run.reference_comparisons:
            lines.append(f"| {reference.label} | {reference.value} | {reference.source} | {reference.comparability} |")
    else:
        lines.append("No reference values were attached.")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in run.limitations]
    lines += [
        "",
        "## Reproducibility Notes",
        "",
        f"- Model: {run.run['model']} via {run.run['endpoint_type']}",
        f"- Sampling: {json.dumps(run.run.get('sampling', {}), sort_keys=True)}",
        f"- Generation limits: {json.dumps(run.run.get('generation_limits', {}), sort_keys=True)}",
        f"- Run id: {run.run['run_id']}; generated {generated_at}",
        "",
    ]
    return "\n".join(lines)


def build_blade_files(run: NormalizedRun, blade_dir: Path, *, generated_at: str) -> None:
    blade_dir.mkdir(parents=True, exist_ok=True)
    d1 = d1_metrics(run)
    d1_text = json.dumps(d1, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (blade_dir / "d1-metrics.json").write_text(d1_text, encoding="utf-8")
    (blade_dir / "metrics.json").write_text(d1_text, encoding="utf-8")
    (blade_dir / "d2-anchor-facts.json").write_text(
        json.dumps(d2_anchor_facts(run), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (blade_dir / "d3-shallow-baseline.md").write_text(d3_shallow_baseline(run), encoding="utf-8")
    (blade_dir / "BLADE-report.md").write_text(blade_report(run, generated_at), encoding="utf-8")
