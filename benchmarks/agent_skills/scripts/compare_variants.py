# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired comparison utilities for agent-skill experiment rollouts."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
    "reward",
    "correctness",
    "completeness",
    "convention_compliance",
    "sandbox_elapsed_seconds",
    "verifier_elapsed_seconds",
    "input_tokens",
    "output_tokens",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def rollout_key(row: dict[str, Any]) -> tuple[str, int]:
    task_id = row.get("task_id")
    if task_id is None:
        task_id = (row.get("verifier_metadata") or {}).get("task_id")
    if task_id is None:
        task_id = row.get("_ng_task_index")
    rollout_index = row.get("_ng_rollout_index", 0)
    if task_id is None:
        raise ValueError("Rollout has no task_id, verifier_metadata.task_id, or _ng_task_index")
    return str(task_id), int(rollout_index)


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric in {"input_tokens", "output_tokens"}:
        value = ((row.get("response") or {}).get("usage") or {}).get(metric)
    else:
        value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = rollout_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate rollout key: {key}")
        indexed[key] = row
    return indexed


def pass_at_k(rows: list[dict[str, Any]]) -> float:
    task_rewards: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        task_id, _ = rollout_key(row)
        task_rewards[task_id].append(metric_value(row, "reward") or 0.0)
    if not task_rewards:
        return 0.0
    return mean(float(any(reward > 0 for reward in rewards)) for rewards in task_rewards.values())


def paired_bootstrap_ci(
    task_deltas: dict[str, float],
    *,
    seed: int,
    samples: int = 2000,
) -> tuple[float, float] | None:
    if not task_deltas:
        return None
    values = list(task_deltas.values())
    if len(values) == 1:
        return None
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    lower_index = max(0, int(0.025 * samples) - 1)
    upper_index = min(samples - 1, int(0.975 * samples))
    return estimates[lower_index], estimates[upper_index]


def compare_rollouts(
    control_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    *,
    control_arm: str,
    treatment_arm: str,
    seed: int,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    control = _index_rows(control_rows)
    treatment = _index_rows(treatment_rows)
    control_keys = set(control)
    treatment_keys = set(treatment)
    if control_keys != treatment_keys:
        missing_control = sorted(treatment_keys - control_keys)
        missing_treatment = sorted(control_keys - treatment_keys)
        raise ValueError(
            "Control and treatment rollout keys differ: "
            f"missing_from_control={missing_control}, missing_from_treatment={missing_treatment}"
        )
    paired_keys = sorted(control_keys & treatment_keys)
    if not paired_keys:
        raise ValueError("Control and treatment have no paired rollout keys")

    pairs: list[dict[str, Any]] = []
    wins = losses = ties = 0
    task_metric_deltas: dict[str, dict[str, list[float]]] = {metric: defaultdict(list) for metric in METRICS}
    metric_control_values: dict[str, list[float]] = defaultdict(list)
    metric_treatment_values: dict[str, list[float]] = defaultdict(list)

    for task_id, rollout_index in paired_keys:
        control_row = control[(task_id, rollout_index)]
        treatment_row = treatment[(task_id, rollout_index)]
        control_reward = metric_value(control_row, "reward") or 0.0
        treatment_reward = metric_value(treatment_row, "reward") or 0.0
        if treatment_reward > control_reward:
            wins += 1
            outcome = "win"
        elif treatment_reward < control_reward:
            losses += 1
            outcome = "loss"
        else:
            ties += 1
            outcome = "tie"

        metric_deltas: dict[str, float] = {}
        for metric in METRICS:
            control_value = metric_value(control_row, metric)
            treatment_value = metric_value(treatment_row, metric)
            if control_value is None or treatment_value is None:
                continue
            delta = treatment_value - control_value
            metric_deltas[metric] = delta
            metric_control_values[metric].append(control_value)
            metric_treatment_values[metric].append(treatment_value)
            task_metric_deltas[metric][task_id].append(delta)
        pairs.append(
            {
                "task_id": task_id,
                "rollout_index": rollout_index,
                "outcome": outcome,
                "control_reward": control_reward,
                "treatment_reward": treatment_reward,
                "metric_deltas": metric_deltas,
            }
        )

    metrics: dict[str, Any] = {}
    for metric in METRICS:
        control_values = metric_control_values.get(metric, [])
        treatment_values = metric_treatment_values.get(metric, [])
        if not control_values or not treatment_values:
            continue
        per_task_delta = {task_id: mean(deltas) for task_id, deltas in task_metric_deltas[metric].items()}
        lower, upper = paired_bootstrap_ci(per_task_delta, seed=seed, samples=bootstrap_samples) or (None, None)
        control_mean = mean(control_values)
        treatment_mean = mean(treatment_values)
        metrics[metric] = {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "delta": treatment_mean - control_mean,
            "paired_task_ci_95": [lower, upper],
            "paired_tasks": len(per_task_delta),
        }

    return {
        "control_arm": control_arm,
        "treatment_arm": treatment_arm,
        "control_rollouts": len(control_rows),
        "treatment_rollouts": len(treatment_rows),
        "paired_rollouts": len(paired_keys),
        "missing_from_control": [list(key) for key in sorted(treatment_keys - control_keys)],
        "missing_from_treatment": [list(key) for key in sorted(control_keys - treatment_keys)],
        "outcomes": {"wins": wins, "losses": losses, "ties": ties},
        "pass_at_k": {
            "control": pass_at_k(control_rows),
            "treatment": pass_at_k(treatment_rows),
        },
        "metrics": metrics,
        "pairs": pairs,
    }


def render_report(comparison: dict[str, Any]) -> str:
    outcomes = comparison["outcomes"]
    lines = [
        "# Agent Skill A/B Comparison",
        "",
        f"- Control: `{comparison['control_arm']}`",
        f"- Treatment: `{comparison['treatment_arm']}`",
        f"- Paired rollouts: {comparison['paired_rollouts']}",
        f"- Wins / losses / ties: {outcomes['wins']} / {outcomes['losses']} / {outcomes['ties']}",
        f"- Pass@k: {comparison['pass_at_k']['control']:.4f} → {comparison['pass_at_k']['treatment']:.4f}",
        "",
        "## Metrics",
        "",
        "| Metric | Control | Treatment | Delta | Paired task 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in comparison["metrics"].items():
        lower, upper = values["paired_task_ci_95"]
        ci = "n/a" if lower is None or upper is None else f"[{lower:+.4f}, {upper:+.4f}]"
        lines.append(
            f"| {metric} | {values['control_mean']:.4f} | "
            f"{values['treatment_mean']:.4f} | {values['delta']:+.4f} | "
            f"{ci} |"
        )

    lines.extend(["", "## Paired outcomes", "", "| Task | Rollout | Outcome | Reward delta |", "|---|---:|---|---:|"])
    for pair in comparison["pairs"]:
        reward_delta = pair["treatment_reward"] - pair["control_reward"]
        lines.append(f"| {pair['task_id']} | {pair['rollout_index']} | {pair['outcome']} | {reward_delta:+.4f} |")

    if comparison["missing_from_control"] or comparison["missing_from_treatment"]:
        lines.extend(
            [
                "",
                "## Pairing warnings",
                "",
                f"- Missing from control: {len(comparison['missing_from_control'])}",
                f"- Missing from treatment: {len(comparison['missing_from_treatment'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-name", default="discovery_control")
    parser.add_argument("--treatment-name", default="treatment")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    comparison = compare_rollouts(
        load_jsonl(args.control),
        load_jsonl(args.treatment),
        control_arm=args.control_name,
        treatment_arm=args.treatment_name,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    args.report.write_text(render_report(comparison))


if __name__ == "__main__":
    main()
