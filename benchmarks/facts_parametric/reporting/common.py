# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the benchmark-specific normalizers: run-directory loading, outcome reconciliation, excerpts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .schema import OutcomeCounts, RolloutSummary


@dataclass
class RunArtifacts:
    run_dir: Path
    stem: str
    rollouts: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    materialized: list[dict[str, Any]]
    aggregate: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def rollouts_path(self) -> Path:
        return self.run_dir / f"{self.stem}.jsonl"

    @property
    def failures_path(self) -> Path:
        return self.run_dir / f"{self.stem}_failures.jsonl"

    @property
    def materialized_path(self) -> Path:
        return self.run_dir / f"{self.stem}_materialized_inputs.jsonl"

    @property
    def aggregate_path(self) -> Path:
        return self.run_dir / f"{self.stem}_aggregate_metrics.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(run_dir: Path, stem: str) -> RunArtifacts:
    run_dir = Path(run_dir)
    aggregate_path = run_dir / f"{stem}_aggregate_metrics.json"
    aggregate: dict[str, Any] = {}
    if aggregate_path.exists():
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            aggregate = payload[0].get("agent_metrics", {})
        elif isinstance(payload, dict):
            aggregate = payload.get("agent_metrics", payload)
    quality_path = run_dir / "quality_summary.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    return RunArtifacts(
        run_dir=run_dir,
        stem=stem,
        rollouts=read_jsonl(run_dir / f"{stem}.jsonl"),
        failures=read_jsonl(run_dir / f"{stem}_failures.jsonl"),
        materialized=read_jsonl(run_dir / f"{stem}_materialized_inputs.jsonl"),
        aggregate=aggregate,
        quality=quality,
    )


def rollout_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row.get("_ng_task_index", 0)), int(row.get("_ng_rollout_index", 0))


def classify_failure(row: dict[str, Any]) -> str:
    failure_class = str(row.get("_ng_failure_class") or "")
    if failure_class == "judge_failed":
        return "judge_failed"
    if failure_class.endswith("simulation_failed"):
        return "simulation_failed"
    return "infrastructure_failed"


def reconcile(
    artifacts: RunArtifacts, *, task_id_of: Callable[[dict[str, Any]], str], num_repeats: int = 1
) -> tuple[OutcomeCounts, dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    """Reconcile expected (materialized x repeats) rollouts against scored rows and sidecar attempts.

    Returns the outcome counts, the scored rows keyed by (task, rollout), and the unresolved failure rows (latest
    attempt per rollout that never scored).
    """
    expected_keys = {(index, repeat) for index in range(len(artifacts.materialized)) for repeat in range(num_repeats)}
    scored: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in artifacts.rollouts:
        key = rollout_key(row)
        if key in scored:
            duplicates += 1
        scored[key] = row
    latest_failure: dict[tuple[int, int], dict[str, Any]] = {}
    for row in artifacts.failures:
        latest_failure[rollout_key(row)] = row
    replaced = sum(1 for key in latest_failure if key in scored)
    unresolved = [row for key, row in latest_failure.items() if key not in scored]
    counts = {"judge_failed": 0, "simulation_failed": 0, "infrastructure_failed": 0}
    for row in unresolved:
        counts[classify_failure(row)] += 1
    observed = set(scored) | set(latest_failure)
    missing = sorted(expected_keys - observed)
    missing_task_ids = sorted(
        {task_id_of(artifacts.materialized[index]) for index, _ in missing if index < len(artifacts.materialized)}
    )
    task_ids = [task_id_of(row) for row in artifacts.materialized]
    duplicate_task_ids = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    outcomes = OutcomeCounts(
        expected_tasks=len(artifacts.materialized),
        expected_rollouts=len(expected_keys),
        scored_rollouts=len(scored),
        judge_failed=counts["judge_failed"],
        simulation_failed=counts["simulation_failed"],
        infrastructure_failed=counts["infrastructure_failed"],
        missing_rollouts=len(missing),
        duplicate_rollouts=duplicates,
        replaced_attempts=replaced,
        missing_task_ids=missing_task_ids,
        duplicate_task_ids=duplicate_task_ids,
    )
    return outcomes, scored, unresolved


def excerpt(text: Any, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def failure_summaries(
    unresolved: list[dict[str, Any]], *, task_id_of: Callable[[dict[str, Any]], str]
) -> list[RolloutSummary]:
    summaries = []
    for row in unresolved:
        task, rollout = rollout_key(row)
        summaries.append(
            RolloutSummary(
                task_id=task_id_of(row),
                rollout_id=f"{task}-{rollout}",
                outcome_class=classify_failure(row),  # type: ignore[arg-type]
                reward=None,
                components={
                    "failure_class": row.get("_ng_failure_class"),
                    "error": excerpt(
                        row.get("_ng_failure_judge_error") or row.get("error") or row.get("_ng_failure_error")
                    ),
                },
            )
        )
    return summaries


def git_revision(repo_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - only outside a checkout
        return "unknown"


def wilson_ci(successes: float, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))
