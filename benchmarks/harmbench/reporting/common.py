# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the benchmark-specific normalizers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import MetricValue, OutcomeCounts


_WHITESPACE = re.compile(r"\s+")


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def excerpt(text: str | None, limit: int = 240) -> str:
    """Whitespace-collapsed prefix; long enough to identify behavior, short enough to stay sanitized."""
    if not text:
        return ""
    collapsed = _WHITESPACE.sub(" ", text).strip()
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def assistant_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for part in item.get("content", []):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    return "\n".join(texts).strip()


def user_text(row: dict[str, Any]) -> str:
    for item in row.get("responses_create_params", {}).get("input", []):
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
            return item["content"]
    return ""


def rollout_id(row: dict[str, Any]) -> str:
    return f"rollout-{row.get('_ng_task_index', 'na')}-{row.get('_ng_rollout_index', 0)}"


def rate_metric(
    metric_id: str,
    name: str,
    numerator: int,
    denominator: int,
    *,
    direction: str,
    kind: str,
    definition: str,
    slice_: dict[str, str] | None = None,
) -> MetricValue:
    return MetricValue(
        id=metric_id,
        name=name,
        value=(numerator / denominator) if denominator else None,
        numerator=numerator,
        denominator=denominator,
        unit="rate",
        direction=direction,
        kind=kind,
        slice=slice_,
        definition=definition,
    )


def count_metric(
    metric_id: str, name: str, value: int, *, kind: str = "operational", definition: str, direction: str = "neutral"
) -> MetricValue:
    return MetricValue(
        id=metric_id, name=name, value=value, unit="count", direction=direction, kind=kind, definition=definition
    )


def outcome_counts(
    expected_ids: list[str],
    scored_ids: list[str],
    failed_ids: list[str],
    *,
    policy_scored: int,
    invalid_model_output: int,
    judge_missing: int,
    pending_annotation: int,
) -> OutcomeCounts:
    scored_counter = Counter(scored_ids)
    duplicates = sorted(task for task, count in scored_counter.items() if count > 1)
    expected = set(expected_ids)
    failed_only = sorted(set(failed_ids) - set(scored_ids))
    missing = sorted(expected - set(scored_ids) - set(failed_only))
    return OutcomeCounts(
        expected_tasks=len(expected),
        expected_rollouts=len(expected_ids),
        scored_rollouts=len(scored_ids),
        policy_scored=policy_scored,
        invalid_model_output=invalid_model_output,
        infrastructure_failure=len(failed_only),
        judge_missing=judge_missing,
        pending_annotation=pending_annotation,
        missing_task_ids=missing,
        duplicate_task_ids=duplicates,
    )


def reconcile(agent_metrics: dict[str, Any], expected: dict[str, float | None], *, tolerance: float = 1e-9) -> None:
    """Fail loudly when a recomputed metric disagrees with Gym's aggregate-metrics file."""
    problems = []
    for key, value in expected.items():
        if key not in agent_metrics:
            continue
        observed = agent_metrics[key]
        if value is None or observed is None:
            if value != observed:
                problems.append(f"{key}: recomputed {value!r} vs aggregate {observed!r}")
        elif abs(float(observed) - float(value)) > tolerance:
            problems.append(f"{key}: recomputed {value!r} vs aggregate {observed!r}")
    if problems:
        raise ValueError("aggregate metrics do not reconcile: " + "; ".join(problems))


def trajectory_bounds(rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Earliest and latest model-call timestamps recorded in the rollouts."""
    from datetime import datetime, timezone

    stamps: list[float] = []
    for row in rows:
        for turn in (row.get("ng_trajectory") or {}).get("turns", []):
            if isinstance(turn.get("timestamp"), (int, float)):
                stamps.append(float(turn["timestamp"]))
        for call in (
            (row.get("ng_model_call_capture") or {}).get("calls", [])
            if isinstance(row.get("ng_model_call_capture"), dict)
            else []
        ):
            for key in ("started_at", "completed_at"):
                if isinstance(call.get(key), (int, float)):
                    stamps.append(float(call[key]))
    if not stamps:
        return None, None
    fmt = lambda value: datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    return fmt(min(stamps)), fmt(max(stamps))


def agent_metrics_from(aggregate_path: Path | None) -> dict[str, Any]:
    if aggregate_path is None or not aggregate_path.exists():
        return {}
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[0].get("agent_metrics", {})
    return payload.get("agent_metrics", payload) if isinstance(payload, dict) else {}
