# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate NeMo Gym observability artifacts into summary.json."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, skipping malformed rows."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "max": None,
        }
    sorted_values = sorted(numeric)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": _percentile(sorted_values, 0.50),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
        "mean": mean(sorted_values),
        "max": sorted_values[-1],
    }


def summarize_observability(output_dir: Path) -> dict[str, Any]:
    """Summarize observability JSONL files under one run directory."""
    events = load_jsonl(output_dir / "events.jsonl")
    resources = load_jsonl(output_dir / "resource_samples.jsonl")
    durations_by_name: dict[str, list[float]] = defaultdict(list)
    durations_by_phase: dict[str, list[float]] = defaultdict(list)
    stop_reasons: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    rewards = []
    trajectory_durations = []

    for event in events:
        attrs = event.get("attributes") or {}
        if event.get("event_type") == "span_end":
            duration_s = attrs.get("duration_s")
            if isinstance(duration_s, (int, float)):
                name = str(event.get("name") or "unknown")
                durations_by_name[name].append(float(duration_s))
                phase = attrs.get("phase") or event.get("phase")
                if phase:
                    durations_by_phase[str(phase)].append(float(duration_s))
            if attrs.get("status") == "error":
                error_type = attrs.get("error_type") or "error"
                errors[str(error_type)] += 1
        if event.get("name") == "trajectory.complete":
            stop_reasons[str(attrs.get("stop_reason") or "unknown")] += 1
            reward = attrs.get("reward")
            if isinstance(reward, (int, float)):
                rewards.append(float(reward))
            duration_s = attrs.get("duration_s")
            if isinstance(duration_s, (int, float)):
                trajectory_durations.append(float(duration_s))
        if event.get("name") == "trajectory.masked":
            stop_reasons[str(attrs.get("stop_reason") or "masked")] += 1

    memory_values = [
        int(sample["memory_usage_bytes"]) for sample in resources if isinstance(sample.get("memory_usage_bytes"), int)
    ]
    cpu_values = [
        float(sample["cpu_utilization"])
        for sample in resources
        if isinstance(sample.get("cpu_utilization"), (int, float))
    ]
    process_counts = [
        int(sample["process_count"]) for sample in resources if isinstance(sample.get("process_count"), int)
    ]

    return {
        "schema_version": 1,
        "events_count": len(events),
        "resource_samples_count": len(resources),
        "durations_by_name": {name: _stats(values) for name, values in sorted(durations_by_name.items())},
        "durations_by_phase": {phase: _stats(values) for phase, values in sorted(durations_by_phase.items())},
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "errors": dict(sorted(errors.items())),
        "reward": _stats(rewards),
        "trajectory_duration_s": _stats(trajectory_durations),
        "resource_peaks": {
            "memory_usage_bytes": max(memory_values) if memory_values else None,
            "cpu_utilization": max(cpu_values) if cpu_values else None,
            "process_count": max(process_counts) if process_counts else None,
        },
    }


def write_summary(output_dir: Path) -> dict[str, Any]:
    """Write ``summary.json`` and return the aggregate summary."""
    summary = summarize_observability(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def flatten_summary_metrics(
    summary: dict[str, Any],
    *,
    prefix: str = "gym/observability",
) -> dict[str, int | float]:
    """Flatten stable summary fields into W&B-friendly scalar metrics."""
    metrics: dict[str, int | float] = {}
    _add_metric(metrics, f"{prefix}/events_count", summary.get("events_count"))
    _add_metric(
        metrics,
        f"{prefix}/resource_samples_count",
        summary.get("resource_samples_count"),
    )

    for group_name in ("reward", "trajectory_duration_s"):
        group = summary.get(group_name)
        if isinstance(group, dict):
            _add_stats(metrics, f"{prefix}/{group_name}", group)

    for key, value in (summary.get("resource_peaks") or {}).items():
        _add_metric(metrics, f"{prefix}/resource_peaks/{key}", value)

    for collection_name in ("durations_by_name", "durations_by_phase"):
        collection = summary.get(collection_name)
        if not isinstance(collection, dict):
            continue
        for name, stats in collection.items():
            if isinstance(stats, dict):
                safe_name = str(name).replace("/", "_")
                _add_stats(metrics, f"{prefix}/{collection_name}/{safe_name}", stats)

    return metrics


def _add_stats(
    metrics: dict[str, int | float],
    prefix: str,
    stats: dict[str, Any],
) -> None:
    for key in ("count", "min", "p50", "p95", "p99", "mean", "max"):
        _add_metric(metrics, f"{prefix}/{key}", stats.get(key))


def _add_metric(
    metrics: dict[str, int | float],
    key: str,
    value: Any,
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        metrics[key] = value
    elif isinstance(value, float):
        metrics[key] = value
