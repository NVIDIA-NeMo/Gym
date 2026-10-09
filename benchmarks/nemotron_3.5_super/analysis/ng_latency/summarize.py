# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistics of derived metrics across the trials of a run (deterministic, sorted, rounded)."""

from __future__ import annotations

import math
from typing import Any


def _round(value: Any) -> Any:
    return round(value, 6) + 0.0 if isinstance(value, float) else value


def _rank(items: list[float], p: float) -> float:
    return items[max(0, math.ceil(p * len(items)) - 1)]


def stats(values: list[Any]) -> dict[str, Any]:
    items = sorted(float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not items:
        return {
            "count": 0,
            "sum": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    n = len(items)
    median = items[n // 2] if n % 2 else (items[n // 2 - 1] + items[n // 2]) / 2
    return {
        "count": n,
        "sum": _round(sum(items)),
        "mean": _round(sum(items) / n),
        "median": _round(median),
        "p90": _round(_rank(items, 0.9)),
        "p99": _round(_rank(items, 0.99)),
        "min": _round(items[0]),
        "max": _round(items[-1]),
    }


def _is_dist(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"count", "sum", "mean", "median", "p90", "p99", "max"}


def _outcome_group(trial: dict[str, Any]) -> str:
    success = (trial.get("outcome") or {}).get("success")
    return "unknown" if success is None else ("success" if success else "failure")


def _merge_nested(target: dict[str, Any], value: dict[str, Any]) -> None:
    """Sum numeric leaves of a nested count/time dict across trials; collect distinct strings."""
    for key, item in value.items():
        if isinstance(item, dict):
            _merge_nested(target.setdefault(key, {}), item)
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, (int, float)):
            target[key] = _round(target.get(key, 0) + item)
        elif isinstance(item, str):
            target.setdefault(key, set()).add(item)


def _finish_nested(target: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(target):
        item = target[key]
        if isinstance(item, dict):
            out[key] = _finish_nested(item)
        elif isinstance(item, set):
            values = sorted(item)
            out[key] = values[0] if len(values) == 1 else values
        else:
            out[key] = item
    return out


def _dist_entry() -> dict[str, list[float]]:
    return {"pooled": [], "per_trial_sum": [], "per_trial_count": []}


def _dist_stats(entry: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "pooled": stats(entry["pooled"]) if entry["pooled"] else None,
        "per_trial_sum": stats(entry["per_trial_sum"]),
        "per_trial_count": stats(entry["per_trial_count"]),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda t: (str(t.get("replica")), str(t.get("rollout_id"))))
    agent_kinds: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    scalar: dict[str, dict[str, list[float]]] = {}
    dists: dict[str, dict[str, list[float]]] = {}
    dists_by_outcome: dict[str, dict[str, dict[str, list[float]]]] = {}
    nested: dict[str, dict[str, Any]] = {}
    by_purpose: dict[str, dict[str, Any]] = {}
    by_operation: dict[str, dict[str, Any]] = {}
    flags: dict[str, int] = {}
    gaps: dict[str, int] = {}
    # Per-field coverage: how many trials carry a value for each metric (None, an empty distribution and an
    # empty nested counter do not count), so a reader knows the denominator behind every number.
    present: dict[str, int] = {}
    for trial in ordered:
        kind = str(trial.get("agent_kind"))
        agent_kinds[kind] = agent_kinds.get(kind, 0) + 1
        group = _outcome_group(trial)
        by_outcome[group] = by_outcome.get(group, 0) + 1
        for name, value in (trial.get("metrics") or {}).items():
            if _is_dist(value):
                present[name] = present.get(name, 0) + (1 if value["count"] else 0)
            elif isinstance(value, dict):
                present[name] = present.get(name, 0) + (
                    1 if any(v not in (None, 0, {}) for v in value.values()) else 0
                )
            else:
                present[name] = present.get(name, 0) + (1 if value is not None else 0)
            if _is_dist(value):
                series = [v for v in (trial.get("series") or {}).get(name, []) if isinstance(v, (int, float))]
                for entry in (
                    dists.setdefault(name, _dist_entry()),
                    dists_by_outcome.setdefault(name, {}).setdefault(group, _dist_entry()),
                ):
                    entry["pooled"].extend(series)
                    if value["sum"] is not None:
                        entry["per_trial_sum"].append(value["sum"])
                    entry["per_trial_count"].append(value["count"])
            elif isinstance(value, dict):
                _merge_nested(nested.setdefault(name, {}), value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                entry = scalar.setdefault(name, {"all": [], "success": [], "failure": [], "unknown": []})
                entry["all"].append(value)
                entry[group].append(value)
        for purpose, data in (trial.get("by_purpose") or {}).items():
            target = by_purpose.setdefault(
                purpose,
                {"count": 0, "total_time": 0.0, "tokens": {"input": 0, "output": 0, "reasoning": 0, "cached": 0}},
            )
            target["count"] += data.get("count", 0)
            target["total_time"] = _round(target["total_time"] + (data.get("total_time") or 0.0))
            for key in target["tokens"]:
                target["tokens"][key] += (data.get("tokens") or {}).get(key) or 0
        for operation, data in (trial.get("by_operation") or {}).items():
            target = by_operation.setdefault(operation, {"count": 0, "total_time": 0.0, "errors": 0})
            target["count"] += data.get("count", 0)
            target["total_time"] = _round(target["total_time"] + (data.get("total_time") or 0.0))
            target["errors"] += round((data.get("error_rate") or 0.0) * data.get("count", 0))
            for key in ("output_chars", "output_tokens"):
                if key in data:
                    target[key] = target.get(key, 0) + data[key]
        for flag in trial.get("flags") or []:
            flags[flag] = flags.get(flag, 0) + 1
        for gap in trial.get("gaps") or []:
            code = str(gap.get("code"))
            gaps[code] = gaps.get(code, 0) + 1
    metrics: dict[str, Any] = {}
    for name in sorted(set(scalar) | set(dists) | set(nested)):
        if name in scalar:
            metrics[name] = {grp: stats(values) for grp, values in scalar[name].items() if grp == "all" or values}
        elif name in dists:
            metrics[name] = _dist_stats(dists[name])
            metrics[name]["by_outcome"] = {
                grp: _dist_stats(entry) for grp, entry in sorted(dists_by_outcome[name].items())
            }
        else:
            metrics[name] = _finish_nested(nested[name])
    for target in by_operation.values():
        errors = target.pop("errors")
        target["error_rate"] = _round(errors / target["count"]) if target["count"] else None
    return {
        "trials": len(ordered),
        "agent_kinds": dict(sorted(agent_kinds.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "metrics": metrics,
        "by_purpose": dict(sorted(by_purpose.items())),
        "by_operation": dict(sorted(by_operation.items())),
        "flags": dict(sorted(flags.items())),
        "gaps": dict(sorted(gaps.items())),
        "field_coverage": {
            name: {"trials_with_value": count, "fraction": _round(count / len(ordered)) if ordered else None}
            for name, count in sorted(present.items())
        },
    }
