# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .models import AnalysisResult, Table


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def quantile(values, q):
    """Inclusive linear quantile; identical to a sorted sample's interpolated rank."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def mean(values):
    return sum(values) / len(values) if values else None


def finish(
    name,
    description,
    rollouts,
    *,
    rows,
    columns,
    metrics,
    used,
    options,
    notes=(),
    reason="Required observations are unavailable",
    extra_coverage=None,
):
    coverage = {
        "rollouts_total": len(rollouts),
        "rollouts_excluded_health": sum(not r.eligible for r in rollouts),
        "rollouts_health_unknown": sum(r.health == "unknown" for r in rollouts),
        "rollouts_capture_complete": sum(r.capture_complete is True for r in rollouts),
        "observations_used": used,
    }
    coverage.update(extra_coverage or {})
    status = "available" if used else "unavailable"
    if used and (
        coverage["rollouts_excluded_health"]
        or any(r.gaps or (r.calls and r.capture_complete is not True) for r in rollouts if r.eligible)
        or coverage.get("observations_missing", 0)
    ):
        status = "partial"
    return AnalysisResult(
        name=name,
        description=description,
        status=status,
        reason=None if used else reason,
        metrics=metrics,
        coverage=coverage,
        tables={"detail": Table.build(columns, rows, options.max_rows)},
        notes=list(notes),
    )
