# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed built-in registry. Add analyses through a reviewed library change."""

from __future__ import annotations

from .models import Options, Report, Rollout
from .quality import (
    compaction_impact,
    effort_outliers,
    long_calls,
    pass_at_k,
    repeat_excess,
    token_accumulation,
)
from .systems import runtime_tails, time_distribution


ANALYSES = (
    long_calls,
    repeat_excess,
    effort_outliers,
    token_accumulation,
    compaction_impact,
    pass_at_k,
    runtime_tails,
    time_distribution,
)


def analyze(
    rollouts: list[Rollout], options: Options | None = None, *, disabled: tuple[str, ...] | list[str] = ()
) -> Report:
    """Analyze one run/benchmark/agent/model/config cohort. No network or SDK imports."""
    options = options or Options()
    names = {fn.__name__ for fn in ANALYSES}
    unknown = set(disabled) - names
    if unknown:
        raise ValueError(f"Unknown analyses: {sorted(unknown)}")
    unique = {}
    for rollout in rollouts:
        previous = unique.get(rollout.key)
        if previous is not None and previous != rollout:
            raise ValueError(f"Conflicting attempts for {rollout.key}; select canonical outcomes first")
        unique[rollout.key] = rollout
    ordered = sorted(unique.values(), key=lambda r: r.key)
    if not ordered:
        raise ValueError("At least one rollout is required")
    if len({r.cohort for r in ordered}) != 1:
        raise ValueError("Analyze one run/benchmark/agent/model/config cohort at a time")
    for r in ordered:
        tool_indices = [(t.invocation_id, t.index) for t in r.tools if t.index is not None]
        if len(tool_indices) != len(set(tool_indices)):
            raise ValueError("Tool indices must be unique within an invocation")
    cohort = dict(zip(("run_id", "benchmark", "agent", "model", "config_id"), ordered[0].cohort))
    return Report([fn(ordered, options) for fn in ANALYSES if fn.__name__ not in disabled], cohort, options)
