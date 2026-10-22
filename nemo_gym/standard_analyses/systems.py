# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime and measured time-distribution analyses."""

from __future__ import annotations

from collections import defaultdict

from .common import finish, mean, quantile, ratio
from .models import AnalysisResult, Options, Rollout


def runtime_tails(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    tasks = defaultdict(list)
    for r in rollouts:
        if r.eligible and r.wall_s is not None:
            tasks[r.task_id].append(r.wall_s)
    medians = {task: quantile(times, 0.5) for task, times in tasks.items()}
    p10, p90 = quantile(list(medians.values()), 0.1), quantile(list(medians.values()), 0.9)
    rows = [
        [task, len(tasks[task]), seconds, seconds <= p10, seconds > p90]
        for task, seconds in sorted(medians.items(), key=lambda x: (-x[1], x[0]))
    ]
    return finish(
        "runtime_tails",
        "Fast and tail tasks by median observed rollout wall time",
        rollouts,
        rows=rows,
        columns=[
            "task_id",
            "measured_repeats",
            "median_wall_s",
            "fast_at_or_below_p10",
            "tail_above_p90",
        ],
        metrics={"task_wall_s_p10": p10, "task_wall_s_p90": p90, "measured_tasks": len(tasks)},
        used=len(rows),
        options=options,
        extra_coverage={"observations_missing": sum(r.eligible and r.wall_s is None for r in rollouts)},
        reason="Need measured rollout wall time; summed model time is not substituted",
        notes=["One median per task avoids weighting tasks by their number of repeats."],
    )


def _union(intervals):
    if not intervals:
        return None
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def time_distribution(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    rows = []
    model_total = tool_total = 0
    measured = missing = 0
    for r in rollouts:
        if r.eligible:
            model = [c.duration_s for c in r.calls if c.duration_s is not None]
            tools = [t.duration_s for t in r.tools if t.duration_s is not None]
            measured += len(model) + len(tools)
            missing += len(r.calls) + len(r.tools) - len(model) - len(tools)
            model_s = sum(model) if model else None
            tool_s = sum(tools) if tools else None
            model_total += model_s or 0
            tool_total += tool_s or 0
            interval_calls = (*r.calls, *r.tools)
            complete_intervals = bool(interval_calls) and all(
                c.started_at is not None and c.duration_s is not None for c in interval_calls
            )
            activity_union = (
                _union([(c.started_at, c.started_at + c.duration_s) for c in interval_calls])
                if complete_intervals
                else None
            )
            gross = sum(model) + sum(tools)
            rows.append(
                [
                    r.task_id,
                    r.repeat,
                    model_s,
                    tool_s,
                    ratio(model_s, gross) if model_s is not None else None,
                    activity_union,
                    gross - activity_union if activity_union is not None else None,
                    r.wall_s,
                ]
            )
    model_durations = [c.duration_s for r in rollouts if r.eligible for c in r.calls if c.duration_s is not None]
    tool_durations = [t.duration_s for r in rollouts if r.eligible for t in r.tools if t.duration_s is not None]
    return finish(
        "time_distribution",
        "Measured time in model calls and tools, with interval overlap when available",
        rollouts,
        rows=rows,
        columns=[
            "task_id",
            "repeat",
            "model_s",
            "tool_s",
            "model_fraction_of_summed_call_time",
            "observed_activity_union_s",
            "overlapping_call_seconds",
            "rollout_wall_s",
        ],
        metrics={
            "model_s": model_total if model_durations else None,
            "tool_s": tool_total if tool_durations else None,
            "model_call_mean_s": mean(model_durations),
            "model_call_p90_s": quantile(model_durations, 0.9),
            "tool_call_mean_s": mean(tool_durations),
            "tool_call_p90_s": quantile(tool_durations, 0.9),
        },
        used=measured,
        options=options,
        extra_coverage={"observations_missing": missing},
        notes=[
            "Model and tool sums can overlap and do not decompose agent wall time.",
            "Union covers only captured activity. All starts must use one comparable clock.",
            "Overlapping call-seconds measure multiplicity beyond union duration; no critical path is inferred.",
        ],
    )
