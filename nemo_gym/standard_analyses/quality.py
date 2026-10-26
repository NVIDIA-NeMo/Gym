# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Effort, reasoning and outcome analyses over one consistent comparison cohort."""

from __future__ import annotations

import re
from collections import defaultdict

from .common import finish, mean, quantile, ratio
from .models import AnalysisResult, Options, Rollout, Table


def long_calls(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    calls = [c for r in rollouts if r.eligible for c in r.calls]
    rows, metrics = [], {}
    charts = {name: [] for name in ("call", "time", "token")}
    for kind, field in (("output", "output_tokens"), ("reasoning", "reasoning_tokens")):
        measured = [c for c in calls if getattr(c, field) is not None]
        timed = [c for c in measured if c.duration_s is not None]
        for threshold in options.thresholds:
            long = [c for c in measured if getattr(c, field) > threshold]
            long_timed = [c for c in timed if getattr(c, field) > threshold]
            count_fraction = ratio(len(long), len(measured))
            time_fraction = ratio(sum(c.duration_s for c in long_timed), sum(c.duration_s for c in timed))
            token_fraction = ratio(sum(getattr(c, field) for c in long), sum(getattr(c, field) for c in measured))
            for label, value in (("call", count_fraction), ("time", time_fraction), ("token", token_fraction)):
                if value is not None:
                    charts[label].append([f"{kind} >{threshold}", value])
            rows.append(
                [
                    kind,
                    threshold,
                    len(calls),
                    len(measured),
                    len(timed),
                    len(long),
                    count_fraction,
                    time_fraction,
                    token_fraction,
                ]
            )
            for key, value in (
                ("call_fraction", count_fraction),
                ("time_fraction", time_fraction),
                ("token_fraction", token_fraction),
            ):
                metrics[f"{kind}_gt_{threshold}_{key}"] = value
    measured = sum(c.output_tokens is not None for c in calls)
    result = finish(
        "long_calls",
        "Long-output and long-reasoning shares of captured calls, time and tokens",
        rollouts,
        rows=rows,
        columns=[
            "token_kind",
            "threshold",
            "captured_calls",
            "calls_with_tokens",
            "calls_with_tokens_and_time",
            "long_calls",
            "call_fraction",
            "model_time_fraction",
            "token_fraction",
        ],
        metrics=metrics,
        used=sum(c.output_tokens is not None or c.reasoning_tokens is not None for c in calls),
        options=options,
        extra_coverage={
            "observations_missing": sum(
                c.output_tokens is None or c.reasoning_tokens is None or c.duration_s is None for c in calls
            )
            + sum(r.eligible and r.capture_complete is not True for r in rollouts),
            "calls_missing_output": len(calls) - measured,
            "calls_missing_reasoning": sum(c.reasoning_tokens is None for c in calls),
            "calls_missing_time": sum(c.duration_s is None for c in calls),
        },
        notes=[
            "Strictly greater than threshold. Time denominator is measured model-call seconds, including failed attempts.",
            "Output tokens include reasoning once. Reasoning token shares use their own measured denominator.",
            "Captured calls are not necessarily agent turns. Missing reasoning remains unknown.",
        ],
    )

    result.tables.update(
        {
            f"{name}_share_table": Table.build(["threshold", "fraction"], values, options.max_rows)
            for name, values in charts.items()
        }
    )
    return result


def token_accumulation(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    rows, chart_rows = [], []
    measured = missing = 0
    for r in rollouts:
        if r.eligible:
            output, reasoning = 0, 0
            for c in sorted(r.calls, key=lambda c: c.index):
                output = output + c.output_tokens if output is not None and c.output_tokens is not None else None
                reasoning = (
                    reasoning + c.reasoning_tokens
                    if reasoning is not None and c.reasoning_tokens is not None
                    else None
                )
                measured += c.output_tokens is not None
                missing += c.output_tokens is None or c.reasoning_tokens is None
                if output is not None:
                    chart_rows.append([c.index, f"{r.task_id} / repeat {r.repeat}", output])
                rows.append(
                    [
                        r.task_id,
                        r.repeat,
                        c.index,
                        c.id,
                        c.output_tokens,
                        c.reasoning_tokens,
                        c.input_tokens,
                        output,
                        reasoning,
                        r.success,
                    ]
                )
    result = finish(
        "token_accumulation",
        "Per-call output and reasoning accumulation within each rollout",
        rollouts,
        rows=rows,
        columns=[
            "task_id",
            "repeat",
            "call_index",
            "call_id",
            "output_tokens",
            "reasoning_tokens",
            "input_tokens",
            "cumulative_output_tokens",
            "cumulative_reasoning_tokens",
            "success",
        ],
        metrics={"calls_measured": measured},
        used=measured,
        options=options,
        extra_coverage={
            "observations_missing": missing + sum(r.eligible and r.capture_complete is not True for r in rollouts)
        },
        notes=[
            "Call index is canonical trajectory order, not a semantic agent step or causal order for parallel calls.",
            "A missing token measurement makes subsequent cumulative totals unknown.",
        ],
    )

    series = sorted({row[1] for row in chart_rows})[:24]
    result.tables["chart_table"] = Table.build(
        ["step", "lineKey", "lineVal"], [row for row in chart_rows if row[1] in series], options.max_rows
    )
    return result


def repeat_excess(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    tasks = defaultdict(list)
    eligible = [r for r in rollouts if r.complete_time and r.success is not None]
    for r in eligible:
        tasks[(r.task_id, str(r.stage))].append(r)
    rows, chart_rows, metrics = [], [], {}
    for scope in ("all_scored", "successful"):
        observed, excess, extra_calls, compared_tasks = [], [], [], set()
        for (task, stage), samples in sorted(tasks.items()):
            successes = [r for r in samples if r.success]
            if len(samples) >= 2 and successes and (scope == "all_scored" or len(successes) >= 2):
                baseline = min(successes, key=lambda r: (r.model_s, r.repeat))
                selected = samples if scope == "all_scored" else successes
                for r in sorted(selected, key=lambda r: r.repeat):
                    delta = max(r.model_s - baseline.model_s, 0)
                    observed.append(r.model_s)
                    excess.append(delta)
                    extra = max(len(r.calls) - len(baseline.calls), 0)
                    extra_calls.append(extra)
                    compared_tasks.add((task, stage))
                    if scope == "all_scored":
                        chart_rows.append([f"{task} / repeat {r.repeat}", delta])
                    rows.append(
                        [
                            scope,
                            task,
                            r.repeat,
                            r.success,
                            r.model_s,
                            baseline.repeat,
                            baseline.model_s,
                            delta,
                            extra,
                        ]
                    )
        metrics.update(
            {
                f"{scope}_comparable_tasks": len(compared_tasks),
                f"{scope}_compared_rollouts": len(observed),
                f"{scope}_model_s": sum(observed) if observed else None,
                f"{scope}_excess_s": sum(excess) if observed else None,
                f"{scope}_excess_fraction": ratio(sum(excess), sum(observed)),
                f"{scope}_mean_extra_calls": mean(extra_calls),
            }
        )
    result = finish(
        "repeat_excess",
        "Observed excess model-call effort versus the fastest successful repeat",
        rollouts,
        rows=rows,
        columns=[
            "scope",
            "task_id",
            "repeat",
            "success",
            "model_s",
            "baseline_repeat",
            "baseline_model_s",
            "excess_model_s",
            "extra_calls",
        ],
        metrics=metrics,
        used=len(rows),
        options=options,
        reason="Need two complete scored repeats of one task, including at least one success",
        extra_coverage={
            "rollouts_usable": len(eligible),
            "observations_missing": sum(r.eligible for r in rollouts) - len(eligible),
        },
        notes=[
            "Same run, benchmark, agent, model and configuration. Baseline ties use repeat index.",
            "Baseline rows remain in the denominator. Success-only scope needs two successes.",
            "Descriptive lower-envelope comparison, not proven wasted reasoning or recoverable wall time.",
            "Requires declared complete capture; unknown completeness is excluded. Unknown health is reported separately.",
        ],
    )

    result.tables["chart_table"] = Table.build(
        ["task_repeat", "excess_model_s"], chart_rows, min(100, options.max_rows)
    )
    return result


def effort_outliers(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    eligible = [r for r in rollouts if r.eligible]
    values = []
    for r in eligible:
        tokens = (
            sum(c.output_tokens for c in r.calls)
            if r.capture_complete is True and r.calls and all(c.output_tokens is not None for c in r.calls)
            else None
        )
        values.append((r, tokens, r.model_s if r.complete_time else None))
    token_p90 = quantile([tokens for _, tokens, _ in values if tokens is not None], 0.9)
    model_p90 = quantile([seconds for _, _, seconds in values if seconds is not None], 0.9)
    rows = [
        [
            r.task_id,
            r.repeat,
            r.success,
            tokens,
            seconds,
            tokens > token_p90 if tokens is not None else None,
            seconds > model_p90 if seconds is not None else None,
        ]
        for r, tokens, seconds in values
    ]
    used = sum(t is not None or s is not None for _, t, s in values)
    return finish(
        "effort_outliers",
        "Rollouts with outlying total output tokens or measured model time",
        rollouts,
        rows=rows,
        columns=[
            "task_id",
            "repeat",
            "success",
            "output_tokens",
            "model_s",
            "tokens_above_p90",
            "model_time_above_p90",
        ],
        metrics={"output_tokens_p90": token_p90, "model_s_p90": model_p90},
        used=used,
        options=options,
        extra_coverage={"observations_missing": sum(t is None or s is None for _, t, s in values)},
        notes=[
            "P90 is an inclusive linear quantile over complete eligible rollouts in this cohort.",
            "An outlier label identifies unusual effort, not unnecessary effort.",
        ],
    )


def pass_at_k(
    rollouts: list[Rollout], options: Options, *, metrics: dict[str, float | int] | None = None
) -> AnalysisResult:
    """Display the environment's authoritative aggregate pass@k; never rescore rollouts."""
    rows = []
    for key, value in (metrics or {}).items():
        match = re.fullmatch(r"pass@(\d+)/(.+)", key)
        if match and isinstance(value, (float, int)):
            rows.append([int(match[1]), match[2], value])
    rows.sort(key=lambda row: (row[1], row[0]))
    return finish(
        "pass_at_k",
        "Pass@k percentages from Gym aggregate metrics",
        rollouts,
        rows=rows,
        columns=["k", "score", "pass_at_k_percent"],
        metrics={},
        used=len(rows),
        options=options,
        reason="No aggregate pass@k metrics with an unambiguous cohort binding",
        notes=[
            "Uses the environment's aggregate results and its masking, failure and missing-rollout policy.",
            "Percentages (0–100). Health exclusions apply to effort diagnostics, not these benchmark scores.",
            "Original scalar metrics remain under the environment's existing exporter keys.",
        ],
    )
