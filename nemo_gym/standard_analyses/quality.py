# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Effort, reasoning and outcome analyses over one consistent comparison cohort."""

from __future__ import annotations

import math
from collections import defaultdict

from .common import finish, mean, quantile, ratio
from .models import AnalysisResult, Options, Rollout


def long_calls(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    calls = [c for r in rollouts if r.eligible for c in r.calls]
    rows, metrics = [], {}
    for kind, field in (("output", "output_tokens"), ("reasoning", "reasoning_tokens")):
        measured = [c for c in calls if getattr(c, field) is not None]
        timed = [c for c in measured if c.duration_s is not None]
        for threshold in options.thresholds:
            long = [c for c in measured if getattr(c, field) > threshold]
            long_timed = [c for c in timed if getattr(c, field) > threshold]
            count_fraction = ratio(len(long), len(measured))
            time_fraction = ratio(sum(c.duration_s for c in long_timed), sum(c.duration_s for c in timed))
            token_fraction = ratio(sum(getattr(c, field) for c in long), sum(getattr(c, field) for c in measured))
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
    return finish(
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
            ),
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


def token_accumulation(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    rows = []
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
                rows.append(
                    [
                        r.task_id,
                        r.repeat,
                        c.index,
                        c.id,
                        c.invocation_id,
                        c.output_tokens,
                        c.reasoning_tokens,
                        c.input_tokens,
                        output,
                        reasoning,
                        r.success,
                    ]
                )
    return finish(
        "token_accumulation",
        "Per-call output and reasoning accumulation within each rollout",
        rollouts,
        rows=rows,
        columns=[
            "task_id",
            "repeat",
            "call_index",
            "call_id",
            "invocation_id",
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
        extra_coverage={"observations_missing": missing},
        notes=[
            "Call index is capture order, not a semantic main-agent step or a causal order for parallel calls.",
            "A missing token measurement makes subsequent cumulative totals unknown.",
        ],
    )


def repeat_excess(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    tasks = defaultdict(list)
    eligible = [r for r in rollouts if r.complete_time and r.success is not None]
    for r in eligible:
        tasks[r.task_id].append(r)
    rows, metrics = [], {}
    for scope in ("all_scored", "successful"):
        observed, excess, extra_calls, compared_tasks = [], [], [], set()
        for task, samples in sorted(tasks.items()):
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
                    compared_tasks.add(task)
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
    return finish(
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
            "observations_missing": len(rollouts) - len(eligible),
        },
        notes=[
            "Same run, benchmark, agent, model and configuration. Baseline ties use repeat index.",
            "Baseline rows remain in the denominator. Success-only scope needs two successes.",
            "Descriptive lower-envelope comparison, not proven wasted reasoning or recoverable wall time.",
            "Requires declared complete capture; unknown completeness is excluded. Unknown health is reported separately.",
        ],
    )


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
        extra_coverage={"observations_missing": len(eligible) - used},
        notes=[
            "P90 is an inclusive linear quantile over complete eligible rollouts in this cohort.",
            "An outlier label identifies unusual effort, not unnecessary effort.",
        ],
    )


def pass_at_k(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    tasks = defaultdict(list)
    for r in rollouts:
        if r.eligible and r.success is not None:
            tasks[r.task_id].append(r.success)
    rows, metrics = [], {}
    for k in options.pass_k:
        estimates = []
        for task, outcomes in sorted(tasks.items()):
            n, c = len(outcomes), sum(outcomes)
            if n >= k:
                estimate = 1 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0
                estimates.append(estimate)
                rows.append([task, k, n, c, estimate, c / n])
        metrics[f"pass_at_{k}"] = mean(estimates)
        metrics[f"tasks_at_{k}"] = len(estimates)
    return finish(
        "pass_at_k",
        "Task difficulty and pass@k headroom from scored repeats",
        rollouts,
        rows=rows,
        columns=["task_id", "k", "scored_repeats", "successes", "pass_at_k", "empirical_pass_rate"],
        metrics=metrics,
        used=len(rows),
        options=options,
        extra_coverage={"observations_missing": sum(r.eligible and r.success is None for r in rollouts)},
        reason="Need scored binary outcomes; k requires at least k eligible repeats per task",
        notes=[
            "Unbiased estimator 1-C(n-c,k)/C(n,k), averaged equally over tasks eligible at each k.",
            "Assumes exchangeable samples of the same policy/configuration. Adaptive retries violate that assumption.",
            "Task cohorts can differ across k; compare tasks_at_k alongside the estimates.",
        ],
    )


def compaction_impact(rollouts: list[Rollout], options: Options) -> AnalysisResult:
    rows = []
    missing = 0
    for r in rollouts:
        if r.eligible:
            by_id = {c.id: c for c in r.calls}
            boundaries = defaultdict(list)
            for event in r.compactions:
                call = by_id.get(event.after_call_id)
                if call and call.invocation_id == event.invocation_id:
                    boundaries[event.invocation_id].append(call.index)
                else:
                    missing += 1
            for invocation, indices in sorted(boundaries.items()):
                indices = sorted(set(indices))
                calls = sorted((c for c in r.calls if c.invocation_id == invocation), key=lambda c: c.index)
                missing += sum(c.reasoning_tokens is None for c in calls)
                for i, boundary in enumerate(indices):
                    low = indices[i - 1] if i else -1
                    high = indices[i + 1] if i + 1 < len(indices) else math.inf
                    before = [
                        c.reasoning_tokens
                        for c in calls
                        if low <= c.index < boundary and c.reasoning_tokens is not None
                    ]
                    after = [
                        c.reasoning_tokens
                        for c in calls
                        if boundary <= c.index < high and c.reasoning_tokens is not None
                    ]
                    missing += int(not before or not after)
                    rows.append(
                        [
                            r.task_id,
                            r.repeat,
                            invocation,
                            boundary,
                            len(before),
                            len(after),
                            mean(before),
                            mean(after),
                            ratio(mean(after), mean(before)) if before and after else None,
                        ]
                    )
    used = sum(row[4] > 0 and row[5] > 0 for row in rows)
    return finish(
        "compaction_impact",
        "Reasoning before and after explicitly joined compaction boundaries",
        rollouts,
        rows=rows,
        columns=[
            "task_id",
            "repeat",
            "invocation_id",
            "after_call_index",
            "calls_before",
            "calls_after",
            "mean_reasoning_before",
            "mean_reasoning_after",
            "after_before_ratio",
        ],
        metrics={"comparable_boundaries": used},
        used=used,
        options=options,
        extra_coverage={"observations_missing": missing},
        reason="Need explicit compaction boundaries, invocation ownership and reasoning token measurements",
        notes=[
            "Comparison windows stop at adjacent compactions in the same invocation.",
            "Observed association; context, task phase and sample selection can confound the comparison.",
        ],
    )
