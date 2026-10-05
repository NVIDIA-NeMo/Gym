# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analysis Plan 2 derived metrics, computed per rollout from ``rollouts.jsonl``.

Everything here is derived from the recorded ``ng_trajectory`` / ``ng_perf`` fields; nothing is
re-measured. Rules that matter (plan sections 0 and 2):

* Durations are never summed across spans. Model, tool and subagent spans overlap, so time is
  accounted through interval unions clipped to the agent interval.
* Timestamps from the sandbox clock (``timing_source == "artifact"``) are shifted onto the
  harness clock with the sandbox's measured offset when one was recorded; otherwise the rollout
  is flagged and cross-clock gaps are reported but not trusted.
* Retries are separate attempts. Their time is already inside the logical call's span, so it is
  reported as overhead, never added again to end-to-end time.
* Both harnesses run one decision cycle at a time, so the critical path is the agent interval
  itself and splits exactly into model / tool / other along it. ``parallel_execution_time`` is
  the model-tool overlap, which is zero unless a background command outlived its batch.

Fields that a rollout does not carry (older captures, or harness limits) are reported as
``None`` and listed under ``coverage`` rather than guessed.

Usage::

    python -m benchmarks.nemotron_3.5_super.analysis.derived_metrics \\
        --rollouts results/.../replica_0/rollouts.jsonl results/.../replica_1/rollouts.jsonl \\
        --out metrics.jsonl --summary summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


Interval = tuple[float, float]

UNEXPLAINED_FLAG_FRACTION = 0.05
AGENT_PURPOSES = ("agent_step", "subagent_step")
COMPACTION_PURPOSES = ("compaction_summary", "compaction_question")


# --------------------------------------------------------------------------- interval algebra


def union(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping or touching intervals; returns a sorted, disjoint list."""
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] >= i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def total(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in union(intervals))


def clip(intervals: Iterable[Interval], window: Optional[Interval]) -> list[Interval]:
    if window is None:
        return union(intervals)
    lo, hi = window
    return union((max(s, lo), min(e, hi)) for s, e in intervals if min(e, hi) > max(s, lo))


def intersect(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for s1, e1 in union(a):
        for s2, e2 in union(b):
            lo, hi = max(s1, s2), min(e1, e2)
            if hi > lo:
                out.append((lo, hi))
    return union(out)


def subtract(a: Iterable[Interval], b: Iterable[Interval]) -> list[Interval]:
    """Parts of ``a`` not covered by ``b``."""
    remaining = union(a)
    for s2, e2 in union(b):
        next_remaining: list[Interval] = []
        for s1, e1 in remaining:
            if e2 <= s1 or s2 >= e1:
                next_remaining.append((s1, e1))
                continue
            if s1 < s2:
                next_remaining.append((s1, s2))
            if e2 < e1:
                next_remaining.append((e2, e1))
        remaining = next_remaining
    return remaining


# --------------------------------------------------------------------------- helpers


def _f(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _span(record: dict[str, Any], start_key: str = "started_at", end_key: str = "completed_at") -> Optional[Interval]:
    start, end = _f(record.get(start_key)), _f(record.get(end_key))
    if start is None or end is None or end < start:
        return None
    return (start, end)


def _shift(interval: Optional[Interval], offset: float) -> Optional[Interval]:
    return None if interval is None else (interval[0] - offset, interval[1] - offset)


def _stats(values: list[float]) -> Optional[dict[str, float]]:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    values.sort()
    q = lambda p: values[min(len(values) - 1, int(len(values) * p))]  # noqa: E731
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": values[-1],
    }


def _operation_category(operation: Optional[str]) -> str:
    if not operation:
        return "(none)"
    head = operation.strip().split("\n", 1)[0].strip()
    for prefix in ("cd ", "sudo "):
        if head.startswith(prefix):
            parts = head.split("&&", 1)
            head = parts[1].strip() if len(parts) == 2 else head[len(prefix) :].strip()
    token = head.split(" ", 1)[0] if head else "(none)"
    return token[:32]


# --------------------------------------------------------------------------- per rollout


def derive(row: dict[str, Any]) -> dict[str, Any]:
    """Compute the derived metrics for one rollout row of ``rollouts.jsonl``."""
    trajectory = row.get("ng_trajectory") or {}
    perf = row.get("ng_perf") or {}
    invocations: list[dict[str, Any]] = trajectory.get("invocations") or []
    model_calls: list[dict[str, Any]] = trajectory.get("model_calls") or []
    tool_calls: list[dict[str, Any]] = trajectory.get("tool_calls") or []
    compactions: list[dict[str, Any]] = trajectory.get("compactions") or []
    gaps = {g.get("code") for g in trajectory.get("gaps") or []}
    flags: list[str] = []

    # ---- clocks: shift artifact-timed spans onto the harness clock when an offset was measured
    sandbox_records = [
        r for r in (row.get("ng_agent_observations") or {}).get("records") or [] if r.get("kind") == "sandbox"
    ]
    agent_sandbox = next((r for r in sandbox_records if r.get("role") == "agent"), None)
    clock_offset = _f((agent_sandbox or {}).get("clock_offset_s"))
    clock_uncertainty = _f((agent_sandbox or {}).get("clock_offset_uncertainty_s"))
    artifact_timed = any(t.get("timing_source") == "artifact" for t in tool_calls)
    if artifact_timed and clock_offset is None:
        flags.append("clock_offset_unavailable")
        clock_offset = 0.0
    offset = clock_offset or 0.0

    # ---- rollout and agent bounds
    rollout = _span(perf, "rollout_started_at", "rollout_completed_at")
    root = next((i for i in invocations if i.get("parent_invocation_id") is None), None)
    agent = _span(root) if root is not None else None
    if agent is not None and artifact_timed:
        agent = _shift(agent, offset)
    e2e_rollout = (rollout[1] - rollout[0]) if rollout else (_f(perf.get("total_latency_ms")) or 0) / 1000 or None
    e2e_agent = (agent[1] - agent[0]) if agent else None
    pre_agent = (agent[0] - rollout[0]) if (agent and rollout) else None
    post_agent = (rollout[1] - agent[1]) if (agent and rollout) else None

    # ---- spans
    model_spans = [s for s in (_span(c) for c in model_calls) if s]
    tool_spans = [
        s
        for s in (_shift(_span(t), offset) if t.get("timing_source") == "artifact" else _span(t) for t in tool_calls)
        if s
    ]
    compaction_spans = [s for s in (_span(c, "observed_at", "completed_at") for c in compactions) if s]
    u_model = clip(model_spans, agent)
    u_tool = clip(tool_spans, agent)
    u_both = intersect(u_model, u_tool)
    covered = union(u_model + u_tool)
    non_model_tool = (e2e_agent - total(covered)) if e2e_agent is not None else None
    known_no_mt = total(subtract(clip(compaction_spans, agent), covered)) if agent else None
    unexplained = (non_model_tool - known_no_mt) if (non_model_tool is not None and known_no_mt is not None) else None
    if unexplained is not None and e2e_agent and unexplained / e2e_agent > UNEXPLAINED_FLAG_FRACTION:
        flags.append("unexplained_agent_time_over_threshold")

    # ---- counts
    by_purpose = Counter(c.get("model_call_purpose") or "unknown" for c in model_calls)
    subagents = [i for i in invocations if i.get("parent_invocation_id") is not None]
    kinds = Counter(c.get("model_response_kind") or "unknown" for c in model_calls)
    agent_steps = [c for c in model_calls if c.get("model_call_purpose") == "agent_step"]
    last_agent_step = max(agent_steps, key=lambda c: _f(c.get("started_at")) or 0, default=None)

    # ---- tokens (agent + subagent steps vs compaction, per plan)
    def toks(calls: list[dict[str, Any]], key: str) -> Optional[int]:
        vals = [(c.get("token_stats") or {}).get(key) for c in calls]
        vals = [v for v in vals if isinstance(v, int)]
        return sum(vals) if vals else None

    tokens = {
        "input": toks(model_calls, "prompt_tokens"),
        "output": toks(model_calls, "completion_tokens"),
        "reasoning": toks(model_calls, "reasoning_tokens"),
        "cached": toks(model_calls, "cached_tokens"),
    }
    reasoning_share = (
        tokens["reasoning"] / (tokens["reasoning"] + tokens["output"])
        if tokens["reasoning"] is not None and tokens["output"] and (tokens["reasoning"] + tokens["output"])
        else None
    )
    cache_share = tokens["cached"] / tokens["input"] if tokens["cached"] is not None and tokens["input"] else None
    step_inputs = [
        (c.get("token_stats") or {}).get("prompt_tokens")
        for c in sorted(agent_steps, key=lambda c: _f(c.get("started_at")) or 0)
    ]
    step_inputs = [v for v in step_inputs if isinstance(v, int)]
    context_growth = (
        {
            "first": step_inputs[0],
            "max": max(step_inputs),
            "last": step_inputs[-1],
            "cumulative": sum(step_inputs),
            "mean_delta_per_call": (step_inputs[-1] - step_inputs[0]) / (len(step_inputs) - 1)
            if len(step_inputs) > 1
            else None,
        }
        if step_inputs
        else None
    )

    # ---- retries
    attempts = [a for c in model_calls for a in (c.get("attempts") or [])]
    attempts_visible = any(c.get("attempts") for c in model_calls)
    failed_attempts = [a for a in attempts if a.get("status") != "completed"]
    calls_with_retry = sum(1 for c in model_calls if len(c.get("attempts") or []) > 1)
    retry_overhead_s = (
        sum((_f(a.get("duration_ms")) or 0) for a in failed_attempts) / 1000 if attempts_visible else None
    )

    # ---- model time by purpose, tool time by operation
    model_time_by_purpose = {
        purpose: _stats(
            [
                (_f(c.get("duration_ms")) or 0) / 1000
                for c in model_calls
                if (c.get("model_call_purpose") or "unknown") == purpose
            ]
        )
        for purpose in by_purpose
    }
    tool_by_op: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "failed": 0, "timeout": 0}
    )
    for t in tool_calls:
        bucket = tool_by_op[
            _operation_category(t.get("operation")) if t.get("operation") else (t.get("tool_name") or "(none)")
        ]
        bucket["count"] += 1
        bucket["total_s"] += (_f(t.get("duration_ms")) or 0) / 1000
        bucket["failed"] += t.get("status") == "failed"
        bucket["timeout"] += t.get("status") == "timeout"

    dispatch = [
        _f(t.get("started_at")) - _f(t.get("requested_at"))
        for t in tool_calls
        if _f(t.get("started_at")) is not None and _f(t.get("requested_at")) is not None
    ]
    observation = [
        _f(t.get("response_received_at")) - _f(t.get("completed_at"))
        for t in tool_calls
        if _f(t.get("response_received_at")) is not None and _f(t.get("completed_at")) is not None
    ]
    engine = [(c.get("response_metadata") or {}).get("engine") or {} for c in model_calls]
    queue_ms = [_f(e.get("queue_time_ms")) for e in engine if _f(e.get("queue_time_ms")) is not None]

    cp_model = total(u_model)
    cp_tool = total(subtract(u_tool, u_model))
    cp_other = (e2e_agent - cp_model - cp_tool) if e2e_agent is not None else None

    return {
        "task_id": trajectory.get("task_id"),
        "rollout_id": trajectory.get("rollout_id"),
        "reward": row.get("reward"),
        "e2e_rollout_time_s": e2e_rollout,
        "e2e_agent_time_s": e2e_agent,
        "pre_agent_time_s": pre_agent,
        "post_agent_time_s": post_agent,
        "interval_union": {
            "model_s": cp_model,
            "tool_s": total(u_tool),
            "overlap_s": total(u_both),
            "non_model_tool_s": non_model_tool,
            "known_no_model_tool_s": known_no_mt,
            "unexplained_s": unexplained,
            "unexplained_fraction": (unexplained / e2e_agent) if (unexplained is not None and e2e_agent) else None,
        },
        "critical_path": {
            "duration_s": e2e_agent,
            "model_s": cp_model,
            "tool_s": cp_tool,
            "other_s": cp_other,
            "model_fraction": (cp_model / e2e_agent) if e2e_agent else None,
            "tool_fraction": (cp_tool / e2e_agent) if e2e_agent else None,
            "other_fraction": (cp_other / e2e_agent) if (e2e_agent and cp_other is not None) else None,
            "parallel_execution_s": total(u_both),
            "assumption": "sequential decision cycle; path == agent interval",
        },
        "counts": {
            "model_calls": len(model_calls),
            "model_calls_by_purpose": dict(by_purpose),
            "agent_steps": by_purpose.get("agent_step", 0),
            "tool_calls": len(tool_calls),
            "subagent_invocations": len(subagents),
            "compactions": len(compactions),
            "valid_tool_action_responses": kinds.get("tool_call", 0),
            "valid_final_responses": 1
            if last_agent_step is not None and last_agent_step.get("model_response_kind") == "text"
            else 0,
            "no_action_turn": kinds.get("other", 0),
            "response_kind_unknown": kinds.get("unknown", 0),
        },
        "tokens": tokens
        | {
            "reasoning_share_of_generated": reasoning_share,
            "prompt_cache_hit_share": cache_share,
            "output_tokens_per_agent_call": (toks(agent_steps, "completion_tokens") / len(agent_steps))
            if agent_steps and toks(agent_steps, "completion_tokens") is not None
            else None,
            "context_growth": context_growth,
        },
        "retries": {
            "attempts_visible": attempts_visible,
            "attempts_total": len(attempts) if attempts_visible else None,
            "failed_model_attempts": len(failed_attempts) if attempts_visible else None,
            "calls_with_retry": calls_with_retry if attempts_visible else None,
            "retry_rate": (calls_with_retry / len(model_calls)) if attempts_visible and model_calls else None,
            "model_retry_overhead_s": retry_overhead_s,
            "retry_delay_fraction": (retry_overhead_s / e2e_agent)
            if (retry_overhead_s is not None and e2e_agent)
            else None,
        },
        "model_time_by_purpose_s": model_time_by_purpose,
        "tool_time_by_operation": dict(tool_by_op),
        "tool_dispatch_delay_s": _stats(dispatch),
        "tool_observation_delay_s": _stats(observation),
        "engine_queue_time_ms": _stats(queue_ms),
        "compaction_time_s": _stats([e - s for s, e in compaction_spans]),
        "clock": {
            "artifact_timed_spans": artifact_timed,
            "offset_s": clock_offset if artifact_timed else None,
            "uncertainty_s": clock_uncertainty if artifact_timed else None,
        },
        "coverage": {
            "rollout_bounds": rollout is not None,
            "agent_span": agent is not None,
            "model_response_kind": kinds.get("unknown", 0) == 0 and bool(model_calls),
            "attempts": attempts_visible,
            "tool_boundaries": bool(dispatch),
            "tool_response_boundary_approximate": "tool_response_boundary_approximate" in gaps,
            "engine_metrics": bool(queue_ms),
            "observation_capture_failed": "observation_capture_failed" in gaps,
        },
        "flags": flags,
    }


# --------------------------------------------------------------------------- aggregate


def iter_rows(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with open(path, "rb") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    per: dict[str, list[float]] = defaultdict(list)
    coverage: Counter = Counter()
    flags: Counter = Counter()
    n = 0
    for m in rows:
        n += 1
        for key in ("e2e_rollout_time_s", "e2e_agent_time_s", "pre_agent_time_s", "post_agent_time_s"):
            if m.get(key) is not None:
                per[key].append(m[key])
        for key in ("model_fraction", "tool_fraction", "other_fraction"):
            if m["critical_path"].get(key) is not None:
                per[f"critical_path.{key}"].append(m["critical_path"][key])
        if m["interval_union"].get("unexplained_fraction") is not None:
            per["unexplained_fraction"].append(m["interval_union"]["unexplained_fraction"])
        for key in ("reasoning_share_of_generated", "prompt_cache_hit_share", "output_tokens_per_agent_call"):
            if m["tokens"].get(key) is not None:
                per[f"tokens.{key}"].append(m["tokens"][key])
        if m["retries"].get("retry_rate") is not None:
            per["retry_rate"].append(m["retries"]["retry_rate"])
        for key, present in m["coverage"].items():
            coverage[key] += bool(present)
        flags.update(m["flags"])
    return {
        "rollouts": n,
        "metrics": {key: _stats(values) for key, values in sorted(per.items())},
        "coverage": {
            key: {"rollouts": count, "fraction": count / n if n else None} for key, count in sorted(coverage.items())
        },
        "flags": dict(flags),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Per-rollout derived metrics, JSONL.")
    parser.add_argument("--summary", type=Path, help="Aggregate over all rollouts, JSON.")
    args = parser.parse_args(argv)

    derived: list[dict[str, Any]] = []
    with open(args.out, "w") as out:
        for row in iter_rows(args.rollouts):
            metrics = derive(row)
            derived.append(metrics)
            out.write(json.dumps(metrics) + "\n")
    if args.summary:
        args.summary.write_text(json.dumps(summarize(derived), indent=2))
    print(f"derived {len(derived)} rollouts -> {args.out}" + (f", summary -> {args.summary}" if args.summary else ""))


if __name__ == "__main__":
    main()
