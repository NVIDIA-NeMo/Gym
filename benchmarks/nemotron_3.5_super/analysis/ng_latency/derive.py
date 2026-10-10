# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived per-trial metrics from one rollouts_timeline record (+ its rollouts.jsonl row)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ng_latency.build import _tool_output_texts
from ng_latency.graph import build_graph, critical_path, parallel_execution_time
from ng_latency.intervals import Interval, intersect, normalize, overlap_time, subtract, total, union


UNEXPLAINED_FLAG_FRACTION = 0.05
# Per-rollout conditions that change how a number must be read; the static limitations section covers the rest.
FLAGS: dict[str, str] = {
    "agent_end_unobserved": (
        "The OpenCode process was killed (sandbox timeout or failure) and left no exit mark, so the agent span ends "
        "at the last event OpenCode persisted: e2e_agent_time is a lower bound for this rollout."
    ),
    "clock_offset_unavailable": (
        "The sandbox clock probe failed, so sandbox-clock spans cannot be placed against the node clock: cross-clock "
        "values are null for this rollout and its timeline lane is drawn unplaced."
    ),
    "unexplained_gap_over_5pct": "Agent time covered by no logged event exceeds 5 % of this rollout's agent time.",
    "critical_path_exceeds_agent_time": (
        "The computed critical path is longer than the agent span, so the dependency graph or the clocks are "
        "inconsistent for this rollout; its critical-path fractions should not be used."
    ),
    "tool_response_boundary_approximate": (
        "OpenCode's tool_call_response_received is the creation of the next assistant message: an upper bound on "
        "when the result was in the prompt, so tool_observation_delay is an upper bound too."
    ),
    "open_span_excluded": "A span with no end was excluded from durations and unions: totals are lower bounds.",
    "agent_artifact_unavailable": (
        "The agent's own record (OpenCode database or Terminus invocation) was not captured: model calls are known "
        "from the server only and the rollout is excluded from agent-side metrics."
    ),
    "server_spans_missing": (
        "No model-server records joined this rollout, so engine time, tokens and response classification are null."
    ),
    "model_call_join_unverified": (
        "A captured call was paired with an OpenCode step by session order and the token counts disagreed; per-step "
        "attribution for that call may be off by one step, per-rollout totals are unaffected."
    ),
    "model_steps_unavailable": "OpenCode's steps could not be read from its database; model calls are server-side only.",
    "row_missing": "The timeline record has no matching row in rollouts.jsonl, so reward and outcome are unknown.",
}
GENERATION_TAIL_TOKENS = 10_000  # a model call generating more than this is part of the generation tail
_TRIVIAL_COMMANDS = ("cd", "cat", "ls", "echo", "pwd")  # Terminus commands that never need a long wait
_TRIVIAL_WAIT_S = 5.0
_READ_TOOLS = ("read", "grep", "glob")  # OpenCode tools that only look

DEFINITIONS: dict[str, str] = {
    "e2e_rollout_time": "rollout span duration (collector monotonic clock)",
    "e2e_agent_time": "agent span duration (harness clock)",
    "pre_agent_time": (
        "Terminus: agent.start - rollout.start; "
        "OpenCode: (agent_sandbox_exec.start - rollout.start) + opencode_startup_s"
    ),
    "post_agent_time": (
        "Terminus: rollout.end - agent.end; OpenCode: (rollout.end - agent_sandbox_exec.end) + opencode_shutdown_s"
    ),
    "lifecycle_residual_time": (
        "e2e_rollout_time - (pre_agent_time + e2e_agent_time + post_agent_time); OpenCode: sandbox-exec time outside "
        "the OpenCode process lifetime (exec start -> process launch, process exit -> exec end); Terminus: 0"
    ),
    "model_union_time": "|U(model_call)| over every purpose, clipped to the agent window",
    "tool_union_time": "|U(tool_call)| (OpenCode) or |U(tool_batch)| (Terminus), clipped to the agent window",
    "model_tool_overlap_time": "|U_model ∩ U_tool|",
    "non_model_tool_time": "e2e_agent_time - |U_model ∪ U_tool|",
    "known_no_model_tool_time": (
        "time outside U_model ∪ U_tool covered by a logged event "
        "(compaction, subagent, step tail, observation assembly, shutdown)"
    ),
    "unexplained_no_model_tool_time": (
        "non_model_tool_time - known_no_model_tool_time; flagged when > 5 % of e2e_agent_time"
    ),
    "unexplained_fraction": "unexplained_no_model_tool_time / e2e_agent_time",
    "agent_unattributed_time": "same quantity as unexplained_no_model_tool_time",
    "overlap_time": "time with two or more leaf work spans (model_call, tool_call/tool_batch) active",
    "critical_path_duration": "longest start-to-end path through the dependency graph including waits",
    "cp_model_time": "model-call time on the critical path",
    "cp_tool_time": "tool time on the critical path",
    "cp_other_time": "waits and everything else on the critical path",
    "cp_model_fraction": "cp_model_time / critical_path_duration",
    "cp_tool_fraction": "cp_tool_time / critical_path_duration",
    "cp_other_fraction": "cp_other_time / critical_path_duration",
    "parallel_execution_time": "time with two or more independent work nodes active",
    "agent_steps": "main-session model calls with purpose agent_step",
    "model_calls": "all logical model calls; model_calls_by_purpose groups agent_step, subagent_step, compaction",
    "model_call_time": "logical model_call duration (agent-observed; OpenCode: send to first part arrival)",
    "model_time_by_purpose": "per purpose: count, total model_call_time, token sums from the linked server span",
    "model_engine_queue_time": "vLLM queue_time per request, from the engine's per-request metrics in the response",
    "model_engine_prefill_time": "vLLM time_to_first_token - queue_time per request",
    "model_engine_decode_time": "vLLM generation_time per request",
    "model_engine_e2e_time": "vLLM time_to_first_token + generation_time per request",
    "gym_hop_overhead_time": "model-server duration - engine e2e per call (the Gym hop and vLLM's API server)",
    "context_length_at_call": "server input tokens (including cached) per call in start order",
    "context_growth": "last - first input tokens over main-session agent steps",
    "max_context": "largest input token count over main-session agent steps",
    "cumulative_input_tokens": "sum of input tokens over main-session agent steps",
    "context_growth_per_call": "successive input-token differences over main-session agent steps",
    "failed_model_attempts": (
        "attempts ending in error/timeout/cancelled (Terminus attempt spans; OpenCode server-visible)"
    ),
    "attempts_total": "attempts per logical call",
    "model_retry_overhead_time": (
        "Terminus: logical duration - successful attempt duration (0 for a single ok attempt; a call with no "
        "successful attempt, e.g. cancelled by the rollout timeout, contributes its whole duration; calls without "
        "attempt spans are left out); OpenCode: sum of failed server-visible attempts"
    ),
    "retry_rate": "logical calls with more than one attempt / logical calls",
    "retry_delay_fraction": "sum of model_retry_overhead_time / e2e_agent_time",
    "valid_tool_action_responses": "completed responses with at least one valid tool action",
    "valid_final_responses": "main-session responses accepted as the final answer",
    "produced_action": "responses that were a valid final answer or a valid tool action",
    "no_action_turn": "completed step responses with neither, by reason (parse_failure, empty, other)",
    "unusable_model_outputs": (
        "no_action_turn plus completed responses of kind other and responses the harness discarded and answered "
        "with a compaction, by reason: context_overflow (Gym's empty answer to an over-context request) or "
        "length_cap (the response hit the output cap)"
    ),
    "tool_calls": "tool_call spans (per command for Terminus, per tool for OpenCode)",
    "tool_batches": "Terminus command batches",
    "tool_execution_time": "tool_call duration",
    "tool_dispatch_delay": (
        "tool_call.start - requested_wall_s (Terminus per command; OpenCode: 0, the span starts at the request)"
    ),
    "tool_observation_delay": (
        "Terminus: response_received_wall_s - tool_batch.end per batch; OpenCode: 0 (not observable)"
    ),
    "tool_time_by_operation": (
        "per category: count, total duration, error_rate (and Terminus timed_out_batches, error_batches, empty_batches)"
    ),
    "observation_volume": "tool output chars and tokens, total and per category (Terminus per batch)",
    "subagent_invocations": "subagent spans; resumed when the child's first message precedes the request",
    "subagent_time": "subagent span duration",
    "agent_to_subagent_time": "subagent.start - requested_wall_s",
    "subagent_to_agent_time": "response_observed_wall_s - subagent.end",
    "compactions": "compaction spans",
    "compaction_time": "compaction span duration",
    "reasoning_share_of_generated_tokens": "Σ reasoning / (Σ reasoning + Σ output)",
    "prompt_cache_hit_share": "Σ cached / Σ input (null when any call lacks cached tokens)",
    "tool_arg_share_of_output": "Σ tool_args / Σ output",
    "output_tokens_per_agent_call": "Σ output over agent steps / agent_steps",
    "repeated_actions": (
        "identical tool invocations beyond the first (OpenCode: same tool and title; Terminus: same command text)"
    ),
    "repeated_reads": "OpenCode read calls of a title already read in the rollout (null for Terminus)",
    "interrupts_sent": "Terminus C-c commands (null for OpenCode)",
    "stall_before_end_s": (
        "silence before the agent was stopped: the sandbox exec duration minus the agent-clock time from the agent's "
        "start to the end of its last model call or tool call (agent end minus last activity when no exec span "
        "exists); a hang when large"
    ),
    "generation_tail_calls": f"model calls generating more than {GENERATION_TAIL_TOKENS} tokens (reasoning + output)",
    "generation_tail_tokens": "reasoning + output tokens of the generation-tail calls",
    "generation_tail_decode_time": "vLLM decode time of the generation-tail calls",
    "length_stop_calls": "model calls whose stop reason was length (the generation cap)",
    "reasoning_per_step_drift": "mean reasoning tokens of the last three main agent steps minus the first three",
    "wait_analysis": (
        "Terminus waits: count and time of all commands, of bare waits (no command) and of waits of at least "
        f"{_TRIVIAL_WAIT_S:g} s after {', '.join(_TRIVIAL_COMMANDS)}"
    ),
    "tool_wait_share": "Terminus: Σ tool_call duration / e2e_agent_time",
    "per_step_unexplained_time": (
        "unexplained_no_model_tool_time / agent_steps: the per-step tax of harness work no logged event covers"
    ),
}


def _round(value: Any) -> Any:
    return round(value, 6) + 0.0 if isinstance(value, float) else value  # + 0.0 turns -0.0 into 0.0


DIST_KEYS = ("count", "sum", "mean", "median", "p90", "p99", "max")


def dist(values: list[float]) -> dict[str, Any]:
    """count/sum/mean/median/p90/p99/max with nearest-rank percentiles; None-filled when empty."""
    items = sorted(float(v) for v in values if v is not None)
    if not items:
        return {"count": 0, "sum": None, "mean": None, "median": None, "p90": None, "p99": None, "max": None}
    n = len(items)
    median = items[n // 2] if n % 2 else (items[n // 2 - 1] + items[n // 2]) / 2
    p90 = items[max(0, math.ceil(0.9 * n) - 1)]
    p99 = items[max(0, math.ceil(0.99 * n) - 1)]
    return {
        "count": n,
        "sum": _round(sum(items)),
        "mean": _round(sum(items) / n),
        "median": _round(median),
        "p90": _round(p90),
        "p99": _round(p99),
        "max": _round(items[-1]),
    }


@dataclass
class _Context:
    record: dict[str, Any]
    row: Optional[dict[str, Any]]
    spans: list[dict[str, Any]]
    by_name: dict[str, list[dict[str, Any]]]
    kind: str
    agent: Optional[dict[str, Any]]
    main_session: Optional[str]
    domain: Optional[str]
    window: Optional[Interval]
    tool_span_name: str
    codes: set[str] = field(default_factory=set)
    flags: list[str] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    last_action: Optional[dict[str, Any]] = None

    @staticmethod
    def interval(span: dict[str, Any]) -> Optional[Interval]:
        start, end = span.get("start_wall_s"), span.get("end_wall_s")
        if start is None or end is None or end < start:
            return None
        return float(start), float(end)

    @staticmethod
    def duration(span: dict[str, Any]) -> Optional[float]:
        if span.get("duration_ns") is not None:
            return span["duration_ns"] / 1e9
        interval = _Context.interval(span)
        return interval[1] - interval[0] if interval else None

    def harness(self, name: str) -> list[dict[str, Any]]:
        """Spans of ``name`` on the agent's clock, sorted by start."""
        return sorted(
            (s for s in self.by_name.get(name, []) if self.domain is None or s["clock"]["domain"] == self.domain),
            key=lambda s: (s["start_wall_s"], s.get("end_wall_s") or s["start_wall_s"], s["span_id"]),
        )

    def closed(self, spans: list[dict[str, Any]]) -> list[Interval]:
        result = []
        for span in spans:
            interval = self.interval(span)
            if interval is None:
                self.codes.add("open_span_excluded")
            else:
                result.append(interval)
        return result

    @classmethod
    def build(cls, record: dict[str, Any], row: Optional[dict[str, Any]]) -> "_Context":
        spans = [s for s in record.get("spans") or [] if isinstance(s, dict)]
        by_name: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            by_name.setdefault(span["name"], []).append(span)
        kind = "unknown"
        if row is not None and "opencode_finished" in row:
            kind = "opencode"
        elif row is not None and "terminus2_completed" in row:
            kind = "terminus2"
        agents = sorted(by_name.get("agent", []), key=lambda s: s["start_wall_s"])
        agent = agents[0] if agents else None
        domain = agent["clock"]["domain"] if agent else None
        main_session = (agent.get("ids") or {}).get("invocation_id") if agent else None
        window = cls.interval(agent) if agent else None
        tool_span_name = "tool_batch" if kind == "terminus2" else "tool_call"
        return cls(record, row, spans, by_name, kind, agent, main_session, domain, window, tool_span_name)


def _same_domain(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["clock"]["domain"] == b["clock"]["domain"]


def _outcome(ctx: _Context) -> dict[str, Any]:
    row = ctx.row or {}
    reward = row.get("reward")
    outcome: dict[str, Any] = {"reward": reward, "success": (reward > 0) if isinstance(reward, (int, float)) else None}
    if ctx.kind == "opencode":
        outcome["resolved"] = row.get("resolved")
    elif ctx.kind == "terminus2":
        outcome["completed"] = row.get("terminus2_completed")
    return outcome


def _lifecycle(ctx: _Context, metrics: dict[str, Any]) -> None:
    _lifecycle_parts(ctx, metrics)
    parts = (metrics.get("pre_agent_time"), metrics.get("e2e_agent_time"), metrics.get("post_agent_time"))
    rollout_time = metrics.get("e2e_rollout_time")
    metrics["lifecycle_residual_time"] = None
    if rollout_time is not None and None not in parts:
        metrics["lifecycle_residual_time"] = rollout_time - sum(parts)


def _lifecycle_parts(ctx: _Context, metrics: dict[str, Any]) -> None:
    rollout = (ctx.by_name.get("rollout") or [None])[0]
    metrics["e2e_rollout_time"] = ctx.duration(rollout) if rollout else None
    agent_duration = ctx.duration(ctx.agent) if ctx.agent else None
    metrics["e2e_agent_time"] = agent_duration
    if ctx.agent is not None and agent_duration is None:
        ctx.codes.add("open_span_excluded")
    metrics["pre_agent_time"] = metrics["post_agent_time"] = None
    if rollout is None or ctx.agent is None or ctx.interval(rollout) is None:
        return
    r_start, r_end = ctx.interval(rollout)
    if ctx.kind == "opencode":
        execs = ctx.by_name.get("agent_sandbox_exec") or []
        exec_span = execs[0] if execs else None
        exec_interval = ctx.interval(exec_span) if exec_span else None
        startup = ctx.agent["attrs"].get("opencode_startup_s")
        shutdown = ctx.agent["attrs"].get("opencode_shutdown_s")
        if exec_interval is None:
            return
        if not _same_domain(exec_span, rollout):
            ctx.codes.add("clock_domain_mismatch")
            return
        if startup is not None:
            metrics["pre_agent_time"] = (exec_interval[0] - r_start) + startup
        if shutdown is not None and agent_duration is not None:
            metrics["post_agent_time"] = (r_end - exec_interval[1]) + shutdown
        if metrics["pre_agent_time"] is not None or metrics["post_agent_time"] is not None:
            ctx.codes.add("two_clock_sum")
        return
    if not _same_domain(ctx.agent, rollout):
        ctx.codes.add("clock_domain_mismatch")
        return
    metrics["pre_agent_time"] = ctx.agent["start_wall_s"] - r_start
    if ctx.window is not None:
        metrics["post_agent_time"] = r_end - ctx.window[1]


def _empty_decisions(ctx: _Context, *, ran_batch: bool) -> list[dict[str, Any]]:
    """Terminus agent steps that parsed no command. Harbor still runs its (empty) command batch for them, except
    after a parse error, when it goes straight to the next prompt; ``ran_batch`` keeps only the former."""
    return [
        c
        for c in ctx.harness("model_call")
        if c.get("end_wall_s") is not None
        and c["attrs"].get("purpose") == "agent_step"
        and c.get("status") == "ok"
        and not c["attrs"].get("parsed_commands")
        and not (ran_batch and c["attrs"].get("parse_error") is True)
    ]


def _coverage(ctx: _Context) -> dict[str, Any]:
    """Intervals on the harness clock: window, model, tool, known events (by name) and unexplained time."""
    if ctx.window is None:
        return {"window": None, "model": [], "tool": [], "known_by_event": {}, "known": [], "unexplained": []}
    window = ctx.window
    u_model = intersect(ctx.closed(ctx.harness("model_call")), [window])
    u_tool = intersect(ctx.closed(ctx.harness(ctx.tool_span_name)), [window])
    covered = union(u_model, u_tool)
    events: dict[str, list[Interval]] = {
        "compaction": ctx.closed(ctx.harness("compaction")),
        "subagent": ctx.closed(ctx.harness("subagent")),
    }
    if ctx.kind == "opencode":
        tails = []
        for step in ctx.harness("model_step"):
            first = step["attrs"].get("first_part_wall_s")
            if first is not None and step.get("end_wall_s") is not None and step["end_wall_s"] > first:
                tails.append((float(first), float(step["end_wall_s"])))
        events["opencode_step_tail"] = tails
        last_output = ctx.agent["attrs"].get("last_output_wall_s")
        exit_wall = ctx.agent["attrs"].get("process_exit_wall_s")
        shutdown = last_output is not None and exit_wall is not None and exit_wall > last_output
        events["opencode_shutdown"] = [(float(last_output), float(exit_wall))] if shutdown else []
    else:
        assembly = []
        for batch in ctx.harness("tool_batch"):
            received = batch["attrs"].get("response_received_wall_s")
            if received is not None and batch.get("end_wall_s") is not None and received > batch["end_wall_s"]:
                assembly.append((float(batch["end_wall_s"]), float(received)))
        # A decision without commands (the completion claim, its confirmation, a malformed response) still
        # costs Terminus the screen read and the next prompt: the time to the next call, or to the agent's
        # end, is the same harness work as the assembly after a batch.
        starts = [float(c["start_wall_s"]) for c in ctx.harness("model_call")]
        for call in _empty_decisions(ctx, ran_batch=False):
            end = float(call["end_wall_s"])
            following = [s for s in starts if s > end]
            until = min(following) if following else float(window[1])
            if until > end:
                assembly.append((end, until))
        events["terminus_observation_assembly"] = assembly
    known_by_event: dict[str, list[Interval]] = {}
    known_total: list[Interval] = []
    for name in sorted(events):
        extra = subtract(intersect(normalize(events[name]), [window]), covered)
        known_by_event[name] = extra
        known_total = union(known_total, extra)
    unexplained = subtract([window], union(covered, known_total))
    return {
        "window": window,
        "model": u_model,
        "tool": u_tool,
        "known_by_event": known_by_event,
        "known": known_total,
        "unexplained": unexplained,
    }


def coverage_intervals(record: dict[str, Any], agent_kind: str) -> dict[str, Any]:
    """Public entry for the visualisation: the same intervals ``_unions`` measures, keyed by agent kind."""
    row = {"opencode_finished": True} if agent_kind == "opencode" else None
    if agent_kind == "terminus2":
        row = {"terminus2_completed": True}
    return _coverage(_Context.build(record, row))


def _unions(ctx: _Context, metrics: dict[str, Any]) -> None:
    names = (
        "model_union_time",
        "tool_union_time",
        "model_tool_overlap_time",
        "non_model_tool_time",
        "known_no_model_tool_time",
        "known_no_model_tool_time_by_event",
        "unexplained_no_model_tool_time",
        "unexplained_fraction",
        "agent_unattributed_time",
        "overlap_time",
    )
    cov = _coverage(ctx)
    if cov["window"] is None:
        metrics.update({name: None for name in names})
        return
    window = cov["window"]
    agent_time = window[1] - window[0]
    covered = union(cov["model"], cov["tool"])
    non_model_tool = agent_time - total(covered)
    known = total(cov["known"])
    unexplained = max(0.0, non_model_tool - known)
    leaf = ctx.closed(ctx.harness("model_call")) + ctx.closed(ctx.harness(ctx.tool_span_name))
    metrics.update(
        {
            "model_union_time": _round(total(cov["model"])),
            "tool_union_time": _round(total(cov["tool"])),
            "model_tool_overlap_time": _round(total(intersect(cov["model"], cov["tool"]))),
            "non_model_tool_time": _round(non_model_tool),
            "known_no_model_tool_time": _round(known),
            "known_no_model_tool_time_by_event": {k: _round(total(v)) for k, v in cov["known_by_event"].items()},
            "unexplained_no_model_tool_time": _round(unexplained),
            "unexplained_fraction": _round(unexplained / agent_time) if agent_time > 0 else None,
            "agent_unattributed_time": _round(unexplained),
            "overlap_time": _round(overlap_time(intersect_each(leaf, window))),
        }
    )
    if agent_time > 0 and unexplained / agent_time > UNEXPLAINED_FLAG_FRACTION:
        ctx.flags.append("unexplained_gap_over_5pct")
        ctx.codes.add("unexplained_gap_over_5pct")


def intersect_each(intervals: list[Interval], window: Interval) -> list[Interval]:
    """Clip each interval to the window without merging (overlap_time needs the raw multiset)."""
    result = []
    for start, end in intervals:
        clipped = (max(start, window[0]), min(end, window[1]))
        if clipped[1] > clipped[0]:
            result.append(clipped)
    return result


def derive_trial(record: dict[str, Any], row: Optional[dict[str, Any]], *, replica: str = "") -> dict[str, Any]:
    ctx = _Context.build(record, row)
    ctx.gaps.extend(g for g in record.get("gaps") or [] if isinstance(g, dict))
    metrics: dict[str, Any] = {}
    series: dict[str, list[Any]] = {}
    by_purpose: dict[str, Any] = {}
    by_operation: dict[str, Any] = {}
    _lifecycle(ctx, metrics)
    if ctx.kind == "unknown":
        ctx.gaps.append({"code": "agent_kind_unknown"})
    else:
        _unions(ctx, metrics)
        for section in _SECTIONS:
            section(ctx, metrics, series, by_purpose, by_operation)
    # Flags: the record's own (projection) plus the conditions the derivation met.
    flags = set(record.get("flags") or []) | set(ctx.flags)
    flags |= {code for code in ctx.codes if code in FLAGS}
    flags |= {str(gap.get("code")) for gap in ctx.gaps if gap.get("code") in FLAGS}
    if ctx.kind != "unknown" and metrics.get("e2e_agent_time") and metrics.get("agent_steps"):
        unexplained = metrics.get("unexplained_no_model_tool_time")
        metrics["per_step_unexplained_time"] = (
            _round(unexplained / metrics["agent_steps"]) if isinstance(unexplained, (int, float)) else None
        )
    else:
        metrics["per_step_unexplained_time"] = None
    return {
        "kind": "trial",
        "rollout_id": record.get("rollout_id"),
        "task_id": record.get("task_id"),
        "repeat": record.get("repeat"),
        "attempt": record.get("attempt"),
        "replica": replica,
        "agent_kind": ctx.kind,
        "outcome": _outcome(ctx),
        "clock_domains": {
            "rollout": ((ctx.by_name.get("rollout") or [{}])[0].get("clock") or {}).get("domain"),
            "agent": ctx.domain,
        },
        "metrics": {k: _round(v) for k, v in metrics.items()},
        "series": series,
        "by_purpose": by_purpose,
        "by_operation": by_operation,
        "flags": sorted(flags),
        "gaps": ctx.gaps,
        "last_action": ctx.last_action,
    }


_TRUE_WORDS = ("true", "1", "yes")


def terminus_task_complete(text: str) -> Optional[bool]:
    """Read Terminus 2's completion flag the way its JSON parser does; XML form as fallback; None if unparseable."""
    body = text
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        body = fence.group(1)
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            value = data.get("task_complete", False)
            return value.strip().lower() in _TRUE_WORDS if isinstance(value, str) else bool(value)
    match = re.search(r"<task_complete>\s*(.*?)\s*</task_complete>|<task_complete\s*/>", text, re.DOTALL)
    if match:
        inner = (match.group(1) or "").strip().lower()
        return inner in _TRUE_WORDS or inner == ""
    return None


def _assistant_messages(row: Optional[dict[str, Any]]) -> list[tuple[int, str]]:
    """(output index, text) of every assistant message in the recorded trajectory, in order."""
    messages: list[tuple[int, str]] = []
    for index, item in enumerate(((row or {}).get("response") or {}).get("output") or []):
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content")
            if isinstance(content, str):
                messages.append((index, content))
            else:
                messages.append((index, "".join(str(p.get("text", "")) for p in content or [] if isinstance(p, dict))))
    return messages


def _assistant_texts(row: Optional[dict[str, Any]]) -> list[str]:
    return [text for _, text in _assistant_messages(row)]


def _last_assistant_text(row: Optional[dict[str, Any]]) -> Optional[str]:
    texts = _assistant_texts(row)
    return texts[-1] if texts else None


def _purpose_group(purpose: Optional[str]) -> str:
    if purpose and purpose.startswith("compaction"):
        return "compaction"
    return purpose or "agent_step"


def _linked_servers(ctx: _Context) -> dict[str, dict[str, Any]]:
    """Logical model_call span_id -> its server span (status ok preferred, else the last by start)."""
    servers = sorted(ctx.by_name.get("model_call_server", []), key=lambda s: (s["start_wall_s"], s["span_id"]))
    by_id = {s["span_id"]: s for s in servers}
    by_client: dict[str, list[dict[str, Any]]] = {}
    for server in servers:
        client = (server.get("ids") or {}).get("client_call_id")
        if client:
            by_client.setdefault(client, []).append(server)
    linked: dict[str, dict[str, Any]] = {}
    for call in ctx.harness("model_call"):
        candidates: list[dict[str, Any]] = []
        if call["attrs"].get("server_span_id") in by_id:
            candidates = [by_id[call["attrs"]["server_span_id"]]]
        else:
            candidates = by_client.get((call.get("ids") or {}).get("client_call_id") or "", [])
        ok = [c for c in candidates if c.get("status") == "ok"]
        chosen = (ok or candidates)[-1] if (ok or candidates) else None
        if chosen is not None:
            linked[call["span_id"]] = chosen
    return linked


def _server_attempts(ctx: _Context, call: dict[str, Any]) -> list[dict[str, Any]]:
    client = (call.get("ids") or {}).get("client_call_id")
    servers = ctx.by_name.get("model_call_server", [])
    return sorted(
        (s for s in servers if client and (s.get("ids") or {}).get("client_call_id") == client),
        key=lambda s: (s["start_wall_s"], s["span_id"]),
    )


def _engine(ctx: _Context, server: dict[str, Any]) -> dict[str, Any]:
    """The engine's own per-request durations, as the model server recorded them from the response body."""
    return server.get("attrs", {}).get("vllm") or {}


def _is_main(ctx: _Context, span: dict[str, Any]) -> bool:
    ids = span.get("ids") or {}
    return ids.get("invocation_id") == ctx.main_session and not ids.get("parent_invocation_id")


def _model_section(
    ctx: _Context,
    metrics: dict[str, Any],
    series: dict[str, Any],
    by_purpose: dict[str, Any],
    _by_operation: dict[str, Any],
) -> None:
    calls = ctx.harness("model_call")
    linked = _linked_servers(ctx)
    tokens_keys = ("input", "output", "reasoning", "cached")
    purposes: dict[str, int] = {}
    times: list[float] = []
    for call in calls:
        group = _purpose_group(call["attrs"].get("purpose"))
        purposes[group] = purposes.get(group, 0) + 1
        entry = by_purpose.setdefault(group, {"count": 0, "total_time": 0.0, "tokens": {k: 0 for k in tokens_keys}})
        entry["count"] += 1
        duration = ctx.duration(call)
        if duration is None:
            ctx.codes.add("open_span_excluded")
        else:
            times.append(duration)
            entry["total_time"] = _round(entry["total_time"] + duration)
        server_tokens = (linked.get(call["span_id"]) or {}).get("attrs", {}).get("tokens") or {}
        for key in tokens_keys:
            if isinstance(server_tokens.get(key), int):
                entry["tokens"][key] += server_tokens[key]
    main_steps = [c for c in calls if c["attrs"].get("purpose") == "agent_step" and _is_main(ctx, c)]
    metrics["agent_steps"] = len(main_steps)
    metrics["model_calls"] = len(calls)
    metrics["model_calls_by_purpose"] = dict(sorted(purposes.items()))
    metrics["model_call_time"] = dist(times)
    series["model_call_time"] = [_round(t) for t in times]

    # Engine split and proxy overhead, from the linked server span and its vLLM span.
    engine: dict[str, list[float]] = {"queue_s": [], "prefill_s": [], "decode_s": [], "e2e_s": []}
    hops: list[float] = []
    for call in calls:
        server = linked.get(call["span_id"])
        if server is None:
            continue
        attrs = _engine(ctx, server)
        for key in engine:
            if isinstance(attrs.get(key), (int, float)):
                engine[key].append(float(attrs[key]))
        if isinstance(server["attrs"].get("gym_hop_overhead_s"), (int, float)):
            hops.append(float(server["attrs"]["gym_hop_overhead_s"]))
    metrics["model_engine_queue_time"] = dist(engine["queue_s"])
    metrics["model_engine_prefill_time"] = dist(engine["prefill_s"])
    metrics["model_engine_decode_time"] = dist(engine["decode_s"])
    metrics["model_engine_e2e_time"] = dist(engine["e2e_s"])
    metrics["gym_hop_overhead_time"] = dist(hops)
    series["model_engine_queue_time"] = [_round(v) for v in engine["queue_s"]]
    series["model_engine_prefill_time"] = [_round(v) for v in engine["prefill_s"]]
    series["model_engine_decode_time"] = [_round(v) for v in engine["decode_s"]]
    series["model_engine_e2e_time"] = [_round(v) for v in engine["e2e_s"]]
    series["gym_hop_overhead_time"] = [_round(v) for v in hops]

    # Context: main-session agent steps in start order.
    inputs: list[int] = []
    cached: list[Optional[int]] = []
    for call in main_steps:
        tokens = (linked.get(call["span_id"]) or {}).get("attrs", {}).get("tokens") or {}
        if isinstance(tokens.get("input"), int):
            inputs.append(tokens["input"])
            cached.append(tokens.get("cached"))
    series["context_length_at_call"] = inputs
    series["cached_tokens_at_call"] = cached
    diffs = [b - a for a, b in zip(inputs, inputs[1:])]
    metrics["context_growth"] = (inputs[-1] - inputs[0]) if inputs else None
    metrics["max_context"] = max(inputs) if inputs else None
    metrics["cumulative_input_tokens"] = sum(inputs) if inputs else None
    metrics["context_growth_per_call"] = dist([float(d) for d in diffs])
    series["context_growth_per_call"] = diffs

    # Response classification over agent and subagent steps whose server call completed.
    # Terminus: a step is a final response when its own message says task_complete (the harness asks the model
    # to confirm, so a completed trial has two: the claim and the confirmation). Each step's message is the first
    # assistant message after its prompt_trajectory_index (compaction exchanges also sit in the trajectory);
    # without that attribute, messages map to main steps by position, and on a count mismatch only the last
    # step is judged, by the last message.
    last_main = main_steps[-1]["span_id"] if main_steps else None
    main_index = {c["span_id"]: i for i, c in enumerate(main_steps)}
    messages = _assistant_messages(ctx.row) if ctx.kind == "terminus2" else []
    flags_by_output_index = [(index, terminus_task_complete(text)) for index, text in messages]
    terminus_flags = [flag for _, flag in flags_by_output_index]
    per_message = ctx.kind == "terminus2" and len(terminus_flags) == len(main_steps)
    terminus_final = terminus_flags[-1] if terminus_flags else None

    def terminus_is_final(call: dict[str, Any]) -> bool:
        declared = call["attrs"].get("task_complete")
        if isinstance(declared, bool):
            return declared  # the harness parsed the response and recorded the flag itself
        prompt_index = call["attrs"].get("prompt_trajectory_index")
        if isinstance(prompt_index, int):
            return (
                any(flag is True for index, flag in flags_by_output_index if index > prompt_index)
                and next((flag for index, flag in flags_by_output_index if index > prompt_index), None) is True
            )
        if per_message and call["span_id"] in main_index:
            return terminus_flags[main_index[call["span_id"]]] is True
        return call["span_id"] == last_main and terminus_final is True

    step_calls = [c for c in calls if c["attrs"].get("purpose") in ("agent_step", "subagent_step")]
    if step_calls and not linked:
        # Classification needs the server's view of each response; without capture it is not observable.
        for name in ("valid_tool_action_responses", "valid_final_responses", "produced_action"):
            metrics[name] = None
        metrics["no_action_turn"] = metrics["unusable_model_outputs"] = None
        ctx.gaps.append({"code": "server_spans_missing"})
    tool_actions = finals = produced = 0
    no_action: dict[str, int] = {}
    unusable: dict[str, int] = {}
    for call in step_calls if linked else []:
        server = linked.get(call["span_id"])
        if call["attrs"].get("context_overflow") is True:
            # The harness discarded the response and compacted: Gym's empty answer to an over-context request, or
            # a response that ran into the output cap. A length finish the harness did use is judged like any other.
            attrs = (server or {}).get("attrs") or {}
            stop = attrs.get("stop_reason") or attrs.get("gym_finish_reason")
            reason = "length_cap" if stop in ("length", "max_output_tokens") else "context_overflow"
            unusable[reason] = unusable.get(reason, 0) + 1
            continue
        if server is None or server.get("status") != "ok":
            continue
        kind = server["attrs"].get("response_kind")
        chars = (server["attrs"].get("tokens") or {}).get("output")
        if ctx.kind == "terminus2":
            is_tool = (call["attrs"].get("parsed_commands") or 0) > 0
            is_final = terminus_is_final(call)
        else:
            is_tool = kind == "tool_call"
            is_final = kind == "text"
        if is_tool:
            tool_actions += 1
        if is_final and _is_main(ctx, call):
            finals += 1
        if is_tool or is_final:
            produced += 1
            continue
        parse_error = call["attrs"].get("parse_error") is True or (ctx.kind == "terminus2" and kind == "text")
        reason = "empty" if chars == 0 else ("parse_failure" if parse_error else "other")
        if kind != "other":
            no_action[reason] = no_action.get(reason, 0) + 1
        unusable[reason] = unusable.get(reason, 0) + 1
    if linked or not step_calls:
        metrics["valid_tool_action_responses"] = tool_actions
        metrics["valid_final_responses"] = finals
        metrics["produced_action"] = produced
        metrics["no_action_turn"] = {"total": sum(no_action.values()), "by_reason": dict(sorted(no_action.items()))}
        metrics["unusable_model_outputs"] = {
            "total": sum(unusable.values()),
            "by_reason": dict(sorted(unusable.items())),
        }
    if unusable:
        ctx.codes.add("reason_taxonomy_limited")

    # Attempts and retries.
    attempts_by_call: dict[str, list[dict[str, Any]]] = {}
    for attempt in ctx.harness("model_call_attempt"):
        attempts_by_call.setdefault(attempt.get("parent_span_id") or "", []).append(attempt)
    failed_by_status: dict[str, int] = {}
    totals: list[int] = []
    overheads: list[float] = []
    for call in calls:
        overhead: Optional[float]
        if ctx.kind == "terminus2":
            attempts = attempts_by_call.get(call["span_id"], [])
            ok = [a for a in attempts if a.get("status") == "ok"]
            logical = ctx.duration(call)
            ok_duration = ctx.duration(ok[-1]) if ok else None
            if not attempts:
                overhead = None  # no attempt spans: nothing to attribute
            elif len(attempts) == 1 and ok:
                overhead = 0.0  # the wrapper around a single successful attempt is not retry time
            elif logical is not None and ok_duration is not None:
                overhead = logical - ok_duration
            else:
                overhead = logical or 0.0
        else:
            attempts = _server_attempts(ctx, call)
            ctx.codes.add("server_visible_attempts_only")
            overhead = sum(ctx.duration(a) or 0.0 for a in attempts if a.get("status") != "ok") if attempts else None
        totals.append(max(1, len(attempts)))
        if overhead is not None:
            overheads.append(max(0.0, overhead))
        for attempt in attempts:
            if attempt.get("status") != "ok":
                failed_by_status[attempt["status"]] = failed_by_status.get(attempt["status"], 0) + 1
    metrics["failed_model_attempts"] = {
        "total": sum(failed_by_status.values()),
        "by_status": dict(sorted(failed_by_status.items())),
    }
    metrics["attempts_total"] = dist([float(t) for t in totals])
    series["attempts_total"] = totals
    metrics["model_retry_overhead_time"] = dist(overheads)
    series["model_retry_overhead_time"] = [_round(o) for o in overheads]
    metrics["retry_rate"] = (sum(1 for t in totals if t > 1) / len(totals)) if totals else None
    agent_time = metrics.get("e2e_agent_time")
    metrics["retry_delay_fraction"] = (sum(overheads) / agent_time) if agent_time else None

    # Token efficiency ratios over the linked server spans.
    linked_tokens = [s["attrs"].get("tokens") or {} for s in linked.values()]
    total_reasoning = sum(t.get("reasoning") or 0 for t in linked_tokens)
    total_output = sum(t.get("output") or 0 for t in linked_tokens)
    total_input = sum(t.get("input") or 0 for t in linked_tokens)
    total_tool_args = sum(t.get("tool_args") or 0 for t in linked_tokens)
    generated = total_reasoning + total_output
    metrics["reasoning_share_of_generated_tokens"] = (total_reasoning / generated) if generated else None
    cached_known = bool(linked_tokens) and all(isinstance(t.get("cached"), int) for t in linked_tokens)
    total_cached = sum(t["cached"] for t in linked_tokens) if cached_known else None
    metrics["prompt_cache_hit_share"] = (total_cached / total_input) if cached_known and total_input else None
    metrics["tool_arg_share_of_output"] = (total_tool_args / total_output) if total_output else None
    step_output = sum(
        (linked.get(c["span_id"]) or {}).get("attrs", {}).get("tokens", {}).get("output") or 0 for c in main_steps
    )
    metrics["output_tokens_per_agent_call"] = (step_output / len(main_steps)) if main_steps else None


# Metric sections appended by later tasks: each is f(ctx, metrics, series, by_purpose, by_operation) -> None.
_SECTIONS: list = []
_SECTIONS.append(_model_section)


def _operation_category(ctx: _Context, tool: dict[str, Any]) -> str:
    if ctx.kind == "opencode":
        return str(tool["attrs"].get("tool") or "unknown")
    tokens = str(tool["attrs"].get("operation") or "").split()
    return tokens[0] if tokens else "wait"


def _tool_section(
    ctx: _Context,
    metrics: dict[str, Any],
    series: dict[str, Any],
    _by_purpose: dict[str, Any],
    by_operation: dict[str, Any],
) -> None:
    tools = ctx.harness("tool_call")
    batches = ctx.harness("tool_batch") if ctx.kind == "terminus2" else []
    metrics["tool_calls"] = len(tools)
    if ctx.kind == "terminus2":
        metrics["tool_batches"] = len(batches)
    exec_times: list[float] = []
    operations: list[str] = []  # the category of each timed tool call, parallel to exec_times
    dispatch: list[float] = []
    counts: dict[str, dict[str, Any]] = {}
    for tool in tools:
        duration = ctx.duration(tool)
        category = _operation_category(ctx, tool)
        entry = counts.setdefault(
            category, {"count": 0, "total_time": 0.0, "errors": 0, "output_chars": 0, "output_tokens": 0}
        )
        entry["count"] += 1
        if duration is None:
            ctx.codes.add("open_span_excluded")
        else:
            exec_times.append(duration)
            operations.append(category)
            entry["total_time"] = _round(entry["total_time"] + duration)
        if tool.get("status") == "error":
            entry["errors"] += 1
        requested = tool["attrs"].get("requested_wall_s")
        if requested is not None:
            dispatch.append(tool["start_wall_s"] - float(requested))
        if ctx.kind == "opencode":
            entry["output_chars"] += int(tool["attrs"].get("output_chars") or 0)
            entry["output_tokens"] += int((tool["attrs"].get("tokens") or {}).get("tool_output") or 0)
    for category in sorted(counts):
        entry = counts[category]
        summary = {
            "count": entry["count"],
            "total_time": entry["total_time"],
            "error_rate": _round(entry["errors"] / entry["count"]),
        }
        if ctx.kind == "opencode":
            summary.update({"output_chars": entry["output_chars"], "output_tokens": entry["output_tokens"]})
        by_operation[category] = summary
    metrics["tool_execution_time"] = dist(exec_times)
    metrics["tool_dispatch_delay"] = dist(dispatch)
    series["tool_execution_time"] = [_round(t) for t in exec_times]
    series["tool_operation"] = operations
    series["tool_dispatch_delay"] = [_round(d) for d in dispatch]
    if ctx.kind == "terminus2":
        observation = []
        for batch in batches:
            received = batch["attrs"].get("response_received_wall_s")
            if received is not None and batch.get("end_wall_s") is not None:
                observation.append(float(received) - float(batch["end_wall_s"]))
        timed_out = sum(1 for b in batches if b["attrs"].get("timeout") is True)
        errors = sum(1 for b in batches if b.get("status") == "error")
        empty = len(_empty_decisions(ctx, ran_batch=True))  # a screen read without commands, no batch span
        metrics["tool_time_by_operation"] = {
            "timed_out_batches": timed_out,
            "error_batches": errors,
            "empty_batches": empty,
        }
        volume_spans = batches
        chars_key = "observation_chars"
        unit = "batch"
    else:
        observation = [0.0] * len(tools)
        if tools:
            ctx.codes.add("not_observable_zero")
        metrics["tool_time_by_operation"] = {}
        volume_spans = tools
        chars_key = "output_chars"
        unit = "tool"
    metrics["tool_observation_delay"] = dist(observation)
    series["tool_observation_delay"] = [_round(o) for o in observation]
    texts = _tool_output_texts(ctx.row or {}, ctx.spans) if ctx.row else {}
    metrics["observation_volume"] = {
        "unit": unit,
        "output_chars_total": sum(int(s["attrs"].get(chars_key) or 0) for s in volume_spans),
        "output_tokens_total": sum(
            int((s["attrs"].get("tokens") or {}).get("tool_output") or 0) for s in volume_spans
        ),
        "output_bytes_total": sum(len(texts.get(s["span_id"], "").encode("utf-8")) for s in volume_spans),
    }


def _subagent_compaction_section(
    ctx: _Context,
    metrics: dict[str, Any],
    series: dict[str, Any],
    _by_purpose: dict[str, Any],
    _by_operation: dict[str, Any],
) -> None:
    subagents = ctx.harness("subagent")
    times: list[float] = []
    to_sub: list[float] = []
    to_agent: list[float] = []
    resumed = 0
    entries: list[dict[str, Any]] = []
    for sub in subagents:
        duration = ctx.duration(sub)
        if duration is not None:
            times.append(duration)
        requested = sub["attrs"].get("requested_wall_s")
        observed = sub["attrs"].get("response_observed_wall_s")
        is_resumed = requested is not None and sub["start_wall_s"] < float(requested)
        handoff_in = max(0.0, sub["start_wall_s"] - float(requested)) if requested is not None else None
        handoff_out = None
        if observed is not None and sub.get("end_wall_s") is not None:
            handoff_out = float(observed) - float(sub["end_wall_s"])
        if handoff_in is not None:
            to_sub.append(handoff_in)
        if handoff_out is not None:
            to_agent.append(handoff_out)
        resumed += int(is_resumed)
        if is_resumed:
            ctx.codes.add("subagent_handoff_clamped")
        entries.append(
            {
                "session": (sub.get("ids") or {}).get("invocation_id"),
                "spawned_by_tool_call_id": sub["attrs"].get("spawned_by_tool_call_id"),
                "start": _round(float(sub["start_wall_s"])),
                "duration": _round(duration),
                "resumed": is_resumed,
                "agent_to_subagent_time": _round(handoff_in),
                "subagent_to_agent_time": _round(handoff_out),
            }
        )
    metrics["subagent_invocations"] = {"total": len(subagents), "resumed": resumed}
    metrics["subagent_time"] = dist(times)
    metrics["agent_to_subagent_time"] = dist(to_sub)
    metrics["subagent_to_agent_time"] = dist(to_agent)
    series["subagent_time"] = [_round(t) for t in times]
    series["agent_to_subagent_time"] = [_round(t) for t in to_sub]
    series["subagent_to_agent_time"] = [_round(t) for t in to_agent]
    series["subagents"] = entries
    compactions = ctx.harness("compaction")
    c_times = [d for d in (ctx.duration(c) for c in compactions) if d is not None]
    reasons: dict[str, int] = {}
    for compaction in compactions:
        reason = str(compaction["attrs"].get("reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    metrics["compactions"] = len(compactions)
    metrics["compaction_time"] = dist(c_times)
    metrics["compaction_reasons"] = dict(sorted(reasons.items()))
    series["compaction_time"] = [_round(t) for t in c_times]
    series["compaction_tokens_before"] = [c["attrs"].get("tokens_before") for c in compactions]
    series["compaction_tokens_after"] = [c["attrs"].get("tokens_after") for c in compactions]


def _critical_path_section(
    ctx: _Context,
    metrics: dict[str, Any],
    _series: dict[str, Any],
    _by_purpose: dict[str, Any],
    _by_operation: dict[str, Any],
) -> None:
    names = (
        "critical_path_duration",
        "cp_model_time",
        "cp_tool_time",
        "cp_other_time",
        "cp_model_fraction",
        "cp_tool_fraction",
        "cp_other_fraction",
        "parallel_execution_time",
    )
    if ctx.window is None or ctx.main_session is None:
        metrics.update({name: None for name in names})
        metrics["critical_path_node_ids"] = []
        return
    graph = build_graph(
        ctx.spans,
        agent_start=ctx.window[0],
        agent_end=ctx.window[1],
        main_session=ctx.main_session,
        tool_span_name=ctx.tool_span_name,
    )
    path = critical_path(graph)
    duration = path.duration
    metrics["critical_path_duration"] = duration
    metrics["cp_model_time"] = path.by_category["model"]
    metrics["cp_tool_time"] = path.by_category["tool"]
    metrics["cp_other_time"] = path.by_category["other"]
    for key in ("model", "tool", "other"):
        metrics[f"cp_{key}_fraction"] = (path.by_category[key] / duration) if duration > 0 else None
    metrics["parallel_execution_time"] = parallel_execution_time(graph)
    metrics["critical_path_node_ids"] = path.node_ids
    if duration > (ctx.window[1] - ctx.window[0]) + 1e-3:
        ctx.gaps.append({"code": "critical_path_exceeds_agent_time"})


def reasoning_drift(values: list[int]) -> Optional[float]:
    """Mean reasoning tokens of the last three steps minus the first three; None under six steps."""
    if len(values) < 6:
        return None
    return _round(sum(values[-3:]) / 3 - sum(values[:3]) / 3)


def _action_name(ctx: _Context, tool: dict[str, Any]) -> str:
    attrs = tool.get("attrs") or {}
    if ctx.kind == "opencode":
        return str(attrs.get("tool") or "tool")
    words = str(attrs.get("operation") or "").split()
    return words[0] if words else "wait"


def _waiting_on(ctx: _Context, after: float) -> dict[str, Any]:
    """What the agent was waiting on when its records stop: a tool call, or a model step (OpenCode) or model call
    (Terminus), that began after its last completed action and never ended. For a model call, the server's record
    of the request, when there is one, is matched by start time within a minute: a join across clocks, not a
    duration."""
    none: dict[str, Any] = {"waiting_on": None, "waiting_name": None, "open_since_wall_s": None, "in_flight": None}

    def open_after(name: str) -> list[dict[str, Any]]:
        return [
            s for s in ctx.harness(name) if s.get("end_wall_s") is None and float(s["start_wall_s"]) >= after - 1.0
        ]

    tools = open_after(ctx.tool_span_name) + (open_after("tool_call") if ctx.tool_span_name != "tool_call" else [])
    if tools:
        tool = max(tools, key=lambda s: s["start_wall_s"])
        return {
            **none,
            "waiting_on": "tool",
            "waiting_name": _action_name(ctx, tool),
            "open_since_wall_s": _round(tool["start_wall_s"]),
        }
    calls = open_after("model_step") + open_after("model_call")
    if not calls:
        return none
    start = min(float(s["start_wall_s"]) for s in calls)
    servers = [
        s for s in ctx.by_name.get("model_call_server", []) if start - 1.0 <= float(s["start_wall_s"]) <= start + 60.0
    ]
    flight = None
    if servers:
        server = max(servers, key=lambda s: s["start_wall_s"])
        attrs = server.get("attrs") or {}
        tokens = attrs.get("tokens") or {}
        engine = _engine(ctx, server)
        completion = tokens.get("completion_total")
        if completion is None:  # a cancelled call has no usage at the server; vLLM still counted what it generated
            completion = (engine.get("tokens") or {}).get("completion")
        flight = {
            "completion_tokens": completion,
            "stop_reason": attrs.get("stop_reason") or attrs.get("gym_finish_reason") or attrs.get("finish_reason"),
            "server_s": _round(ctx.duration(server)),
            "vllm_s": _round(engine.get("e2e_s")),
        }
    return {**none, "waiting_on": "model_call", "open_since_wall_s": _round(start), "in_flight": flight}


def _terminus_action(commands: list[dict[str, Any]], batch: Optional[dict[str, Any]], *, last: bool) -> str:
    if batch is None:
        return "final text" if last else "no command parsed"
    if not commands:
        return "no command parsed"
    first = str(commands[0]["attrs"].get("operation") or "").strip()
    if not first:
        return "wait"
    if first == "C-c":
        return "interrupt"
    return first.split()[0]


def _preceded_by(ctx: _Context, previous: Optional[dict[str, Any]], compacted: bool) -> str:
    if previous is None:
        return "first step"
    if compacted:
        return "right after a compaction"
    if previous["failed"]:
        return "after a failed tool call"
    names, action = previous["names"], previous["action"]
    if ctx.kind == "terminus2":
        if action == "wait":
            return "after a wait"
        if action == "interrupt":
            return "after an interrupt"
        return "after a text response" if action in ("final text", "no command parsed") else "after a command"
    if "task" in names:
        return "after a sub-agent returned"
    if not names:
        return "after a text response"
    return "after a read / grep / glob" if all(n in _READ_TOOLS for n in names) else "after another tool"


def _step_tags(
    ctx: _Context, main_steps: list[dict[str, Any]], linked: dict[str, dict[str, Any]]
) -> tuple[list[str], list[Optional[int]], list[str]]:
    """Per main-agent step: the action it emitted (OpenCode: the step's tool names joined by +, or "final text";
    Terminus: the first word of the first command, "wait", "interrupt", "no command parsed" or "final text"), the
    context it saw (the server's input tokens) and what preceded it. OpenCode tools hang off the step span; when
    that link is missing they are taken by time, between this call's end and the next call's start. Terminus
    command batches are paired to the call they follow the same way."""
    terminus = ctx.kind == "terminus2"
    tools = ctx.harness("tool_call")
    batches = ctx.harness("tool_batch") if terminus else []
    commands_by_batch: dict[str, list[dict[str, Any]]] = {}
    if terminus:
        for tool in tools:
            commands_by_batch.setdefault(str(tool.get("parent_span_id")), []).append(tool)
    main_tools = [] if terminus else [t for t in tools if _is_main(ctx, t)]
    compactions = [s["start_wall_s"] for s in ctx.harness("compaction")]
    compactions += [
        c["start_wall_s"] for c in ctx.harness("model_call") if c["attrs"].get("purpose") == "compaction_summary"
    ]
    actions: list[str] = []
    contexts: list[Optional[int]] = []
    preceded: list[str] = []
    previous: Optional[dict[str, Any]] = None
    previous_start = None
    for index, call in enumerate(main_steps):
        start = float(call["start_wall_s"])
        end = float(call.get("end_wall_s") or start)
        following = float(main_steps[index + 1]["start_wall_s"]) if index + 1 < len(main_steps) else math.inf
        names: list[str] = []
        if terminus:
            batch = next((b for b in batches if end - 1.0 <= float(b["start_wall_s"]) < following), None)
            commands = commands_by_batch.get(str(batch["span_id"]), []) if batch else []
            action = _terminus_action(commands, batch, last=index == len(main_steps) - 1)
            failed = batch is not None and (batch.get("status") == "error" or batch["attrs"].get("timeout") is True)
        else:
            step_tools = [t for t in main_tools if t.get("parent_span_id") == call.get("parent_span_id")]
            if not step_tools:
                step_tools = [t for t in main_tools if end - 1e-3 <= float(t["start_wall_s"]) < following]
            for tool in step_tools:
                name = str(tool["attrs"].get("tool") or "tool")
                if name not in names:
                    names.append(name)
            action = "+".join(names) if names else "final text"
            failed = any(t.get("status") == "error" for t in step_tools)
        tokens = ((linked.get(call["span_id"]) or {}).get("attrs") or {}).get("tokens") or {}
        context = tokens.get("input")
        compacted = previous_start is not None and any(previous_start < c < start for c in compactions)
        actions.append(action)
        contexts.append(int(context) if isinstance(context, int) and not isinstance(context, bool) else None)
        preceded.append(_preceded_by(ctx, previous, compacted))
        previous = {"names": names, "action": action, "failed": failed}
        previous_start = start
    return actions, contexts, preceded


def _behaviour_section(
    ctx: _Context,
    metrics: dict[str, Any],
    series: dict[str, Any],
    _by_purpose: dict[str, Any],
    _by_operation: dict[str, Any],
) -> None:
    """Behaviour the policy controls: repetition, interrupts, stalls, the generation tail, reasoning drift and,
    for Terminus, the waits it asks for."""
    calls = ctx.harness("model_call")
    linked = _linked_servers(ctx)
    main_steps = [c for c in calls if c["attrs"].get("purpose") == "agent_step" and _is_main(ctx, c)]
    generated: list[int] = []
    tail_calls = tail_tokens = length_stops = 0
    tail_decode = 0.0
    for call in calls:
        server = linked.get(call["span_id"])
        tokens = (server or {}).get("attrs", {}).get("tokens") or {}
        gen = int(tokens.get("reasoning") or 0) + int(tokens.get("output") or 0)
        generated.append(gen)
        if server is None:
            continue
        attrs = server["attrs"]
        if attrs.get("stop_reason") == "length" or attrs.get("gym_finish_reason") == "length":
            length_stops += 1
        if gen > GENERATION_TAIL_TOKENS:
            tail_calls += 1
            tail_tokens += gen
            decode = _engine(ctx, server).get("decode_s")
            if isinstance(decode, (int, float)):
                tail_decode += float(decode)
    reasoning_steps = [
        int(((linked.get(c["span_id"]) or {}).get("attrs", {}).get("tokens") or {}).get("reasoning") or 0)
        for c in main_steps
    ]
    series["reasoning_tokens_per_step"] = reasoning_steps
    series["step_action"], series["step_context"], series["step_preceded_by"] = _step_tags(ctx, main_steps, linked)
    series["generated_tokens_per_call"] = generated
    metrics["generation_tail_calls"] = tail_calls
    metrics["generation_tail_tokens"] = tail_tokens
    metrics["generation_tail_decode_time"] = _round(tail_decode)
    metrics["length_stop_calls"] = length_stops
    metrics["reasoning_per_step_drift"] = reasoning_drift(reasoning_steps)

    tools = ctx.harness("tool_call")
    actions: dict[Any, int] = {}
    reads: dict[str, int] = {}
    interrupts = 0
    for tool in tools:
        attrs = tool["attrs"]
        if ctx.kind == "opencode":
            key: Any = (str(attrs.get("tool")), str(attrs.get("title")))
            if attrs.get("tool") == "read":
                reads[str(attrs.get("title"))] = reads.get(str(attrs.get("title")), 0) + 1
        else:
            key = str(attrs.get("operation") or "").strip()
            if key == "C-c":
                interrupts += 1
        actions[key] = actions.get(key, 0) + 1
    metrics["repeated_actions"] = sum(c - 1 for c in actions.values() if c > 1)
    metrics["repeated_reads"] = sum(c - 1 for c in reads.values() if c > 1) if ctx.kind == "opencode" else None
    metrics["interrupts_sent"] = interrupts if ctx.kind == "terminus2" else None

    ends = [(s["end_wall_s"], "model", str(s["attrs"].get("purpose") or "model_call")) for s in calls]
    ends += [(s.get("end_wall_s"), "tool", _action_name(ctx, s)) for s in tools]
    ends = [e for e in ends if e[0] is not None]
    if ends and ctx.window:
        end, kind, name = max(ends, key=lambda e: e[0])
        # The agent span of a killed OpenCode process ends at its last persisted event, so the silence before the
        # kill is only visible against the exec that was killed. Two durations are compared, each on its own clock.
        execs = [e for e in ctx.by_name.get("agent_sandbox_exec") or ctx.by_name.get("agent_execute") or []]
        exec_durations = [d for d in (ctx.duration(e) for e in execs) if d is not None]
        if exec_durations:
            stall = _round(max(0.0, max(exec_durations) - (float(end) - ctx.window[0])))
            if any((e.get("clock") or {}).get("domain") != ctx.domain for e in execs):
                ctx.codes.add("two_clock_sum")
        else:
            stall = _round(max(0.0, ctx.window[1] - float(end)))
        metrics["stall_before_end_s"] = stall
        ctx.last_action = {"kind": kind, "name": name, "stall_before_end_s": stall, **_waiting_on(ctx, float(end))}
    else:
        metrics["stall_before_end_s"] = None

    if ctx.kind == "terminus2":
        waits = bare = trivial = 0
        wait_time = bare_time = trivial_time = 0.0
        for tool in tools:
            duration = ctx.duration(tool)
            if duration is None:
                continue
            words = str(tool["attrs"].get("operation") or "").split()
            waits += 1
            wait_time += duration
            if not words:
                bare += 1
                bare_time += duration
            elif words[0] in _TRIVIAL_COMMANDS and duration >= _TRIVIAL_WAIT_S:
                trivial += 1
                trivial_time += duration
        metrics["wait_analysis"] = {
            "waits": waits,
            "wait_time": _round(wait_time),
            "bare_waits": bare,
            "bare_wait_time": _round(bare_time),
            "trivial_waits": trivial,
            "trivial_wait_time": _round(trivial_time),
        }
        agent_time = (ctx.window[1] - ctx.window[0]) if ctx.window else None
        metrics["tool_wait_share"] = _round(wait_time / agent_time) if agent_time else None


_SECTIONS.extend([_tool_section, _subagent_compaction_section, _critical_path_section, _behaviour_section])
