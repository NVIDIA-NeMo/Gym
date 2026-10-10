# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One ``rollouts.jsonl`` row -> the spans the analysis reads.

The row carries Gym's observability records: ``ng_perf`` (rollout bounds), ``ng_trajectory`` (invocations, model
calls with the harness's and the server's bounds, tool calls, compactions, phases) and ``ng_agent_observations``
(the sandbox record with the clock probe). Every record names its clock domain; nothing here subtracts a
timestamp across domains.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from ng_latency.spans import Span, SpanIds, clock_for, measured_span, serialize, timed_span


_INVOCATION_STATUS = {"completed": "ok", "failed": "error", "incomplete": "incomplete", "unknown": "unknown"}
_EXEC_STATUS = {"timeout": "timeout", "failed": "error", "sandbox_error": "error", "cancelled": "cancelled"}
_TOOL_STATUS = {
    "completed": "ok",
    "failed": "error",
    "timeout": "timeout",
    "cancelled": "cancelled",
    "incomplete": "incomplete",
    "unknown": "unknown",
}
_ATTEMPT_STATUS = {"completed": "ok", "error": "error", "timeout": "timeout", "cancelled": "cancelled"}
_PHASE_STATUS = {
    "completed": "ok",
    "failed": "error",
    "timeout": "timeout",
    "cancelled": "cancelled",
    "incomplete": "incomplete",
    "unknown": "unknown",
}
_SERVER_STATUS_BY_CATEGORY = {"timeout": "timeout", "cancelled": "cancelled", "incomplete": "incomplete"}
# Observation gaps that change how a rollout's numbers must be read; everything else stays in the row.
LIFTED_GAP_CODES = (
    "tool_response_boundary_approximate",
    "model_call_join_unverified",
    "agent_artifact_unavailable",
    "model_steps_unavailable",
)


def _f(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _i(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def identity(record: dict[str, Any]) -> dict[str, Any]:
    task_id = next(
        (str(record[k]) for k in ("task_id", "problem_id", "instance_id", "task_name") if record.get(k) is not None),
        str(record.get("_ng_task_index")),
    )
    rollout = record.get("_ng_rollout_index")
    attempt = record.get("_ng_attempt_index")
    rollout_id = record.get("_ng_rollout_id") or f"{record.get('_ng_task_index')}-{rollout}"
    if attempt:
        rollout_id = f"{rollout_id}-a{attempt}"
    return {"task_id": task_id, "rollout_id": rollout_id, "repeat": rollout, "attempt": attempt}


def agent_kind(record: dict[str, Any]) -> str:
    if "opencode_finished" in record:
        return "opencode"
    if "terminus2_completed" in record:
        return "terminus2"
    return "unknown"


def _engine_attrs(engine: Optional[dict[str, Any]], server_duration_s: Optional[float]) -> dict[str, Any]:
    """vLLM's per-request durations, in seconds, with the derived prefill and end-to-end values and the hop
    overhead (server duration minus engine end-to-end). Durations only: the engine reports no timestamps."""
    if not isinstance(engine, dict):
        return {}
    queue_ms = _f(engine.get("queue_time_ms"))
    ttft_ms = _f(engine.get("time_to_first_token_ms"))
    generation_ms = _f(engine.get("generation_time_ms"))
    out: dict[str, Any] = {
        "queue_s": queue_ms / 1000 if queue_ms is not None else None,
        "prefill_s": (ttft_ms - queue_ms) / 1000 if ttft_ms is not None and queue_ms is not None else None,
        "decode_s": generation_ms / 1000 if generation_ms is not None else None,
        "e2e_s": (ttft_ms + generation_ms) / 1000 if ttft_ms is not None and generation_ms is not None else None,
        "ttft_s": ttft_ms / 1000 if ttft_ms is not None else None,
        "mean_itl_ms": _f(engine.get("mean_itl_ms")),
        "tokens_per_second": _f(engine.get("tokens_per_second")),
        "speculative_decoding": engine.get("speculative_decoding"),
    }
    if out["prefill_s"] is not None and out["prefill_s"] < 0:
        out["prefill_s"] = 0.0
    out["gym_hop_overhead_s"] = (
        server_duration_s - out["e2e_s"] if server_duration_s is not None and out["e2e_s"] is not None else None
    )
    return out


def _tokens(stats: dict[str, Any]) -> dict[str, Optional[int]]:
    completion = _i(stats.get("completion_tokens"))
    reasoning = _i(stats.get("reasoning_tokens"))
    return {
        "input": _i(stats.get("prompt_tokens")),
        "output": completion - reasoning if completion is not None and reasoning is not None else completion,
        "reasoning": reasoning,
        "completion_total": completion,
        "cached": _i(stats.get("cached_tokens")),
    }


def _server_status(call: dict[str, Any]) -> str:
    metadata = call.get("response_metadata") or {}
    category = metadata.get("error_category")
    if category in _SERVER_STATUS_BY_CATEGORY:
        return _SERVER_STATUS_BY_CATEGORY[category]
    if category:
        return "error"
    status_code = metadata.get("status_code")
    if isinstance(status_code, int) and status_code >= 400:
        return "error"
    if call.get("context_overflow") is True:
        return "incomplete"
    return "ok"


class _Projection:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.trajectory: dict[str, Any] = (
            record.get("ng_trajectory") if isinstance(record.get("ng_trajectory"), dict) else {}
        )
        self.perf: dict[str, Any] = record.get("ng_perf") if isinstance(record.get("ng_perf"), dict) else {}
        bundle = record.get("ng_agent_observations")
        self.records: list[dict[str, Any]] = (bundle.get("records") or []) if isinstance(bundle, dict) else []
        self.kind = agent_kind(record)
        self.ids = identity(record)
        self.base_ids = SpanIds(
            task_id=self.ids["task_id"],
            repeat=_i(self.ids["repeat"]),
            attempt=_i(self.ids["attempt"]),
            rollout_id=self.ids["rollout_id"],
        )
        self.spans: list[Span] = []
        self.flags: set[str] = set()
        self.gaps: list[dict[str, Any]] = []
        phases = self.trajectory.get("phases") or []
        self.node_domain = (
            next((p.get("clock_domain") for p in phases if isinstance(p, dict) and p.get("clock_domain")), None)
            or "node:unknown"
        )
        self.sandbox = (
            next((r for r in self.records if r.get("kind") == "sandbox" and r.get("role") == "agent"), None) or {}
        )
        self.session_spans: dict[str, Span] = {}
        self.parent_of: dict[str, Optional[str]] = {}
        self.agent: Optional[Span] = None

    # -- helpers --------------------------------------------------------------------------------------------
    def ids_for(self, invocation_id: Optional[str] = None, **extra: Any) -> SpanIds:
        return self.base_ids.merged(
            SpanIds(
                invocation_id=invocation_id,
                parent_invocation_id=self.parent_of.get(invocation_id) if invocation_id else None,
                **extra,
            )
        )

    def add(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    # -- rollout and phases -----------------------------------------------------------------------------------
    def rollout(self) -> Optional[Span]:
        start = _f(self.perf.get("rollout_started_at"))
        if start is None:
            return None
        duration_ms = _f(self.perf.get("rollout_duration_ms"))
        wall_ms = _f(self.perf.get("total_latency_ms"))
        return self.add(
            timed_span(
                "rollout",
                clock=clock_for(self.node_domain, "collector"),
                start_wall_s=start,
                duration_ms=duration_ms if duration_ms is not None else wall_ms,
                end_wall_s=_f(self.perf.get("rollout_completed_at")),
                ids=self.base_ids,
                attrs={"wall_latency_ms": wall_ms},
            )
        )

    def phases(self, parent: Optional[Span]) -> None:
        for phase in self.trajectory.get("phases") or []:
            if not isinstance(phase, dict) or _f(phase.get("started_at")) is None:
                continue
            self.add(
                timed_span(
                    str(phase.get("name") or "phase"),
                    clock=clock_for(phase.get("clock_domain") or self.node_domain, "agent_server"),
                    start_wall_s=float(phase["started_at"]),
                    duration_ms=_f(phase.get("duration_ms")),
                    end_wall_s=_f(phase.get("completed_at")),
                    ids=self.base_ids,
                    parent=parent,
                    status=_PHASE_STATUS.get(str(phase.get("status")), "unknown"),
                    attrs={"error_type": phase.get("error_type")},
                )
            )

    # -- invocations ------------------------------------------------------------------------------------------
    def invocations(self, parent: Optional[Span]) -> None:
        invocations = [i for i in self.trajectory.get("invocations") or [] if isinstance(i, dict)]
        for invocation in invocations:
            self.parent_of[str(invocation["invocation_id"])] = invocation.get("parent_invocation_id")
        exec_outcome = self.sandbox.get("outcome")
        roots = [i for i in invocations if not i.get("parent_invocation_id")]
        children = [i for i in invocations if i.get("parent_invocation_id")]
        for invocation in roots:
            start = _f(invocation.get("started_at"))
            if start is None:
                continue
            end = _f(invocation.get("completed_at"))
            launch = _f(invocation.get("process_started_at"))
            exited = _f(invocation.get("process_exited_at"))
            status = _EXEC_STATUS.get(str(exec_outcome), None) or _INVOCATION_STATUS.get(
                str(invocation.get("status")), "unknown"
            )
            if status == "incomplete" and invocation.get("error_type") == "TimeoutError":
                status = (
                    "timeout"  # Harbor's loop cancelled at the sandbox cap (OpenCode's exec outcome says so itself)
                )
            if self.kind == "opencode" and exec_outcome in ("timeout", "failed", "sandbox_error") and exited is None:
                # The process was killed: the span ends at the last event OpenCode persisted, so agent time is a
                # lower bound for this rollout.
                self.flags.add("agent_end_unobserved")
            span = timed_span(
                "agent",
                clock=clock_for(invocation.get("clock_domain") or self.node_domain, "agent"),
                start_wall_s=start,
                duration_ms=_f(invocation.get("duration_ms")),
                end_wall_s=end,
                ids=self.ids_for(str(invocation["invocation_id"])),
                parent=parent,
                status=status,
                attrs={
                    "process_launch_wall_s": launch,
                    "process_exit_wall_s": exited,
                    "last_output_wall_s": end,
                    "exec_outcome": exec_outcome,
                    "error_type": invocation.get("error_type"),
                    "opencode_startup_s": start - launch if launch is not None else None,
                    "opencode_shutdown_s": exited - end if exited is not None and end is not None else None,
                },
            )
            self.session_spans[str(invocation["invocation_id"])] = self.add(span)
            if self.agent is None:
                self.agent = span
        for invocation in children:  # parents precede children in the record
            start = _f(invocation.get("started_at"))
            if start is None:
                continue
            session_id = str(invocation["invocation_id"])
            parent_span = self.session_spans.get(str(invocation.get("parent_invocation_id")))
            span = measured_span(
                "subagent",
                clock=clock_for(
                    invocation.get("clock_domain") or (self.agent.clock.domain if self.agent else None), "agent"
                ),
                start_wall_s=start,
                end_wall_s=_f(invocation.get("completed_at")),
                ids=self.ids_for(session_id),
                parent=parent_span,
                status=_INVOCATION_STATUS.get(str(invocation.get("status")), "unknown"),
                attrs={
                    "requested_wall_s": _f(invocation.get("requested_at")),
                    "response_observed_wall_s": _f(invocation.get("response_observed_at")),
                    "spawned_by_tool_call_id": invocation.get("spawned_by_tool_call_id"),
                    "last_output_wall_s": _f(invocation.get("completed_at")),
                },
            )
            self.session_spans[session_id] = self.add(span)

    # -- model calls ------------------------------------------------------------------------------------------
    def model_calls(self) -> list[Span]:
        steps: list[Span] = []
        calls = [c for c in self.trajectory.get("model_calls") or [] if isinstance(c, dict)]
        calls.sort(
            key=lambda c: (
                _f(c.get("started_at")) or _f(c.get("server_started_at")) or _f(c.get("completed_at")) or 0.0
            )
        )
        for call in calls:
            session = call.get("invocation_id")
            session_id = str(session) if session else None
            session_span = self.session_spans.get(session_id or "")
            purpose = call.get("model_call_purpose") or "agent_step"
            domain = call.get("clock_domain") or self.node_domain
            metadata = call.get("response_metadata") or {}
            start = _f(call.get("started_at"))
            completed = _f(call.get("completed_at"))
            processed = _f(call.get("processed_at"))
            step: Optional[Span] = None
            if self.kind == "opencode":
                # OpenCode's step: from the request to the end of OpenCode's processing of the response.
                step_start = start if start is not None else completed
                if step_start is not None:
                    step = self.add(
                        measured_span(
                            "model_step",
                            clock=clock_for(domain, "opencode"),
                            start_wall_s=step_start,
                            end_wall_s=processed if processed is not None else completed,
                            ids=self.ids_for(session_id, model_call_id=call.get("model_call_id")),
                            parent=session_span,
                            attrs={
                                "first_part_wall_s": completed,
                                "summary_message": purpose == "compaction_summary",
                                "tokens": _tokens(call.get("token_stats") or {}),
                                "purpose": purpose,
                            },
                        )
                    )
                    steps.append(step)
            logical: Optional[Span] = None
            if start is not None:
                # No status from the harness: a rejected request is incomplete, and so is a call it never saw end
                # (the OpenCode process was killed with the request in flight).
                never_ended = completed is None and _f(call.get("duration_ms")) is None
                status = call.get("status") or ("incomplete" if call.get("context_overflow") or never_ended else "ok")
                logical = self.add(
                    timed_span(
                        "model_call",
                        clock=clock_for(domain, "agent"),
                        start_wall_s=start,
                        duration_ms=_f(call.get("duration_ms")),
                        end_wall_s=completed,
                        ids=self.ids_for(
                            session_id,
                            model_call_id=call.get("model_call_id"),
                            client_call_id=call.get("client_call_id"),
                        ),
                        parent=step or session_span,
                        status=status,
                        attrs={
                            "purpose": purpose,
                            "parsed_commands": _i(call.get("parsed_tool_calls")),
                            "task_complete": call.get("task_complete"),
                            "parse_error": call.get("parse_error"),
                            "context_overflow": call.get("context_overflow"),
                            "processed_wall_s": processed,
                            "response_id": metadata.get("response_id"),
                        },
                    )
                )
                for attempt in call.get("attempts") or []:
                    if not isinstance(attempt, dict) or _f(attempt.get("started_at")) is None:
                        continue
                    self.add(
                        timed_span(
                            "model_call_attempt",
                            clock=clock_for(attempt.get("clock_domain") or domain, "agent"),
                            start_wall_s=float(attempt["started_at"]),
                            duration_ms=_f(attempt.get("duration_ms")),
                            end_wall_s=_f(attempt.get("completed_at")),
                            ids=self.ids_for(session_id, client_call_id=call.get("client_call_id")),
                            parent=logical,
                            status=_ATTEMPT_STATUS.get(str(attempt.get("status")), "unknown"),
                            attrs={
                                "attempt": attempt.get("attempt_index"),
                                "status_code": attempt.get("status_code"),
                                "error": attempt.get("error_type"),
                            },
                        )
                    )
            elif self.kind != "opencode" or step is None:
                self.gaps.append({"code": "model_call_without_bounds", "model_call_id": call.get("model_call_id")})
            server_start = _f(call.get("server_started_at"))
            if server_start is not None:
                server_duration_ms = _f(call.get("server_duration_ms"))
                server_end = _f(call.get("server_completed_at"))
                server_duration_s = (
                    server_duration_ms / 1000
                    if server_duration_ms is not None
                    else (server_end - server_start if server_end else None)
                )
                engine = _engine_attrs(metadata.get("engine"), server_duration_s)
                server = self.add(
                    timed_span(
                        "model_call_server",
                        clock=clock_for(self.node_domain, "model_server"),
                        start_wall_s=server_start,
                        duration_ms=server_duration_ms,
                        end_wall_s=server_end,
                        ids=self.ids_for(
                            session_id,
                            model_call_id=call.get("model_call_id"),
                            client_call_id=call.get("client_call_id"),
                        ),
                        parent=logical or step or session_span,
                        status=_server_status(call),
                        attrs={
                            "purpose": purpose,
                            "response_kind": call.get("model_response_kind"),
                            "stop_reason": metadata.get("finish_reason"),
                            "gym_finish_reason": metadata.get("finish_reason"),
                            "status_code": metadata.get("status_code"),
                            "error_category": metadata.get("error_category"),
                            "latency_ttfb_ms": _f(metadata.get("latency_ttft_ms")),
                            "dialect": metadata.get("dialect"),
                            "model": metadata.get("model"),
                            "tokens": _tokens(call.get("token_stats") or {}),
                            "vllm": engine,
                            "gym_hop_overhead_s": engine.get("gym_hop_overhead_s"),
                            "rejected_request_tokens": _i(call.get("rejected_request_tokens")),
                            "context_overflow": call.get("context_overflow"),
                        },
                    )
                )
                if logical is not None:
                    logical.attrs["server_span_id"] = server.span_id
                elif step is not None:
                    step.attrs["server_span_id"] = server.span_id
        return steps

    # -- tools ---------------------------------------------------------------------------------------------------
    def tools(self, steps: list[Span]) -> None:
        tools = [t for t in self.trajectory.get("tool_calls") or [] if isinstance(t, dict)]
        tools.sort(
            key=lambda t: (_f(t.get("started_at")) or _f(t.get("requested_at")) or 0.0, str(t.get("batch_index") or 0))
        )
        batches: dict[str, list[Span]] = defaultdict(list)
        batch_spans: dict[str, Span] = {}
        if self.kind == "terminus2":
            # One batch span per decision, the frame the per-command spans hang under.
            for batch_id, group in self._batches(tools).items():
                starts = [_f(t.get("started_at")) for t in group]
                ends = [_f(t.get("completed_at")) for t in group]
                if any(s is None for s in starts):
                    continue
                first = group[0]
                session_id = str(first.get("invocation_id")) if first.get("invocation_id") else None
                end = max(ends) if all(e is not None for e in ends) else None
                statuses = [str(t.get("status")) for t in group]
                status = "cancelled" if "cancelled" in statuses else ("error" if "failed" in statuses else "ok")
                batch_spans[batch_id] = self.add(
                    measured_span(
                        "tool_batch",
                        clock=clock_for(first.get("clock_domain") or self.node_domain, "agent"),
                        start_wall_s=min(starts),
                        end_wall_s=end,
                        ids=self.ids_for(session_id),
                        parent=self.session_spans.get(session_id or ""),
                        status=status,
                        attrs={
                            "num_commands": len(group),
                            "timeout": "timeout" in statuses,
                            "observation_chars": _i(first.get("output_chars")),
                            "response_received_wall_s": _f(first.get("response_received_at")),
                            "observation_trajectory_index": _i(first.get("output_trajectory_index")),
                        },
                    )
                )
        for tool in tools:
            start = _f(tool.get("started_at"))
            if start is None:
                continue
            session_id = str(tool.get("invocation_id")) if tool.get("invocation_id") else None
            domain = tool.get("clock_domain") or (self.node_domain if tool.get("timing_source") == "harness" else None)
            if domain is None and self.agent is not None:
                domain = self.agent.clock.domain
            parent: Optional[Span] = None
            if self.kind == "terminus2":
                parent = batch_spans.get(str(tool.get("batch_id")))
            else:
                requested = _f(tool.get("requested_at")) or start
                parent = next(
                    (
                        s
                        for s in steps
                        if s.ids.invocation_id == session_id
                        and s.start_wall_s <= requested
                        and (s.end_wall_s is None or requested <= s.end_wall_s)
                    ),
                    None,
                )
            span = timed_span(
                "tool_call",
                clock=clock_for(domain, "agent" if tool.get("timing_source") == "harness" else "opencode"),
                start_wall_s=start,
                duration_ms=_f(tool.get("duration_ms")) if tool.get("timing_source") == "harness" else None,
                end_wall_s=_f(tool.get("completed_at")),
                ids=self.ids_for(session_id, tool_call_id=tool.get("tool_call_id")),
                parent=parent or self.session_spans.get(session_id or ""),
                status=_TOOL_STATUS.get(str(tool.get("status")), "unknown"),
                attrs={
                    "tool": tool.get("tool_name"),
                    "operation": tool.get("operation"),
                    "title": tool.get("operation"),
                    "requested_wall_s": _f(tool.get("requested_at")),
                    "response_received_wall_s": _f(tool.get("response_received_at")),
                    "requested_wait_s": _f(tool.get("requested_wait_s")),
                    "output_chars": _i(tool.get("output_chars")),
                    "batch_id": tool.get("batch_id"),
                    "batch_index": _i(tool.get("batch_index")),
                    "observation_trajectory_index": _i(tool.get("output_trajectory_index")),
                    "error_type": tool.get("error_type"),
                },
            )
            self.add(span)
            if tool.get("batch_id"):
                batches[str(tool["batch_id"])].append(span)

    @staticmethod
    def _batches(tools: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for tool in tools:
            key = str(tool.get("batch_id") or f"single:{tool.get('tool_call_id')}")
            groups[key].append(tool)
        for group in groups.values():
            group.sort(key=lambda t: (_i(t.get("batch_index")) or 0, _f(t.get("started_at")) or 0.0))
        return groups

    # -- compactions ----------------------------------------------------------------------------------------------
    def compactions(self) -> None:
        for compaction in self.trajectory.get("compactions") or []:
            if not isinstance(compaction, dict):
                continue
            start = _f(compaction.get("observed_at"))
            if start is None:
                continue
            session_id = str(compaction.get("invocation_id")) if compaction.get("invocation_id") else None
            outcome = str(compaction.get("outcome") or "unknown")
            status = {"completed": "ok", "failed": "error", "aborted": "cancelled"}.get(outcome, "unknown")
            span = self.add(
                timed_span(
                    "compaction",
                    clock=clock_for(
                        compaction.get("clock_domain")
                        or (self.agent.clock.domain if self.agent else self.node_domain),
                        "agent",
                    ),
                    start_wall_s=start,
                    duration_ms=_f(compaction.get("duration_ms")),
                    end_wall_s=_f(compaction.get("completed_at")),
                    ids=self.ids_for(session_id),
                    parent=self.session_spans.get(session_id or ""),
                    status=status,
                    attrs={
                        "reason": compaction.get("trigger"),
                        "outcome": outcome,
                        "tokens_before": _i(compaction.get("tokens_before")),
                        "tokens_after": _i(compaction.get("tokens_after")),
                        "rejected_request_tokens": _i(compaction.get("rejected_request_tokens")),
                    },
                )
            )
            self._adopt_compaction_calls(span, compaction, session_id)

    def _adopt_compaction_calls(self, compaction: Span, record: dict[str, Any], session_id: Optional[str]) -> None:
        """The compaction is the parent of the model calls made for it. Terminus names them (``model_calls`` refs,
        renamed by the collector to the merged ids); OpenCode does not, so a call with a compaction purpose that
        starts inside the compaction's window in the same session is its child. An OpenCode call moves together
        with its harness step."""
        named = {
            str(ref.get("model_call_id"))
            for ref in record.get("model_calls") or []
            if isinstance(ref, dict) and ref.get("model_call_id")
        }
        by_id = {s.span_id: s for s in self.spans}
        for call in [s for s in self.spans if s.name == "model_call"]:
            inside = (
                str(call.attrs.get("purpose") or "").startswith("compaction")
                and (session_id is None or call.ids.invocation_id == session_id)
                and call.start_wall_s >= compaction.start_wall_s
                and (compaction.end_wall_s is None or call.start_wall_s <= compaction.end_wall_s)
            )
            if call.ids.model_call_id in named or inside:
                parent = by_id.get(call.parent_span_id or "")
                target = parent if parent is not None and parent.name == "model_step" else call
                target.parent_span_id = compaction.span_id

    # -- clocks -----------------------------------------------------------------------------------------------------
    def clock_offsets(self) -> dict[str, dict[str, Any]]:
        offsets: dict[str, dict[str, Any]] = {}
        sandbox_domain = (
            self.agent.clock.domain if self.agent and self.agent.clock.domain.startswith("sandbox:") else None
        )
        probe = _f(self.sandbox.get("clock_offset_s"))
        if sandbox_domain is not None:
            entry: dict[str, Any] = {}
            if probe is not None:
                # Her probe: sandbox clock minus harness clock at sandbox start, with half the round trip as the
                # error bar. The only offset the analysis may apply, and only with that uncertainty attached.
                entry["offset_s"] = probe
                entry["uncertainty_s"] = _f(self.sandbox.get("clock_offset_uncertainty_s"))
                entry["source"] = "probe"
            else:
                self.flags.add("clock_offset_unavailable")
            # Offset plus the smallest network latency, from calls whose send time (sandbox clock) and server
            # arrival (node clock) are both known. Reported, never applied.
            servers = {
                s.ids.model_call_id: s for s in self.spans if s.name == "model_call_server" and s.ids.model_call_id
            }
            deltas = [
                servers[s.ids.model_call_id].start_wall_s - s.start_wall_s
                for s in self.spans
                if s.name == "model_call" and s.clock.domain == sandbox_domain and s.ids.model_call_id in servers
            ]
            if deltas:
                entry["offset_plus_min_latency_s"] = min(deltas)
                entry["samples"] = len(deltas)
            if entry:
                offsets[sandbox_domain] = entry
        return offsets

    # -- all together -------------------------------------------------------------------------------------------
    def run(self) -> Optional[dict[str, Any]]:
        if not self.trajectory and not self.perf:
            return None
        rollout = self.rollout()
        self.phases(rollout)
        self.invocations(rollout)
        steps = self.model_calls()
        self.tools(steps)
        self.compactions()
        offsets = self.clock_offsets()
        for gap in self.trajectory.get("gaps") or []:
            if isinstance(gap, dict) and gap.get("code") in LIFTED_GAP_CODES:
                self.flags.add(str(gap["code"]))
        if self.agent is None:
            self.flags.add("agent_artifact_unavailable")
        return {
            "kind": "rollout",
            **self.ids,
            "agent_kind": self.kind,
            "clock_offsets": offsets,
            "spans": serialize(self.spans),
            "gaps": self.gaps,
            "flags": sorted(self.flags),
        }


def project_rollout(record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Spans for one rollout row, or None when the row carries no observability records."""
    return _Projection(record).run()
