# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local canonical records and Gym capture/trajectory adapters. No service access."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from .models import Call, Compaction, Rollout, Tool


def _strict(cls, row):
    unknown = set(row) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**row)


def from_dict(row: dict) -> Rollout:
    row = dict(row)
    for name, cls in (("calls", Call), ("tools", Tool), ("compactions", Compaction)):
        row[name] = tuple(_strict(cls, x) for x in row.get(name, []))
    row["gaps"] = tuple(row.get("gaps", []))
    return _strict(Rollout, row)


def _seconds(ms):
    return ms / 1000 if ms is not None else None


def from_gym(row: dict, *, run_id: str, benchmark: str, config_id: str) -> Rollout:
    """One materialized evaluator rollout. Capture beats transcript copies."""
    trajectory = row.get("ng_trajectory") or {}
    capture = row.get("ng_model_call_capture")
    bundle = row.get("ng_agent_observations") or {}
    records = bundle.get("records", [])
    gaps = [g.get("code", "unknown") for g in bundle.get("gaps", [])]
    invocation_by_call = {}
    for invocation in records + trajectory.get("invocations", []):
        if invocation.get("kind") == "agent_invocation" or "conversation" in invocation:
            for ref in invocation.get("model_calls", []):
                cid = ref.get("model_call_id")
                if cid:
                    existing = invocation_by_call.get(cid)
                    if existing is not None and existing != invocation["invocation_id"]:
                        raise ValueError(f"Ambiguous invocation ownership for call {cid}")
                    invocation_by_call[cid] = invocation["invocation_id"]
    raw_calls = capture.get("calls", []) if capture is not None else trajectory.get("model_calls", [])
    calls = []
    for index, raw in enumerate(raw_calls):
        if not raw.get("is_copied_context", False):
            usage = raw.get("token_stats") or {}
            meta = raw.get("response_metadata") or {}
            cid = raw.get("model_call_id") or f"call:{index}"
            status = raw.get("error_category") or meta.get("error_category")
            if status is None:
                code = raw.get("status_code", meta.get("status_code"))
                status = (
                    "error"
                    if code is not None and code >= 400
                    else (raw.get("finish_reason") or meta.get("finish_reason") or "unknown")
                )
            calls.append(
                Call(
                    id=cid,
                    index=raw.get("call_index", index),
                    output_tokens=raw.get("tokens_out", usage.get("completion_tokens")),
                    reasoning_tokens=raw.get("tokens_reasoning", usage.get("reasoning_tokens")),
                    input_tokens=raw.get("tokens_in", usage.get("prompt_tokens")),
                    duration_s=_seconds(raw.get("latency_total_ms", raw.get("duration_ms"))),
                    started_at=raw.get("started_at"),
                    invocation_id=raw.get("client_session_id") or invocation_by_call.get(cid),
                    status=status,
                )
            )
    if capture is not None:
        capture_gaps = [g.get("code", "unknown") for g in capture.get("gaps", [])]
        gaps.extend(capture_gaps)
        complete = bool(calls) and not any(g.startswith("model_call_capture_") for g in capture_gaps)
    else:
        # A trajectory may contain only the joined calls, so absence of gaps is
        # insufficient to assert full model-call capture coverage.
        complete = None
        gaps.append("capture_completeness_unobserved")
    tools = []
    raw_tools = [r for r in records if r.get("kind") == "tool_call"]
    if not raw_tools:
        raw_tools = trajectory.get("tool_calls", [])
    for raw in raw_tools:
        tools.append(
            Tool(
                id=raw["tool_call_id"],
                index=raw.get("index"),
                name=raw.get("tool_name") or "unknown",
                duration_s=_seconds(raw.get("duration_ms")),
                started_at=raw.get("started_at"),
                invocation_id=raw.get("invocation_id"),
                status=raw.get("status", "unknown"),
                action=raw.get("action"),
                target=raw.get("target"),
                content_hash=raw.get("content_hash"),
            )
        )
    compactions = []
    for raw in records:
        if raw.get("kind") == "context_compaction" and raw.get("outcome") == "completed":
            ref = raw.get("after_model_call") or {}
            if ref.get("model_call_id"):
                compactions.append(Compaction(raw["invocation_id"], ref["model_call_id"]))
            else:
                gaps.append("compaction_boundary_unjoined")
    environments = [r.get("outcome", "unknown") for r in records if r.get("kind") == "sandbox"]
    env_status = next((s for s in environments if s not in ("completed", "unknown")), "unknown")
    if environments and all(s == "completed" for s in environments):
        env_status = "completed"
    reward = row.get("reward")
    success = bool(reward) if reward in (0, 1) else None
    if reward is not None and reward not in (0, 1):
        gaps.append("nonbinary_reward_requires_success_policy")
    task = row.get("_ng_task_index", row.get("task_id", row.get("task_index", trajectory.get("task_id"))))
    repeat = row.get("_ng_rollout_index", row.get("rollout_index", row.get("repeat")))
    if task is None or repeat is None:
        raise ValueError("Gym input requires task_id/task_index and rollout_index/repeat")
    agent_ref = row.get("agent_ref")
    agent = agent_ref.get("name") if isinstance(agent_ref, dict) else agent_ref
    models = {r.get("model") or (r.get("response_metadata") or {}).get("model") for r in raw_calls}
    models.discard(None)
    if len(models) > 1:
        raise ValueError("Mixed-model Gym capture requires explicit canonical cohort normalization")
    model = next(iter(models), row.get("model"))
    if not model:
        raise ValueError("Gym input must identify its model")
    return Rollout(
        run_id=run_id,
        task_id=str(task),
        repeat=repeat,
        agent=agent,
        model=model,
        benchmark=benchmark,
        config_id=config_id,
        success=success,
        health="quarantined" if row.get("mask_sample") else row.get("health_verdict") or "unknown",
        capture_complete=complete,
        wall_s=row.get("wall_s", _seconds((row.get("ng_perf") or {}).get("total_latency_ms"))),
        calls=tuple(calls),
        tools=tuple(tools),
        compactions=tuple(compactions),
        environment_status=env_status,
        gaps=tuple(sorted(set(gaps))),
    )


def read_records(paths: list[Path], *, format="canonical", **gym_context) -> list[Rollout]:
    if format not in ("canonical", "gym"):
        raise ValueError("format must be canonical or gym")
    records = []
    for path in paths:
        with path.open() as stream:
            if path.suffix == ".jsonl":
                rows = (json.loads(line) for line in stream if line.strip())
            else:
                data = json.load(stream)
                if not isinstance(data, list):
                    raise ValueError("JSON input must be an array of rollout records")
                rows = data
            for row in rows:
                records.append(from_dict(row) if format == "canonical" else from_gym(row, **gym_context))
    return records
