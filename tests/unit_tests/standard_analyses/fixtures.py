# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic fixtures for numerical, adapter and export tests."""

from __future__ import annotations

from dataclasses import asdict

from nemo_gym.standard_analyses.models import Call, Compaction, Rollout, Tool


def demo_records():
    records = []
    for task_index, task in enumerate(("small", "long-tail", "compaction")):
        for repeat in range(4):
            invocation = f"{task}/{repeat}/main"
            lengths = (120, 400, 1600, 250, 12000 if repeat == 3 else 800, 300)
            calls = tuple(
                Call(
                    id=f"c{index}",
                    index=index,
                    output_tokens=length * (task_index + 1),
                    reasoning_tokens=int(length * 0.7) if repeat != 2 or index != 5 else None,
                    input_tokens=1000 + index * 800,
                    duration_s=length / 70 + repeat * 2,
                    started_at=1000 + index * 300,
                    invocation_id=invocation,
                    status="timeout" if repeat == 3 and index == 5 else "completed",
                )
                for index, length in enumerate(lengths)
            )
            tools = tuple(
                Tool(
                    id=f"t{index}",
                    index=index,
                    name=name,
                    duration_s=1 + index,
                    started_at=1010 + index * 300,
                    invocation_id=invocation,
                    status="completed",
                    action=action,
                    target="example.py",
                    content_hash=content_hash,
                )
                for index, (name, action, content_hash) in enumerate(
                    (
                        ("read_file", "read", "h1"),
                        ("read_file", "read", "h1"),
                        ("edit_file", "write", "h2"),
                        ("read_file", "read", "h2"),
                        ("run_tests", "verify", None),
                        ("shell", None, None),
                    )
                )
            )
            records.append(
                Rollout(
                    run_id="synthetic-demo",
                    task_id=task,
                    repeat=repeat,
                    agent="demo-agent",
                    model="synthetic-policy",
                    benchmark="synthetic-example",
                    config_id="demo-v1",
                    success=repeat != 3,
                    health="healthy",
                    capture_complete=True,
                    wall_s=1800 + task_index * 500 + repeat * 50,
                    calls=calls,
                    tools=tools,
                    compactions=(Compaction(invocation, "c3"),),
                    environment_status="completed",
                )
            )
    return records


def gym_rows():
    """Shapes produced by Gym, with unsupported action-state fields omitted."""
    rows = []
    for r in demo_records():
        rows.append(
            {
                "task_id": r.task_id,
                "rollout_index": r.repeat,
                "agent_ref": {"name": r.agent},
                "reward": int(r.success),
                "health_verdict": r.health,
                "ng_model_call_capture": {
                    "rollout_id": f"{r.task_id}/{r.repeat}",
                    "calls": [
                        {
                            "model_call_id": c.id,
                            "call_index": c.index,
                            "tokens_out": c.output_tokens,
                            "tokens_reasoning": c.reasoning_tokens,
                            "tokens_in": c.input_tokens,
                            "latency_total_ms": c.duration_s * 1000,
                            "started_at": c.started_at,
                            "client_session_id": c.invocation_id,
                            "finish_reason": c.status,
                            "model": r.model,
                        }
                        for c in r.calls
                    ],
                },
                "ng_agent_observations": {
                    "source": "synthetic-demo",
                    "records": [
                        {
                            "kind": "tool_call",
                            "tool_call_id": t.id,
                            "tool_name": t.name,
                            "duration_ms": t.duration_s * 1000,
                            "started_at": t.started_at,
                            "invocation_id": t.invocation_id,
                            "status": t.status,
                        }
                        for t in r.tools
                    ]
                    + [
                        {
                            "kind": "context_compaction",
                            "invocation_id": e.invocation_id,
                            "outcome": "completed",
                            "after_model_call": {"model_call_id": e.after_call_id},
                        }
                        for e in r.compactions
                    ],
                },
            }
        )
    return rows


def canonical_rows():
    return [asdict(r) for r in demo_records()]
