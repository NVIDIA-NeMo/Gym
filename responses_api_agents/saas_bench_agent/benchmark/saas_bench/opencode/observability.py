# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/opencode/observability.py
# Copyright: SaaS-Bench Authors; see bundled licenses.
# Vendored for native benchmark evaluation; original behavior retained.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish native actor turns to Gym without inventing model-call joins."""

from typing import get_args

from nemo_gym.rollout_observability import TrajectoryRecord

from saas_bench.opencode.core import response_items


def native_trajectory(
    events: list[dict], task_id: str, rollout_id: str, health: dict
) -> dict:
    invocation = next(
        (e["sessionID"] for e in events if e.get("sessionID")), "opencode"
    )
    groups = {}
    tools = []
    for event in events:
        part = event.get("part", {})
        message_id = part.get("messageID")
        if event.get("type") == "step_start":
            groups[message_id] = {"events": [], "timestamp": event["timestamp"] / 1000}
        elif message_id in groups:
            groups[message_id]["events"].append(event)
        if event.get("type") == "tool_use":
            state = part.get("state", {})
            timing = state.get("time", {})
            tools.append(
                {
                    "invocation_id": invocation,
                    "tool_call_id": part["callID"],
                    "tool_name": part["tool"],
                    "status": "failed"
                    if state.get("status") == "error"
                    else "completed"
                    if state.get("status") == "completed"
                    else "unknown",
                    "started_at": timing["start"] / 1000
                    if timing.get("start")
                    else None,
                    "completed_at": timing["end"] / 1000 if timing.get("end") else None,
                    "timing_source": "artifact",
                    "output": state.get("output", state.get("error")),
                }
            )
    turns = []
    turn_schema = get_args(TrajectoryRecord.model_fields["turns"].annotation)[0]
    for message_id, group in groups.items():
        answer = response_items(group["events"])
        if answer:
            turn = {
                "invocation_id": invocation,
                "task_id": task_id,
                "rollout_id": rollout_id,
                "turn_no": len(turns) + 1,
                "timestamp": group["timestamp"],
                "answer": answer,
                "step_count": len(turns) + 1,
            }
            if "source_message_id" in turn_schema.model_fields:
                turn["source_message_id"] = message_id
            turns.append(turn)
    termination = health.get("termination", "unknown")
    invocation = {
        "invocation_id": invocation,
        "status": "completed" if termination == "final_response" else "incomplete",
        "conversation": response_items(events),
    }
    invocation_schema = get_args(
        TrajectoryRecord.model_fields["invocations"].annotation
    )[0]
    if "stop_reason" in invocation_schema.model_fields:
        invocation["stop_reason"] = termination
    return TrajectoryRecord.model_validate(
        {
            "task_id": task_id,
            "rollout_id": rollout_id,
            "invocations": [invocation],
            "turns": turns,
            "tool_calls": tools,
            "gaps": [
                {
                    "code": "native_model_call_join_unavailable",
                    "detail": "Native message IDs and tool outcomes are observed; exact model-call references are not supplied by OpenCode. Compaction/title calls remain in model captures.",
                }
            ],
        }
    ).model_dump(mode="json")
