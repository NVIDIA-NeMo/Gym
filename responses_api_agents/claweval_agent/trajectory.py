# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert native trace messages, retaining tool pairs and simulated-user turns."""

import json
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    SandboxObservation,
    ToolCallObservation,
)


class ClawEvalExecutionError(ValueError):
    """Native execution failed after its task and grading identities were validated."""


def trace_observations(
    path: Path | None,
    *,
    invocation_id: str,
    response: NeMoGymResponse,
    prompt: str,
    output_dir: Path,
    duration_ms: float,
    failed: bool = False,
) -> AgentObservationBundle:
    """Join native tool timing and sandbox outcomes to the captured policy invocation."""
    bundle = AgentObservationBundle(
        source="claweval_native",
        records=[
            AgentInvocation(
                invocation_id=invocation_id,
                status="failed" if failed else "completed",
                duration_ms=duration_ms,
                conversation=[{"role": "user", "content": prompt}, *response.output],
            )
        ],
    )
    sandbox_path = output_dir / "sandboxes.json"
    sandboxes = (
        [SandboxObservation.model_validate(row) for row in json.loads(sandbox_path.read_text())]
        if sandbox_path.exists()
        else []
    )
    bundle.records.extend(sandboxes)
    sandbox_id = sandboxes[0].sandbox_id if len(sandboxes) == 1 else None
    if not sandboxes:
        bundle.gaps.append(ObservationGap(code="sandbox_outcome_unavailable", invocation_id=invocation_id))
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path else []
    error_calls = {
        block["tool_use_id"]
        for event in events
        if event["type"] == "message"
        for block in event["message"]["content"]
        if isinstance(block, dict)
        if block.get("type") == "tool_result" and block.get("is_error")
    }
    for event in events:
        if event["type"] == "tool_dispatch":
            completed = datetime.fromisoformat(event["timestamp"]).timestamp()
            duration = event["latency_ms"]
            failed_tool = event["response_status"] >= 400 or event["tool_use_id"] in error_calls
            bundle.records.append(
                ToolCallObservation(
                    invocation_id=invocation_id,
                    tool_call_id=event["tool_use_id"],
                    sandbox_id=sandbox_id,
                    tool_name=event["tool_name"],
                    started_at=completed - duration / 1000,
                    completed_at=completed,
                    duration_ms=duration,
                    timing_source="artifact",
                    status="failed" if failed_tool else "completed",
                    error_type="native_tool_error" if failed_tool else None,
                )
            )
    return bundle


def trace_to_response(path: Path, task_id: str, model: str, expected_score: float | None = None) -> NeMoGymResponse:
    output: list[dict[str, Any]] = []
    start = end = grade = None
    seen_initial_user = False
    calls: set[str] = set()
    results: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event["type"]
            if kind == "trace_start":
                start = event
            elif kind == "trace_end":
                end = event
            elif kind == "grading_result":
                grade = event
            elif kind == "message":
                message = event["message"]
                role = message["role"]
                if role == "system":
                    continue
                blocks = message["content"]
                if isinstance(blocks, str):
                    blocks = [{"type": "text", "text": blocks}]
                # The original prompt is already in responses_create_params.input.
                if role == "user" and not seen_initial_user:
                    seen_initial_user = True
                    if not any(block["type"] == "tool_result" for block in blocks):
                        continue
                if role == "assistant" and message.get("reasoning_content"):
                    output.append(
                        {
                            "type": "reasoning",
                            "id": f"reasoning-{len(output)}",
                            "summary": [{"type": "summary_text", "text": message["reasoning_content"]}],
                        }
                    )
                for block in blocks:
                    block_type = block["type"]
                    if block_type == "text" and block["text"]:
                        if role == "assistant":
                            output.append(
                                {
                                    "type": "message",
                                    "id": f"msg-{len(output)}",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "output_text", "text": block["text"], "annotations": []}],
                                }
                            )
                        else:
                            output.append({"type": "message", "role": "user", "content": block["text"]})
                    elif block_type == "tool_use":
                        call_id = block["id"]
                        if call_id in calls:
                            raise ValueError(f"Duplicate tool call ID in Claw-Eval trace: {call_id}")
                        calls.add(call_id)
                        output.append(
                            {
                                "type": "function_call",
                                "id": call_id,
                                "call_id": call_id,
                                "name": block["name"],
                                "arguments": json.dumps(block["input"], ensure_ascii=False),
                                "status": "completed",
                            }
                        )
                    elif block_type == "tool_result":
                        call_id = block["tool_use_id"]
                        if call_id not in calls or call_id in results:
                            raise ValueError(f"Unmatched or duplicate tool result: {call_id}")
                        results.add(call_id)
                        output.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": "\n".join(
                                    part["text"] for part in block.get("content", []) if part.get("type") == "text"
                                ),
                            }
                        )
                    # Media bytes remain in the native trace. This Gym projection is
                    # for evaluation/inspection, not lossless training replay.
    if start is None or end is None or grade is None:
        raise ValueError("Claw-Eval trace must contain trace_start, trace_end, and grading_result")
    if start["task_id"] != task_id or grade["task_id"] != task_id:
        raise ValueError("Claw-Eval trace task identity mismatch")
    if expected_score is not None and grade["task_score"] != expected_score:
        raise ValueError("Claw-Eval result disagrees with its native grading event")
    if any(event["trace_id"] != start["trace_id"] for event in (end, grade)):
        raise ValueError("Claw-Eval trace ID mismatch")
    if end.get("failure_modes"):
        raise ClawEvalExecutionError(f"Claw-Eval execution failed: {end['failure_modes']}")
    if calls != results:
        raise ValueError("Claw-Eval trace has tool calls without results")
    input_tokens = end.get("model_input_tokens", end.get("input_tokens", 0))
    output_tokens = end.get("model_output_tokens", end.get("output_tokens", 0))
    return NeMoGymResponse.model_validate(
        {
            "id": start["trace_id"],
            "created_at": int(time()),
            "object": "response",
            "model": model,
            "output": output,
            "parallel_tool_calls": False,
            "tools": [],
            "tool_choice": "auto",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": end.get("cached_tokens")},
                "output_tokens_details": {"reasoning_tokens": end.get("reasoning_tokens")},
            },
        }
    )
