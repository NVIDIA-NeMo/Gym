# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retain evidence from the original SpatialClaw trace and Gym HTTP transport."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponseInputItem,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)


_ITEM = TypeAdapter(NeMoGymResponseInputItem)


def response_usage(journal: Path) -> NeMoGymResponseUsage | None:
    """Sum observed provider usage, preserving missing usage as unknown."""
    if not journal.is_file():
        return None
    calls = [json.loads(line) for line in journal.read_text().splitlines()]
    usages = [call.get("usage") for call in calls]
    if not usages or any(not usage for usage in usages):
        return None
    if any(
        usage.get(key) is None for usage in usages for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return None

    def optional_total(details: str, key: str) -> int | None:
        values = [(usage.get(details) or {}).get(key) for usage in usages]
        return None if any(value is None for value in values) else sum(values)

    return NeMoGymResponseUsage(
        input_tokens=sum(usage["prompt_tokens"] for usage in usages),
        output_tokens=sum(usage["completion_tokens"] for usage in usages),
        total_tokens=sum(usage["total_tokens"] for usage in usages),
        input_tokens_details=NeMoGymResponseInputTokensDetails(
            cached_tokens=optional_total("prompt_tokens_details", "cached_tokens")
        ),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(
            reasoning_tokens=optional_total("completion_tokens_details", "reasoning_tokens")
        ),
    )


def read_observations(
    workspace: Path,
    session_id: str,
    *,
    model_ref: ModelServerRef,
    result: dict[str, Any],
    failure: dict[str, str] | None,
    duration_ms: float,
) -> AgentObservationBundle:
    """Translate native events, without guessing model/tool joins from their order.

    Each execution event identifies its code by session and step_index. This is
    the native Jupyter execution boundary; nested Python/GPU functions are not
    claimed as independently timed calls. All HTTP responses belong to the one
    workflow invocation, including VLM calls made inside its kernel.
    """
    conversation = []
    tools = []
    gaps = []
    calls = []
    journal = workspace / "gym-model-responses.jsonl"
    if journal.is_file():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            if record.get("response_id"):
                calls.append(ModelCallRef(model_ref=model_ref, response_id=record["response_id"]))
            else:
                gaps.append(ObservationGap(code="model_response_id_unavailable", invocation_id=session_id))
    trace = workspace / f"session-{session_id}" / "trace.jsonl"
    if not trace.is_file():
        gaps.append(ObservationGap(code="agent_artifact_unavailable", invocation_id=session_id))
    else:
        for line in trace.read_text().splitlines():
            try:
                event = json.loads(line)
                kind = event["event_type"]
                if kind == "init":
                    conversation.append(_ITEM.validate_python({"role": "system", "content": event["system_prompt"]}))
                    conversation.append(_ITEM.validate_python({"role": "user", "content": event["instruction"]}))
                elif kind in {"plan", "llm_call"}:
                    content = event.get("plan_text") if kind == "plan" else event.get("raw_response")
                    if content:
                        conversation.append(_ITEM.validate_python({"role": "assistant", "content": content}))
                elif kind == "execution":
                    call_id = f"{session_id}:execute:{event['step_index']}"
                    completed_at = datetime.fromisoformat(event["timestamp"]).timestamp()
                    duration = event.get("execution_time_sec")
                    error = event.get("error")
                    conversation.append(
                        _ITEM.validate_python(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": "python",
                                "arguments": json.dumps({"code": event["code"]}),
                            }
                        )
                    )
                    conversation.append(
                        _ITEM.validate_python(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps({key: event.get(key) for key in ("stdout", "stderr", "error")}),
                            }
                        )
                    )
                    tools.append(
                        ToolCallObservation(
                            invocation_id=session_id,
                            tool_call_id=call_id,
                            tool_name="python",
                            completed_at=completed_at,
                            duration_ms=duration * 1000 if duration is not None else None,
                            timing_source="artifact",
                            status="failed" if error else "completed",
                            error_type="python_execution_error" if error else None,
                        )
                    )
                elif kind == "feedback":
                    conversation.append(_ITEM.validate_python({"role": "user", "content": event["feedback_text"]}))
            except (ValueError, TypeError, KeyError, ValidationError) as exc:
                gaps.append(
                    ObservationGap(code="agent_artifact_record_unparseable", invocation_id=session_id, detail=str(exc))
                )
    reason = result.get("termination_reason")
    return AgentObservationBundle(
        source="spatialclaw",
        records=[
            AgentInvocation(
                invocation_id=session_id,
                status="failed" if failure else "completed",
                stop_reason="max_iterations" if reason == "max_steps" else reason,
                error_type=failure["failure_kind"] if failure else None,
                duration_ms=duration_ms,
                conversation=conversation,
                model_calls=calls,
            ),
            *tools,
        ],
        gaps=gaps,
    )
