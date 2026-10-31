# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project OSWorld's recorded policy calls into Gym's shared trajectory contract."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)


def _conversation(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    # OSWorld windows its history. This is the actual final context, not an
    # invented append-only conversation; earlier requests remain in captured calls.
    messages = []
    for message in call["prompt_messages"] or []:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"type": "input_text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    image = part["image_url"]
                    image = {"url": image} if isinstance(image, str) else image
                    parts.append(
                        {"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")}
                    )
                else:
                    parts.append(part)
            content = parts
        messages.append({"role": message["role"], "content": content})
    response = call.get("response") or {}
    raw_content = response.get("raw_content")
    messages.append(
        {
            "role": "assistant",
            "content": raw_content if isinstance(raw_content, str) else response.get("content") or "",
        }
    )
    return messages


def _valid_usage(usage: Any) -> bool:
    return isinstance(usage, Mapping) and all(
        type(usage.get(key)) is int and usage[key] >= 0 for key in ("prompt_tokens", "completion_tokens")
    )


def build_gym_observability(
    *,
    request: BaseRunRequest,
    model_calls: Sequence[Mapping[str, Any]],
    model_ref: ModelServerRef,
    evaluation_completed: bool,
    model_call_records_complete: bool,
) -> TrajectoryRecord | None:
    """Keep adapter-observed calls, including parser retries, and explicit capture references.

    OSWorld's separate semantic IDs are not capture IDs. Shared references join
    by the actual model server and completion response ID. Missing evidence is
    retained as a gap instead of manufacturing a successful observation.
    """
    extra = request.model_extra or {}
    identity = extra.get("trajectory_identity") or {}
    rollout_id = identity.get("rollout_id") or maybe_rollout_id_from_run_body(request)
    if rollout_id is None:
        return None
    task_id = identity.get("task_id") or next(
        (
            str(extra[key])
            for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
            if extra.get(key) is not None
        ),
        "unknown",
    )
    # Keep failed requests in OSWorld's records and shared captures, but only
    # completions own turns. Missing evidence alone is not proof of a failed request.
    completion_calls = [
        call for call in model_calls if not (call.get("completion_received") is False and call.get("response") is None)
    ]
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    if not model_call_records_complete:
        trajectory.gaps.append(
            ObservationGap(code="model_calls_unavailable", detail="osworld_step_model_call_records")
        )
        trajectory.gaps.append(
            ObservationGap(code="turn_model_call_scope_incomplete", detail="osworld_step_model_call_records")
        )
        if not completion_calls:
            trajectory.gaps.append(ObservationGap(code="turns_unavailable", detail="osworld_step_model_call_records"))
    invocation = AgentInvocation(
        invocation_id="root",
        status="completed" if evaluation_completed else "incomplete",
        conversation=_conversation(completion_calls[-1]) if completion_calls else [],
    )
    trajectory.invocations.append(invocation)
    for position, call in enumerate(completion_calls):
        response = call.get("response") or {}
        observed = response.get("observation") or {}
        if not _valid_usage(observed.get("usage")):
            trajectory.gaps.append(ObservationGap(code="model_call_usage_unavailable", detail=f"call:{position}"))
        response_id = observed.get("response_id")
        references = []
        if isinstance(response_id, str) and response_id:
            references.append(ModelCallRef(model_ref=model_ref, response_id=response_id))
            invocation.model_calls.extend(references)
        else:
            trajectory.gaps.append(ObservationGap(code="turn_model_call_scope_incomplete", detail=f"call:{position}"))
        timestamp = observed.get("started_at")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            trajectory.gaps.append(ObservationGap(code="turns_unavailable", detail=f"call:{position}:timestamp"))
            trajectory.gaps.append(ObservationGap(code="turn_model_call_scope_incomplete", detail=f"call:{position}"))
        else:
            trajectory.turns.append(
                TrajectoryTurn(
                    invocation_id=invocation.invocation_id,
                    task_id=task_id,
                    rollout_id=rollout_id,
                    turn_no=position + 1,
                    timestamp=timestamp,
                    # Exact requests, including screenshots, live in the referenced
                    # captures. Avoid duplicating every image again on each turn.
                    answer=response.get("content"),
                    reasoning_content=response.get("reasoning_content"),
                    step_count=call["step_position"],
                    model_calls=references,
                )
            )
    return trajectory
