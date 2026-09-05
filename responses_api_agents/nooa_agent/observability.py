# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from datetime import datetime
from time import time
from typing import Any

from nooa.atif import Trajectory

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part.text for part in value if getattr(part, "type", None) == "text")
    return str(value)


def _json_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _step_response_items(step: Any, prefix: str) -> list[Any]:
    items: list[Any] = []
    if step.reasoning_content:
        items.append(
            NeMoGymResponseReasoningItem(
                id=f"{prefix}-reasoning-{step.step_id}",
                summary=[NeMoGymSummary(text=step.reasoning_content, type="summary_text")],
            )
        )
    if message := _text(step.message):
        items.append(
            NeMoGymResponseOutputMessage(
                id=f"{prefix}-message-{step.step_id}",
                content=[NeMoGymResponseOutputText(annotations=[], text=message)],
            )
        )
    for call in step.tool_calls or []:
        items.append(
            NeMoGymResponseFunctionToolCall(
                id=call.tool_call_id,
                call_id=call.tool_call_id,
                name=call.function_name,
                arguments=json.dumps(call.arguments),
                status="completed",
            )
        )
    for result in step.observation.results if step.observation is not None else []:
        if result.source_call_id is not None:
            items.append(
                NeMoGymFunctionCallOutput(
                    call_id=result.source_call_id,
                    output=_json_output(result.content or ""),
                    status="completed",
                )
            )
    return items


def _response_items(trajectory: Trajectory) -> list[Any]:
    prefix = trajectory.trajectory_id or trajectory.session_id or "root"
    return [item for step in trajectory.steps if step.source == "agent" for item in _step_response_items(step, prefix)]


def _input_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [NeMoGymEasyInputMessage(role="user", content=value)]
    return list(value)


def _conversation(trajectory: Trajectory) -> list[Any]:
    items: list[Any] = []
    prefix = trajectory.trajectory_id or trajectory.session_id or "root"
    for step in trajectory.steps:
        if step.source in {"system", "user"}:
            items.append(NeMoGymEasyInputMessage(role=step.source, content=_text(step.message)))
        elif step.source == "agent":
            items.extend(_step_response_items(step, prefix))
    return items


def _timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _trajectory_records(
    trajectory: Trajectory,
    *,
    parent_invocation_id: str | None = None,
    path: tuple[int, ...] = (),
) -> tuple[list[AgentInvocation], list[ToolCallObservation]]:
    invocation_id = trajectory.trajectory_id or trajectory.session_id or f"nooa-{'-'.join(map(str, path)) or 'root'}"
    invocation = AgentInvocation(
        invocation_id=invocation_id,
        parent_invocation_id=parent_invocation_id,
        status="failed" if (trajectory.extra or {}).get("crashed") else "completed",
        conversation=_conversation(trajectory),
    )
    tools: list[ToolCallObservation] = []
    for step in trajectory.steps:
        results = {
            result.source_call_id
            for result in (step.observation.results if step.observation is not None else [])
            if result.source_call_id is not None
        }
        for call in step.tool_calls or []:
            tools.append(
                ToolCallObservation(
                    invocation_id=invocation_id,
                    tool_call_id=call.tool_call_id,
                    tool_name=call.function_name,
                    started_at=_timestamp(step.timestamp),
                    timing_source="artifact",
                    status="completed" if call.tool_call_id in results else "incomplete",
                )
            )

    invocations = [invocation]
    for index, child in enumerate(trajectory.subagent_trajectories or []):
        child_invocations, child_tools = _trajectory_records(
            child,
            parent_invocation_id=invocation_id,
            path=(*path, index),
        )
        invocations.extend(child_invocations)
        tools.extend(child_tools)
    return invocations, tools


def project_nooa_episode(
    *,
    create_params: NeMoGymResponseCreateParamsNonStreaming,
    trajectory: Trajectory,
    model_calls: list[ModelCallRef],
) -> AgentEpisode:
    """Project one native NOOA trajectory into Gym's standard episode contract."""

    output = _response_items(trajectory)
    gaps: list[ObservationGap] = []

    invocations, tools = _trajectory_records(trajectory)
    if len(invocations) == 1:
        invocations[0].model_calls = list(model_calls)
        if not invocations[0].conversation:
            invocations[0].conversation = [*_input_items(create_params.input), *output]
    elif model_calls:
        gaps.append(
            ObservationGap(
                code="model_call_ownership_unavailable",
                detail="ATIF does not expose response IDs needed to assign model calls across nested invocations.",
            )
        )

    response = NeMoGymResponse(
        id=f"nooa-{trajectory.trajectory_id or trajectory.session_id or 'embedded'}",
        created_at=time(),
        model=create_params.model or trajectory.agent.model_name or "nooa",
        object="response",
        output=output,
        tools=create_params.tools,
        tool_choice=create_params.tool_choice,
        parallel_tool_calls=create_params.parallel_tool_calls,
    )
    return AgentEpisode(
        response=response,
        observations=AgentObservationBundle(source="nooa", records=[*invocations, *tools], gaps=gaps),
    )


def _ends_with_completed_assistant_message(response: NeMoGymResponse) -> bool:
    if not response.output:
        return False
    last = response.output[-1]
    return isinstance(last, NeMoGymResponseOutputMessage) and last.status == "completed"


def ensure_verifier_final_message(
    response: NeMoGymResponse,
    return_value: Any,
) -> tuple[NeMoGymResponse, list[ObservationGap]]:
    """Add a verifier-facing fallback message without mutating the ATIF episode."""

    gaps: list[ObservationGap] = []
    if _ends_with_completed_assistant_message(response) or return_value is None:
        return response, gaps

    # Verifiers commonly grade the last assistant message, so preserve the full ATIF trace and append the
    # entrypoint return only when that trace does not already end with a completed assistant message.
    output = list(response.output)
    output.append(
        NeMoGymResponseOutputMessage(
            id="nooa_fallback",
            content=[NeMoGymResponseOutputText(annotations=[], text=_json_output(return_value))],
        )
    )
    gaps.append(
        ObservationGap(
            code="non_trainable_fallback_output",
            detail="NOOA returned a value without a final model-authored message.",
        )
    )
    return response.model_copy(update={"output": output}), gaps


def finalize_observations(
    observations: AgentObservationBundle,
    *,
    extra_gaps: list[ObservationGap] | None = None,
    termination_reason: str | None = None,
    termination_error: str | None = None,
) -> AgentObservationBundle:
    """Merge lifecycle-only observation gaps onto the ATIF projection."""

    gaps = [*observations.gaps, *(extra_gaps or [])]
    if termination_reason is not None:
        gaps.append(
            ObservationGap(
                code=termination_reason,
                detail=termination_error or f"NOOA execution terminated with {termination_reason}.",
            )
        )
    if not gaps:
        return observations
    return observations.model_copy(update={"gaps": gaps})
