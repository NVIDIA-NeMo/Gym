# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project native Tau tool exchanges and generated participant turns."""

import json
from collections import defaultdict
from datetime import datetime

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymFunctionCallOutput, NeMoGymResponseFunctionToolCall
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    TrajectoryTurn,
)
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun


def build_tool_observations(result: SimulationRun) -> AgentObservationBundle:
    """Retain uniquely matched native tool results, separating policy and simulator.

    Conversation items here preserve tool exchanges only, not complete model prompts.
    Native message timestamps are not executor timings. Invocation outcomes and model
    ownership also remain unknown until Tau supplies those observations separately.
    """
    bundle = AgentObservationBundle(source="tau2")
    requests: dict[tuple[str, str], list[tuple[int, ToolCall]]] = defaultdict(list)
    results: dict[tuple[str, str], list[tuple[int, ToolMessage]]] = defaultdict(list)
    for index, message in enumerate(result.messages or []):
        if isinstance(message, (AssistantMessage, UserMessage)):
            for call in message.tool_calls or []:
                if call.requestor != message.role:
                    bundle.gaps.append(ObservationGap(code="tau_tool_requestor_mismatch", detail=call.id))
                    continue
                requests[(call.requestor, call.id)].append((index, call))
        elif isinstance(message, (ToolMessage, MultiToolMessage)):
            for output in message.tool_messages if isinstance(message, MultiToolMessage) else [message]:
                results[(output.requestor, output.id)].append((index, output))

    invocations: dict[str, AgentInvocation] = {}
    for owner, call_id in dict.fromkeys([*requests, *results]):
        invocation = invocations.setdefault(
            owner,
            AgentInvocation(invocation_id=f"{result.id}:{'agent' if owner == 'assistant' else 'user_simulator'}"),
        )
        key = (owner, call_id)
        calls, outputs = requests[key], results[key]
        if not call_id or len(calls) != 1 or len(outputs) > 1:
            bundle.gaps.append(
                ObservationGap(
                    code="tau_tool_join_unavailable", invocation_id=invocation.invocation_id, detail=call_id
                )
            )
            continue
        request_index, call = calls[0]
        # The shared collector joins the result to this invocation's conversation by call ID.
        invocation.conversation.append(
            NeMoGymResponseFunctionToolCall(
                type="function_call", call_id=call.id, name=call.name, arguments=json.dumps(call.arguments)
            )
        )
        if not outputs or outputs[0][0] <= request_index:
            bundle.gaps.append(
                ObservationGap(code="tau_tool_result_missing", invocation_id=invocation.invocation_id, detail=call_id)
            )
            continue
        output = outputs[0][1]
        if output.content is not None:
            invocation.conversation.append(
                NeMoGymFunctionCallOutput(type="function_call_output", call_id=call.id, output=output.content)
            )
        bundle.records.append(
            ToolCallObservation(
                invocation_id=invocation.invocation_id,
                tool_call_id=call.id,
                tool_name=call.name,
                status="failed" if output.error else "completed",
            )
        )

    for invocation in invocations.values():
        bundle.records.append(invocation)
        bundle.gaps.append(
            ObservationGap(
                code="tau_tool_transcript_only",
                invocation_id=invocation.invocation_id,
                detail="Tool exchanges only; full model context and executor timings are unavailable.",
            )
        )
    return bundle


def build_trajectory(
    result: SimulationRun,
    *,
    observations: AgentObservationBundle,
    task_id: str,
    rollout_id: str,
    policy_model: ModelServerRef,
    user_model: ModelServerRef,
) -> TrajectoryRecord:
    """Associate native generated messages with captured calls for shared ng_perf aggregation.

    Tau retains the raw response on each generated participant message. Scripted
    greetings and tool outputs have no raw response and are not decision turns.
    Both policy and simulator consumption contribute to rollout-wide ng_perf.
    Rejected attempts/internal retries are not reconstructed from accepted messages.
    """
    invocations = {r.invocation_id: r for r in observations.records if isinstance(r, AgentInvocation)}
    trajectory = TrajectoryRecord(task_id=task_id, rollout_id=rollout_id)
    turn_counts: dict[str, int] = defaultdict(int)
    for message in result.messages or []:
        if not isinstance(message, (AssistantMessage, UserMessage)) or message.raw_data is None:
            continue
        invocation_id = f"{result.id}:{'agent' if message.role == 'assistant' else 'user_simulator'}"
        if invocation_id not in invocations:
            invocations[invocation_id] = AgentInvocation(invocation_id=invocation_id)
            observations.records.append(invocations[invocation_id])
        invocation = invocations[invocation_id]
        response_id = message.raw_data.get("id")
        refs = (
            [
                ModelCallRef(
                    model_ref=policy_model if message.role == "assistant" else user_model, response_id=response_id
                )
            ]
            if isinstance(response_id, str) and response_id.strip()
            else []
        )
        invocation.model_calls.extend(refs)
        if not refs:
            trajectory.gaps.append(
                ObservationGap(code="model_call_reference_unavailable", invocation_id=invocation_id)
            )
        if message.timestamp is None:
            trajectory.gaps.append(ObservationGap(code="tau_turn_timestamp_missing", invocation_id=invocation_id))
            continue
        turn_counts[invocation_id] += 1
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id=invocation_id,
                task_id=trajectory.task_id,
                rollout_id=rollout_id,
                turn_no=turn_counts[invocation_id],
                timestamp=datetime.fromisoformat(message.timestamp).timestamp(),
                answer={
                    "content": message.content,
                    "tool_calls": [call.model_dump(mode="json") for call in message.tool_calls or []],
                },
                step_count=turn_counts[invocation_id],
                model_calls=refs,
            )
        )
    trajectory.invocations = list(invocations.values())
    return trajectory
