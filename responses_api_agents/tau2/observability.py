# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project native Tau tool exchanges without inferring execution or model-call ownership."""

import json
from collections import defaultdict

from nemo_gym.openai_utils import NeMoGymFunctionCallOutput, NeMoGymResponseFunctionToolCall
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    ToolCallObservation,
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
                detail="Tool exchanges only; full model context, call ownership and executor timings are unavailable.",
            )
        )
    return bundle
