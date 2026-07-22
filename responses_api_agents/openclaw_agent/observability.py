# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observability exposed by Gym's existing OpenClaw response parser."""

from collections.abc import Iterable
from typing import Any

from nemo_gym.openai_utils import NeMoGymResponseInputItem
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    ToolCallObservation,
)


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def build_openclaw_observations(
    invocation_id: str,
    conversation: Iterable[NeMoGymResponseInputItem],
    *,
    transcript_available: bool,
    source: str = "openclaw",
) -> AgentObservationBundle:
    """Describe signals present in Gym's normalized OpenClaw response."""
    items = list(conversation)
    result_ids = {
        call_id
        for item in items
        if _field(item, "type") == "function_call_output" and isinstance((call_id := _field(item, "call_id")), str)
    }

    def tool_status(call_id: str) -> str:
        # OpenClaw's normalized output marks result presence, not execution
        # success. Do not turn that transport status into a success claim.
        return "unknown" if call_id in result_ids else "incomplete"

    tools = [
        ToolCallObservation(
            invocation_id=invocation_id,
            tool_call_id=call_id,
            tool_name=_field(item, "name"),
            status=tool_status(call_id),
        )
        for item in items
        if _field(item, "type") == "function_call"
        and isinstance((call_id := _field(item, "call_id")), str)
        and call_id
    ]

    gaps = [
        ObservationGap(code="subagent_hierarchy_unavailable", source=source),
        ObservationGap(code="model_call_ownership_unavailable", source=source),
        ObservationGap(code="context_compaction_unavailable", source=source),
    ]
    if not transcript_available:
        gaps.append(ObservationGap(code="agent_transcript_unavailable", source=source))
    if tools:
        gaps.append(ObservationGap(code="tool_timing_unavailable", source=source, invocation_id=invocation_id))

    return AgentObservationBundle(
        source=source,
        invocations=[AgentInvocation(invocation_id=invocation_id, conversation=items)],
        tool_calls=tools,
        gaps=gaps,
    )
