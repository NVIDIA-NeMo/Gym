# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project dependency-light Hermes runtime events into Gym observability records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any
from uuid import uuid4

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseInputItem,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from responses_api_agents.hermes_agent.raw_observability import RawHermesObserver


_SOURCE = "hermes"
_CONVERTER = ResponsesConverter(return_token_id_information=False)


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def normalize_hermes_chat_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Expose Hermes dispatcher calls as the actual tool calls they represent."""
    normalized = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        message = deepcopy(raw_message)
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments", "")
            parsed_arguments = arguments
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass

            if function.get("name") == "tool_call" and isinstance(parsed_arguments, dict):
                target_name = parsed_arguments.get("name")
                if isinstance(target_name, str) and target_name and "arguments" in parsed_arguments:
                    function["name"] = target_name
                    arguments = parsed_arguments["arguments"]
            function["arguments"] = _json_arguments(arguments)
        normalized.append(message)
    return normalized


def project_hermes_response_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Remove Hermes-internal dispatch steps from the external action trajectory."""
    messages = [deepcopy(message) for message in messages if isinstance(message, dict)]
    omitted_call_ids = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if isinstance(function, dict) and function.get("name") in {"tool_describe", "tool_search"}:
                call_id = call.get("id")
                if call_id:
                    omitted_call_ids.add(call_id)

        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str) and (
                "does not exist. Available tools:" in content or "The tool was NOT invoked." in content
            ):
                call_id = message.get("tool_call_id")
                if call_id:
                    omitted_call_ids.add(call_id)

    projected = []
    for message in messages:
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            message["tool_calls"] = [
                call for call in calls if not isinstance(call, dict) or call.get("id") not in omitted_call_ids
            ]
        if message.get("role") == "tool" and message.get("tool_call_id") in omitted_call_ids:
            continue
        if message.get("role") == "assistant" and not message.get("content") and not message.get("tool_calls"):
            continue
        projected.append(message)
    return normalize_hermes_chat_messages(projected)


def normalize_hermes_messages(
    messages: Iterable[Any],
    *,
    id_prefix: str = "hermes",
) -> list[NeMoGymResponseInputItem]:
    """Convert Hermes Chat Completions messages with Hermes reasoning into Responses items."""
    output: list[NeMoGymResponseInputItem] = []
    for index, message in enumerate(normalize_hermes_chat_messages(messages)):
        reasoning = message.pop("reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            output.append(
                NeMoGymResponseReasoningItem(
                    id=f"{id_prefix}-reasoning-{index}",
                    summary=[NeMoGymSummary(text=reasoning, type="summary_text")],
                )
            )
        output.extend(_CONVERTER.chat_completions_messages_to_responses_items([message]))
    return output


def build_hermes_observations(
    raw: dict[str, Any],
    *,
    model_ref: ModelServerRef | None = None,
) -> AgentObservationBundle:
    """Validate raw child-process events against Gym's canonical observation schema."""
    gaps = [ObservationGap.model_validate(gap) for gap in raw.get("gaps") or [] if isinstance(gap, dict)]

    def add_gap(code: str, invocation_id: str | None, detail: str | None = None) -> None:
        gap = ObservationGap(code=code, invocation_id=invocation_id, detail=detail)
        if gap not in gaps:
            gaps.append(gap)

    records = []
    for invocation in raw.get("invocations") or []:
        if not isinstance(invocation, dict):
            continue
        invocation_id = str(invocation.get("invocation_id") or f"hermes-{uuid4().hex}")
        model_calls = []
        response_ids = [
            response_id
            for response_id in invocation.get("model_response_ids") or []
            if isinstance(response_id, str) and response_id
        ]
        if model_ref is None:
            if response_ids:
                add_gap("model_response_id_unavailable", invocation_id, "model_ref")
                add_gap("model_call_ownership_unavailable", invocation_id)
        else:
            model_calls = [ModelCallRef(model_ref=model_ref, response_id=response_id) for response_id in response_ids]

        conversation = normalize_hermes_messages(
            invocation.get("messages") or [],
            id_prefix=invocation_id,
        )
        system_message = invocation.get("system_message")
        if (
            isinstance(system_message, str)
            and system_message
            and not (conversation and getattr(conversation[0], "role", None) == "system")
        ):
            conversation.insert(0, NeMoGymEasyInputMessage(role="system", content=system_message))

        records.append(
            AgentInvocation(
                invocation_id=invocation_id,
                parent_invocation_id=invocation.get("parent_invocation_id"),
                spawned_by_tool_call_id=invocation.get("spawned_by_tool_call_id"),
                status=invocation.get("status", "unknown"),
                model_calls=model_calls,
                conversation=conversation,
            )
        )

    records.extend(
        ToolCallObservation.model_validate(tool) for tool in raw.get("tools") or [] if isinstance(tool, dict)
    )
    records.extend(
        ContextCompactionObservation.model_validate(compaction)
        for compaction in raw.get("compactions") or []
        if isinstance(compaction, dict)
    )
    return AgentObservationBundle(source=str(raw.get("source") or _SOURCE), records=records, gaps=gaps)


class HermesAgentObserver:
    """Compatibility wrapper for in-process callers and unit tests."""

    def __init__(
        self,
        *,
        root_invocation_id: str = "root",
        model_ref: ModelServerRef | None = None,
    ) -> None:
        self._raw = RawHermesObserver(root_invocation_id=root_invocation_id)
        self._model_ref = model_ref

    def instrument(self, agent: Any) -> "HermesAgentObserver":
        self._raw.instrument(agent)
        return self

    def finish(
        self,
        result: dict[str, Any] | None = None,
        *,
        error: BaseException | None = None,
    ) -> AgentObservationBundle:
        return build_hermes_observations(
            self._raw.finish(result, error=error),
            model_ref=self._model_ref,
        )
