# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stirrup-specific helpers shared across all task strategies.

These functions deal with Stirrup message types and history format —
nothing task-specific lives here.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, List, Tuple, cast

from stirrup.clients.utils import to_openai_messages
from stirrup.core.models import AssistantMessage, ChatMessage, SystemMessage, ToolMessage, UserMessage

from nemo_gym.openai_utils import (
    NeMoGymChatCompletionMessageParam,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from responses_api_agents.stirrup_agent.nemo_agent import NeMoUserMessage


LOGGER = logging.getLogger(__name__)

# A provider parser can occasionally leave an otherwise valid tool invocation in
# assistant ``content`` instead of returning it through ``tool_calls``. Feeding
# that raw block back on the next turn teaches the model to repeat the malformed
# format. Strip it only from the provider-bound copy; the original Stirrup
# history remains untouched for debugging and trajectory export.
_UNPARSED_TOOL_BLOCKS = (
    re.compile(r"<tool_call\b[^>]*>.*?(?:</tool_call>|$)", re.IGNORECASE | re.DOTALL),
    re.compile(r"<function=[^>]*>.*?(?:</function>|$)", re.IGNORECASE | re.DOTALL),
)


def _strip_unparsed_tool_blocks(content: str) -> str:
    cleaned = content
    for pattern in _UNPARSED_TOOL_BLOCKS:
        cleaned = pattern.sub("", cleaned)
    if cleaned == content:
        return content
    # History is serialized again on every model turn. Keep this at debug so a
    # malformed turn does not emit the same warning hundreds of times as the
    # immutable history grows; trajectory export still preserves the evidence.
    LOGGER.debug(
        "Stripped an unparsed tool-call block from provider-bound assistant history; "
        "the original trajectory remains unchanged."
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def restore_tool_messages_for_model(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Return provider-valid history for OpenAI-compatible model calls.

    NeMoUserMessage intentionally presents tool results to the agent as user
    turns during normal execution. OpenAI-compatible chat completions APIs,
    however, require assistant messages with tool_calls to be followed by
    matching tool-role messages, so provider-bound serialization must restore
    those tool results first.
    """
    pending_tool_call_ids: set[str] = set()
    restored: list[ChatMessage] = []

    for message in messages:
        if isinstance(message, AssistantMessage):
            pending_tool_call_ids = {tc.tool_call_id for tc in message.tool_calls if tc.tool_call_id}
            provider_message = message
            if not message.tool_calls and isinstance(message.content, str):
                cleaned = _strip_unparsed_tool_blocks(message.content)
                if cleaned != message.content:
                    provider_message = message.model_copy(update={"content": cleaned})
            restored.append(provider_message)
            continue

        if isinstance(message, NeMoUserMessage) and message.tool_call_id in pending_tool_call_ids:
            restored.append(
                ToolMessage(
                    content=message.content,
                    name=message.name,
                    success=message.success,
                    args_was_valid=message.args_was_valid,
                    tool_call_id=message.tool_call_id,
                    tool_start_time=message.tool_start_time,
                    tool_end_time=message.tool_end_time,
                )
            )
            pending_tool_call_ids.discard(message.tool_call_id)
            continue

        restored.append(message)

    return restored


# Stirrup's ``to_openai_messages`` leaks its internal models onto the wire:
# ``ToolCall.model_dump()`` (``tool_call_id``/``name``/``arguments``/``signature``)
# beside the canonical ``{id, type, function}`` envelope, ``name`` on tool-role
# messages, and ``reasoning_content``/``thinking_blocks``/``metadata`` on the
# assistant message. Gym's proxy ingress validated with ``extra="ignore"`` for
# every campaign to date, so the policy only ever saw the canonical keys below.
# The ingress schema is strict now; emit exactly what it accepted before rather
# than widening the schema, so the provider-bound payload stays byte-identical.
_PROVIDER_ASSISTANT_KEYS = frozenset({"role", "content", "tool_calls"})
_PROVIDER_TOOL_CALL_KEYS = frozenset({"id", "type", "function"})
_PROVIDER_TOOL_MESSAGE_KEYS = frozenset({"role", "content", "tool_call_id"})


def _canonical_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    # The flat duplicates come from the same ToolCall object as the canonical
    # envelope, so a disagreement means the serializer changed underneath us;
    # dropping the alias would then hide which value the model actually saw.
    for alias, canonical in (
        (tool_call.get("tool_call_id"), tool_call.get("id")),
        (tool_call.get("name"), function.get("name")),
        (tool_call.get("arguments"), function.get("arguments")),
    ):
        if alias is not None and canonical is not None and alias != canonical:
            raise ValueError(f"tool call alias disagrees with its canonical field: {alias!r} != {canonical!r}")
    return {key: value for key, value in tool_call.items() if key in _PROVIDER_TOOL_CALL_KEYS}


def _canonical_provider_message(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role == "assistant":
        canonical = {key: value for key, value in message.items() if key in _PROVIDER_ASSISTANT_KEYS}
        if canonical.get("tool_calls"):
            canonical["tool_calls"] = [_canonical_tool_call(tool_call) for tool_call in canonical["tool_calls"]]
        return canonical
    if role == "tool":
        return {key: value for key, value in message.items() if key in _PROVIDER_TOOL_MESSAGE_KEYS}
    return message


def to_provider_openai_messages(messages: list[ChatMessage]) -> list[NeMoGymChatCompletionMessageParam]:
    """Serialize Stirrup history for an OpenAI-compatible provider."""
    serialized = to_openai_messages(restore_tool_messages_for_model(messages))
    return cast(list[NeMoGymChatCompletionMessageParam], [_canonical_provider_message(m) for m in serialized])


def convert_stirrup_history_to_output_items(
    history: List[List[Any]],
) -> Tuple[List, List]:
    """Convert Stirrup message history into NeMoGym input/output items.

    Returns ``(input_items, output_items)`` where *input_items* are
    system/user messages and *output_items* are assistant messages +
    tool calls/results.
    """
    input_items: list = []
    output_items: list = []

    # Some serving layers mint tool_call_id per turn (e.g. "code_exec:0"), so the same id
    # recurs every turn and a trajectory ends up with many function_call / function_call_output
    # pairs sharing one call_id. Anything that later pairs them by id -- replay, training,
    # trajectory analysis -- then silently attributes the wrong output to a call. Disambiguate
    # by occurrence: the nth call and the nth output for a given raw id get the same suffix, so
    # pairing is preserved while ids become unique within the response. The first occurrence
    # keeps the raw id, so a trajectory that never repeats one is unchanged.
    call_seq: dict[str, int] = {}
    output_seq: dict[str, int] = {}
    occurrence_ids: dict[tuple[str, int], str] = {}
    used_ids: set[str] = set()

    def _disambiguate(raw: str, seen: dict[str, int]) -> str:
        seen[raw] = seen.get(raw, 0) + 1
        occurrence = seen[raw]
        key = (raw, occurrence)
        if key in occurrence_ids:
            return occurrence_ids[key]

        preferred = raw if occurrence == 1 else f"{raw}#{occurrence}"
        candidate = preferred
        collision_index = 2
        while candidate in used_ids:
            candidate = f"{preferred}#{collision_index}"
            collision_index += 1

        occurrence_ids[key] = candidate
        used_ids.add(candidate)
        return candidate

    for turn in history:
        for msg in turn:
            if isinstance(msg, SystemMessage):
                input_items.append(
                    NeMoGymEasyInputMessage(
                        role="system",
                        content=msg.content if isinstance(msg.content, str) else str(msg.content),
                    )
                )

            elif isinstance(msg, NeMoUserMessage) and msg.tool_call_id:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=_disambiguate(msg.tool_call_id, output_seq),
                        output=content,
                        type="function_call_output",
                    )
                )

            elif isinstance(msg, UserMessage):
                content_text = ""
                if isinstance(msg.content, str):
                    content_text = msg.content
                elif isinstance(msg.content, list):
                    content_text = " ".join(part.text if hasattr(part, "text") else str(part) for part in msg.content)
                else:
                    content_text = str(msg.content)

                input_items.append(NeMoGymEasyInputMessage(role="user", content=content_text))

            elif isinstance(msg, AssistantMessage):
                content_text = msg.content if isinstance(msg.content, str) else ""
                if content_text:
                    output_items.append(
                        NeMoGymResponseOutputMessage(
                            id=f"msg-{uuid.uuid4().hex[:8]}",
                            content=[
                                NeMoGymResponseOutputText(
                                    type="output_text",
                                    text=content_text,
                                    annotations=[],
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message",
                        )
                    )

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        call_id = (
                            getattr(tc, "tool_call_id", None)
                            or getattr(tc, "id", None)
                            or f"call-{uuid.uuid4().hex[:8]}"
                        )
                        output_items.append(
                            NeMoGymResponseFunctionToolCall(
                                id=f"fc-{uuid.uuid4().hex[:8]}",
                                arguments=tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments),
                                call_id=_disambiguate(call_id, call_seq),
                                name=tc.name,
                                type="function_call",
                                status="completed",
                            )
                        )

            elif isinstance(msg, ToolMessage):
                call_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else f"call-{uuid.uuid4().hex[:8]}"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        call_id=_disambiguate(call_id, output_seq),
                        output=content,
                        type="function_call_output",
                    )
                )

    return input_items, output_items


def extract_deliverable_text(history: List[List[Any]], finish_params: Any) -> str:
    """Extract the final deliverable text from a Stirrup agent run.

    Combines the ``finish_params.reason`` (if present) with the last
    assistant message in *history*.
    """
    parts: list[str] = []

    if finish_params and hasattr(finish_params, "reason") and finish_params.reason:
        parts.append(finish_params.reason)

    for turn in reversed(history):
        for msg in reversed(turn):
            if isinstance(msg, AssistantMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                if content and content not in parts:
                    parts.append(content)
                    break
        if len(parts) > 1:
            break

    return "\n\n".join(parts) if parts else ""
