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
            restored.append(message)
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


def to_provider_openai_messages(messages: list[ChatMessage]) -> list[NeMoGymChatCompletionMessageParam]:
    """Serialize Stirrup history for an OpenAI-compatible provider."""
    return cast(list[NeMoGymChatCompletionMessageParam], to_openai_messages(restore_tool_messages_for_model(messages)))


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
                        call_id=msg.tool_call_id,
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
                                call_id=call_id,
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
                        call_id=call_id,
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
