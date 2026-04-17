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
from typing import Any, Dict, List, Optional, Tuple

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)


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
            msg_type = type(msg).__name__

            if msg_type == "SystemMessage":
                input_items.append(
                    NeMoGymEasyInputMessage(
                        role="system",
                        content=msg.content if isinstance(msg.content, str) else str(msg.content),
                    )
                )

            elif msg_type == "UserMessage":
                content_text = ""
                if isinstance(msg.content, str):
                    content_text = msg.content
                elif isinstance(msg.content, list):
                    content_text = " ".join(
                        part.text if hasattr(part, "text") else str(part)
                        for part in msg.content
                    )
                else:
                    content_text = str(msg.content)

                input_items.append(
                    NeMoGymEasyInputMessage(role="user", content=content_text)
                )

            elif msg_type == "AssistantMessage":
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
                        call_id = tc.id if hasattr(tc, "id") else f"call-{uuid.uuid4().hex[:8]}"
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

            elif msg_type == "ToolMessage":
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
            if type(msg).__name__ == "AssistantMessage":
                content = msg.content if isinstance(msg.content, str) else ""
                if content and content not in parts:
                    parts.append(content)
                    break
        if len(parts) > 1:
            break

    return "\n\n".join(parts) if parts else ""
