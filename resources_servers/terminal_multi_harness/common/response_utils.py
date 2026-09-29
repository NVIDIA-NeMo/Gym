# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import TYPE_CHECKING, Any, Optional

from resources_servers.terminal_multi_harness.common.verification_utils import (
    ExpectedAction,
    FunctionCallAction,
    FunctionCallBatchAction,
    MessageAction,
)


if TYPE_CHECKING:
    from nemo_gym.openai_utils import NeMoGymResponse


def _build_action(tool_calls: list[FunctionCallAction], assistant_text: str | None) -> Optional[ExpectedAction]:
    if tool_calls:
        if len(tool_calls) == 1:
            return tool_calls[0]
        return FunctionCallBatchAction(
            type="function_call_batch",
            calls=tool_calls,
            ordered=True,
        )

    if assistant_text is not None:
        return MessageAction(
            type="message",
            content=assistant_text,
        )

    return None


def _extract_assistant_text_from_content(content: Any) -> str | None:
    text_parts: list[str] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for content_item in content:
            if hasattr(content_item, "type") and getattr(content_item, "type", None) == "output_text":
                text = getattr(content_item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
                continue

            if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                text = content_item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

    if not text_parts:
        return None

    return "".join(text_parts)


def extract_action(response: "NeMoGymResponse") -> Optional[ExpectedAction]:
    assistant_text_parts: list[str] = []
    tool_calls: list[FunctionCallAction] = []

    for output_item in response.output:
        if output_item.type == "function_call":
            tool_calls.append(
                FunctionCallAction(
                    type="function_call",
                    name=output_item.name,
                    arguments=output_item.arguments,
                )
            )
            continue

        if output_item.type == "message" and output_item.role == "assistant":
            assistant_text = _extract_assistant_text_from_content(output_item.content)
            if assistant_text is not None:
                assistant_text_parts.append(assistant_text)

    assistant_text = "".join(assistant_text_parts) if assistant_text_parts else None
    return _build_action(tool_calls, assistant_text)


def extract_action_from_responses_api_response(response: dict[str, Any]) -> Optional[ExpectedAction]:
    assistant_text_parts: list[str] = []
    tool_calls: list[FunctionCallAction] = []

    for output_item in response.get("output") or []:
        if not isinstance(output_item, dict):
            continue

        if output_item.get("type") == "function_call":
            name = output_item.get("name")
            arguments = output_item.get("arguments")
            if isinstance(name, str) and isinstance(arguments, str):
                tool_calls.append(
                    FunctionCallAction(
                        type="function_call",
                        name=name,
                        arguments=arguments,
                    )
                )
            continue

        if output_item.get("type") == "message" and output_item.get("role") == "assistant":
            assistant_text = _extract_assistant_text_from_content(output_item.get("content"))
            if assistant_text is not None:
                assistant_text_parts.append(assistant_text)

    assistant_text = "".join(assistant_text_parts) if assistant_text_parts else None
    return _build_action(tool_calls, assistant_text)


def extract_action_from_backend_response(backend_response: dict[str, Any]) -> Optional[ExpectedAction]:
    """Normalize Aspen's raw chat-completions backend response into an expected action.

    For Codex collection, this payload is the teacher-model source of truth for the
    next-step expected answer. Aspen's synthesized `responses_api_response` should
    normalize to the same canonical action.
    """
    choices = backend_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None

    tool_calls: list[FunctionCallAction] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue

        function_payload = tool_call.get("function")
        if not isinstance(function_payload, dict):
            continue

        name = function_payload.get("name")
        arguments = function_payload.get("arguments")
        if isinstance(name, str) and isinstance(arguments, str):
            tool_calls.append(
                FunctionCallAction(
                    type="function_call",
                    name=name,
                    arguments=arguments,
                )
            )

    assistant_text = _extract_assistant_text_from_content(message.get("content"))

    return _build_action(tool_calls, assistant_text)
