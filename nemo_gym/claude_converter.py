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
"""Converter between the Anthropic Messages API (``/v1/messages``) and the OpenAI
Chat Completions API.

Mirrors the structure of ``VLLMConverter`` (Responses <-> Chat Completions) in
``responses_api_models/vllm_model/app.py``: a small Pydantic ``BaseModel`` with one
method per direction plus private ``_convert_*`` helpers. The field mappings follow
``nemo_switchyard``'s ``anthropic_openai`` translation module.

Two directions, both non-streaming:

- ``messages_to_chat_completion_create_params``: an Anthropic Messages request
  (``NeMoGymAnthropicMessageCreateParamsNonStreaming``) ->
  ``NeMoGymChatCompletionCreateParamsNonStreaming``.
- ``chat_completion_to_message``: a Chat Completions response
  (``NeMoGymChatCompletion``) -> ``NeMoGymAnthropicMessage``.

Mapping summary:
- Anthropic ``system`` param -> OpenAI system message
- Anthropic ``text`` block -> OpenAI content string
- Anthropic ``tool_use`` block -> OpenAI assistant ``tool_calls`` entry
- Anthropic ``tool_result`` block -> OpenAI ``tool`` role message
- Anthropic tool definitions -> OpenAI ``tools`` (JSON schema under ``parameters``)
- Anthropic ``tool_choice`` <-> OpenAI ``tool_choice``
- OpenAI ``finish_reason`` -> Anthropic ``stop_reason``

Out of scope for now (raise / drop rather than silently mishandle): streaming,
training token-id/logprob threading, image/document content blocks. ``thinking`` /
``redacted_thinking`` blocks are dropped on the request path.
"""

import json
from typing import Any, ClassVar, Dict, List, Optional, Union
from uuid import uuid4

from anthropic.types import StopReason
from pydantic import BaseModel

from nemo_gym.anthropic_utils import (
    NeMoGymAnthropicContentBlock,
    NeMoGymAnthropicMessage,
    NeMoGymAnthropicMessageCreateParamsNonStreaming,
    NeMoGymAnthropicTextBlock,
    NeMoGymAnthropicToolUseBlock,
    NeMoGymAnthropicUsage,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChatCompletionMessageParam,
    NeMoGymChatCompletionMessageToolCallFunctionParam,
    NeMoGymChatCompletionMessageToolCallParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    NeMoGymChatCompletionToolParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymFunctionDefinition,
)


class ClaudeConverter(BaseModel):
    # Anthropic tool_choice (string or {"type": ...}) -> OpenAI tool_choice string.
    _TOOL_CHOICE_MAP: ClassVar[Dict[str, str]] = {
        "auto": "auto",
        "any": "required",
        "none": "none",
    }

    # OpenAI finish_reason -> Anthropic stop_reason.
    _FINISH_REASON_MAP: ClassVar[Dict[str, StopReason]] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",  # legacy
    }

    # =======================================================
    # Messages create params -> Chat Completion create params
    # =======================================================

    def messages_to_chat_completion_create_params(
        self,
        body: NeMoGymAnthropicMessageCreateParamsNonStreaming,
    ) -> NeMoGymChatCompletionCreateParamsNonStreaming:
        """Convert an Anthropic Messages request to a Chat Completions request."""
        messages: List[NeMoGymChatCompletionMessageParam] = []

        # Anthropic carries the system prompt as a top-level param; Chat
        # Completions carries it as the first message.
        system_message = self._convert_system(body.system)
        if system_message is not None:
            messages.append(system_message)

        # body.messages entries are validated TypedDicts, i.e. plain dicts here.
        for message in body.messages:
            messages.extend(self._convert_message(message))

        kwargs: Dict[str, Any] = {"messages": messages, "max_tokens": body.max_tokens}
        if body.model is not None:
            kwargs["model"] = body.model
        if body.temperature is not None:
            kwargs["temperature"] = body.temperature
        if body.top_p is not None:
            kwargs["top_p"] = body.top_p
        if body.stop_sequences:
            kwargs["stop"] = body.stop_sequences
        if body.tools:
            kwargs["tools"] = self._convert_tools(body.tools)
        if body.tool_choice:
            kwargs["tool_choice"] = self._convert_tool_choice(body.tool_choice)
        # Note: Anthropic `top_k` has no Chat Completions equivalent and is dropped.

        return NeMoGymChatCompletionCreateParamsNonStreaming(**kwargs)

    def _convert_system(
        self,
        system: Optional[Union[str, List[Dict[str, Any]]]],
    ) -> Optional[NeMoGymChatCompletionSystemMessageParam]:
        if not system:
            return None
        if isinstance(system, str):
            text = system
        else:
            # A validated Anthropic system list contains only text blocks.
            text = " ".join(block.get("text", "") for block in system)
        return NeMoGymChatCompletionSystemMessageParam(role="system", content=text)

    def _convert_message(self, message: Dict[str, Any]) -> List[NeMoGymChatCompletionMessageParam]:
        # ``message`` is a validated Anthropic MessageParam: ``role`` is one of
        # user/assistant/system and ``content`` is a str or a list of typed blocks.
        role = message["role"]
        content = message["content"]

        if isinstance(content, str):
            return [self._simple_message(role, content)]

        text_parts: List[str] = []
        tool_calls: List[NeMoGymChatCompletionMessageToolCallParam] = []
        tool_messages: List[NeMoGymChatCompletionMessageParam] = []

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type in ("thinking", "redacted_thinking"):
                # Dropped on the request path for now.
                continue
            elif block_type == "tool_use":
                # Anthropic tool_use input is always an object; serialize it to
                # the JSON string OpenAI tool_calls expect.
                tool_calls.append(
                    NeMoGymChatCompletionMessageToolCallParam(
                        id=block["id"],
                        type="function",
                        function=NeMoGymChatCompletionMessageToolCallFunctionParam(
                            name=block["name"],
                            arguments=json.dumps(block["input"]),
                        ),
                    )
                )
            elif block_type == "tool_result":
                tool_messages.append(
                    NeMoGymChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=block.get("tool_use_id", ""),
                        content=self._flatten_tool_result_content(block.get("content", "")),
                    )
                )
            else:
                # image / document blocks have no Chat Completions equivalent yet.
                raise NotImplementedError(f"Unsupported Anthropic content block type: {block_type!r}")

        # An assistant turn requesting tools becomes a single assistant message
        # carrying both any text and the tool_calls.
        if tool_calls:
            return [
                NeMoGymChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=" ".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls,
                )
            ]

        # Otherwise emit any tool_result messages, then a text message. In
        # Anthropic format tool_use and tool_result don't share a message, so
        # these branches are mutually exclusive in practice.
        result: List[NeMoGymChatCompletionMessageParam] = list(tool_messages)
        if text_parts:
            result.append(self._simple_message(role, " ".join(text_parts)))
        elif not result:
            result.append(self._simple_message(role, ""))
        return result

    def _simple_message(self, role: str, content: str) -> NeMoGymChatCompletionMessageParam:
        match role:
            case "user":
                return NeMoGymChatCompletionUserMessageParam(role="user", content=content)
            case "assistant":
                return NeMoGymChatCompletionAssistantMessageParam(role="assistant", content=content)
            case "system":
                return NeMoGymChatCompletionSystemMessageParam(role="system", content=content)
            case _:
                raise NotImplementedError(f"Unsupported Anthropic message role: {role!r}")

    def _flatten_tool_result_content(self, content: Any) -> str:
        """Flatten Anthropic tool_result content (string or block list) to a string.

        Text blocks contribute their text; non-text blocks are JSON-encoded so their
        data is preserved rather than silently dropped.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                else:
                    parts.append(json.dumps(block))
            return " ".join(parts)
        return str(content)

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[NeMoGymChatCompletionToolParam]:
        converted: List[NeMoGymChatCompletionToolParam] = []
        for tool in tools:
            # Custom tools carry an input_schema; Anthropic server tools (web_search,
            # bash, ...) don't and have no Chat Completions equivalent.
            if "input_schema" not in tool:
                raise NotImplementedError(f"Unsupported Anthropic tool (no input_schema): {tool.get('name', tool)!r}")
            converted.append(
                NeMoGymChatCompletionToolParam(
                    type="function",
                    function=NeMoGymFunctionDefinition(
                        name=tool.get("name", ""),
                        description=tool.get("description", ""),
                        parameters=tool["input_schema"],
                    ),
                )
            )
        return converted

    def _convert_tool_choice(self, tool_choice: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        if isinstance(tool_choice, str):
            return self._TOOL_CHOICE_MAP.get(tool_choice, tool_choice)
        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
        return self._TOOL_CHOICE_MAP.get(choice_type, "auto")

    # =======================================================
    # Chat Completion -> Messages response
    # =======================================================

    def chat_completion_to_message(
        self,
        chat_completion: NeMoGymChatCompletion,
        model: Optional[str] = None,
    ) -> NeMoGymAnthropicMessage:
        """Convert a Chat Completions response to an Anthropic Messages response."""
        choice = chat_completion.choices[0]
        content_blocks = self._message_to_content_blocks(choice.message)
        stop_reason = self._FINISH_REASON_MAP.get(choice.finish_reason or "stop", "end_turn")

        usage = chat_completion.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return NeMoGymAnthropicMessage(
            id=f"msg_{uuid4().hex}",
            type="message",
            role="assistant",
            model=model or chat_completion.model,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=NeMoGymAnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def _message_to_content_blocks(
        self,
        message: NeMoGymChatCompletionMessage,
    ) -> List[NeMoGymAnthropicContentBlock]:
        blocks: List[NeMoGymAnthropicContentBlock] = []

        content = message.content
        if content:
            blocks.append(NeMoGymAnthropicTextBlock(type="text", text=content))

        for tool_call in message.tool_calls or []:
            arguments = tool_call.function.arguments
            try:
                tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                # Preserve the raw arguments rather than dropping the tool call.
                tool_input = {"raw": arguments}
            blocks.append(
                NeMoGymAnthropicToolUseBlock(
                    type="tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=tool_input,
                )
            )

        # Anthropic responses always carry at least one content block.
        if not blocks:
            blocks.append(NeMoGymAnthropicTextBlock(type="text", text=""))

        return blocks
