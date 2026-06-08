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
import asyncio
import json
from contextlib import nullcontext
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.server_utils import request as aiohttp_request


class ClaudeModelConfig(BaseResponsesAPIModelConfig):
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_api_key: str
    anthropic_model: str
    max_tokens: int
    anthropic_version: str = "2023-06-01"
    thinking_budget_tokens: Optional[int] = None
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Cap on in-flight upstream requests from this server (per-process asyncio.Semaphore). None = unlimited."
        ),
    )
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class ClaudeModel(SimpleResponsesAPIModel):
    config: ClaudeModelConfig

    def model_post_init(self, context):
        self._converter = ClaudeConverter()
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )
        return super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        anthropic_body = self._converter.responses_to_anthropic(
            body=body,
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            thinking_budget_tokens=self.config.thinking_budget_tokens,
            extra_body=self.config.extra_body,
        )
        async with self._semaphore:
            anthropic_response = await self._messages_create(anthropic_body, cookies=request.cookies)

        return self._converter.anthropic_to_responses(
            anthropic_response=anthropic_response,
            request_body=body,
            model=self.config.anthropic_model,
        )

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError("claude_model supports /v1/responses only")

    async def _messages_create(self, body: Dict[str, Any], cookies: Dict[str, str]) -> Dict[str, Any]:
        request_kwargs = {
            "url": self._messages_url(),
            "json": body,
            "headers": {
                "x-api-key": self.config.anthropic_api_key,
                "anthropic-version": self.config.anthropic_version,
            },
            "cookies": cookies,
        }
        response = await aiohttp_request(method="POST", **request_kwargs)
        await raise_for_status(response)
        return await get_response_json(response)

    def _messages_url(self) -> str:
        base_url = self.config.anthropic_base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return f"{base_url}/messages"
        return f"{base_url}/v1/messages"


class ClaudeConverter:
    def responses_to_anthropic(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        model: str,
        max_tokens: int,
        thinking_budget_tokens: Optional[int],
        extra_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        body_dict = body.model_dump(exclude_unset=True)
        anthropic_body = dict(extra_body)
        anthropic_body.update(
            {
                "model": model,
                "max_tokens": body_dict.pop("max_output_tokens", None) or max_tokens,
                "messages": [],
            }
        )

        system_parts = []
        if body.instructions:
            system_parts.append(body.instructions)

        response_input = body_dict.pop("input")
        input_items = self._normalize_input(response_input)
        for item in input_items:
            item_type = item.get("type") or "message"
            if item_type == "message":
                self._append_message_item(item, anthropic_body["messages"], system_parts)
            elif item_type == "reasoning":
                self._append_content(
                    anthropic_body["messages"],
                    "assistant",
                    self._reasoning_item_to_anthropic_blocks(item),
                )
            elif item_type == "function_call":
                self._append_content(
                    anthropic_body["messages"],
                    "assistant",
                    [self._function_call_to_tool_use(item)],
                )
            elif item_type == "function_call_output":
                self._append_content(
                    anthropic_body["messages"],
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": item["call_id"],
                            "content": item["output"],
                        }
                    ],
                )
            else:
                raise NotImplementedError(f"Unsupported Responses API item type for Claude: {item_type}")

        if system_parts:
            anthropic_body["system"] = self._system_parts_to_anthropic_blocks(system_parts)

        self._copy_sampling_params(body_dict, anthropic_body)
        self._copy_tools(body_dict, anthropic_body)
        self._copy_tool_choice(body_dict, anthropic_body)
        if thinking_budget_tokens is not None:
            anthropic_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }

        return anthropic_body

    def anthropic_to_responses(
        self,
        anthropic_response: Dict[str, Any],
        request_body: NeMoGymResponseCreateParamsNonStreaming,
        model: str,
    ) -> NeMoGymResponse:
        output = []
        pending_text = []
        for block in anthropic_response.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                pending_text.append(block.get("text", ""))
            elif block_type == "thinking":
                self._flush_text_output(pending_text, output)
                output.append(
                    NeMoGymResponseReasoningItem(
                        id=f"rs_{uuid4().hex}",
                        summary=[
                            NeMoGymSummary(
                                text=block.get("thinking") or block.get("text", ""),
                                type="summary_text",
                            )
                        ],
                        encrypted_content=block.get("signature"),
                    )
                )
            elif block_type == "tool_use":
                self._flush_text_output(pending_text, output)
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=json.dumps(block.get("input", {})),
                        call_id=block["id"],
                        name=block["name"],
                        id=block["id"],
                        status="completed",
                    )
                )
            else:
                raise NotImplementedError(f"Unsupported Anthropic content block type: {block_type}")

        self._flush_text_output(pending_text, output)
        if not output:
            self._flush_text_output([""], output)

        usage = self._usage_to_responses_usage(anthropic_response.get("usage"))
        stop_reason = anthropic_response.get("stop_reason")
        incomplete_details = {"reason": "max_output_tokens"} if stop_reason == "max_tokens" else None

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model,
            object="response",
            output=[item.model_dump() for item in output],
            tool_choice=request_body.tool_choice,
            parallel_tool_calls=request_body.parallel_tool_calls,
            tools=request_body.tools,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            background=request_body.background,
            max_output_tokens=request_body.max_output_tokens,
            max_tool_calls=request_body.max_tool_calls,
            previous_response_id=request_body.previous_response_id,
            prompt=request_body.prompt,
            reasoning=request_body.reasoning,
            service_tier=request_body.service_tier,
            text=request_body.text,
            top_logprobs=request_body.top_logprobs,
            truncation=request_body.truncation,
            metadata=request_body.metadata,
            instructions=request_body.instructions,
            user=request_body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )

    def _normalize_input(self, response_input: Any) -> List[Dict[str, Any]]:
        if isinstance(response_input, str):
            return [NeMoGymEasyInputMessage(content=response_input, role="user").model_dump(exclude_unset=True)]
        return [
            item.model_dump(exclude_unset=True) if hasattr(item, "model_dump") else item for item in response_input
        ]

    def _append_message_item(
        self,
        item: Dict[str, Any],
        messages: List[Dict[str, Any]],
        system_parts: List[str],
    ) -> None:
        role = item["role"]
        content = item.get("content", "")
        if role in ("system", "developer"):
            system_parts.append(self._content_to_text(content))
            return
        if role not in ("user", "assistant"):
            raise NotImplementedError(f"Unsupported Responses API role for Claude: {role}")
        self._append_content(messages, role, self._content_to_anthropic_blocks(content, role))

    def _append_content(
        self,
        messages: List[Dict[str, Any]],
        role: str,
        content_blocks: List[Dict[str, Any]],
    ) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(content_blocks)
        else:
            messages.append({"role": role, "content": content_blocks})

    def _content_to_anthropic_blocks(self, content: Any, role: str) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks = []
        for part in content:
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text"):
                blocks.append({"type": "text", "text": part["text"]})
            elif part_type == "refusal" and role == "assistant":
                blocks.append({"type": "text", "text": part["refusal"]})
            else:
                raise NotImplementedError(f"Unsupported content part for Claude: {part_type}")
        return blocks

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        texts = []
        for part in content:
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text"):
                texts.append(part["text"])
            else:
                raise NotImplementedError(f"Unsupported system content part for Claude: {part_type}")
        return "\n".join(texts)

    def _system_parts_to_anthropic_blocks(self, system_parts: List[str]) -> List[Dict[str, str]]:
        return [{"type": "text", "text": text} for text in system_parts if text]

    def _reasoning_item_to_anthropic_blocks(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = []
        for summary in item.get("summary", []):
            block = {
                "type": "thinking",
                "thinking": summary["text"],
            }
            if item.get("encrypted_content"):
                block["signature"] = item["encrypted_content"]
            blocks.append(block)
        return blocks

    def _function_call_to_tool_use(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": item["call_id"],
            "name": item["name"],
            "input": self._json_object_from_arguments(item["arguments"]),
        }

    def _json_object_from_arguments(self, arguments: str) -> Dict[str, Any]:
        parsed = json.loads(arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError(f"Claude tool_use input must be a JSON object, got {type(parsed).__name__}")
        return parsed

    def _copy_sampling_params(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        for source, target in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
        ):
            value = body_dict.get(source)
            if value is not None:
                anthropic_body[target] = value

    def _copy_tools(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        tools = body_dict.get("tools") or []
        if not tools:
            return

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") != "function":
                raise NotImplementedError(f"Unsupported Responses API tool type for Claude: {tool.get('type')}")
            anthropic_tool = {
                "name": tool["name"],
                "input_schema": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            if tool.get("description"):
                anthropic_tool["description"] = tool["description"]
            anthropic_tools.append(anthropic_tool)
        anthropic_body["tools"] = anthropic_tools

    def _copy_tool_choice(self, body_dict: Dict[str, Any], anthropic_body: Dict[str, Any]) -> None:
        tool_choice = body_dict.get("tool_choice")
        if tool_choice is None:
            return
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                anthropic_body["tool_choice"] = {"type": "any"}
            elif tool_choice in ("auto", "none"):
                anthropic_body["tool_choice"] = {"type": tool_choice}
            else:
                raise NotImplementedError(f"Unsupported tool_choice for Claude: {tool_choice}")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            anthropic_body["tool_choice"] = {"type": "tool", "name": tool_choice["name"]}
        else:
            raise NotImplementedError(f"Unsupported tool_choice for Claude: {tool_choice}")

    def _flush_text_output(self, pending_text: List[str], output: List[Any]) -> None:
        if not pending_text:
            return
        output.append(
            NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text="".join(pending_text),
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )
        pending_text.clear()

    def _usage_to_responses_usage(self, usage: Optional[Dict[str, Any]]) -> Optional[NeMoGymResponseUsage]:
        if usage is None:
            return None
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        return NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(
                cached_tokens=usage.get("cache_read_input_tokens", 0)
            ),
            output_tokens=output_tokens,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens + output_tokens,
        )


if __name__ == "__main__":
    ClaudeModel.run_webserver()
