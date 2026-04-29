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
"""
Anthropic model server for NeMo-Gym.

Accepts OpenAI Responses API format (NeMoGymResponseCreateParamsNonStreaming),
translates to Anthropic Messages API, calls the Anthropic backend, and
translates the response back to NeMoGymResponse.

All context management (conversation history, trimming) is handled by the
agent/adapter layer — this server is a stateless translator + API relay.
"""

import logging
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from anthropic import AsyncAnthropic
from fastapi import HTTPException

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymComputerToolCall,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)


logger = logging.getLogger(__name__)


class AnthropicModelServerConfig(BaseResponsesAPIModelConfig):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_timeout: float = 300.0
    anthropic_max_tokens: int = 4096
    effort_level: str = "high"
    computer_tool_version: str = "computer_20250124"
    computer_betas: List[str] = ["computer-use-2025-01-24", "token-efficient-tools-2025-02-19"]
    zoom_enabled: bool = False


class AnthropicModelServer(SimpleResponsesAPIModel):
    config: AnthropicModelServerConfig

    def model_post_init(self, context):
        self._client = AsyncAnthropic(
            api_key=self.config.anthropic_api_key,
            max_retries=4,
            timeout=self.config.anthropic_timeout,
        )
        return super().model_post_init(context)

    # ── OpenAI Responses API endpoint ────────────────────────────

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        messages = self._translate_input_to_messages(body)
        system = body.instructions or ""

        tools, betas = self._derive_tools_and_betas(body)

        api_params: Dict[str, Any] = {
            "model": self.config.anthropic_model,
            "max_tokens": body.max_output_tokens or self.config.anthropic_max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            api_params["tools"] = tools

        if self.config.effort_level != "high":
            api_params["output_config"] = {"effort": self.config.effort_level}

        if betas:
            api_params["betas"] = betas
            logger.info(
                "Anthropic proxy (beta): model=%s, num_messages=%d, betas=%s",
                api_params["model"],
                len(messages),
                betas,
            )
            response = await self._client.beta.messages.create(**api_params)
        else:
            logger.info(
                "Anthropic proxy: model=%s, num_messages=%d",
                api_params["model"],
                len(messages),
            )
            response = await self._client.messages.create(**api_params)

        logger.info(
            "Anthropic proxy response: stop_reason=%s, blocks=%d",
            response.stop_reason,
            len(response.content),
        )

        return self._translate_response(response, body)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise HTTPException(status_code=501, detail="Anthropic model server does not support /v1/chat/completions")

    # ── Tool & beta derivation from body.tools ────────────────────

    def _derive_tools_and_betas(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> tuple:
        """Inspect body.tools to derive provider-specific tools and betas.

        If body.tools contains a computer_use_preview tool, translates it to
        the Anthropic computer tool format specified by config (tool version,
        betas, zoom). Different Anthropic models require different versions —
        e.g. Sonnet 4 uses computer_20250124 while Opus 4 uses computer_20251124.

        Otherwise returns empty lists (generic text mode).
        """
        if not body.tools:
            return [], []

        anthropic_tools: List[Dict[str, Any]] = []
        betas: List[str] = []

        for t in body.tools:
            t_dict = t.model_dump() if hasattr(t, "model_dump") else t
            tool_type = t_dict.get("type", "")

            if tool_type == "computer_use_preview":
                w = t_dict.get("display_width", 1280)
                h = t_dict.get("display_height", 720)
                tool: Dict[str, Any] = {
                    "type": self.config.computer_tool_version,
                    "name": "computer",
                    "display_width_px": w,
                    "display_height_px": h,
                    "display_number": 1,
                }
                if self.config.zoom_enabled:
                    tool["enable_zoom"] = True
                anthropic_tools.append(tool)
                betas = list(self.config.computer_betas)

        return anthropic_tools, betas

    # ── Inbound: OpenAI input items → Anthropic messages ─────────

    def _translate_input_to_messages(self, body: NeMoGymResponseCreateParamsNonStreaming) -> List[Dict[str, Any]]:
        raw_input = body.input
        if isinstance(raw_input, str):
            return [{"role": "user", "content": [{"type": "text", "text": raw_input}]}]

        messages: List[Dict[str, Any]] = []
        input_items = [item.model_dump() if hasattr(item, "model_dump") else item for item in raw_input]

        for item in input_items:
            item_type = item.get("type", "")

            if item_type == "message":
                messages.append(self._translate_message_item(item))

            elif item_type == "computer_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": item["call_id"],
                                "name": "computer",
                                "input": item["action"],
                            }
                        ],
                    }
                )

            elif item_type == "computer_call_output":
                content: List[Dict[str, Any]] = []
                output = item.get("output", {})
                screenshot_b64 = self._extract_screenshot_b64(output)
                if screenshot_b64:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        }
                    )
                error_url = output.get("current_url", "")
                if error_url.startswith("error:"):
                    content.insert(0, {"type": "text", "text": f"Action failed: {error_url}"})

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": item["call_id"],
                                "is_error": error_url.startswith("error:"),
                                "content": content,
                            }
                        ],
                    }
                )

        return messages

    @staticmethod
    def _translate_message_item(item: Dict[str, Any]) -> Dict[str, Any]:
        role = item.get("role", "user")
        content = item.get("content", "")

        if isinstance(content, str):
            return {"role": role, "content": [{"type": "text", "text": content}]}

        parts: List[Dict[str, Any]] = []
        for part in content:
            part_type = part.get("type", "")
            if part_type in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "input_image":
                image_url = part.get("image_url", "")
                b64_data = _extract_b64_from_data_url(image_url)
                if b64_data:
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            },
                        }
                    )

        if not parts:
            parts.append({"type": "text", "text": str(content)})

        return {"role": role, "content": parts}

    @staticmethod
    def _extract_screenshot_b64(output: Dict[str, Any]) -> Optional[str]:
        image_url = output.get("image_url", "")
        if not image_url:
            return None
        return _extract_b64_from_data_url(image_url)

    # ── Outbound: Anthropic response → NeMoGymResponse ───────────

    def _translate_response(
        self,
        response,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        output_items = []

        for block in response.content:
            if block.type == "tool_use":
                output_items.append(
                    NeMoGymComputerToolCall(
                        id=f"cu_{uuid4().hex}",
                        action=block.input or {},
                        call_id=block.id,
                        pending_safety_checks=[],
                        status="completed",
                        type="computer_call",
                    ).model_dump()
                )
            elif block.type == "text":
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}",
                        content=[
                            NeMoGymResponseOutputText(
                                type="output_text",
                                text=block.text,
                                annotations=[],
                            )
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    ).model_dump()
                )

        if not output_items:
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text="",
                            annotations=[],
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ).model_dump()
            )

        usage = None
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens or 0
            output_tokens = response.usage.output_tokens or 0
            usage = NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            )

        incomplete_details = None
        if response.stop_reason == "max_tokens":
            incomplete_details = {"reason": "max_output_tokens"}

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.anthropic_model,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice if hasattr(body, "tool_choice") else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            reasoning=body.reasoning,
            truncation=body.truncation,
            metadata=body.metadata,
            instructions=body.instructions,
            user=body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )


def _extract_b64_from_data_url(data_url: str) -> Optional[str]:
    """Extract base64 data from a data URL like 'data:image/png;base64,...'."""
    if data_url.startswith("data:"):
        parts = data_url.split(",", 1)
        return parts[1] if len(parts) == 2 else None
    return data_url if data_url else None


if __name__ == "__main__":
    AnthropicModelServer.run_webserver()
