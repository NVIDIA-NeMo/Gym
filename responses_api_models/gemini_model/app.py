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
Gemini model server for NeMo-Gym.

Accepts OpenAI Responses API format (NeMoGymResponseCreateParamsNonStreaming),
translates to Gemini generate_content API, calls the Gemini backend, and
translates the response back to NeMoGymResponse.

All context management (conversation history, trimming) is handled by the
agent/adapter layer — this server is a stateless translator + API relay.
"""

import asyncio
import base64
import logging
import random
from time import time
from typing import Any, Dict, Optional
from uuid import uuid4

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


_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "ResourceExhausted",
        "ServiceUnavailable",
        "InternalServerError",
        "DeadlineExceeded",
        "Aborted",
        "ServerError",
        "TooManyRequests",
    }
)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 520})


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """Determine if a Gemini API exception is transient and safe to retry."""
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    if type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES:
        return True
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
        return True
    return False


class GeminiModelServerConfig(BaseResponsesAPIModelConfig):
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-computer-use-preview-10-2025"
    gemini_timeout: float = 300.0
    gemini_max_retries: int = 4
    thinking_level: str = "MEDIUM"
    include_thoughts: bool = True


class GeminiModelServer(SimpleResponsesAPIModel):
    config: GeminiModelServerConfig

    def model_post_init(self, context):
        from google import genai

        self._client = genai.Client(api_key=self.config.gemini_api_key)
        return super().model_post_init(context)

    # ── OpenAI Responses API endpoint ────────────────────────────

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        model_name = self.config.gemini_model

        contents = self._translate_input_to_contents(body)
        gen_config = self._build_generate_config(body)

        logger.info(
            "Gemini proxy: model=%s, num_contents=%d",
            model_name,
            len(contents),
        )

        max_attempts = self.config.gemini_max_retries + 1
        last_exception: Optional[Exception] = None
        attempt = 0

        while attempt < max_attempts:
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=gen_config,
                    ),
                    timeout=self.config.gemini_timeout,
                )

                candidates_count = len(response.candidates) if response.candidates else 0
                logger.info("Gemini proxy response: candidates=%d", candidates_count)
                return self._translate_response(response, body)

            except Exception as e:
                last_exception = e
                if not _is_retryable_gemini_error(e):
                    logger.error("Gemini API call failed (non-retryable): %s", repr(e))
                    raise

                attempt += 1
                logger.warning(
                    "Gemini API call failed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    repr(e),
                )

                if attempt < max_attempts:
                    wait = min((1.0 + random.uniform(0, 1)) * (2**attempt), 60)
                    logger.info("Retrying in %.1fs...", wait)
                    await asyncio.sleep(wait)

        logger.error(
            "Gemini API call failed after %d attempts: %s",
            attempt,
            repr(last_exception),
        )
        raise last_exception  # type: ignore[misc]

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise HTTPException(status_code=501, detail="Gemini model server does not support /v1/chat/completions")

    # ── Provider config helpers ──────────────────────────────────

    def _build_generate_config(self, body: NeMoGymResponseCreateParamsNonStreaming):
        from google.genai import types

        is_gemini_3 = "gemini-3" in self.config.gemini_model.lower()

        kwargs: Dict[str, Any] = {}

        gemini_tools = self._derive_tools(body)
        if gemini_tools:
            kwargs["tools"] = gemini_tools

        if body.instructions:
            kwargs["system_instruction"] = body.instructions

        if is_gemini_3:
            kwargs["temperature"] = body.temperature if body.temperature is not None else 1.0
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.config.thinking_level,
                include_thoughts=self.config.include_thoughts,
            )
        else:
            kwargs["temperature"] = body.temperature if body.temperature is not None else 0.0

        if body.max_output_tokens is not None:
            kwargs["max_output_tokens"] = body.max_output_tokens
        if body.top_p is not None:
            kwargs["top_p"] = body.top_p

        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _derive_tools(body: NeMoGymResponseCreateParamsNonStreaming) -> list:
        """Inspect body.tools to derive Gemini-native tools.

        If body.tools contains a computer_use_preview tool, translates it to
        Gemini's ComputerUse tool. Otherwise returns an empty list.
        """
        from google.genai import types

        if not body.tools:
            return []

        gemini_tools = []
        for t in body.tools:
            t_dict = t.model_dump() if hasattr(t, "model_dump") else t
            if t_dict.get("type") == "computer_use_preview":
                gemini_tools.append(
                    types.Tool(
                        computer_use=types.ComputerUse(
                            environment=types.Environment.ENVIRONMENT_BROWSER,
                        ),
                    )
                )

        return gemini_tools

    # ── Inbound: OpenAI input items → Gemini contents ────────────

    def _translate_input_to_contents(self, body: NeMoGymResponseCreateParamsNonStreaming) -> list:
        from google.genai.types import Content, FunctionCall, FunctionResponse, Part

        raw_input = body.input
        if isinstance(raw_input, str):
            return [Content(role="user", parts=[Part(text=raw_input)])]

        input_items = [item.model_dump() if hasattr(item, "model_dump") else item for item in raw_input]

        contents: list = []
        for item in input_items:
            item_type = item.get("type", "")

            if item_type == "message":
                contents.append(self._translate_message_to_content(item))

            elif item_type == "computer_call":
                action = item.get("action", {})
                fn_name = action.get("type", "click_at")
                fn_args = {k: v for k, v in action.items() if k not in ("type", "safety_decision")}
                safety = action.get("safety_decision")
                if safety:
                    fn_args["safety_decision"] = safety
                contents.append(
                    Content(
                        role="model",
                        parts=[Part(function_call=FunctionCall(name=fn_name, args=fn_args))],
                    )
                )

            elif item_type == "computer_call_output":
                output = item.get("output", {})
                call_id = item.get("call_id", "")
                parts: list = []

                current_url = output.get("current_url", "about:blank")
                response_dict: Dict[str, Any] = {"url": current_url, "status": "success"}
                if current_url.startswith("error:"):
                    response_dict["status"] = "error"
                    response_dict["error"] = current_url
                if item.get("acknowledged_safety_checks"):
                    response_dict["safety_acknowledgement"] = "true"

                fn_name_for_resp = self._find_function_name_for_call_id(
                    contents, call_id, item.get("action_name", "click_at")
                )
                parts.append(
                    Part(
                        function_response=FunctionResponse(
                            name=fn_name_for_resp,
                            response=response_dict,
                        )
                    )
                )

                screenshot_b64 = self._extract_screenshot_b64(output)
                if screenshot_b64:
                    screenshot_bytes = base64.b64decode(screenshot_b64)
                    parts.append(Part.from_bytes(data=screenshot_bytes, mime_type="image/png"))

                contents.append(Content(role="user", parts=parts))

        return contents

    @staticmethod
    def _translate_message_to_content(item: Dict[str, Any]):
        from google.genai.types import Content, Part

        role = item.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        content = item.get("content", "")

        if isinstance(content, str):
            return Content(role=gemini_role, parts=[Part(text=content)])

        parts = []
        for part in content:
            part_type = part.get("type", "")
            if part_type in ("input_text", "output_text", "text"):
                parts.append(Part(text=part.get("text", "")))
            elif part_type == "input_image":
                image_url = part.get("image_url", "")
                b64_data = _extract_b64_from_data_url(image_url)
                if b64_data:
                    img_bytes = base64.b64decode(b64_data)
                    parts.append(Part.from_bytes(data=img_bytes, mime_type="image/png"))

        if not parts:
            parts.append(Part(text=str(content)))

        return Content(role=gemini_role, parts=parts)

    @staticmethod
    def _find_function_name_for_call_id(contents: list, call_id: str, fallback: str) -> str:
        """Walk backwards through contents to find the function_call name matching a call_id.

        Since Gemini doesn't use call_ids natively, we match by looking at the
        immediately preceding model content's function_call names.
        """
        for i in range(len(contents) - 1, -1, -1):
            c = contents[i]
            if c.role == "model":
                for part in c.parts or []:
                    if hasattr(part, "function_call") and part.function_call:
                        return part.function_call.name
        return fallback

    @staticmethod
    def _extract_screenshot_b64(output: Dict[str, Any]) -> Optional[str]:
        image_url = output.get("image_url", "")
        if not image_url:
            return None
        return _extract_b64_from_data_url(image_url)

    # ── Outbound: Gemini response → NeMoGymResponse ──────────────

    def _translate_response(
        self,
        response,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> NeMoGymResponse:
        output_items = []

        candidates = response.candidates or []
        if candidates:
            candidate = candidates[0]
            if candidate.content:
                for part in candidate.content.parts or []:
                    if hasattr(part, "function_call") and part.function_call:
                        fn_name = part.function_call.name
                        fn_args = dict(part.function_call.args or {})
                        safety_decision = fn_args.pop("safety_decision", None)
                        action_dict = {"type": fn_name, **fn_args}
                        pending_safety = []
                        if safety_decision:
                            action_dict["safety_decision"] = safety_decision
                            pending_safety = [safety_decision]
                        output_items.append(
                            NeMoGymComputerToolCall(
                                id=f"cu_{uuid4().hex}",
                                action=action_dict,
                                call_id=f"call_{uuid4().hex}",
                                pending_safety_checks=pending_safety,
                                status="completed",
                                type="computer_call",
                            ).model_dump()
                        )
                    elif hasattr(part, "text") and part.text and not getattr(part, "thought", False):
                        output_items.append(
                            NeMoGymResponseOutputMessage(
                                id=f"msg_{uuid4().hex}",
                                content=[
                                    NeMoGymResponseOutputText(
                                        type="output_text",
                                        text=part.text,
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
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            usage = NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            )

        incomplete_details = None
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
            if "MAX_TOKENS" in finish_reason:
                incomplete_details = {"reason": "max_output_tokens"}

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.gemini_model,
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
    GeminiModelServer.run_webserver()
