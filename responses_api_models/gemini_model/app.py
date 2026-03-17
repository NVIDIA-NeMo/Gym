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

Stateless proxy that receives ready-to-send Gemini API parameters,
forwards them to the Google Gemini generate_content API, and returns the raw response.
All context management (conversation history, trimming) is handled by the
agent/adapter layer — this server is a pure API relay.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
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


class GeminiProxyRequest(BaseModel):
    """Request body for the Gemini proxy endpoint.

    Contains Gemini-native API parameters, prepared by the adapter.
    The 'contents' field holds serialized Content objects; 'config' holds
    the serialized GenerateContentConfig.
    """

    model_config = ConfigDict(extra="allow")

    contents: List[Dict[str, Any]]
    config: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None


class GeminiModelServer(SimpleResponsesAPIModel):
    config: GeminiModelServerConfig

    def model_post_init(self, context):
        from google import genai

        self._client = genai.Client(api_key=self.config.gemini_api_key)
        return super().model_post_init(context)

    async def responses(self, body: GeminiProxyRequest = Body()):  # type: ignore[override]
        model_name = body.model or self.config.gemini_model

        contents = self._deserialize_contents(body.contents)
        gen_config = self._deserialize_config(body.config)

        logger.info(
            "Gemini proxy: model=%s, num_contents=%d",
            model_name,
            len(contents),
        )

        max_attempts = self.config.gemini_max_retries + 1
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
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
                return self._serialize_response(response)

            except Exception as e:
                last_exception = e
                if not _is_retryable_gemini_error(e):
                    logger.error("Gemini API call failed (non-retryable): %s", repr(e))
                    raise

                logger.warning(
                    "Gemini API call failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    repr(e),
                )

                if attempt < self.config.gemini_max_retries:
                    wait = 0.5 * (2**attempt)
                    logger.info("Retrying in %.1fs...", wait)
                    await asyncio.sleep(wait)

        logger.error(
            "Gemini API call failed after %d attempts: %s",
            max_attempts,
            repr(last_exception),
        )
        raise last_exception  # type: ignore[misc]

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError("Gemini model server does not support /v1/chat/completions")

    @staticmethod
    def _deserialize_contents(raw_contents: List[Dict[str, Any]]):
        """Convert serialized content dicts back into google.genai Content objects."""
        import base64

        from google.genai.types import Content, FunctionCall, FunctionResponse, Part

        contents = []
        for raw in raw_contents:
            role = raw.get("role", "user")
            parts = []
            for raw_part in raw.get("parts", []):
                if "text" in raw_part:
                    parts.append(Part(text=raw_part["text"]))
                elif "function_call" in raw_part:
                    fc = raw_part["function_call"]
                    parts.append(Part(function_call=FunctionCall(name=fc["name"], args=fc.get("args", {}))))
                elif "function_response" in raw_part:
                    fr = raw_part["function_response"]
                    parts.append(
                        Part(
                            function_response=FunctionResponse(
                                name=fr["name"],
                                response=fr.get("response", {}),
                            )
                        )
                    )
                elif "inline_data" in raw_part:
                    data_info = raw_part["inline_data"]
                    mime_type = data_info.get("mime_type", "image/png")
                    data_b64 = data_info.get("data", "")
                    data_bytes = base64.b64decode(data_b64) if isinstance(data_b64, str) else data_b64
                    parts.append(Part.from_bytes(data=data_bytes, mime_type=mime_type))
                elif raw_part.get("thought"):
                    parts.append(Part(text=raw_part.get("text", ""), thought=True))
            contents.append(Content(role=role, parts=parts))
        return contents

    @staticmethod
    def _deserialize_config(raw_config: Dict[str, Any]):
        """Convert serialized config dict back into GenerateContentConfig."""
        from google.genai import types

        if not raw_config:
            return types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        computer_use=types.ComputerUse(
                            environment=types.Environment.ENVIRONMENT_BROWSER,
                        )
                    )
                ],
                temperature=0.0,
            )

        tools = []
        for raw_tool in raw_config.get("tools", []):
            if "computer_use" in raw_tool:
                cu = raw_tool["computer_use"]
                env_str = cu.get("environment", "ENVIRONMENT_BROWSER")
                env_val = getattr(types.Environment, env_str, types.Environment.ENVIRONMENT_BROWSER)
                tools.append(types.Tool(computer_use=types.ComputerUse(environment=env_val)))
            else:
                tools.append(
                    types.Tool(
                        computer_use=types.ComputerUse(
                            environment=types.Environment.ENVIRONMENT_BROWSER,
                        )
                    )
                )

        kwargs = {
            "tools": tools,
            "temperature": raw_config.get("temperature", 0.0),
        }

        if "system_instruction" in raw_config:
            kwargs["system_instruction"] = raw_config["system_instruction"]
        if "top_p" in raw_config:
            kwargs["top_p"] = raw_config["top_p"]
        if "top_k" in raw_config:
            kwargs["top_k"] = raw_config["top_k"]
        if "max_output_tokens" in raw_config:
            kwargs["max_output_tokens"] = raw_config["max_output_tokens"]

        if "thinking_config" in raw_config:
            tc = raw_config["thinking_config"]
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=tc.get("thinking_level", "THINKING_LEVEL_MEDIUM"),
                include_thoughts=tc.get("include_thoughts", True),
            )

        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _serialize_response(response) -> Dict[str, Any]:
        """Convert Gemini response into a JSON-serializable dict."""
        result: Dict[str, Any] = {"candidates": []}

        for candidate in response.candidates or []:
            cand_dict: Dict[str, Any] = {}

            if candidate.finish_reason:
                cand_dict["finish_reason"] = str(candidate.finish_reason)

            if candidate.content:
                parts_list = []
                for part in candidate.content.parts or []:
                    part_dict: Dict[str, Any] = {}
                    if part.text:
                        part_dict["text"] = part.text
                        if getattr(part, "thought", False):
                            part_dict["thought"] = True
                        thought_sig = getattr(part, "thoughtSignature", None) or getattr(
                            part, "thought_signature", None
                        )
                        if thought_sig:
                            part_dict["thought_signature"] = thought_sig
                    elif part.function_call:
                        part_dict["function_call"] = {
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args or {}),
                        }
                    elif part.function_response:
                        fr_dict: Dict[str, Any] = {
                            "name": part.function_response.name,
                            "response": dict(part.function_response.response or {}),
                        }
                        part_dict["function_response"] = fr_dict
                    parts_list.append(part_dict)

                cand_dict["content"] = {
                    "role": candidate.content.role,
                    "parts": parts_list,
                }

            result["candidates"].append(cand_dict)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            result["usage_metadata"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
                "cached_content_token_count": getattr(usage, "cached_content_token_count", 0),
                "thoughts_token_count": getattr(usage, "thoughts_token_count", 0),
            }

        return result


if __name__ == "__main__":
    GeminiModelServer.run_webserver()
