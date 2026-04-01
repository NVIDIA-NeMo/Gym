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
import logging
import re
from typing import Any, Dict, Optional

from pydantic import Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


logger = logging.getLogger(__name__)

_SENSITIVE_HEADER_RE = re.compile(r"('Authorization': ')[^']*(')", re.IGNORECASE)
_SENSITIVE_COOKIE_RE = re.compile(r"('(?:Set-)?Cookie': ')[^']*(')", re.IGNORECASE)


def _sanitize_error(e: Exception) -> str:
    """Strip sensitive headers (API keys, cookies) from error repr for safe logging."""
    msg = repr(e)
    msg = _SENSITIVE_HEADER_RE.sub(r"\1[REDACTED]\2", msg)
    msg = _SENSITIVE_COOKIE_RE.sub(r"\1[REDACTED]\2", msg)
    return msg


def _normalize_to_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a chat.completion response to the Responses API format.

    Some LiteLLM proxies downgrade /v1/responses calls to chat completions
    internally and return object='chat.completion' instead of 'response'.
    """
    # Fix fields that cause validation errors even in native response format.
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") == "none":
        reasoning["effort"] = None

    if data.get("object") not in ("chat.completion",):
        return data

    logger.info("Normalizing chat.completion response to Responses API format")

    # LiteLLM proxies may return a hybrid format: object="chat.completion" but
    # content in either choices[] (standard) or output[] (Responses API style).
    text = ""

    # Try output[] first (LiteLLM hybrid format)
    for item in data.get("output", []):
        if isinstance(item, dict):
            for block in item.get("content", []):
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text = block.get("text", "") or ""
                    if text:
                        break
        if text:
            break

    # Fall back to choices[] (standard chat completion format)
    if not text:
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            text = msg.get("content", "") or ""
            if text:
                break

    # Build a Responses API shaped dict
    usage = data.get("usage", {}) or {}
    return {
        "id": data.get("id", ""),
        "created_at": data.get("created", 0),
        "model": data.get("model", ""),
        "object": "response",
        "output": [
            {
                "id": f"msg_{data.get('id', '')[-16:]}",
                "content": [
                    {"annotations": [], "text": text, "type": "output_text", "logprobs": None}
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_organization: Optional[str] = None

    extra_body: Dict[str, Any] = Field(default_factory=dict)


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            organization=self.config.openai_organization,
        )

        return super().model_post_init(context)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        try:
            openai_response_dict = await self._client.create_response(**body_dict)
        except Exception as e:
            logger.error("OpenAI API call failed: %s", _sanitize_error(e))
            raise
        try:
            # Some LiteLLM proxies return chat.completion format instead of response format.
            # Normalize to the expected Responses API shape so downstream validation succeeds.
            openai_response_dict = _normalize_to_response(openai_response_dict)
            return NeMoGymResponse.model_validate(openai_response_dict)
        except Exception as e:
            logger.error("NeMoGymResponse validation failed: %s", repr(e))
            raise

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        openai_response_dict = await self._client.create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
