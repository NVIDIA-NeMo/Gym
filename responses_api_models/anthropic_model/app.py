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
from contextlib import nullcontext
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from pydantic import Field

from nemo_gym.anthropic_converter import (
    SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES,
    AnthropicConverter,
)
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.server_utils import request as aiohttp_request


# Re-exported for backwards-compatible imports; the converter now lives in nemo_gym core so it
# can be shared with the inverse-direction (ingress) Anthropic Messages proxy.
__all__ = ["AnthropicModel", "AnthropicModelConfig", "AnthropicConverter", "SUPPORTED_ANTHROPIC_IMAGE_MEDIA_TYPES"]


class AnthropicModelConfig(BaseResponsesAPIModelConfig):
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_api_key: str
    anthropic_model: str
    max_tokens: int
    anthropic_version: str = "2023-06-01"
    thinking: Optional[Dict[str, Any]] = None
    thinking_budget_tokens: Optional[int] = None
    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Cap on in-flight upstream requests from this server (per-process asyncio.Semaphore). None = unlimited."
        ),
    )
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class AnthropicModel(SimpleResponsesAPIModel):
    config: AnthropicModelConfig

    def model_post_init(self, context):
        self._converter = AnthropicConverter()
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )
        return super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        try:
            anthropic_body = self._converter.responses_to_anthropic(
                body=body,
                model=self.config.anthropic_model,
                max_tokens=self.config.max_tokens,
                thinking=self.config.thinking,
                thinking_budget_tokens=self.config.thinking_budget_tokens,
                extra_body=self.config.extra_body,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        raise NotImplementedError("anthropic_model supports /v1/responses only")

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


if __name__ == "__main__":
    AnthropicModel.run_webserver()
