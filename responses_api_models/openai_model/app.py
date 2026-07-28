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
import asyncio
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

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


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: Union[str, List[str]]
    openai_api_key: str
    openai_model: str

    extra_body: Dict[str, Any] = Field(default_factory=dict)
    openai_default_headers: Dict[str, str] = Field(default_factory=dict)

    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Cap on in-flight upstream requests from this server (per-process "
            "asyncio.Semaphore). Set on rate-limited endpoints (e.g. Gemini) "
            "to stay under quota; None = unlimited."
        ),
    )


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        base_urls = (
            [self.config.openai_base_url]
            if isinstance(self.config.openai_base_url, str)
            else self.config.openai_base_url
        )
        if not base_urls:
            raise ValueError("openai_base_url must contain at least one endpoint")
        self._clients = [
            NeMoGymAsyncOpenAI(
                base_url=base_url,
                api_key=self.config.openai_api_key,
                default_headers=self.config.openai_default_headers,
            )
            for base_url in base_urls
        ]
        # Keep the original attribute for backwards compatibility with tests
        # and callers that replace a single upstream client.
        self._client = self._clients[0]
        self._next_client_idx = 0
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )

        return super().model_post_init(context)

    def _resolve_client(self) -> NeMoGymAsyncOpenAI:
        if len(self._clients) == 1:
            return self._client
        client = self._clients[self._next_client_idx]
        self._next_client_idx = (self._next_client_idx + 1) % len(self._clients)
        return client

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        client = self._resolve_client()
        async with self._semaphore:
            openai_response_dict = await client.create_response(**body_dict)
        return NeMoGymResponse.model_validate(openai_response_dict)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        client = self._resolve_client()
        async with self._semaphore:
            openai_response_dict = await client.create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
