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


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
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

    drop_input_reasoning_items: bool = Field(
        default=False,
        description=(
            "Strip type=reasoning items from the Responses API input before the "
            "upstream call. Workaround for endpoints (e.g. NVIDIA-hosted gpt-oss) "
            "that 500 with KeyError 'content' on their own content-less reasoning "
            "items when echoed back across tool-use turns."
        ),
    )

    retry_empty_completions: int = Field(
        default=0,
        description=(
            "Retry the upstream chat completion up to N extra times when the "
            "returned message has neither content nor tool_calls (reasoning-only "
            "turns from always-thinking models, e.g. nemotron-3.5-lightning stops "
            "after emitting only reasoning_content on ~1 in 10 agentic turns; "
            "OpenHands aborts the episode on such an empty message). 0 disables."
        ),
    )


class SimpleModelServer(SimpleResponsesAPIModel):
    config: SimpleModelServerConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            default_headers=self.config.openai_default_headers,
        )
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )

        return super().model_post_init(context)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        if self.config.drop_input_reasoning_items:
            input_items = body_dict.get("input")
            if isinstance(input_items, list):
                body_dict["input"] = [
                    item for item in input_items if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]
        async with self._semaphore:
            openai_response_dict = await self._client.create_response(**body_dict)
        return NeMoGymResponse.model_validate(openai_response_dict)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        async with self._semaphore:
            openai_response_dict = await self._client.create_chat_completion(**body_dict)
            for _ in range(self.config.retry_empty_completions):
                if not self._completion_is_empty(openai_response_dict):
                    break
                openai_response_dict = await self._client.create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)

    @staticmethod
    def _completion_is_empty(response_dict: Dict[str, Any]) -> bool:
        choices = response_dict.get("choices") or []
        message = (choices[0] or {}).get("message") or {} if choices else {}
        return not message.get("content") and not message.get("tool_calls")


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
