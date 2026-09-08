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
"""Server for any hosted inference provider that exposes an OpenAI-compatible /v1/chat/completions endpoint.

Supports: Fireworks, Together.ai, Baseten, DeepInfra, Nebius, Friendli,
OpenRouter, HF Inference, Gemini and any other OpenAI-compatible provider.

For training workloads that require token IDs, use vllm_model instead.
"""

from asyncio import Semaphore
from time import time
from typing import Any, Dict

from fastapi import Request
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
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


class InferenceProviderConfig(BaseResponsesAPIModelConfig):
    base_url: str
    api_key: str
    model: str

    uses_reasoning_parser: bool = False
    num_concurrent_requests: int = 1000
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class InferenceProvider(SimpleResponsesAPIModel):
    config: InferenceProviderConfig

    def model_post_init(self, context):
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self._converter = ResponsesConverter(
            return_token_id_information=False,
            uses_reasoning_parser=self.config.uses_reasoning_parser,
        )
        self._semaphore = Semaphore(self.config.num_concurrent_requests)
        return super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
        body.model = self.config.model

        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        response = self._converter.chat_completion_to_response(
            responses_create_params=body,
            chat_completion=chat_completion_response,
        )
        return response.model_copy(update={"created_at": int(time())})

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.model

        if self.config.extra_body:
            body_dict = self.config.extra_body | body_dict

        if self.config.uses_reasoning_parser:
            for message_dict in body_dict.get("messages", []):
                if message_dict.get("role") != "assistant" or "content" not in message_dict:
                    continue
                content = message_dict["content"]
                if isinstance(content, str):
                    _, remaining_content = self._converter._extract_reasoning_from_content(content)
                    message_dict["content"] = remaining_content

        async with self._semaphore:
            chat_completion_dict = await self._client.create_chat_completion(**body_dict)

        choice_dict = chat_completion_dict["choices"][0]
        if self.config.uses_reasoning_parser:
            reasoning_content = choice_dict["message"].get("reasoning_content") or choice_dict["message"].get(
                "reasoning"
            )
            if reasoning_content:
                choice_dict["message"].pop("reasoning_content", None)
                choice_dict["message"].pop("reasoning", None)
                choice_dict["message"]["content"] = self._converter._wrap_reasoning_in_think_tags(
                    [reasoning_content]
                ) + (choice_dict["message"].get("content") or "")

        return NeMoGymChatCompletion.model_validate(chat_completion_dict)


if __name__ == "__main__":
    InferenceProvider.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = InferenceProvider.run_webserver()  # noqa: F401
