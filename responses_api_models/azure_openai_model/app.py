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
from asyncio import Semaphore

from fastapi import Request
from openai import AsyncAzureOpenAI

from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models import BaseResponsesAPIModelConfig, Body, ChatCompletionsConverter, SimpleResponsesAPIModel


class AzureOpenAIModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    default_query: dict
    num_concurrent_requests: int


class AzureOpenAIModelServer(SimpleResponsesAPIModel):
    config: AzureOpenAIModelServerConfig

    def model_post_init(self, context):
        self._client = AsyncAzureOpenAI(
            azure_endpoint=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            api_version=self.config.default_query.get("api-version"),
        )
        self._converter = ChatCompletionsConverter(
            return_token_id_information=False,
            custom_tool_strategy=self.config.custom_tool_strategy,
        )
        self._semaphore: Semaphore = Semaphore(self.config.num_concurrent_requests)
        return super().model_post_init(context)

    def _mints_response_ids(self) -> bool:
        # responses() builds each response from a chat completion.
        return True

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        async with self._semaphore:
            chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
            chat_completion_params_dict = chat_completion_create_params.model_dump(exclude_unset=True)
            chat_completion_params_dict["model"] = self.config.openai_model
            chat_completion_response = await self._client.chat.completions.create(**chat_completion_params_dict)

        return self._converter.chat_completion_to_response(
            responses_create_params=body.model_copy(update={"model": self.config.openai_model}),
            chat_completion=NeMoGymChatCompletion.model_validate(chat_completion_response.model_dump()),
        )

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        openai_response_dict = await self._client.chat.completions.create(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    AzureOpenAIModelServer.run_webserver()
