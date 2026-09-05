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
from urllib.parse import quote

from fastapi import Request
from pydantic import model_validator

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
from nemo_gym.responses_converter import VLLMConverter


class AzureOpenAIModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    default_query: dict
    num_concurrent_requests: int

    @model_validator(mode="after")
    def _require_api_version(self) -> "AzureOpenAIModelServerConfig":
        if not self.default_query.get("api-version"):
            raise ValueError("default_query must include a non-empty 'api-version'")
        if self.num_concurrent_requests < 1:
            raise ValueError("num_concurrent_requests must be at least 1")
        return self


class AzureOpenAIModelServer(SimpleResponsesAPIModel):
    config: AzureOpenAIModelServerConfig

    def model_post_init(self, context):
        endpoint = self.config.openai_base_url.rstrip("/")
        deployment = quote(self.config.openai_model, safe="")
        self._client = NeMoGymAsyncOpenAI(
            base_url=f"{endpoint}/openai/deployments/{deployment}",
            api_key=self.config.openai_api_key,
            default_query=self.config.default_query,
            auth_header_name="api-key",
            auth_header_prefix="",
        )
        self._converter = VLLMConverter(return_token_id_information=False)
        self._semaphore: Semaphore = Semaphore(self.config.num_concurrent_requests)
        return super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        async with self._semaphore:
            chat_completion_create_params = self._converter.responses_to_chat_completion_create_params(body)
            chat_completion_params_dict = chat_completion_create_params.model_dump(exclude_unset=True)
            chat_completion_params_dict["model"] = self.config.openai_model
            response_dict = await self._client.create_chat_completion(**chat_completion_params_dict)
            chat_completion_response = NeMoGymChatCompletion.model_validate(response_dict)

        return self._converter.chat_completion_to_response(
            responses_create_params=body.model_copy(update={"model": self.config.openai_model}),
            chat_completion=chat_completion_response,
        )

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        async with self._semaphore:
            openai_response_dict = await self._client.create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    AzureOpenAIModelServer.run_webserver()
