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
from uuid import uuid4

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
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
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

        chat_completion_response = await self.chat_completions(request, chat_completion_create_params)

        choice = chat_completion_response.choices[0]
        response_output = self._converter.postprocess_chat_response(choice)
        response_output_dicts = [item.model_dump() for item in response_output]

        usage = None
        if chat_completion_response.usage:
            usage = NeMoGymResponseUsage(
                input_tokens=chat_completion_response.usage.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=chat_completion_response.usage.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_completion_response.usage.prompt_tokens
                + chat_completion_response.usage.completion_tokens,
            )

        incomplete_details = None
        if choice.finish_reason == "length":
            incomplete_details = {"reason": "max_output_tokens"}
        elif choice.finish_reason == "content_filter":
            incomplete_details = {"reason": "content_filter"}

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model,
            object="response",
            output=response_output_dicts,
            tool_choice=body.tool_choice if body.tool_choice is not None else "auto",
            parallel_tool_calls=body.parallel_tool_calls,
            tools=body.tools,
            temperature=body.temperature,
            top_p=body.top_p,
            background=body.background,
            max_output_tokens=body.max_output_tokens,
            max_tool_calls=body.max_tool_calls,
            previous_response_id=body.previous_response_id,
            prompt=body.prompt,
            reasoning=body.reasoning,
            service_tier=body.service_tier,
            text=body.text,
            top_logprobs=body.top_logprobs,
            truncation=body.truncation,
            metadata=body.metadata,
            instructions=body.instructions,
            user=body.user,
            incomplete_details=incomplete_details,
            usage=usage,
        )

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
