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

import json
from asyncio import Semaphore
from time import time
from typing import Any, Dict
from uuid import uuid4

from aiohttp.client_exceptions import ClientResponseError
from fastapi import HTTPException, Request
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
    _RETRYABLE_PROVIDER_STATUSES = {429, 500, 502, 503, 504, 520}

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
            try:
                chat_completion_dict = await self._client.create_chat_completion(**body_dict)
            except ClientResponseError as e:
                normalized_payload = self._build_provider_error_payload(e)
                raise HTTPException(
                    status_code=normalized_payload["provider_status"], detail=normalized_payload
                ) from e

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

    def _build_provider_error_payload(self, error: ClientResponseError) -> Dict[str, Any]:
        provider_status = error.status if error.status else 500
        message = self._extract_provider_error_message(error)
        category = self._classify_provider_error(provider_status, message)
        return {
            "provider_status": provider_status,
            "retryable": provider_status in self._RETRYABLE_PROVIDER_STATUSES,
            "provider_context": {"base_url": self.config.base_url},
            "model": self.config.model,
            "category": category,
            "message": message,
        }

    def _classify_provider_error(self, status: int, message: str) -> str:
        message_lower = message.lower()
        if status in {401, 403} or "api key" in message_lower or "auth" in message_lower:
            return "authentication"
        if status == 404 or ("model" in message_lower and "not found" in message_lower):
            return "model_not_found"
        if status == 429 or "rate limit" in message_lower:
            return "rate_limit"
        if status in {400, 422}:
            return "request_error"
        if status in self._RETRYABLE_PROVIDER_STATUSES:
            return "transient_upstream_failure"
        return "provider_error"

    def _extract_provider_error_message(self, error: ClientResponseError) -> str:
        response_content = getattr(error, "response_content", b"")
        if isinstance(response_content, bytes):
            response_text = response_content.decode("utf-8", errors="replace").strip()
        elif response_content:
            response_text = str(response_content).strip()
        else:
            response_text = str(error)

        parsed_message = response_text
        if response_text:
            parsed_message = self._extract_error_message_from_response(response_text)

        if parsed_message:
            return self._concise(parsed_message)
        return "Provider request failed"

    @staticmethod
    def _extract_error_message_from_response(response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text

        if isinstance(payload, dict):
            if isinstance(payload.get("error"), dict):
                nested_error = payload["error"]
                if nested_error.get("message"):
                    return str(nested_error["message"])

            if payload.get("message"):
                return str(payload["message"])

            if payload.get("detail"):
                return str(payload["detail"])

        return response_text

    @staticmethod
    def _concise(message: str) -> str:
        compact = " ".join(message.strip().split())
        return compact if len(compact) <= 200 else compact[:197] + "..."


if __name__ == "__main__":
    InferenceProvider.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = InferenceProvider.run_webserver()  # noqa: F401
