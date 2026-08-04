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
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

from aiohttp import ClientConnectionError, ClientError, ClientResponseError
from pydantic import Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    RETRY_ERROR_CODES,
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

    endpoint_failover_enabled: bool = Field(
        default=False,
        description=(
            "Retry a request on another configured upstream after a retryable "
            "network or 5xx failure, and temporarily quarantine the failed "
            "endpoint. Disabled by default for backwards compatibility."
        ),
    )
    endpoint_failure_cooldown_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="How long a retryably failed endpoint is skipped.",
    )
    endpoint_max_attempts: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Maximum distinct upstream endpoints tried for one request. Defaults to every configured endpoint."
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
        # Preserve the original attribute for callers/tests that replace one
        # upstream client and leave failover disabled.
        self._client = self._clients[0]
        self._base_urls = base_urls
        self._next_client_idx = 0
        self._quarantined_until = [0.0 for _ in self._clients]
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )

        return super().model_post_init(context)

    def _resolve_client_index(self, excluded: Optional[set[int]] = None) -> int:
        excluded = excluded or set()
        now = time.monotonic()
        for offset in range(len(self._clients)):
            idx = (self._next_client_idx + offset) % len(self._clients)
            if idx in excluded:
                continue
            if self.config.endpoint_failover_enabled and self._quarantined_until[idx] > now:
                continue
            self._next_client_idx = (idx + 1) % len(self._clients)
            return idx
        raise RuntimeError("No non-quarantined OpenAI upstream endpoints remain")

    def _resolve_client(self) -> NeMoGymAsyncOpenAI:
        if len(self._clients) == 1 and not self.config.endpoint_failover_enabled:
            return self._client
        return self._clients[self._resolve_client_index()]

    @staticmethod
    def _is_retryable_endpoint_error(error: Exception) -> bool:
        if isinstance(error, ClientResponseError):
            return error.status in RETRY_ERROR_CODES
        return isinstance(error, (asyncio.TimeoutError, ClientConnectionError, ClientError, OSError))

    async def _call_upstream_with_failover(
        self,
        operation: str,
        body_dict: Dict[str, Any],
        trace_id: str,
    ) -> Dict[str, Any]:
        failover_enabled = self.config.endpoint_failover_enabled
        max_attempts = (
            min(self.config.endpoint_max_attempts or len(self._clients), len(self._clients)) if failover_enabled else 1
        )
        excluded: set[int] = set()
        trace_enabled = failover_enabled or trace_id != "-"

        for attempt in range(1, max_attempts + 1):
            idx = self._resolve_client_index(excluded) if failover_enabled else 0
            client = self._clients[idx] if failover_enabled else self._client
            endpoint = self._base_urls[idx]
            if trace_enabled:
                print(
                    f"[endpoint_route trace_id={trace_id} operation={operation} "
                    f"endpoint_idx={idx} endpoint={endpoint} attempt={attempt}/{max_attempts} "
                    "event=selected]",
                    flush=True,
                )
            try:
                async with self._semaphore:
                    if operation == "responses":
                        result = await client.create_response(**body_dict)
                    else:
                        result = await client.create_chat_completion(**body_dict)
                if trace_enabled:
                    print(
                        f"[endpoint_route trace_id={trace_id} operation={operation} "
                        f"endpoint_idx={idx} endpoint={endpoint} attempt={attempt}/{max_attempts} "
                        "event=success]",
                        flush=True,
                    )
                return result
            except Exception as error:
                retryable = self._is_retryable_endpoint_error(error)
                if failover_enabled and retryable:
                    excluded.add(idx)
                    self._quarantined_until[idx] = time.monotonic() + self.config.endpoint_failure_cooldown_seconds
                    print(
                        f"[endpoint_route trace_id={trace_id} operation={operation} "
                        f"endpoint_idx={idx} endpoint={endpoint} attempt={attempt}/{max_attempts} "
                        f"event=quarantined cooldown_s={self.config.endpoint_failure_cooldown_seconds:g} "
                        f"error={type(error).__name__}:{str(error)[:240]}]",
                        flush=True,
                    )
                    if attempt < max_attempts:
                        continue
                if trace_enabled:
                    print(
                        f"[endpoint_route trace_id={trace_id} operation={operation} "
                        f"endpoint_idx={idx} endpoint={endpoint} attempt={attempt}/{max_attempts} "
                        f"event=failed retryable={str(retryable).lower()} "
                        f"error={type(error).__name__}:{str(error)[:240]}]",
                        flush=True,
                    )
                raise

        raise RuntimeError("OpenAI upstream endpoint failover exhausted")

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        if self.config.drop_input_reasoning_items:
            input_items = body_dict.get("input")
            if isinstance(input_items, list):
                body_dict["input"] = [
                    item for item in input_items if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]
        trace_id = str(body_dict.get("user") or "-")
        openai_response_dict = await self._call_upstream_with_failover("responses", body_dict, trace_id)
        return NeMoGymResponse.model_validate(openai_response_dict)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        trace_id = str(body_dict.get("user") or "-")
        openai_response_dict = await self._call_upstream_with_failover("chat_completions", body_dict, trace_id)
        return NeMoGymChatCompletion.model_validate(openai_response_dict)


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
