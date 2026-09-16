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
import logging
import random
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Awaitable, Callable, Dict, Literal, Optional, TypeVar

from aiohttp import ClientResponseError
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

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
    ReasoningEffort,
)


LOG = logging.getLogger(__name__)
ResponseT = TypeVar("ResponseT")


class UpstreamRetryPolicy(BaseModel):
    """Retry policy for one logical request to the upstream provider."""

    max_attempts: int = Field(
        default=1,
        ge=1,
        description="Total provider attempts, including the first attempt.",
    )
    pre_request_jitter_seconds: Optional[tuple[float, float]] = Field(
        default=None,
        description=("Optional uniform sleep range sampled before every provider attempt, including the first."),
    )
    backoff_initial_seconds: float = Field(
        default=1.0,
        ge=0,
        description="Base delay after the first failed attempt.",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        gt=0,
        description="Multiplier applied to the base delay after each failure.",
    )
    backoff_jitter_fraction: float = Field(
        default=0.0,
        ge=0,
        description=("Uniform additive jitter from zero through this fraction of the current base delay."),
    )
    terminal_http_status_codes: frozenset[int] = Field(
        default_factory=frozenset,
        description="Upstream HTTP statuses that must fail without retrying.",
    )

    @model_validator(mode="after")
    def validate_pre_request_jitter(self) -> "UpstreamRetryPolicy":
        if self.pre_request_jitter_seconds is None:
            return self

        lower, upper = self.pre_request_jitter_seconds
        if lower < 0 or upper < lower:
            raise ValueError("pre_request_jitter_seconds must be a nonnegative, ordered two-value range")
        return self


class SimpleModelServerConfig(BaseResponsesAPIModelConfig):
    openai_base_url: str
    openai_api_key: str
    openai_model: str

    extra_body: Dict[str, Any] = Field(default_factory=dict)
    openai_default_headers: Dict[str, str] = Field(default_factory=dict)

    reasoning_effort_none_replacement: Optional[ReasoningEffort] = Field(
        default="minimal",
        description=(
            "Some providers (e.g. NVIDIA's Responses-compatible endpoint) return "
            "reasoning.effort: 'none'. For compatibility with callers that expect "
            "a different effort, 'none' is rewritten to this value. Set to null "
            "to preserve the provider's value, including for exact provenance validation."
        ),
    )

    max_concurrent_requests: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cap on in-flight upstream requests from this server (per-process "
            "asyncio.Semaphore). Set on rate-limited endpoints (e.g. Gemini) "
            "to stay under quota; None = unlimited."
        ),
    )
    upstream_max_num_tries: Optional[Literal[1]] = Field(
        default=None,
        description=(
            "Set to 1 to disable the outbound client's inner transport and "
            "HTTP-status retry layers. None preserves existing behavior."
        ),
    )
    upstream_request_timeout_seconds: Optional[float] = Field(default=None, gt=0)
    upstream_connect_timeout_seconds: Optional[float] = Field(default=None, gt=0)
    upstream_pool_timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Optional timeout for acquiring this model server's upstream "
            "concurrency slot. This models a provider client's pool-acquire "
            "timeout without timing or abandoning the internal HTTP hop."
        ),
    )
    propagate_upstream_http_status_codes: frozenset[int] = Field(
        default_factory=frozenset,
        description=(
            "Upstream HTTP error statuses to preserve across this model-server "
            "hop. Empty keeps the historical generic exception behavior."
        ),
    )
    upstream_retry_policy: UpstreamRetryPolicy = Field(
        default_factory=UpstreamRetryPolicy,
        description=(
            "Optional model-server retry layer around provider calls. The "
            "default policy makes one call with no added delay. Each retry "
            "attempt reacquires the upstream concurrency slot."
        ),
    )

    @model_validator(mode="after")
    def validate_upstream_controls(self) -> "SimpleModelServerConfig":
        if self.upstream_retry_policy.max_attempts > 1 and self.upstream_max_num_tries != 1:
            raise ValueError(
                "An upstream retry policy with multiple attempts requires "
                "upstream_max_num_tries=1 so no inner retry layer can add "
                "provider attempts"
            )
        if self.upstream_pool_timeout_seconds is not None and self.max_concurrent_requests is None:
            raise ValueError("upstream_pool_timeout_seconds requires max_concurrent_requests")
        if self.upstream_connect_timeout_seconds is not None and self.upstream_request_timeout_seconds is None:
            raise ValueError("upstream_connect_timeout_seconds requires upstream_request_timeout_seconds")
        invalid_status_codes = sorted(
            status for status in self.propagate_upstream_http_status_codes if status < 400 or status > 599
        )
        if invalid_status_codes:
            raise ValueError(
                "propagate_upstream_http_status_codes must contain HTTP error "
                f"statuses in [400, 599], received {invalid_status_codes}"
            )
        return self

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
        self._client = NeMoGymAsyncOpenAI(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
            default_headers=self.config.openai_default_headers,
            max_num_tries=self.config.upstream_max_num_tries,
            request_timeout_seconds=self.config.upstream_request_timeout_seconds,
            connect_timeout_seconds=self.config.upstream_connect_timeout_seconds,
        )
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )

        return super().model_post_init(context)

    @asynccontextmanager
    async def _upstream_request_slot(self):
        """Acquire one provider slot without wrapping the provider operation.

        A timeout on the resource-server -> model-server HTTP request cannot
        stand in for a provider pool timeout: client disconnect does not
        cancel the server handler, so the provider call can continue as a
        ghost. Time only semaphore acquisition here and always release an
        acquired slot in the serving process.
        """

        if self.config.max_concurrent_requests is None or self.config.upstream_pool_timeout_seconds is None:
            async with self._semaphore:
                yield
            return

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.upstream_pool_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "Timed out acquiring an upstream provider slot after "
                f"{self.config.upstream_pool_timeout_seconds:g} seconds"
            ) from exc

        try:
            yield
        finally:
            self._semaphore.release()

    async def _call_upstream(
        self,
        operation: Callable[[], Awaitable[ResponseT]],
    ) -> ResponseT:
        retry_policy = self.config.upstream_retry_policy
        if retry_policy.max_attempts == 1:
            # Preserve the ordinary model server's historical behavior: one
            # request owns its concurrency slot until its one client call has
            # returned (including any retries performed inside that client).
            if retry_policy.pre_request_jitter_seconds is not None:
                await asyncio.sleep(random.uniform(*retry_policy.pre_request_jitter_seconds))
            async with self._upstream_request_slot():
                return await operation()

        last_error: Exception | None = None
        for attempt in range(retry_policy.max_attempts):
            if retry_policy.pre_request_jitter_seconds is not None:
                await asyncio.sleep(random.uniform(*retry_policy.pre_request_jitter_seconds))
            try:
                # Acquire per attempt so a failure releases its provider slot
                # before retry backoff instead of monopolizing capacity for
                # the full retry loop.
                async with self._upstream_request_slot():
                    return await operation()
            except ClientResponseError as exc:
                if exc.status in retry_policy.terminal_http_status_codes:
                    raise
                last_error = exc
            except Exception as exc:
                last_error = exc

            if attempt + 1 >= retry_policy.max_attempts:
                break

            base_backoff = retry_policy.backoff_initial_seconds * retry_policy.backoff_multiplier**attempt
            retry_wait = base_backoff + random.uniform(
                0.0,
                base_backoff * retry_policy.backoff_jitter_fraction,
            )
            LOG.warning(
                "Upstream provider request failed (%d/%d); retrying after %.1f seconds: %s",
                attempt + 1,
                retry_policy.max_attempts,
                retry_wait,
                last_error,
            )
            await asyncio.sleep(retry_wait)

        raise RuntimeError(
            f"Upstream provider request failed after {retry_policy.max_attempts} attempts"
        ) from last_error

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model
        if self.config.drop_input_reasoning_items:
            input_items = body_dict.get("input")
            if isinstance(input_items, list):
                body_dict["input"] = [
                    item for item in input_items if not (isinstance(item, dict) and item.get("type") == "reasoning")
                ]

        async def create_and_validate() -> NeMoGymResponse:
            response_dict = await self._client.create_response(**body_dict)
            reasoning = response_dict.get("reasoning")
            if (
                self.config.reasoning_effort_none_replacement is not None
                and isinstance(reasoning, dict)
                and reasoning.get("effort") == "none"
            ):
                response_dict = dict(response_dict)
                response_dict["reasoning"] = {
                    **reasoning,
                    "effort": self.config.reasoning_effort_none_replacement,
                }
            return NeMoGymResponse.model_validate(response_dict)

        try:
            return await self._call_upstream(create_and_validate)
        except ClientResponseError as exc:
            if self._should_propagate_http_error(exc):
                # Preserve the provider status across the model-server hop.
                # Otherwise SimpleServer's generic exception middleware turns
                # this into HTTP 500 and an outer caller may retry a request
                # that the provider rejected immediately.
                raise HTTPException(
                    status_code=exc.status,
                    detail=f"Upstream provider request failed with HTTP {exc.status}",
                ) from exc
            raise

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.openai_model

        async def create_and_validate() -> NeMoGymChatCompletion:
            response_dict = await self._client.create_chat_completion(**body_dict)
            return NeMoGymChatCompletion.model_validate(response_dict)

        try:
            return await self._call_upstream(create_and_validate)
        except ClientResponseError as exc:
            if self._should_propagate_http_error(exc):
                raise HTTPException(
                    status_code=exc.status,
                    detail=f"Upstream provider request failed with HTTP {exc.status}",
                ) from exc
            raise

    def _should_propagate_http_error(self, exc: ClientResponseError) -> bool:
        return exc.status in self.config.propagate_upstream_http_status_codes


if __name__ == "__main__":
    SimpleModelServer.run_webserver()
