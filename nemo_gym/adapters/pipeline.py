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
"""Async adapter pipeline — ordered interceptor chain execution."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    InterceptorContext,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
    Stage,
    set_context,
)


UpstreamCall = Callable[[AdapterRequest], Awaitable[AdapterResponse]]

logger = logging.getLogger(__name__)


def _stage_of(
    interceptor: RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor,
) -> Stage:
    return getattr(interceptor, "stage", Stage.REQUEST)


class AdapterPipeline:
    """Async interceptor chain: REQUEST → REQUEST_TO_RESPONSE → RESPONSE."""

    def __init__(
        self,
        interceptors: list[RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor],
    ) -> None:
        self._chain = list(interceptors)
        self._validate_order()
        logger.info(
            "AdapterPipeline ready (%d interceptors: %s)",
            len(self._chain),
            [getattr(i, "_registry_name", type(i).__name__) for i in self._chain],
        )

    @classmethod
    def from_config(
        cls,
        interceptor_specs: list[dict[str, Any]],
    ) -> AdapterPipeline:
        chain: list[RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor] = []
        for spec in interceptor_specs:
            name = spec["name"]
            config = spec.get("config") or {}
            chain.append(InterceptorRegistry.create(name, config))
        return cls(chain)

    _STAGE_ORDER = [Stage.REQUEST, Stage.REQUEST_TO_RESPONSE, Stage.RESPONSE]

    def _validate_order(self) -> None:
        current_idx = 0
        for interceptor in self._chain:
            stage = _stage_of(interceptor)
            try:
                idx = self._STAGE_ORDER.index(stage)
            except ValueError:
                raise ValueError(f"Unknown stage {stage!r} on {type(interceptor).__name__}")
            if idx < current_idx:
                raise ValueError(
                    f"Invalid interceptor order: {type(interceptor).__name__} "
                    f"(stage={stage.value}) appears after stage "
                    f"{self._STAGE_ORDER[current_idx].value}. "
                    f"Required order: request → request_to_response → response"
                )
            current_idx = max(current_idx, idx)

    async def process(
        self,
        request: AdapterRequest,
        upstream_call: UpstreamCall | None = None,
    ) -> AdapterResponse:
        """Run *request* through the chain and return the response.

        If no ``RequestToResponseInterceptor`` short-circuits, ``upstream_call``
        is invoked with the (possibly mutated) request. Response interceptors
        then run in reverse order.
        """
        ctx = InterceptorContext(request_id=request.ctx.request_id)
        set_context(ctx)

        current: AdapterRequest | AdapterResponse = request

        for interceptor in self._chain:
            if isinstance(current, AdapterResponse):
                break

            if isinstance(interceptor, (RequestInterceptor, RequestToResponseInterceptor)):
                try:
                    result = await interceptor.intercept_request(current)  # type: ignore[arg-type]
                    current = result
                except Exception:
                    if getattr(interceptor, "best_effort", False):
                        logger.warning(
                            "Interceptor %s failed (best_effort=True), continuing",
                            type(interceptor).__name__,
                            exc_info=True,
                        )
                        continue
                    raise

        if not isinstance(current, AdapterResponse):
            if upstream_call is None:
                raise RuntimeError("No interceptor produced a response. Is 'endpoint' in the chain?")
            current = await upstream_call(current)

        response = current
        response_interceptors = [ic for ic in reversed(self._chain) if isinstance(ic, ResponseInterceptor)]
        for interceptor in response_interceptors:
            try:
                response = await interceptor.intercept_response(response)
            except Exception:
                if getattr(interceptor, "best_effort", False):
                    logger.warning(
                        "Response interceptor %s failed (best_effort=True), continuing",
                        type(interceptor).__name__,
                        exc_info=True,
                    )
                    continue
                raise

        return response
