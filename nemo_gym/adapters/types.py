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
"""Core types and ABCs for the adapter interceptor pipeline."""

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class InterceptorContext:
    """Per-request state shared across interceptors via ContextVar."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    extra: dict[str, Any] = field(default_factory=dict)


_current_context: ContextVar[InterceptorContext] = ContextVar("adapter_ctx")


def get_context() -> InterceptorContext:
    try:
        return _current_context.get()
    except LookupError:
        ctx = InterceptorContext()
        _current_context.set(ctx)
        return ctx


def set_context(ctx: InterceptorContext) -> None:
    _current_context.set(ctx)


@dataclass
class AdapterRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    ctx: InterceptorContext = field(default_factory=get_context)


@dataclass
class AdapterResponse:
    # ``headers`` is a list of byte-tuples (Starlette's ``raw_headers`` shape)
    # so multi-valued headers like ``Set-Cookie`` survive. A plain ``dict`` is
    # also accepted for convenience.
    status_code: int
    headers: list[tuple[bytes, bytes]] | dict[str, str]
    body: dict[str, Any] | bytes
    latency_ms: float = 0.0
    ctx: InterceptorContext = field(default_factory=get_context)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


class Stage(enum.Enum):
    REQUEST = "request"
    REQUEST_TO_RESPONSE = "request_to_response"
    RESPONSE = "response"


class RequestInterceptor(ABC):
    stage: Stage = Stage.REQUEST
    stream_safe: bool = True
    best_effort: bool = False

    @abstractmethod
    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest: ...


class RequestToResponseInterceptor(ABC):
    """Request-phase interceptor that may short-circuit by returning a response."""

    stage: Stage = Stage.REQUEST_TO_RESPONSE
    stream_safe: bool = True
    best_effort: bool = False

    @abstractmethod
    async def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse: ...


class ResponseInterceptor(ABC):
    stage: Stage = Stage.RESPONSE
    stream_safe: bool = True
    best_effort: bool = False

    @abstractmethod
    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse: ...


class GracefulError(Exception):
    """Terminate the request with a 429 (e.g. session budget exhausted)."""


class InterceptorSpec(BaseModel):
    """One entry in an ``adapters`` config list."""

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class AdapterProxyConfig(BaseModel):
    """Configuration for a localhost adapter proxy in front of an external upstream.

    Used by agents that bring their own inference (e.g. ``claude_code_agent``
    with ``anthropic_base_url``). The agent server starts the proxy alongside
    itself; the agent's SDK ``*_BASE_URL`` is rewritten to the proxy's URL so
    all outbound model traffic flows through the adapter chain.
    """

    upstream_url: str
    adapters: list[InterceptorSpec] = Field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 0
    request_timeout: float = 120.0
    unsafe_allow_remote: bool = False
