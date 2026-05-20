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
"""Core types and ABCs for the Gym adapter interceptor pipeline."""

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# InterceptorContext — per-request cross-cutting state via ContextVar
# ---------------------------------------------------------------------------


@dataclass
class InterceptorContext:
    """Per-request context shared across all interceptors via ContextVar.

    Provides trace/request id and timing without mutable-bag passing.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    extra: dict[str, Any] = field(default_factory=dict)


_current_context: ContextVar[InterceptorContext] = ContextVar("adapter_ctx")


def get_context() -> InterceptorContext:
    """Return the current request's InterceptorContext (or create one)."""
    try:
        return _current_context.get()
    except LookupError:
        ctx = InterceptorContext()
        _current_context.set(ctx)
        return ctx


def set_context(ctx: InterceptorContext) -> None:
    _current_context.set(ctx)


# ---------------------------------------------------------------------------
# Request / Response data objects
# ---------------------------------------------------------------------------


@dataclass
class AdapterRequest:
    """An interceptable HTTP request flowing through the pipeline."""

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any]
    ctx: InterceptorContext = field(default_factory=get_context)


@dataclass
class AdapterResponse:
    """An interceptable HTTP response flowing back through the pipeline.

    ``headers`` is a list of ``(name, value)`` byte tuples — the same shape
    Starlette uses on ``Response.raw_headers``. The list form is load-bearing:
    a Python ``dict`` silently collapses duplicate keys, which breaks any
    multi-valued header (most importantly ``Set-Cookie``). The pre-fix shape
    was ``dict[str, str]``; that collapsed a response with two ``Set-Cookie``
    headers down to one before it reached the client. For backward compat
    with test code that constructs ``AdapterResponse(headers={...})``, a
    ``dict[str, str]`` is also accepted — the middleware re-emit path
    handles both.
    """

    status_code: int
    headers: list[tuple[bytes, bytes]] | dict[str, str]
    body: dict[str, Any] | bytes
    latency_ms: float = 0.0
    ctx: InterceptorContext = field(default_factory=get_context)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


# ---------------------------------------------------------------------------
# Interceptor stage enum
# ---------------------------------------------------------------------------


class Stage(enum.Enum):
    REQUEST = "request"
    REQUEST_TO_RESPONSE = "request_to_response"
    RESPONSE = "response"


# ---------------------------------------------------------------------------
# Interceptor ABCs
# ---------------------------------------------------------------------------


class RequestInterceptor(ABC):
    """Runs during the request phase — may modify the request but cannot
    produce a response (use ``RequestToResponseInterceptor`` for that)."""

    stage: Stage = Stage.REQUEST
    stream_safe: bool = True
    best_effort: bool = False

    @abstractmethod
    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest: ...


class RequestToResponseInterceptor(ABC):
    """Runs during the request phase and may either pass the request through
    or short-circuit by returning an ``AdapterResponse`` directly."""

    stage: Stage = Stage.REQUEST_TO_RESPONSE
    stream_safe: bool = True
    best_effort: bool = False

    @abstractmethod
    async def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse: ...


class ResponseInterceptor(ABC):
    """Runs during the response phase (reverse order). May modify the
    response but cannot re-issue the upstream call."""

    stage: Stage = Stage.RESPONSE
    stream_safe: bool = True
    best_effort: bool = False

    @abstractmethod
    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse: ...


class PostEvalHook(ABC):
    """Optional hook invoked once after the entire evaluation completes."""

    @abstractmethod
    async def post_eval(self) -> None: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GracefulError(Exception):
    """Signal that the pipeline should terminate the current request with a
    429 (session budget exhausted) rather than a generic 500.

    Ported from NEL's ``nemo_evaluator.errors.GracefulError`` so middleware
    can preserve the session-budget-exhausted behavior that clients rely on
    without forcing Gym to depend on the NEL package.
    """
