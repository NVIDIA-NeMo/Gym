# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-scoped model credentials shared by native and subprocess agents."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field


class ModelEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_name: str
    base_url: str
    api_key: str = Field(repr=False)

    def environment(self) -> dict[str, str]:
        """Return standard OpenAI client settings for a child process."""
        return {"OPENAI_BASE_URL": self.base_url, "OPENAI_API_KEY": self.api_key}


_endpoint: ContextVar[ModelEndpoint | None] = ContextVar("model_endpoint", default=None)
_headers = ("x-gym-model-server", "x-gym-model-base-url", "x-gym-model-api-key")


def current_model_endpoint() -> ModelEndpoint | None:
    return _endpoint.get()


@contextmanager
def model_endpoint_context(endpoint: ModelEndpoint | None) -> Iterator[None]:
    token = _endpoint.set(endpoint)
    try:
        yield
    finally:
        _endpoint.reset(token)


def endpoint_headers(endpoint: ModelEndpoint) -> dict[str, str]:
    """Carry credentials only on internal agent-to-agent requests."""
    return dict(zip(_headers, (endpoint.server_name, endpoint.base_url, endpoint.api_key)))


class ModelEndpointMiddleware:
    """Restore context on agent self-calls without parsing the request body."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        headers = dict(scope.get("headers", ()))
        values = [headers.get(name.encode()) for name in _headers]
        endpoint = None
        if all(value is not None for value in values):
            endpoint = ModelEndpoint(
                server_name=values[0].decode(), base_url=values[1].decode(), api_key=values[2].decode()
            )
        with model_endpoint_context(endpoint):
            await self.app(scope, receive, send)
