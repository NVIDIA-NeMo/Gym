# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto Ontology's model calls, sent through Gym with the rollout and the activation that made them.

Auto Ontology keeps one model client for each role for the whole process, and calls it from threads, so a client
cannot hold one episode's URL. Instead its clients get ``ModelCallTransport``. Before an activation steps Auto
Ontology, it sets its ``ModelCallRoute`` in a context variable; ``asyncio.to_thread`` and LangChain's executors copy
the variable into Auto Ontology's threads. The transport reads the route when a call is sent, and sends the call
through ``ServerClient`` on the agent's event loop. So each call takes its rollout's route on Gym's aiohttp client, and
httpx opens no connection.
"""

import asyncio
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

import httpx
from openai.types import CompletionUsage

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseUsage
from nemo_gym.models import chat_completion_usage_to_response_usage


LOG = logging.getLogger(__name__)


@dataclass
class ModelCallRoute:
    """Where one activation's model calls go, and the usage they report.

    ``path_prefix`` is the activation's rollout route: ``/ng-rollout/<capture key>``, with the training-capture segment
    when training capture is on, or empty when the activation came without one. ``invocation_id`` goes in the
    ``x-session-id`` header, which Gym uses to give each captured call to its invocation.
    """

    path_prefix: str
    invocation_id: str
    # One entry for each call: its usage, or None when its reply had none.
    usages: list[Optional[NeMoGymResponseUsage]] = field(default_factory=list)

    def usage(self) -> Optional[NeMoGymResponseUsage]:
        """The total usage of the calls; ``None`` unless each call reported its usage."""
        if any(usage is None for usage in self.usages):
            return None
        return NeMoGymResponseUsage.sum_from_list(self.usages)


_ROUTE: ContextVar[Optional[ModelCallRoute]] = ContextVar("auto_ontology_model_call_route", default=None)


@contextmanager
def model_call_route(route: ModelCallRoute) -> Iterator[None]:
    """Send the model calls made in this context, and in the threads that copy it, on ``route``."""
    token = _ROUTE.set(route)
    try:
        yield
    finally:
        _ROUTE.reset(token)


class ModelCallTransport(httpx.BaseTransport):
    """An httpx transport that sends Auto Ontology's model calls through ``ServerClient`` on ``loop``.

    ``model_servers`` maps the base URL that Auto Ontology's settings give each Model Server to the server's name. A
    call made outside any route, for example from a thread pool that does not copy the context, still reaches its Model
    Server, but without a rollout; ``unrouted_calls`` counts them.
    """

    def __init__(
        self, server_client: ServerClient, model_servers: Mapping[str, str], *, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._server_client = server_client
        self._model_servers = {(httpx.URL(url).host, httpx.URL(url).port): name for url, name in model_servers.items()}
        self._loop = loop
        self.unrouted_calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            raise RuntimeError("Auto Ontology's model calls must run in a thread, not on the agent's event loop")
        server_name = self._model_servers.get((request.url.host, request.url.port))
        if server_name is None:
            raise httpx.ConnectError(f"{request.url} is not one of Auto Ontology's Model Servers", request=request)
        timeout = max((value for value in (request.extensions.get("timeout") or {}).values() if value), default=None)
        future = asyncio.run_coroutine_threadsafe(self._send(request, server_name, _ROUTE.get()), self._loop)
        try:
            status, content_type, content = future.result(timeout)
        except TimeoutError as error:
            future.cancel()
            raise httpx.ReadTimeout(f"{server_name} did not reply within {timeout} s", request=request) from error
        return httpx.Response(status, headers={"content-type": content_type}, content=content, request=request)

    async def _send(
        self, request: httpx.Request, server_name: str, route: Optional[ModelCallRoute]
    ) -> tuple[int, str, bytes]:
        if route is None:
            self.unrouted_calls += 1
            LOG.warning(
                "An Auto Ontology model call to %s ran outside an activation, so it has no rollout", server_name
            )
        response = await self._server_client.post(
            server_name=server_name,
            url_path=f"{route.path_prefix if route else ''}{request.url.raw_path.decode('ascii')}",
            json=json.loads(request.read()),
            headers={"x-session-id": route.invocation_id} if route else {},
        )
        content = await response.read()
        if route is not None:
            route.usages.append(_usage(response.status, content))
        return response.status, response.headers.get("Content-Type", "application/json"), content


def _usage(status: int, content: bytes) -> Optional[NeMoGymResponseUsage]:
    """A chat completion's usage, or ``None`` when its reply has none."""
    if status >= 400:
        return None
    try:
        return chat_completion_usage_to_response_usage(CompletionUsage.model_validate(json.loads(content)["usage"]))
    except (KeyError, TypeError, ValueError):
        return None
