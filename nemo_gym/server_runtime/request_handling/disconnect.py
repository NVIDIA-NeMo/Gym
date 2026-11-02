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
"""Middleware that cancels a request's handler when its client disconnects."""

import asyncio
import time
from typing import Any

from anyio import create_task_group
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.client.http_client import (
    is_global_aiohttp_client_request_debug_enabled,
)
from nemo_gym.server_runtime.request_handling.rollout_context import ROUTED_PATH_STATE_KEY
from nemo_gym.telemetry.observer import request_counts


def _cancelled_request_route(scope: Scope) -> str:
    """Return a registered route template, never a path containing request or rollout IDs."""
    route = scope.get("route")
    if route is not None:
        return getattr(route, "path_format", "<unmatched>")

    # Prefix middleware copies the scope, but shares its actual routed path in state.
    # Respect each server's prefix rules and resolve only registered templates.
    path = scope.get("state", {}).get(ROUTED_PATH_STATE_KEY, scope["path"])
    routed_scope = {**scope, "path": path}
    for candidate in getattr(scope.get("app"), "routes", ()):
        match, _ = candidate.matches(routed_scope)
        if match == Match.FULL:
            return getattr(candidate, "path_format", "<unmatched>")
    return "<unmatched>"


class ClientDisconnectCancellationMiddleware:
    """Cancel an in-flight HTTP request when its client disconnects."""

    # Direct MCP dispatch has no client connection to monitor.
    skipped_on_direct_mcp_dispatch = True

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.num_cancelled = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        counts = request_counts()
        if counts is None:
            await self._handle(scope, receive, send, None)
            return
        counts.in_flight += 1
        try:
            await self._handle(scope, receive, send, counts)
        except Exception:
            # A handled error, such as an HTTPException, became a response inside and is not counted here.
            counts.unhandled_exceptions += 1
            raise
        finally:
            counts.in_flight -= 1

    async def _handle(self, scope: Scope, receive: Receive, send: Send, counts: Any) -> None:
        received_messages: asyncio.Queue[Message] = asyncio.Queue()
        client_disconnected = asyncio.Event()
        response_complete = asyncio.Event()
        debug = is_global_aiohttp_client_request_debug_enabled()
        started_at = time.monotonic() if debug else None
        response_started = False

        async def receive_message() -> Message:
            return await received_messages.get()

        async def send_message(message: Message) -> None:
            nonlocal response_started
            if client_disconnected.is_set():
                return

            if counts is not None and message["type"] == "http.response.start":
                counts.count_response(message["status"])
            await send(message)
            if message["type"] == "http.response.start":
                response_started = True
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_complete.set()

        # The listener is the sole reader of the original ASGI receive channel.
        # Forwarding request messages keeps the body available to the app while
        # also allowing disconnects to cancel handlers that no longer call receive.
        async with create_task_group() as task_group:

            async def run_app() -> None:
                try:
                    await self.app(scope, receive_message, send_message)
                finally:
                    task_group.cancel_scope.cancel()

            async def listen_for_disconnect() -> None:
                while True:
                    message = await receive()
                    if message["type"] != "http.disconnect":
                        await received_messages.put(message)
                        continue

                    # Uvicorn returns http.disconnect from receive() once the response is complete,
                    # even if the peer did not disconnect early. Only cancel requests whose response
                    # has not finished being sent.
                    if response_complete.is_set():
                        return

                    await received_messages.put(message)
                    client_disconnected.set()
                    self.num_cancelled += 1
                    if counts is not None:
                        counts.count_cancellation(_cancelled_request_route(scope), response_started=response_started)
                    if debug or self.num_cancelled % 100 == 0:
                        client = scope.get("client")
                        client_address = f"{client[0]}:{client[1]}" if client else "-:-"
                        detail = ""
                        if started_at is not None:
                            elapsed_s = time.monotonic() - started_at
                            phase = "response_started" if response_started else "before_response_start"
                            detail = f" after {elapsed_s:.1f}s {phase}"
                        print(
                            f'{client_address} - "{scope["method"]} {scope["path"]}" '
                            f"499 CLIENT DISCONNECTED{detail} ({self.num_cancelled} total for this server worker)"
                        )
                    task_group.cancel_scope.cancel()
                    return

            task_group.start_soon(run_app)
            task_group.start_soon(listen_for_disconnect)
