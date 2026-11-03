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
"""Middleware that reads trace context carried in a request's URL path."""

import re
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.client.server_client import (
    _TRACEPARENT_PATTERN,
)
from nemo_gym.config.config_types import (
    ROLLOUT_PATH_PREFIX,
    TOKEN_CAPTURE_PATH_SEGMENT,
    TRACE_CONTEXT_PATH_SEGMENT,
)
from nemo_gym.server_runtime.request_handling.rollout_context import scope_with_routed_path


_ROLLOUT_PREFIX_PATTERN = rf"/{re.escape(ROLLOUT_PATH_PREFIX)}/[^/]+(?:/{re.escape(TOKEN_CAPTURE_PATH_SEGMENT)})?"


_ROLLOUT_PREFIX_RE = re.compile(rf"^{_ROLLOUT_PREFIX_PATTERN}(?=/|$)")


_TRACE_CONTEXT_PATH_RE = re.compile(
    rf"^(?P<head>(?:{_ROLLOUT_PREFIX_PATTERN})?)"
    rf"/{re.escape(TRACE_CONTEXT_PATH_SEGMENT)}/(?P<traceparent>{_TRACEPARENT_PATTERN})(?P<rest>/.*)$"
)


class TraceContextPathMiddleware:
    """Strip ``/ng-trace/<traceparent>`` from the path and supply it as the inbound ``traceparent``.

    The segment comes from ``apply_rollout_prefix``: a model URL an agent hands to a client that
    sends no ``traceparent`` header, such as a harness inside a sandbox. A header the client did
    send wins. The segment is stripped whether or not traces are on, so such a URL always routes.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        # Substring test first: this runs on every request of every server, almost none of which carry the segment.
        match = _TRACE_CONTEXT_PATH_RE.match(path) if f"/{TRACE_CONTEXT_PATH_SEGMENT}/" in path else None
        if match is None:
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers") or [])
        if not any(name.lower() == b"traceparent" for name, _ in headers):
            headers.append((b"traceparent", match.group("traceparent").encode("ascii")))
        scope = scope_with_routed_path(scope, match.group("head") + match.group("rest"))
        scope["headers"] = headers
        segment = f"/{TRACE_CONTEXT_PATH_SEGMENT}/{match.group('traceparent')}"

        async def send_with_segment_on_redirect(message: Message) -> None:
            # The client only ever had the URL, so a redirect it follows must carry the segment again.
            if message["type"] == "http.response.start":
                message = _restore_trace_segment_on_redirect(message, segment=segment, request_headers=headers)
            await send(message)

        await self.app(scope, receive, send_with_segment_on_redirect)


def _restore_trace_segment_on_redirect(
    message: Message, *, segment: str, request_headers: list[tuple[bytes, bytes]]
) -> Message:
    """Put ``/ng-trace/<traceparent>`` back into root-relative and same-origin redirect targets.

    Inserted after a rollout or token-capture prefix (which the capture middleware has already restored)
    or at the root; cross-origin targets and targets that already carry a segment are left alone.
    """
    status = int(message.get("status") or 0)
    if not 300 <= status < 400:
        return message
    request_host = next((value.decode("latin-1") for key, value in request_headers if key.lower() == b"host"), "")
    headers = list(message.get("headers") or [])
    changed = False
    for index, (key, value) in enumerate(headers):
        if key.lower() != b"location":
            continue
        parts = urlsplit(value.decode("latin-1"))
        if (parts.netloc and parts.netloc != request_host) or not parts.path.startswith("/"):
            continue
        if f"/{TRACE_CONTEXT_PATH_SEGMENT}/" in parts.path:
            continue
        prefix = _ROLLOUT_PREFIX_RE.match(parts.path)
        head = prefix.group(0) if prefix else ""
        path = f"{head}{segment}{parts.path[len(head) :]}"
        headers[index] = (key, urlunsplit(parts._replace(path=path)).encode("latin-1"))
        changed = True
    return {**message, "headers": headers} if changed else message


def install_trace_context_path(app: FastAPI) -> None:
    """Install ``TraceContextPathMiddleware`` outside the instrumentor, which reads headers and path when it
    opens the server span. Same mechanism as ``install_request_context_isolation``; call it after
    ``instrument_app_for_telemetry`` and before ``install_request_context_isolation``, which stays outermost.
    """
    if getattr(app.state, "nemo_gym_trace_context_path_installed", False):
        return
    app.state.nemo_gym_trace_context_path_installed = True
    build_inner = app.build_middleware_stack

    def build_middleware_stack() -> ASGIApp:
        return TraceContextPathMiddleware(build_inner())

    app.build_middleware_stack = build_middleware_stack  # type: ignore[method-assign]
    app.middleware_stack = None
