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

import dataclasses
import logging
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.config.config_types import TRACE_CONTEXT_PATH_SEGMENT
from nemo_gym.core.correlation import CorrelationPathError, encode, parse_path, remember, resolve
from nemo_gym.server_runtime.request_handling.rollout_context import scope_with_routed_path


logger = logging.getLogger(__name__)


class TraceContextPathMiddleware:
    """Resolve the request's correlation prefix once, and turn ``/ng-trace/<traceparent>`` into the header.

    This is the outermost layer that reads the path on every Gym server (only request-context isolation is
    outside it), so it resolves the path through ``nemo_gym.core.correlation.resolve``, which shares the
    result with inner middleware through ``scope["state"]``. A malformed prefix (segments out of order, an
    invalid agent instance name or traceparent) is answered with 404, as an unroutable path was before.

    The trace segment comes from ``apply_rollout_prefix``: a model URL an agent hands to a client that
    sends no ``traceparent`` header, such as a harness inside a sandbox. A header the client did
    send wins. The segment is stripped whether or not traces are on, so such a URL always routes.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            resolved = resolve(scope)
        except CorrelationPathError as error:
            # 404, not 400: the agent outcome policy grades a 400 as the model's request error.
            logger.warning(f"Refusing a malformed correlation prefix: {error}")
            await JSONResponse({"detail": str(error)}, status_code=404)(scope, receive, send)
            return
        if resolved is None or resolved[0].trace_parent is None:
            await self.app(scope, receive, send)
            return
        context, rest = resolved
        traceparent = context.trace_parent
        headers = list(scope.get("headers") or [])
        if not any(name.lower() == b"traceparent" for name, _ in headers):
            headers.append((b"traceparent", traceparent.encode("ascii")))
        # The parsed prefix is canonical, so the trace segment sits right before ``rest``: cut it out.
        segment = f"/{TRACE_CONTEXT_PATH_SEGMENT}/{traceparent}"
        path = scope["path"]
        scope = scope_with_routed_path(scope, path[: len(path) - len(rest) - len(segment)] + rest)
        scope["headers"] = headers
        # Inner middleware reads the rewritten path; a trace-only path has no prefix left.
        remaining = dataclasses.replace(context, trace_parent=None)
        remember(scope, (remaining, rest) if encode(remaining) else None)

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

    Inserted after the correlation prefix (capture key, agent instance name, training segment), which the
    capture middleware has already restored, or at the root. Cross-origin targets, targets that already
    carry a segment and targets with a malformed prefix are left alone.
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
        try:
            prefix = parse_path(parts.path, allow_empty_rest=True)
        except CorrelationPathError:
            # Left as is, so the follow-up request is refused with its own reason.
            continue
        head = encode(prefix[0]) if prefix is not None else ""
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
