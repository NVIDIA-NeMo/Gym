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
"""The middleware that handles a model call's ``/ng-sampling/<values|null>`` segment on each kind of server.

``nemo_gym.core.model_call_sampling`` describes the values and where they come from.
"""

import dataclasses
import json
import math
from typing import Any, NamedTuple, Optional

import orjson
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.core.correlation import CorrelationPathError, encode, parse_path
from nemo_gym.core.model_call_sampling import (
    MODEL_SAMPLING_NULL,
    ModelCallSampling,
    decode_model_sampling,
    model_sampling_context,
)
from nemo_gym.server_runtime.request_handling.rollout_context import scope_with_routed_path


# Each generation API's output-token limit fields. The values' cap replaces them all, under the last name.
_TOKEN_LIMIT_FIELDS = {
    "/v1/chat/completions": ("max_tokens", "max_completion_tokens"),
    "/v1/responses": ("max_output_tokens",),
    "/v1/messages": ("max_tokens",),
}


def _route_path(scope: Scope) -> tuple[str, str]:
    """Split the request path into a relay's mount path (``root_path``) and the path this app routes."""
    path = scope.get("path", "")
    # A relay mounted at a path that looks like a rollout prefix or a segment keeps that path.
    root_path = scope.get("root_path", "").rstrip("/")
    if root_path and (path == root_path or path.startswith(f"{root_path}/")):
        return root_path, path[len(root_path) :]
    return "", path


class _SamplingPath(NamedTuple):
    routed_path: str  # The request path without the segment.
    api_path: str  # The path after the segment, such as /v1/responses.
    segment: str
    sampling: Optional[ModelCallSampling]


def _parse_sampling_path(scope: Scope) -> Optional[_SamplingPath]:
    """Split off a request path's sampling segment, or return ``None`` for a path without one.

    Raises ``ValueError`` for a malformed segment.
    """
    root_path, route_path = _route_path(scope)
    # The correlation prefix fixes where the segment sits; a segment out of place raises CorrelationPathError.
    parsed = parse_path(route_path, allow_empty_rest=True)
    if parsed is None or parsed[0].model_sampling is None:
        return None
    context, api_path = parsed
    if not api_path:
        raise ValueError("The /ng-sampling/ segment must precede the API path")
    sampling = decode_model_sampling(context.model_sampling)
    routed_path = f"{root_path}{encode(dataclasses.replace(context, model_sampling=None))}{api_path}"
    return _SamplingPath(routed_path, api_path, context.model_sampling, sampling)


def _api_path(route_path: str) -> str:
    """The path a request routes to after its correlation prefix, such as /v1/responses."""
    try:
        parsed = parse_path(route_path)
    except CorrelationPathError:
        return route_path
    return parsed[1] if parsed is not None else route_path


async def _reject(scope: Scope, receive: Receive, send: Send, detail: str) -> None:
    await JSONResponse({"detail": detail}, status_code=422)(scope, receive, send)


class ModelSamplingMiddleware:
    """Apply a model call's sampling segment on a Model Server, before policy admission and capture see the call.

    Values replace the request's ``temperature`` and ``top_p``. Their ``max_output_tokens`` caps the request's own
    output-token limit, so a smaller limit the caller set is kept. ``null`` leaves the request as its caller sent it.
    Either way the segment is stripped from the path and stays the current segment while the server handles the
    call. With ``required``, a generation call without a segment fails.

    Only the body of a POST to a generation route is rewritten; other routes keep their body. Responses, streamed or
    not, pass through unchanged.
    """

    def __init__(self, app: ASGIApp, *, required: bool = False) -> None:
        self.app = app
        self.required = required

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            parsed = _parse_sampling_path(scope)
        except ValueError as error:
            await _reject(scope, receive, send, f"Invalid model sampling URL: {error}")
            return
        if parsed is None:
            if self.required and scope["method"] == "POST" and _api_path(_route_path(scope)[1]) in _TOKEN_LIMIT_FIELDS:
                await _reject(
                    scope,
                    receive,
                    send,
                    "This Model Server serves an agent that the training framework trains, so every generation call "
                    "must carry the rollout's /ng-sampling/ segment, and this one carries none. Build model URLs with "
                    "Gym's URL helpers, which add it.",
                )
                return
            await self.app(scope, receive, send)
            return

        scope = scope_with_routed_path(scope, parsed.routed_path)
        limit_fields = _TOKEN_LIMIT_FIELDS.get(parsed.api_path) if scope["method"] == "POST" else None
        with model_sampling_context(parsed.segment):
            if parsed.sampling is None or limit_fields is None:
                await self.app(scope, receive, send)
                return
            await self._apply(scope, receive, send, parsed.sampling, limit_fields)

    async def _apply(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        sampling: ModelCallSampling,
        limit_fields: tuple[str, ...],
    ) -> None:
        chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            # Parse as FastAPI does: orjson refuses lone surrogates and turns integers wider than 64 bits into floats.
            body: Any = json.loads(chunks)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            await _reject(scope, receive, send, "A model request with sampling values must be a JSON object")
            return

        values = sampling.model_dump(exclude_none=True)
        cap = values.pop("max_output_tokens", None)
        body.update(values)
        if cap is not None:
            # A harness reserves less output near the end of its context or for a summary; never raise its limit.
            for name in limit_fields:
                limit = body.pop(name, None)
                # JSON may spell a whole number of tokens as a float (50.0); a non-finite float is no limit.
                if isinstance(limit, float) and math.isfinite(limit):
                    limit = int(limit)
                if type(limit) is int and limit > 0:
                    cap = min(cap, limit)
            body[limit_fields[-1]] = cap
        try:
            rewritten = orjson.dumps(body)
        except orjson.JSONEncodeError:
            # What orjson cannot write (a lone surrogate, an integer wider than 64 bits), json writes as it was read.
            rewritten = json.dumps(body, separators=(",", ":")).encode()
        scope = {
            **scope,
            "headers": [(key, value) for key, value in scope.get("headers", []) if key.lower() != b"content-length"]
            + [(b"content-length", str(len(rewritten)).encode())],
        }
        replayed = False

        async def rewritten_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": rewritten, "more_body": False}
            return await receive()

        await self.app(scope, rewritten_receive, send)


class ModelSamplingContextMiddleware:
    """Strip an inbound ``/ng-sampling/`` segment on an agent server and give the agent's model calls the same one.

    An environment server's activation of the agent, or the agent's call to itself, carries the rollout's segment.
    The agent's URL helpers then copy it onto every model URL the agent builds while handling the request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            parsed = _parse_sampling_path(scope)
        except ValueError as error:
            await _reject(scope, receive, send, f"Invalid model sampling URL: {error}")
            return
        if parsed is None:
            await self.app(scope, receive, send)
            return
        with model_sampling_context(parsed.segment):
            await self.app(scope_with_routed_path(scope, parsed.routed_path), receive, send)


class CallerSamplingMiddleware:
    """Give every model call a resources server makes the ``null`` segment, so it keeps its caller's sampling.

    Installed in a run that sets ``model_call_sampling``. A resources server never runs an agent that the framework
    trains, so its judges and tools keep their own sampling, or their Model Server's ``sampling_overrides``.
    """

    # Read by MCP auto-exposure: the segment is set for the whole /mcp request, and direct MCP dispatch runs each tool
    # inside that request, so a tool's model calls keep it.
    applies_to_mcp_requests = True

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with model_sampling_context(MODEL_SAMPLING_NULL):
            await self.app(scope, receive, send)
