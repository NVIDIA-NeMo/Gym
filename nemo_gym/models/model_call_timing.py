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
"""Model-call duration and time to first byte, by dialect and server, for every call a model server answers.

Measured in a raw ASGI middleware, so a streamed response is timed to its last body byte rather than to the moment its
handler returned, and the first byte skips SSE keepalives and Responses lifecycle events, as model-call capture does.
It needs no rollout id and no capture: every call to the three dialect routes records, while metrics are exported.
"""

import asyncio
import re
import time
from typing import Any, Optional

from nemo_gym.config.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.core.responses_api_types import SSE_KEEPALIVE
from nemo_gym.models.responses_streaming import is_responses_lifecycle_chunk
from nemo_gym.telemetry.gym_metrics import record_model_call_duration, record_model_time_to_first_byte


_DIALECT_BY_ROUTE = {"responses": "responses", "chat/completions": "chat", "messages": "messages"}
# The dialect route, optionally under the rollout prefix (`/ng-rollout/<id>[/training-token-capture]`) that capture
# strips further in. Any other path, such as `/foo/v1/responses`, is not a model call.
_MODEL_CALL_PATH_RE = re.compile(
    rf"^(?:/{re.escape(ROLLOUT_PATH_PREFIX)}/[^/]+(?:/{re.escape(TOKEN_CAPTURE_PATH_SEGMENT)})?)?"
    r"/v1/(responses|chat/completions|messages)$"
)


def _dialect(path: str) -> Optional[str]:
    match = _MODEL_CALL_PATH_RE.match(path)
    return _DIALECT_BY_ROUTE[match.group(1)] if match else None


def _outcome(status: Optional[int]) -> str:
    return f"{status // 100}xx" if isinstance(status, int) and 100 <= status < 600 else "error"


class ModelCallTimingMiddleware:
    """Record ``gym.model.call_duration_ms`` and ``gym.model.time_to_first_byte_ms`` for each dialect call."""

    def __init__(self, app: Any, *, server_name: str) -> None:
        self._app = app
        self._server_name = server_name

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        is_post = scope.get("type") == "http" and scope.get("method") == "POST"
        dialect = _dialect(scope.get("path", "")) if is_post else None
        if dialect is None:
            await self._app(scope, receive, send)
            return
        start = time.perf_counter()
        status: Optional[int] = None
        first_byte_recorded = False
        finished = False
        disconnected = False

        def _finish(outcome: str) -> None:
            nonlocal finished
            finished = True
            record_model_call_duration(
                (time.perf_counter() - start) * 1000, dialect=dialect, server_name=self._server_name, outcome=outcome
            )

        async def _receive() -> dict[str, Any]:
            nonlocal disconnected
            message = await receive()
            if message.get("type") == "http.disconnect":
                disconnected = True
            return message

        async def _send(message: dict[str, Any]) -> None:
            nonlocal status, first_byte_recorded
            message_type = message.get("type")
            if message_type == "http.response.start":
                status = message.get("status")
            await send(message)
            if message_type != "http.response.body":
                return
            chunk = message.get("body", b"") or b""
            # After the send returned, so the time includes waiting on the client.
            if (
                not first_byte_recorded
                and chunk
                and chunk != SSE_KEEPALIVE
                and not is_responses_lifecycle_chunk(chunk)
            ):
                first_byte_recorded = True
                record_model_time_to_first_byte(
                    (time.perf_counter() - start) * 1000, dialect=dialect, server_name=self._server_name
                )
            if not message.get("more_body", False):
                _finish(_outcome(status))

        try:
            await self._app(scope, _receive, _send)
        except asyncio.CancelledError:
            if not finished:
                _finish("cancelled")
            raise
        except Exception:
            if not finished:
                _finish("error")
            raise
        if not finished:
            # The app returned without a final body: the client went away, or the response was abandoned.
            _finish("cancelled" if disconnected else "error")
