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
"""Middleware that answers a failed request and reports every request's failures in its reply."""

import asyncio
import json
from typing import Callable

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.core.failure_records import FailureReport
from nemo_gym.core.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.core.typed_exceptions import route_reply_error


def _add_failure_header(message: Message, report: FailureReport) -> None:
    """Put ``report`` in a reply's failure header, merged with any report the reply already carries."""
    headers = MutableHeaders(scope=message)
    # A nested layer of this middleware may have sent the report already; keep its failure_id.
    inner = FailureHeader.decode(headers)
    if inner is not None:
        report = report.merged(inner) if report.failure_id is not None else inner.merged(report)
    headers.update(FailureHeader.encode(report))


def _is_cancelling_this_request(error: BaseException) -> bool:
    """Whether ``error`` is the cancellation of the task serving this request.

    Gym cancels that task when a client disconnects, when a checkpoint retires an activation or a model call, and when
    uvicorn shuts down.
    """
    task = asyncio.current_task()
    return isinstance(error, asyncio.CancelledError) and task is not None and task.cancelling() > 0


class ExceptionHandlingMiddleware:
    """Answer a request whose route failed before replying, and send each reply with its failure report.

    A route that raises, or returns without replying, gets ``failure_reply``'s response.
    That response keeps the partial observations the route left in the request state.
    Every reply, successful or not, carries the failures observed while serving it in its failure header.
    Once a reply has started, its status is already sent, so a later failure propagates.
    A cancelled request still gets the failure reply, and then the cancellation propagates to whoever cancelled the
    task.
    """

    # Direct MCP dispatch handles exceptions in the /mcp request itself.
    skipped_on_direct_mcp_dispatch = True

    def __init__(self, app: ASGIApp, *, failure_reply: Callable[[BaseException], JSONResponse]) -> None:
        self.app = app
        self.failure_reply = failure_reply

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        with RequestFailureLog.capture() as failure_log:
            error: BaseException | None = None
            started = False

            async def send_with_report(message: Message) -> None:
                nonlocal started
                if message["type"] == "http.response.start":
                    started = True
                    # Success or failure, the reply forwards its own failure and those observed while serving it.
                    _add_failure_header(message, failure_log.reply_report(error))
                await send(message)

            try:
                await self.app(scope, receive, send_with_report)
            except BaseException as raised:
                if started:
                    raise
                error = raised
            else:
                if started:
                    return
                # A route that never replied still gets Gym's failure body and header, not uvicorn's bare 500.
                error = RuntimeError("No response returned.")
            await self._failure_response(scope, error)(scope, receive, send_with_report)
            if _is_cancelling_this_request(error):
                raise error

    def _failure_response(self, scope: Scope, error: BaseException) -> JSONResponse:
        # An agent route failure answers with its cause's body and status.
        response = self.failure_reply(route_reply_error(error))
        # These are the same typed observations a successful reply carries.
        # A raised route must not turn its actual partial observations into an escaped error string.
        from nemo_gym.trajectory.trajectory_record import failure_observations

        state = scope.get("state") or {}
        observed = {}
        for slot, key in (
            ("nemo_gym_agent_observations", "ng_agent_observations"),
            ("nemo_gym_agent_trajectory", "ng_trajectory"),
        ):
            value = state.get(slot)
            if isinstance(value, BaseModel):
                observed.update(failure_observations({key: value.model_dump(mode="json")}))
        if not observed:
            return response
        payload = json.loads(response.body)
        payload = payload if isinstance(payload, dict) else {"detail": payload}
        payload.update(observed)
        return JSONResponse(
            payload,
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k != "content-length"},
        )
