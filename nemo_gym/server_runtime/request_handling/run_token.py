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
"""Answer a server's orchestration routes only for callers that send the run's token.

A run's servers and clients send the run's token with every call to each other (``nemo_gym.core.run_token``). A server
that requires it (``require_run_token`` in the global config, or ``requires_run_token`` on its class) answers the routes
``SimpleServer.run_token_routes`` names, such as ``/verify``, only for those callers, so code in a sandbox that reaches
the server over the network cannot call them.

The check wraps each such route, so it runs after routing: it sees the route's own path whatever prefix the request
carried, and it answers before the route reads the request's body. Direct MCP dispatch calls a tool's handler without
its route, so the check does not apply there: the MCP endpoint checks its own session tokens.
"""

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel
from starlette.datastructures import Headers
from starlette.responses import Response
from starlette.routing import Route

from nemo_gym.core.run_token import RUN_TOKEN_HEADER, is_run_token


class RunTokenConfig(BaseModel):
    # Every server of the run answers its orchestration routes (SimpleServer.run_token_routes) only for callers that
    # send the run's token.
    require_run_token: bool = False


def require_run_token_on_routes(app: Any, paths: Iterable[str], refusal: Callable[[], Response]) -> list[str]:
    """Answer the routes of ``app`` whose path is in ``paths`` with ``refusal()`` unless the request carries the run's
    token, and return the paths of the routes it guards, in the app's order.

    Call it once every route is in place: a route added or replaced later is not guarded.
    """
    wanted = frozenset(paths)
    guarded = []
    for route in app.router.routes:
        if isinstance(route, Route) and route.path in wanted:
            route.app = _with_run_token_check(route.app, refusal)
            guarded.append(route.path)
    return guarded


def _with_run_token_check(app: Any, refusal: Callable[[], Response]) -> Callable[..., Any]:
    async def checked(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not is_run_token(Headers(scope=scope).get(RUN_TOKEN_HEADER)):
            await refusal()(scope, receive, send)
            return
        await app(scope, receive, send)

    return checked
