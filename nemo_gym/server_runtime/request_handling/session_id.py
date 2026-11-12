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
"""Middleware that gives every request's session a Gym session ID."""

from uuid import uuid4

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.server_runtime.request_handling.session_routing import SESSION_ACQUIRE_SCOPE_KEY, SESSION_ID_SCOPE_KEY


class _ReplyReplaced(Exception):
    """Stops a route whose reply the 409 for another worker's session replaced, so it streams nothing more."""


class SessionIDMiddleware:
    """Assign the session ID before the route runs, and settle it when the reply starts.

    ``SessionMiddleware`` must wrap this middleware.
    It puts the session in ``scope["session"]``, and writes the cookie when the reply starts, after this middleware has
    settled the session.
    """

    # Direct MCP dispatch materializes the session itself.
    skipped_on_direct_mcp_dispatch = True

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # A new session takes the ID the session router chose, if any (see session_routing):
        # an episode seed's resources_session_id, or an ID the router minted.
        new_session_id = scope.get(SESSION_ID_SCOPE_KEY) or str(uuid4())
        acquire = scope.get(SESSION_ACQUIRE_SCOPE_KEY)
        session = scope["session"]
        minted = SESSION_ID_KEY not in session
        # Always assign so Starlette 1.0+ marks session.modified=True and re-sends Set-Cookie.
        session[SESSION_ID_KEY] = session.get(SESSION_ID_KEY, new_session_id)
        # Starlette's session records each access; only the route's count from here.
        tracks_access = acquire is not None and hasattr(session, "mark_accessed")
        if tracks_access:
            session.accessed = False
        refused = False

        async def send_settled(message: Message) -> None:
            nonlocal refused
            if refused:
                # The route's own reply was replaced by the 409.
                return
            if message["type"] == "http.response.start":
                failed = message["status"] >= 400
                if acquire is None:
                    if minted and failed:
                        # A failed request keeps nothing under a new session, so its reply names none.
                        # A seed refused for another episode's resources_session_id must not hand this client that
                        # session's cookie.
                        del session[SESSION_ID_KEY]
                else:
                    # With several workers, a session nobody owns, new or named by a cookie whose ownership ended,
                    # becomes this worker's only if the route uses it, before the reply leaves this worker,
                    # so the client's next request finds its owner.
                    used = session.accessed if tracks_access else True
                    if tracks_access:
                        # The reply keeps its Vary: Cookie header, as for any request that reads the session.
                        session.mark_accessed()
                    if used and not acquire():
                        # Another worker took ownership of the same ID meanwhile.
                        # Serving the request here would split the session.
                        refused = True
                        conflict = JSONResponse(
                            {"detail": "This session's ID is already owned by another worker. Retry it."}, 409
                        )
                        await conflict(scope, receive, send)
                        raise _ReplyReplaced
                    if minted and (not used or failed):
                        # No cookie names an ID that nothing holds, so a later request cannot name a session nobody
                        # owns.
                        # A failed request hands out no cookie, as with one worker.
                        del session[SESSION_ID_KEY]
            await send(message)

        try:
            await self.app(scope, receive, send_settled)
        except _ReplyReplaced:
            if not refused:
                raise
