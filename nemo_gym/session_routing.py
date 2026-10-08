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
"""Route each session's requests to the uvicorn worker that holds its state.

uvicorn's workers share one listening socket, so a request reaches whichever worker accepts the connection,
while session state (keyed by the signed session cookie) lives in the memory of the worker that created it.

With more than one worker,
each worker also serves the same app on a private Unix socket in a directory the main process creates for the server,
named by a random ID for this worker process.
A new session is stamped with that ID (it travels inside the signed session cookie),
so any worker can tell which worker owns a session from the cookie alone.
A request for a session another worker owns is forwarded to that worker's private socket unchanged,
and its reply is streamed back unchanged.
If the owner's socket is gone (the worker exited and was replaced, or the server was restarted),
the request gets an explicit 410 instead of silently running against empty state.

Clients that reach a resources server's tools over MCP send no cookies.
They send the signed MCP session token minted at ``/seed_session`` (see ``nemo_gym.mcp_auto_exposure``),
which carries the same worker ID, so the router reads the owner from the token on the MCP path,
and on any request without a session cookie.

The owner lives in the cookie or token rather than in a central session table,
so creating or ending a session costs no extra round trip,
and nothing grows with the number of sessions a server has served.

A request that belongs to a worker without a session, such as an agent's call to itself within one ``/run``,
names that worker in the ``x-ng-session-owner`` header instead.

A partial-rollout checkpoint restore installs the sessions of each worker that ran
before a crash on one live worker,
and adds an entry from the old worker ID to the new one to the alias table (see
``nemo_gym._checkpoint.participant_workers``).
Requests whose cookie or token names an old worker then reach the worker that holds its sessions now.
Sessions checkpointed by a single-process server name no worker;
a restore places each on a worker and records it in the placement table, by session ID,
that the router consults for a cookie or token without an owner.
"""

import asyncio
import json
import os
import re
from base64 import b64decode
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import uuid4

import itsdangerous
import uvicorn
from aiohttp import ClientConnectorError, ClientError, ClientSession, ClientTimeout, DummyCookieJar, UnixConnector
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from multidict import CIMultiDict
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send
from yarl import URL

from nemo_gym.runtime_dir import server_runtime_dir


#: Set by the main process of a multi-worker server for the workers it spawns: the directory that holds the
#: workers' private sockets.
SESSION_SOCKET_DIR_ENV = "NEMO_GYM_SESSION_SOCKET_DIR"
#: Session key naming the worker that created the session.
SESSION_OWNER_KEY = "nemo_gym_worker"
#: Session key of the session ID; ``nemo_gym.server_utils.SESSION_ID_KEY`` is defined from it.
SESSION_ID_CLAIM = "session_id"
#: MCP session token key of the session ID (see ``nemo_gym.mcp_auto_exposure``).
_MCP_SESSION_ID_CLAIM = "sid"
#: Header naming the worker a request belongs to, for a request without a session.
SESSION_OWNER_HEADER = "x-ng-session-owner"
#: Scope key marking a request that arrived over a worker's private socket.
_FORWARDED_SCOPE_KEY = "nemo_gym.session_forwarded"

# Matches Starlette's SessionMiddleware default, so a cookie it would accept is routed too.
_SESSION_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
_ROUTING_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
# Each hop frames its own body and connection.
_HOP_BY_HOP_HEADERS = frozenset({"connection", "keep-alive", "transfer-encoding", "content-length", "upgrade"})
# Lower than the private server's keep-alive, so the forwarding side always closes an idle connection first.
_FORWARD_KEEPALIVE_SECONDS = 15.0
_PRIVATE_SERVER_KEEPALIVE_SECONDS = 30


def create_session_socket_dir() -> str:
    """The directory for one server's private worker sockets: this process's runtime directory."""
    return server_runtime_dir()


def worker_socket_path(socket_dir: str, routing_id: str) -> str:
    return os.path.join(socket_dir, f"{routing_id}.sock")


def _valid_owner(claims: Any) -> Optional[str]:
    owner = claims.get(SESSION_OWNER_KEY) if isinstance(claims, dict) else None
    return _valid_routing_id(owner)


def _valid_routing_id(owner: Any) -> Optional[str]:
    return owner if isinstance(owner, str) and _ROUTING_ID_PATTERN.fullmatch(owner) else None


def session_aliases(app: FastAPI) -> dict[str, str]:
    """The app's table of restored session owners: an old worker ID to the live worker that holds its sessions."""
    if not hasattr(app.state, "nemo_gym_session_aliases"):
        app.state.nemo_gym_session_aliases = {}
    return app.state.nemo_gym_session_aliases


def session_placements(app: FastAPI) -> dict[str, str]:
    """The app's table of restored sessions that name no owner: a session ID to the worker that holds it."""
    if not hasattr(app.state, "nemo_gym_session_placements"):
        app.state.nemo_gym_session_placements = {}
    return app.state.nemo_gym_session_placements


def _cookie_claims(cookie: str, *, signer: itsdangerous.TimestampSigner) -> Any:
    try:
        return json.loads(b64decode(signer.unsign(cookie.encode("utf-8"), max_age=_SESSION_MAX_AGE_SECONDS)))
    except (itsdangerous.BadSignature, ValueError):
        return None


def _token_claims(token: str, *, serializer: itsdangerous.URLSafeSerializer) -> Any:
    try:
        return serializer.loads(token)
    except itsdangerous.BadSignature:
        return None


def session_owner(cookie: str, *, signer: itsdangerous.TimestampSigner) -> Optional[str]:
    """Return the worker ID stamped in a session cookie, or None if there is no valid one.

    Decoded exactly as Starlette's SessionMiddleware decodes it.
    A cookie it would reject (badly signed or expired) starts a fresh session,
    which belongs to whichever worker handles the request.
    """
    return _valid_owner(_cookie_claims(cookie, signer=signer))


def mcp_token_owner(token: str, *, serializer: itsdangerous.URLSafeSerializer) -> Optional[str]:
    """Return the worker ID in an MCP session token, or None for a token without one or with a bad signature.

    A rejected token is handled locally, where the MCP endpoint's own token check refuses it as before.
    """
    return _valid_owner(_token_claims(token, serializer=serializer))


class SessionRoutingMiddleware:
    """Forward a request for a session another worker owns to that worker's private socket.

    It sits outside SessionMiddleware and reads the owner by decoding the signed cookie with the same secret.
    A forwarded request therefore never reaches this worker's SessionMiddleware,
    so the reply carries only the owner's Set-Cookie, built from the session as the owner left it.
    Sitting inside SessionMiddleware instead would add this worker's stale copy of the session as a second Set-Cookie.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        routing_id: str,
        socket_dir: str,
        session_cookie: str,
        secret_key: str,
        clients: dict[str, ClientSession],
        aliases: Optional[dict[str, str]] = None,
        placements: Optional[dict[str, str]] = None,
        mcp_token_header: Optional[str] = None,
        mcp_token_serializer: Optional[itsdangerous.URLSafeSerializer] = None,
        mcp_path: str = "/mcp",
    ) -> None:
        self.app = app
        self.routing_id = routing_id
        self.socket_dir = socket_dir
        self.session_cookie = session_cookie
        self.signer = itsdangerous.TimestampSigner(secret_key)
        # One connection pool per owner socket, kept for the life of this worker and closed at its shutdown.
        self.clients = clients
        # Updated in place when a checkpoint restore moves an old worker's sessions to this server's workers.
        self.aliases = aliases if aliases is not None else {}
        # Likewise for restored sessions whose cookie or token names no owner, by session ID.
        self.placements = placements if placements is not None else {}
        self.mcp_token_header = mcp_token_header
        self.mcp_token_serializer = mcp_token_serializer
        self.mcp_path = mcp_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get(_FORWARDED_SCOPE_KEY):
            # Forwarded requests are always handled here, so a request is forwarded at most once.
            await self.app(scope, receive, send)
            return
        owner = self._owner(HTTPConnection(scope))
        owner = self.aliases.get(owner, owner)
        if owner is None or owner == self.routing_id:
            await self.app(scope, receive, send)
            return
        await self._forward(owner, scope, receive, send)

    def _owner(self, connection: HTTPConnection) -> Optional[str]:
        cookie = connection.cookies.get(self.session_cookie)
        token = connection.headers.get(self.mcp_token_header) if self.mcp_token_serializer is not None else None
        path = connection.scope["path"]
        is_mcp = path == self.mcp_path or path.startswith(self.mcp_path + "/")
        # MCP tool calls name their session by the token, never by a cookie.
        if token is not None and (is_mcp or cookie is None):
            claims, session_id_claim = (
                _token_claims(token, serializer=self.mcp_token_serializer),
                _MCP_SESSION_ID_CLAIM,
            )
        elif cookie is not None:
            claims, session_id_claim = _cookie_claims(cookie, signer=self.signer), SESSION_ID_CLAIM
        else:
            return _valid_routing_id(connection.headers.get(SESSION_OWNER_HEADER))
        owner = _valid_owner(claims)
        if owner is None and self.placements and isinstance(claims, dict):
            session_id = claims.get(session_id_claim)
            owner = self.placements.get(session_id) if isinstance(session_id, str) else None
        return owner

    def _client(self, owner: str) -> ClientSession:
        client = self.clients.get(owner)
        if client is None:
            client = self.clients[owner] = ClientSession(
                connector=UnixConnector(
                    worker_socket_path(self.socket_dir, owner), limit=0, keepalive_timeout=_FORWARD_KEEPALIVE_SECONDS
                ),
                cookie_jar=DummyCookieJar(),
                # Pass the reply's bytes through exactly as the owner encoded them.
                auto_decompress=False,
                # The owner's handler takes as long as it takes; the caller holds its own timeout.
                timeout=ClientTimeout(total=None),
            )
        return client

    async def _forward(self, owner: str, scope: Scope, receive: Receive, send: Send) -> None:
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        headers = CIMultiDict(
            (name.decode("latin-1"), value.decode("latin-1"))
            for name, value in scope["headers"]
            if name.decode("latin-1").lower() not in _HOP_BY_HOP_HEADERS
        )
        raw_path = scope.get("raw_path") or scope["path"].encode("utf-8")
        query = scope.get("query_string", b"")
        target = raw_path + (b"?" + query if query else b"")
        url = URL("http://localhost" + target.decode("latin-1"), encoded=True)

        client = self._client(owner)
        try:
            response = await client.request(
                scope["method"],
                url,
                headers=headers,
                data=bytes(body),
                allow_redirects=False,
                # Send only the caller's headers.
                skip_auto_headers=("User-Agent", "Accept", "Accept-Encoding", "Content-Type"),
            )
        except ClientConnectorError:
            # The owner's socket is gone or refuses connections: the worker that held this session exited.
            await self._discard_client(owner)
            await JSONResponse(
                {
                    "detail": (
                        "The server worker that held this session is no longer running, so its state is lost. "
                        "Start a new session."
                    )
                },
                status_code=410,
            )(scope, receive, send)
            return
        except ClientError as error:
            await JSONResponse(
                {"detail": f"Could not reach the server worker that holds this session: {error!r}"}, status_code=502
            )(scope, receive, send)
            return

        async with response:
            await send(
                {
                    "type": "http.response.start",
                    "status": response.status,
                    "headers": [
                        (name, value)
                        for name, value in response.raw_headers
                        if name.decode("latin-1").lower() not in {"connection", "keep-alive", "transfer-encoding"}
                    ],
                }
            )
            async for chunk in response.content.iter_any():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _discard_client(self, owner: str) -> None:
        client = self.clients.pop(owner, None)
        if client is not None:
            await client.close()


def _mark_forwarded(app: ASGIApp) -> ASGIApp:
    async def forwarded_app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = {**scope, _FORWARDED_SCOPE_KEY: True}
        await app(scope, receive, send)

    return forwarded_app


@asynccontextmanager
async def serve_private_socket(app: ASGIApp, path: str) -> AsyncIterator[None]:
    """Serve *app* on a Unix socket for requests other workers forward here, on the running event loop."""
    config = uvicorn.Config(
        _mark_forwarded(app),
        uds=path,
        # This worker's main server already ran the app's lifespan.
        lifespan="off",
        http="httptools",
        # The forwarding worker's own server adds these, so adding them here would duplicate them.
        server_header=False,
        date_header=False,
        proxy_headers=False,
        access_log=False,
        log_level="warning",
        timeout_keep_alive=_PRIVATE_SERVER_KEEPALIVE_SECONDS,
        timeout_graceful_shutdown=0.5,
    )
    config.load()
    server = uvicorn.Server(config)
    # Server.serve would also install signal handlers, which belong to this worker's main server.
    server.lifespan = config.lifespan_class(config)
    await server.startup()
    os.chmod(path, 0o600)
    try:
        yield
    finally:
        await server.shutdown()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def install_session_routing(
    app: FastAPI,
    *,
    socket_dir: str,
    session_cookie: str,
    secret_key: str,
    mcp_token_header: Optional[str] = None,
    mcp_token_serializer: Optional[itsdangerous.URLSafeSerializer] = None,
) -> str:
    """Route this worker's requests by session owner, and serve forwarded requests on a private socket.

    Call once the app's middleware is otherwise in place: the router must sit outside SessionMiddleware.
    New sessions, and MCP session tokens, are stamped with the returned worker ID,
    read from ``app.state.nemo_gym_routing_id``.
    Pass the MCP token header and serializer for a server that exposes its tools over MCP.
    """
    routing_id = uuid4().hex
    app.state.nemo_gym_routing_id = routing_id
    clients: dict[str, ClientSession] = {}
    app.add_middleware(
        SessionRoutingMiddleware,
        routing_id=routing_id,
        socket_dir=socket_dir,
        session_cookie=session_cookie,
        secret_key=secret_key,
        clients=clients,
        aliases=session_aliases(app),
        placements=session_placements(app),
        mcp_token_header=mcp_token_header,
        mcp_token_serializer=mcp_token_serializer,
    )
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_private_socket(application: FastAPI) -> AsyncIterator[Any]:
        async with original_lifespan(application) as state:
            # Started after the app's own startup, so a forwarded request never reaches a half-started app.
            async with serve_private_socket(application, worker_socket_path(socket_dir, routing_id)):
                try:
                    yield state
                finally:
                    await asyncio.gather(*(client.close() for client in clients.values()))
                    clients.clear()

    app.router.lifespan_context = lifespan_with_private_socket
    return routing_id
