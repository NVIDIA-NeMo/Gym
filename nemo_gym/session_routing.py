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

Agent sessions, and the resources sessions an environment server seeds, are seeded and closed by caller-assigned IDs.
Those requests can arrive without the cookie.
A seed that failed carries no ``Set-Cookie`` back, a reply can be lost, and a caller can close before it saw one.
So the first worker to see an ``agent_session_id`` or ``resources_session_id`` claims it.
The claim is a file named for the ID in a subdirectory of the socket directory.
A seed or close for that ID without a cookie naming its owner is forwarded to the claimant (see ``SessionClaims``).
A claim ends once its owner keeps nothing of the session,
and the directory goes with the socket directory when the server exits.
"""

import asyncio
import hashlib
import heapq
import json
import math
import os
import re
from base64 import b64decode
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any, Optional, Protocol
from uuid import uuid4

import itsdangerous
import uvicorn
from aiohttp import ClientConnectorError, ClientError, ClientSession, ClientTimeout, DummyCookieJar, UnixConnector
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from multidict import CIMultiDict
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
from yarl import URL

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.runtime_dir import server_runtime_dir


#: Set by the main process of a multi-worker server for the workers it spawns: the directory that holds the
#: workers' private sockets.
SESSION_SOCKET_DIR_ENV = "NEMO_GYM_SESSION_SOCKET_DIR"
#: Session key naming the worker that created the session.
SESSION_OWNER_KEY = "nemo_gym_worker"
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
# The router reads a seed or close body to find its session ID.
# Real bodies are a few kilobytes; a larger one is refused rather than buffered without limit.
_CLAIMED_BODY_LIMIT_BYTES = 8 * 1024 * 1024
_ROLLOUT_PREFIX = rf"(?:/{ROLLOUT_PATH_PREFIX}/[^/]+(?:/[^/]+)?)?"


@dataclass(frozen=True)
class ClaimedRoutes:
    """The seed and close routes of a session kind whose requests name the session by an ID in the body."""

    #: Subdirectory of the socket directory that holds this kind's claims.
    directory: str
    #: The body field that names the session.
    id_field: str
    #: The seed and close paths, bare or under a rollout prefix.
    path: re.Pattern[str]
    #: How the close path ends.
    close_suffix: str


AGENT_SESSION_ROUTES = ClaimedRoutes(
    directory="agent-sessions",
    id_field="agent_session_id",
    path=re.compile(rf"{_ROLLOUT_PREFIX}/v1/agent_sessions(?:/close)?"),
    close_suffix="/close",
)
# The typed seed and close an environment server sends; a seed without the field is an agent's own /run seed.
RESOURCES_SESSION_ROUTES = ClaimedRoutes(
    directory="resources-sessions",
    id_field="resources_session_id",
    path=re.compile(rf"{_ROLLOUT_PREFIX}/(?:seed_session|close_session)"),
    close_suffix="/close_session",
)
AGENT_SESSION_CLAIMS_DIR = AGENT_SESSION_ROUTES.directory
RESOURCES_SESSION_CLAIMS_DIR = RESOURCES_SESSION_ROUTES.directory


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


def session_owner(cookie: str, *, signer: itsdangerous.TimestampSigner) -> Optional[str]:
    """Return the worker ID stamped in a session cookie, or None if there is no valid one.

    Decoded exactly as Starlette's SessionMiddleware decodes it.
    A cookie it would reject (badly signed or expired) starts a fresh session,
    which belongs to whichever worker handles the request.
    """
    try:
        return _valid_owner(
            json.loads(b64decode(signer.unsign(cookie.encode("utf-8"), max_age=_SESSION_MAX_AGE_SECONDS)))
        )
    except (itsdangerous.BadSignature, ValueError):
        return None


def mcp_token_owner(token: str, *, serializer: itsdangerous.URLSafeSerializer) -> Optional[str]:
    """Return the worker ID in an MCP session token, or None for a token without one or with a bad signature.

    A rejected token is handled locally, where the MCP endpoint's own token check refuses it as before.
    """
    try:
        return _valid_owner(serializer.loads(token))
    except itsdangerous.BadSignature:
        return None


class AgentSessionHolder(Protocol):
    """An agent server, whose sessions are claimed by ID across workers (see ``SessionClaims``)."""

    def holds_agent_session(self, agent_session_id: str) -> bool:
        """Whether this worker still keeps any record of the session, open, failed or closed."""
        ...

    def use_agent_session_claims(self, claims: "SessionClaims") -> None:
        """Keep the claims, to release one when the worker drops the session's last record."""
        ...


class SessionClaims:
    """Which worker first saw each session ID of one kind, shared by the workers of one server.

    A claim is a file in the kind's subdirectory of the socket directory,
    named by a hash of the ID and holding the claimant's worker ID.
    It is created with a hard link from a complete temporary file, which fails if the claim exists,
    so two workers cannot both claim one ID and a reader never sees a partly written claim.

    A claim lives as long as ``holds`` says its owner keeps anything of the session.
    It is not released while a request for the ID is being served here,
    since that request may be about to create a record.
    Only the claimant removes its claim, so a removal cannot delete another worker's claim made since.
    The claims of a worker that exited stay until the server exits, and its sessions get a 410 meanwhile;
    the directory goes with the socket directory.
    """

    def __init__(
        self, socket_dir: str, routing_id: str, holds: Callable[[str], bool], *, routes: ClaimedRoutes
    ) -> None:
        self.routing_id = routing_id
        self.holds = holds
        self.routes = routes
        self.directory = os.path.join(socket_dir, routes.directory)
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        # The IDs this worker has claimed and not released, and the requests for an ID being served here.
        self._mine: set[str] = set()
        self._serving: dict[str, int] = {}

    def _path(self, session_id: str) -> str:
        return os.path.join(self.directory, hashlib.sha256(session_id.encode("utf-8")).hexdigest())

    def claim(self, session_id: str) -> Optional[str]:
        """Claim the ID for this worker unless another worker has; return the claimant.

        Returns None for a claim file that names no valid worker, which nothing here writes.
        """
        if session_id in self._mine:
            return self.routing_id
        path = self._path(session_id)
        while True:
            temporary = f"{path}.{self.routing_id}.tmp"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, self.routing_id.encode("ascii"))
            finally:
                os.close(fd)
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            else:
                self._mine.add(session_id)
                return self.routing_id
            finally:
                os.unlink(temporary)
            try:
                with open(path, encoding="ascii") as claim:
                    return _valid_routing_id(claim.read())
            except FileNotFoundError:
                # The claimant released it between the two calls: claim it again.
                continue

    @contextmanager
    def serving(self, session_id: str) -> Iterator[None]:
        """Hold the claim while a request for the ID is served here, and release it after if nothing is left."""
        self._serving[session_id] = self._serving.get(session_id, 0) + 1
        try:
            yield
        finally:
            remaining = self._serving.pop(session_id) - 1
            if remaining:
                self._serving[session_id] = remaining
            self.release(session_id)

    def release(self, session_id: str) -> None:
        """Remove this worker's claim on the ID, unless a request for it is in progress or the session remains."""
        if session_id not in self._mine or session_id in self._serving or self.holds(session_id):
            return
        self._mine.discard(session_id)
        try:
            os.unlink(self._path(session_id))
        except FileNotFoundError:
            pass


class ResourcesSessionLifetimes:
    """How long a worker keeps its claim on a ``resources_session_id``, from the replies to its seeds and closes.

    A resources server keeps its per-session state in its own subclass, so the router judges by the replies:

    - A seed that succeeded keeps the claim until a close succeeds.
    - A close that succeeded, or a seed or close that failed for a session no seed opened,
      keeps it ``window_seconds`` longer.
      The caller's retried or cleanup close reaches the owner within that window.
      A delayed seed after a close reaches it too, where the server refuses it while it remembers the close.
    - A failed seed or close of an open session leaves it open.

    So a claim outlives the session's last request by at most the window, except for a session never closed,
    which keeps its claim as long as the server keeps the session.
    """

    def __init__(self, window_seconds: float) -> None:
        self.window_seconds = window_seconds
        # Each session's claim lasts until this time, infinite while it is open.
        self._until: dict[str, float] = {}
        # The finite times, earliest first; an entry is stale once ``_until`` has moved on.
        self._expiries: list[tuple[float, str]] = []

    def holds(self, session_id: str) -> bool:
        return self._until.get(session_id, 0.0) > monotonic()

    def finished(self, session_id: str, *, close: bool, succeeded: bool) -> None:
        """Record the outcome of a seed or close of the session served here."""
        if succeeded and not close:
            self._until[session_id] = math.inf
            return
        if not succeeded and self._until.get(session_id) == math.inf:
            return
        until = monotonic() + self.window_seconds
        self._until[session_id] = until
        heapq.heappush(self._expiries, (until, session_id))

    def expired(self) -> list[str]:
        """Forget the sessions whose time has passed, and return them, so their claims can be released."""
        now = monotonic()
        expired = []
        while self._expiries and self._expiries[0][0] <= now:
            until, session_id = heapq.heappop(self._expiries)
            if self._until.get(session_id) == until:
                del self._until[session_id]
                expired.append(session_id)
        return expired


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
        mcp_token_header: Optional[str] = None,
        mcp_token_serializer: Optional[itsdangerous.URLSafeSerializer] = None,
        mcp_path: str = "/mcp",
        session_claims: Optional[SessionClaims] = None,
        resources_lifetimes: Optional[ResourcesSessionLifetimes] = None,
    ) -> None:
        self.app = app
        self.routing_id = routing_id
        self.socket_dir = socket_dir
        self.session_cookie = session_cookie
        self.signer = itsdangerous.TimestampSigner(secret_key)
        # One connection pool per owner socket, kept for the life of this worker and closed at its shutdown.
        self.clients = clients
        self.mcp_token_header = mcp_token_header
        self.mcp_token_serializer = mcp_token_serializer
        self.mcp_path = mcp_path
        self.session_claims = session_claims
        self.resources_lifetimes = resources_lifetimes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        # Forwarded requests are always handled here, so a request is forwarded at most once.
        forwarded = bool(scope.get(_FORWARDED_SCOPE_KEY))
        owner = None if forwarded else self._owner(HTTPConnection(scope))
        if owner is not None and owner != self.routing_id:
            await self._forward(owner, scope, receive, send)
            return
        claims = self.session_claims
        if claims is not None and scope["method"] == "POST" and claims.routes.path.fullmatch(scope["path"]):
            await self._serve_claimed_request(claims, scope, receive, send, routed=forwarded or owner is not None)
            return
        await self.app(scope, receive, send)

    async def _serve_claimed_request(
        self, claims: SessionClaims, scope: Scope, receive: Receive, send: Send, *, routed: bool
    ) -> None:
        """Serve a session seed or close on the worker that claims its ID.

        A request that names no owner goes to the claimant, and claims the ID for this worker if nobody has.
        A request that already reached its owner, by its cookie or by forwarding, is served here.
        If another worker claims the ID, it is refused, since a cookie of another session cannot move the ID.
        """
        body = await _read_body(receive, limit=_CLAIMED_BODY_LIMIT_BYTES)
        if body is None:
            return
        if len(body) > _CLAIMED_BODY_LIMIT_BYTES:
            await JSONResponse(
                {"detail": f"A session seed or close body may not exceed {_CLAIMED_BODY_LIMIT_BYTES} bytes."},
                status_code=413,
            )(scope, receive, send)
            return
        replay = _replay_body(bytes(body), receive)
        session_id = _session_id(body, claims.routes.id_field)
        if session_id is None:
            # A body without a valid ID names no session to route.
            # The app refuses it, or, for a resources seed, starts a session of the cookie's own.
            await self.app(scope, replay, send)
            return
        lifetimes = self.resources_lifetimes
        if lifetimes is not None:
            for expired in lifetimes.expired():
                claims.release(expired)
        claimant = claims.claim(session_id)
        if claimant == self.routing_id:
            with claims.serving(session_id):
                if lifetimes is None:
                    await self.app(scope, replay, send)
                    return
                status: list[int] = []

                async def send_and_record(message: Message) -> None:
                    if message["type"] == "http.response.start":
                        status.append(message["status"])
                    await send(message)

                try:
                    await self.app(scope, replay, send_and_record)
                finally:
                    lifetimes.finished(
                        session_id,
                        close=scope["path"].endswith(claims.routes.close_suffix),
                        succeeded=bool(status) and 200 <= status[0] < 300,
                    )
            return
        if claimant is None or routed:
            # Its cookie names this worker, for another session than the ID's,
            # or the claimant released the ID and another worker claimed it while the request was forwarded here.
            await JSONResponse(
                {
                    "detail": (
                        f"This {claims.routes.id_field} is held by another server worker than the one this request "
                        "was routed to by its session cookie. Do not send another session's cookie with it."
                    )
                },
                status_code=409,
            )(scope, receive, send)
            return
        await self._forward(claimant, scope, receive, send, body=body)

    def _owner(self, connection: HTTPConnection) -> Optional[str]:
        cookie = connection.cookies.get(self.session_cookie)
        token = connection.headers.get(self.mcp_token_header) if self.mcp_token_serializer is not None else None
        path = connection.scope["path"]
        is_mcp = path == self.mcp_path or path.startswith(self.mcp_path + "/")
        # MCP tool calls name their session by the token, never by a cookie.
        if token is not None and (is_mcp or cookie is None):
            return mcp_token_owner(token, serializer=self.mcp_token_serializer)
        if cookie is not None:
            return session_owner(cookie, signer=self.signer)
        return None

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

    async def _forward(
        self, owner: str, scope: Scope, receive: Receive, send: Send, *, body: Optional[bytearray] = None
    ) -> None:
        if body is None:
            body = await _read_body(receive)
            if body is None:
                return

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


async def _read_body(receive: Receive, *, limit: Optional[int] = None) -> Optional[bytearray]:
    """Read the request body, or None if the client disconnected.

    With a limit, stop once the body exceeds it; the caller refuses a body longer than the limit.
    """
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        body += message.get("body", b"")
        if not message.get("more_body", False) or (limit is not None and len(body) > limit):
            return body


def _replay_body(body: bytes, receive: Receive) -> Receive:
    """A receive that gives the app the body already read, then passes later messages, such as a disconnect, on."""
    pending = True

    async def replay() -> Message:
        nonlocal pending
        if pending:
            pending = False
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _session_id(body: bytes | bytearray, field: str) -> Optional[str]:
    """The session ID in ``field`` of a seed or close body, or None for a body that has no string one."""
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    value = payload.get(field) if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _mark_forwarded(app: ASGIApp) -> ASGIApp:
    async def forwarded_app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = {**scope, _FORWARDED_SCOPE_KEY: True}
        await app(scope, receive, send)

    return forwarded_app


class _PrivateSocketProtocol(HttpToolsProtocol):
    """uvicorn's httptools protocol without access logs, which it would otherwise write whenever the logger has handlers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.access_log = False


@asynccontextmanager
async def serve_private_socket(app: ASGIApp, path: str) -> AsyncIterator[None]:
    """Serve *app* on a Unix socket for requests other workers forward here, on the running event loop."""
    config = uvicorn.Config(
        _mark_forwarded(app),
        uds=path,
        # This worker's main server already ran the app's lifespan.
        lifespan="off",
        # The forwarding worker's own server adds these, so adding them here would duplicate them.
        server_header=False,
        date_header=False,
        proxy_headers=False,
        # The forwarding worker logs the request, so this server does not log it again.
        http=_PrivateSocketProtocol,
        # The worker's main server already configured uvicorn's process-wide loggers.
        # Configuring them here would set the worker's log level and drop its access-log handlers.
        log_config=None,
        log_level=None,
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
    agent_sessions: Optional[AgentSessionHolder] = None,
    resources_session_close_window_seconds: Optional[float] = None,
) -> str:
    """Route this worker's requests by session owner, and serve forwarded requests on a private socket.

    Call once the app's middleware is otherwise in place: the router must sit outside SessionMiddleware.
    New sessions, and MCP session tokens, are stamped with the returned worker ID,
    read from ``app.state.nemo_gym_routing_id``.
    Pass the MCP token header and serializer for a server that exposes its tools over MCP.
    Pass the agent for an agent server, whose seeds and closes are then routed by ``agent_session_id``,
    or a close window for a resources server, whose typed seeds and closes are routed by ``resources_session_id``.
    """
    routing_id = uuid4().hex
    app.state.nemo_gym_routing_id = routing_id
    clients: dict[str, ClientSession] = {}
    claims = lifetimes = None
    if agent_sessions is not None:
        claims = SessionClaims(socket_dir, routing_id, agent_sessions.holds_agent_session, routes=AGENT_SESSION_ROUTES)
        agent_sessions.use_agent_session_claims(claims)
    elif resources_session_close_window_seconds is not None:
        lifetimes = ResourcesSessionLifetimes(resources_session_close_window_seconds)
        claims = SessionClaims(socket_dir, routing_id, lifetimes.holds, routes=RESOURCES_SESSION_ROUTES)
    app.add_middleware(
        SessionRoutingMiddleware,
        routing_id=routing_id,
        socket_dir=socket_dir,
        session_cookie=session_cookie,
        secret_key=secret_key,
        clients=clients,
        mcp_token_header=mcp_token_header,
        mcp_token_serializer=mcp_token_serializer,
        session_claims=claims,
        resources_lifetimes=lifetimes,
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
