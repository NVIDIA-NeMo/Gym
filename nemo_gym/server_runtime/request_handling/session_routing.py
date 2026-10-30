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
while a session's state lives in the memory of the worker that created it.
With more than one worker, each worker also serves the same app on a private Unix socket,
in a directory the main process creates for the server, and requests are routed by one rule:

1. Find the request's session ID.
2. Look up the session's claim, which names the worker that created the session.
3. Forward the request to that worker's private socket, unchanged, and stream its reply back unchanged.
   If this worker holds the claim, or the session has no claim, the request is handled here.

A request that arrives over a private socket is always handled where it arrives, so it is forwarded at most once.
If the claimant's socket is gone, because its worker exited, the request gets a 410
instead of silently running against empty state.

Where the session ID comes from:

| Request | Session ID |
| --- | --- |
| Any request to a resources server | The session cookie's ``session_id`` |
| A resources tool call over MCP, or a request with an MCP session token and no cookie | The token's ``sid`` |
| A typed resources ``/seed_session`` or ``/close_session`` | The body's ``resources_session_id`` |
| Any request to an agent server | The session cookie's ``agent_session_id`` |
| An agent session seed or close | The body's ``agent_session_id`` |

The body is read only on the seed and close routes, where it names the session before any cookie exists.
If the cookie and the body name different sessions, the reply is 409.
A typed resources session uses its ``resources_session_id`` as its cookie's ``session_id`` too,
so the seed, its close and every request in between name the session by the same ID.
A resources request with no session ID gets a new ID, which this router mints.
A session start, such as a seed, claims it before its handler runs, since its reply's cookie names the session.
Any other request claims it only if the handler used the session, before the reply leaves the worker.
A request that keeps nothing, such as a health check or ``/aggregate_metrics``, leaves no claim and gets no cookie,
so no cookie names a session nobody claims.
A request whose cookie names a session whose claim ended is claimed the same way by the worker that serves it.

A claim is a file in the socket directory, named by a hash of the session ID and holding the claimant's worker ID.
The worker that creates the session creates the claim, atomically, so two workers can never claim one ID.
Seeds and closes create claims, so a retried seed, a cleanup close, or a delayed seed after a close
all reach the worker that saw the ID first.
A claim ends once its session's last record on the claimant is gone:

- an agent session's claim ends when the agent drops the session's last record,
  which is its close receipt or failed seed, ``session_close_retry_window_seconds`` after it was written;
- a resources session's claim lasts while the session is open, from a successful seed until its close or ``/verify``,
  and ends ``session_close_retry_window_seconds`` after the last request for it otherwise.
  A ``/verify`` that says a close follows, as an environment server's does, leaves the session open for that close.

The claims of a worker that exited stay until the server exits, so its sessions keep getting a 410.
The whole directory goes with the server's runtime directory when the server exits.
With one worker there is no private socket and no claim.

A request without a session ID that belongs to one worker, such as an agent's call to itself within one ``/run``,
names that worker in the ``x-ng-session-owner`` header, and is forwarded there, or gets a 410 if that worker is gone.

A partial-rollout checkpoint restore installs each restored session on one live worker,
which rewrites the session's claim to name itself (see ``SessionClaims.rewrite``).
A cookie or MCP token from before the restore names only the session ID, so its requests reach the new holder.
"""

import asyncio
import fcntl
import hashlib
import heapq
import json
import logging
import math
import os
import re
import secrets
from base64 import b64decode
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
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

from nemo_gym.config.config_types import DEFAULT_UVICORN_BACKLOG, ROLLOUT_PATH_PREFIX
from nemo_gym.server_runtime.runtime_dir import server_runtime_dir


LOG = logging.getLogger(__name__)

#: Set by the main process of a multi-worker server for the workers it spawns: the directory that holds the
#: workers' private sockets and the session claims.
SESSION_SOCKET_DIR_ENV = "NEMO_GYM_SESSION_SOCKET_DIR"
#: Subdirectory of the socket directory that holds the session claims.
SESSION_CLAIMS_DIR = "session-claims"
#: Scope key with the session ID the router chose for a request whose cookie has none:
#: a typed seed's or close's ID, or a new one. The session middleware gives the cookie session this ID.
SESSION_ID_SCOPE_KEY = "nemo_gym.session_id"
#: Scope key with a callable that claims the request's unclaimed session ID for this worker, and says whether it could:
#: a new ID, or one a cookie names whose claim ended.
#: The session middleware calls it once the handler has used the session.
SESSION_CLAIM_SCOPE_KEY = "nemo_gym.claim_session"
#: App state key with the lifespans, each called with the app, that a worker enters once its private socket listens,
#: before it serves any request, and leaves before the socket closes.
#: A checkpoint coordinator's link is one: the coordinator may place restored sessions on the worker once it joins.
ROUTED_LIFESPANS_STATE = "nemo_gym_routed_lifespans"
#: Header naming the worker a request without a session belongs to.
SESSION_OWNER_HEADER = "x-ng-session-owner"
# How many claims a batched rewrite replaces per hold of the workers' claims lock,
# so a release on any worker waits for at most one batch.
_REWRITE_BATCH = 256
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
_ROLLOUT_PREFIX = rf"(?:/{ROLLOUT_PATH_PREFIX}/[^/]+(?:/[^/]+)?)?"
# How often a worker releases the claims whose session has ended, at most.
_MAX_SWEEP_SECONDS = 30.0
_MIN_SWEEP_SECONDS = 0.05


def _paths(paths: tuple[str, ...]) -> re.Pattern[str]:
    """Match any of *paths*, bare or under a rollout prefix."""
    return re.compile(_ROLLOUT_PREFIX + "(?:" + "|".join(re.escape(path) for path in paths) + ")")


@dataclass(frozen=True)
class SessionKind:
    """Where one kind of server's requests name their session."""

    #: The key of the session ID in the session cookie.
    cookie_key: str
    #: The body field that names the session on the seed and close routes.
    id_field: str
    #: The seed and close routes, whose body names the session.
    seed_path: str
    close_path: str
    #: Whether the session ID is the cookie session's own ID, as on a resources server:
    #: a typed seed's ID becomes it, and a request without one gets a new ID,
    #: claimed if the request starts a session or its handler uses the session.
    #: An agent session is named by a key of its own in the cookie instead.
    cookie_session: bool

    @property
    def body_paths(self) -> re.Pattern[str]:
        return _paths((self.seed_path, self.close_path))


def create_session_socket_dir() -> str:
    """The directory for one server's private worker sockets and claims: this process's runtime directory."""
    return server_runtime_dir()


def worker_socket_path(socket_dir: str, routing_id: str) -> str:
    return os.path.join(socket_dir, f"{routing_id}.sock")


def _valid_routing_id(owner: Any) -> Optional[str]:
    return owner if isinstance(owner, str) and _ROUTING_ID_PATTERN.fullmatch(owner) else None


def cookie_session(cookie: str, *, signer: itsdangerous.TimestampSigner) -> Optional[dict[str, Any]]:
    """Return the session stored in a session cookie, or None if Starlette's SessionMiddleware would reject it.

    Decoded exactly as SessionMiddleware decodes it.
    A cookie it rejects (badly signed or expired) starts a fresh session.
    """
    try:
        session = json.loads(b64decode(signer.unsign(cookie.encode("utf-8"), max_age=_SESSION_MAX_AGE_SECONDS)))
    except (itsdangerous.BadSignature, ValueError):
        return None
    return session if isinstance(session, dict) else None


def mcp_token_session_id(token: str, *, serializer: itsdangerous.URLSafeSerializer) -> Optional[str]:
    """Return the session ID in an MCP session token, or None for a token with a bad signature.

    A rejected token is handled locally, where the MCP endpoint's own token check refuses it.
    """
    try:
        payload = serializer.loads(token)
    except itsdangerous.BadSignature:
        return None
    session_id = payload.get("sid") if isinstance(payload, dict) else None
    return session_id if isinstance(session_id, str) else None


class SessionClaims:
    """Which worker created each session, shared by the workers of one server.

    A claim is a file in the claims directory, named by a hash of the session ID and holding the claimant's worker ID.
    It is created with a hard link from a complete temporary file, which fails if the claim exists,
    so two workers cannot both claim one ID and a reader never sees a partly written claim.

    A claim lives as long as ``holds`` says its claimant keeps anything of the session.
    It is not released while a request for the session is being served here,
    since that request may be about to create a record.
    Only the claimant removes its claim, so a removal cannot delete another worker's claim made since.
    A checkpoint restore can rewrite a claim for another worker at any time, though.
    The removal's check of the claimant and its unlink, and every rewrite, hold one lock file shared by the workers,
    so a removal never deletes a claim a rewrite has just written.
    """

    def __init__(self, socket_dir: str, routing_id: str, holds: Callable[[str], bool]) -> None:
        self.routing_id = routing_id
        self.holds = holds
        self.directory = os.path.join(socket_dir, SESSION_CLAIMS_DIR)
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        # Beside the claims, so a listing of the claims directory finds only claims.
        self._lock_path = f"{self.directory}.lock"
        # The IDs this worker has claimed and not released, and the requests for an ID being served here.
        # Only this worker removes its claims, so an ID in the set is claimed by this worker on disk too.
        self._mine: set[str] = set()
        self._serving: dict[str, int] = {}

    def _path(self, session_id: str) -> str:
        return os.path.join(self.directory, hashlib.sha256(session_id.encode("utf-8")).hexdigest())

    def claimant(self, session_id: str) -> Optional[str]:
        """The worker that claims the ID, or None if nobody does."""
        if session_id in self._mine:
            return self.routing_id
        try:
            with open(self._path(session_id), encoding="ascii") as claim:
                return _valid_routing_id(claim.read())
        except FileNotFoundError:
            return None

    def claim(self, session_id: str) -> Optional[str]:
        """Claim the ID for this worker unless another worker has; return the claimant.

        Returns None for a claim file that names no valid worker, which nothing here writes.
        """
        if session_id in self._mine:
            return self.routing_id
        path = self._path(session_id)
        while True:
            # Its own scratch file, so a restore rewriting the same ID in a thread here cannot truncate or remove it.
            temporary = f"{path}.{self.routing_id}.{secrets.token_hex(8)}.tmp"
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
            claimant = self.claimant(session_id)
            if claimant is not None or os.path.exists(path):
                return claimant
            # The claimant released it between the two calls: claim it again.

    def rewrite(self, session_id: str) -> None:
        """Claim the ID for this worker whoever claims it now, for a session a checkpoint restore installs here.

        The claim file is replaced by one rename, so a reader finds either the old claimant or this one.
        A live worker that claimed the ID before must ``disown`` it, since a worker trusts its own claims first.
        """
        self._replace_claims([session_id], [])
        self._mine.add(session_id)

    async def rewrite_all(self, session_ids: list[str]) -> None:
        """``rewrite`` each ID, in a thread, so a restore of many sessions does not hold up the event loop.

        The thread only replaces files, and holds the workers' claims lock once per ``_REWRITE_BATCH`` renames.
        The IDs this worker claims change only on the event loop, once the thread has ended,
        even if the caller was cancelled meanwhile.
        They then name exactly the claims the thread rewrote, so a release after a failure misses none of them.
        Until then this worker does not count them as its own, so a release here leaves them alone.
        The caller holds the sessions (see ``holds``) from before this call, so no release afterwards removes them.
        """
        replaced: list[str] = []
        replacing = asyncio.ensure_future(asyncio.to_thread(self._replace_claims, session_ids, replaced))
        cancelled = False
        while True:
            try:
                await asyncio.shield(replacing)
                break
            except asyncio.CancelledError:
                # The thread cannot be stopped: wait for it, then pass the cancellation on.
                cancelled = True
            except Exception:
                break
        self._mine.update(replaced)
        if cancelled:
            raise asyncio.CancelledError()
        replacing.result()

    def _replace_claims(self, session_ids: list[str], replaced: list[str]) -> None:
        """Replace the claim file of each ID with one naming this worker, and append each ID to ``replaced`` once done.

        It touches only files, so it can run in a thread.
        """
        for start in range(0, len(session_ids), _REWRITE_BATCH):
            batch: list[tuple[str, str, str]] = []
            try:
                for session_id in session_ids[start : start + _REWRITE_BATCH]:
                    path = self._path(session_id)
                    # Its own scratch file, so a claim of the same ID on this worker meanwhile cannot remove it.
                    temporary = f"{path}.{self.routing_id}.{secrets.token_hex(8)}.tmp"
                    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    batch.append((session_id, temporary, path))
                    try:
                        os.write(fd, self.routing_id.encode("ascii"))
                    finally:
                        os.close(fd)
                with self._exclusive():
                    for session_id, temporary, path in batch:
                        os.replace(temporary, path)
                        replaced.append(session_id)
            finally:
                # A batch cut short leaves no scratch file behind.
                for _, temporary, _ in batch:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass

    def disown(self, session_id: str) -> None:
        """Stop holding the ID here, and leave its claim file to the worker a checkpoint restore moved it to."""
        self._mine.discard(session_id)

    def mine(self, session_id: str) -> bool:
        return session_id in self._mine

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
        path = self._path(session_id)
        with self._exclusive():
            # A restore may have rewritten the claim for another worker before this worker disowned it.
            if self.claimant(session_id) != self.routing_id:
                return
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold the workers' claims lock; it is held only for a few file operations, never across an await."""
        fd = os.open(self._lock_path, os.O_RDONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            # Closing the file releases the lock.
            os.close(fd)

    def __len__(self) -> int:
        return len(self._mine)


class SessionLifetimes(Protocol):
    """How long a worker keeps the claims on the sessions it created."""

    def holds(self, session_id: str) -> bool:
        """Whether this worker still keeps any record of the session."""
        ...

    def served(self, session_id: str, scope: Scope, status: Optional[int]) -> None:
        """Record a request for a session this worker claims, once it is served; ``status`` is None if it failed."""
        ...

    def starts(self, path: str) -> bool:
        """Whether a request on ``path`` starts a session, so it claims its session whatever its handler does."""
        ...

    def sweep(self) -> list[str]:
        """Drop the records whose time has passed, and return the sessions whose claims can be released."""
        ...


class ResourcesSessionLifetimes:
    """How long a resources server's worker keeps the claim on a session, judged from the requests it served.

    A resources server keeps its per-session state in its own subclass, so the router judges by the requests:

    - A successful seed, or another declared session start such as a Gymnasium ``/reset``,
      opens the session: its claim lasts until the session ends.
    - A successful ``/close_session`` or ``/verify``, or ``SimpleResourcesServer.checkpoint_session_ended``,
      ends it, unless the request says a close follows, as an environment server's ``/verify`` does.
    - A session that is not open keeps its claim ``window_seconds`` after the last request for it.
      The caller's retried or cleanup close reaches the claimant within that window.
      A delayed seed after a close reaches it too, where the server refuses it while it remembers the close.
    - A failed seed or close of an open session leaves it open.

    So a claim outlives the session's last request by at most the window, except for a session never closed,
    which keeps its claim as long as the server keeps the session.
    """

    def __init__(
        self,
        window_seconds: float,
        *,
        starts: tuple[str, ...],
        ends: tuple[str, ...],
        close_follows_header: Optional[str] = None,
    ) -> None:
        self.window_seconds = window_seconds
        self._starts = _paths(starts)
        self._ends = _paths(ends)
        self._close_follows = close_follows_header.lower().encode("latin-1") if close_follows_header else None
        # Each session's claim lasts until this time, infinite while it is open.
        self._until: dict[str, float] = {}
        # One entry per session with a finite time, earliest first, so the heap never outgrows ``_until``.
        # An entry can be earlier than its session's time, which moved on; the sweep then queues it again.
        self._expiries: list[tuple[float, str]] = []
        self._queued: set[str] = set()
        # Sessions a checkpoint retire freed, whose claims last until ``ended`` whatever requests for them do.
        self._retained: set[str] = set()

    def holds(self, session_id: str) -> bool:
        return session_id in self._retained or self._until.get(session_id, 0.0) > monotonic()

    def served(self, session_id: str, scope: Scope, status: Optional[int]) -> None:
        path = scope["path"]
        succeeded = status is not None and 200 <= status < 300
        is_open = self._until.get(session_id) == math.inf
        if succeeded and self._starts.fullmatch(path):
            self._until[session_id] = math.inf
        elif is_open and not (succeeded and self._ends.fullmatch(path) and not self._close_follows_in(scope)):
            pass
        else:
            self._expire_after_window(session_id)

    def starts(self, path: str) -> bool:
        return self._starts.fullmatch(path) is not None

    def _close_follows_in(self, scope: Scope) -> bool:
        # A close that follows ends the session itself, possibly much later; until then its state stays here.
        return self._close_follows is not None and any(
            name.lower() == self._close_follows and value == b"1" for name, value in scope["headers"]
        )

    def ended(self, session_id: str) -> None:
        """End a session this worker claims, from a route that ends a session under another name."""
        retained = session_id in self._retained
        self._retained.discard(session_id)
        if retained or session_id in self._until:
            self._expire_after_window(session_id)

    def retain(self, session_id: str) -> None:
        """Keep the claim on a session a checkpoint retire freed until ``ended``, when the retire's refusal ends.

        Until then a late request that carries only the session's cookie reaches this worker, which refuses it.
        """
        self._retained.add(session_id)

    def opened(self, session_id: str) -> None:
        """Open a session a checkpoint restore installed on this worker, as a successful seed would."""
        self._until[session_id] = math.inf

    def forget(self, session_id: str) -> None:
        """Drop a session whose install a failed restore undid, so its claim can be released at once."""
        self._until.pop(session_id, None)
        self._retained.discard(session_id)

    def _expire_after_window(self, session_id: str) -> None:
        until = monotonic() + self.window_seconds
        self._until[session_id] = until
        if session_id not in self._queued:
            self._queued.add(session_id)
            heapq.heappush(self._expiries, (until, session_id))

    def sweep(self) -> list[str]:
        now = monotonic()
        expired = []
        while self._expiries and self._expiries[0][0] <= now:
            _, session_id = heapq.heappop(self._expiries)
            until = self._until.get(session_id)
            if until is None or until == math.inf:
                # Opened again since: an end queues it again.
                self._queued.discard(session_id)
            elif until <= now:
                del self._until[session_id]
                self._queued.discard(session_id)
                expired.append(session_id)
            else:
                heapq.heappush(self._expiries, (until, session_id))
        return expired

    def __len__(self) -> int:
        return len(self._until.keys() | self._retained)


class AgentSessionHolder(Protocol):
    """An agent server, whose sessions are claimed by ``agent_session_id`` across workers."""

    def holds_agent_session(self, agent_session_id: str) -> bool:
        """Whether this worker still keeps any record of the session, open, failed or closed."""
        ...

    def use_agent_session_claims(self, claims: SessionClaims) -> None:
        """Keep the claims, to release one when the worker drops the session's last record."""
        ...

    def expire_agent_session_records(self) -> None:
        """Drop the close receipts and failed seeds whose window has passed, releasing their claims."""
        ...


class _AgentSessionLifetimes:
    """An agent's own records decide its claims: it releases a claim itself when it drops a session's last record."""

    def __init__(self, agent: AgentSessionHolder) -> None:
        self.agent = agent

    def holds(self, session_id: str) -> bool:
        return self.agent.holds_agent_session(session_id)

    def served(self, session_id: str, scope: Scope, status: Optional[int]) -> None:
        pass

    def starts(self, path: str) -> bool:
        # An agent session starts at its seed, whose body names it.
        return False

    def sweep(self) -> list[str]:
        self.agent.expire_agent_session_records()
        return []


class SessionRoutingMiddleware:
    """Find a request's session ID, and forward the request to the worker that claims the session.

    It sits outside SessionMiddleware and reads the cookie by decoding it with the same secret.
    A forwarded request therefore never reaches this worker's SessionMiddleware,
    so the reply carries only the claimant's Set-Cookie, built from the session as the claimant left it.
    Sitting inside SessionMiddleware instead would add this worker's stale copy of the session as a second Set-Cookie.

    With one worker there are no claims, and it only reads the session ID of seeds and closes:
    it refuses a request whose cookie and body name different sessions,
    and gives a typed resources session its ID in the cookie.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        kind: SessionKind,
        session_cookie: str,
        secret_key: str,
        routing_id: Optional[str] = None,
        socket_dir: Optional[str] = None,
        clients: Optional[dict[str, ClientSession]] = None,
        claims: Optional[SessionClaims] = None,
        lifetimes: Optional[SessionLifetimes] = None,
        mcp_token_header: Optional[str] = None,
        mcp_token_serializer: Optional[itsdangerous.URLSafeSerializer] = None,
        mcp_path: str = "/mcp",
    ) -> None:
        self.app = app
        self.kind = kind
        self.body_paths = kind.body_paths
        self.session_cookie = session_cookie
        self.signer = itsdangerous.TimestampSigner(secret_key)
        self.routing_id = routing_id
        self.socket_dir = socket_dir
        # One connection pool per claimant socket, kept for the life of this worker and closed at its shutdown.
        # Not Gym's shared aiohttp client: each pool needs a UnixConnector bound to its claimant's socket.
        self.clients = clients if clients is not None else {}
        self.claims = claims
        self.lifetimes = lifetimes
        self.mcp_token_header = mcp_token_header
        self.mcp_token_serializer = mcp_token_serializer
        self.mcp_path = mcp_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        claims = self.claims
        names_by_body = scope["method"] == "POST" and self.body_paths.fullmatch(scope["path"]) is not None
        if claims is None and not names_by_body:
            await self.app(scope, receive, send)
            return

        # 1. Find the session ID: from the cookie or MCP token, else from the body of a seed or close.
        session_id = self._session_id_from_headers(HTTPConnection(scope))
        body = None
        if names_by_body:
            # The whole body is read, whatever its size, since the app would read all of it too.
            # A seed with large task data still has to reach the worker that holds its session.
            body = await _read_body(receive)
            if body is None:
                return
            receive = _replay_body(body, receive)
            body_session_id = _session_id(body, self.kind.id_field)
            if body_session_id is not None:
                if session_id is not None and session_id != body_session_id:
                    await _conflict(
                        f"This request's session cookie names another session than its {self.kind.id_field}. "
                        "Do not send another session's cookie with it.",
                        scope,
                        receive,
                        send,
                    )
                    return
                if session_id is None and self.kind.cookie_session:
                    scope[SESSION_ID_SCOPE_KEY] = body_session_id
                session_id = body_session_id
        if claims is None:
            await self.app(scope, receive, send)
            return

        # 2. Look up the session's claim. A seed, a close or another session start creates it if nobody has.
        forwarded = bool(scope.get(_FORWARDED_SCOPE_KEY))
        starts = self.lifetimes is not None and self.lifetimes.starts(scope["path"])
        if session_id is None:
            owner = _valid_routing_id(HTTPConnection(scope).headers.get(SESSION_OWNER_HEADER))
            if owner is not None and owner != self.routing_id and not forwarded:
                # A request that belongs to one worker without a session, such as an agent's call to itself.
                await self._forward(owner, scope, receive, send, body=body)
                return
            if not self.kind.cookie_session:
                await self.app(scope, receive, send)
                return
            new_session_id = scope[SESSION_ID_SCOPE_KEY] = str(uuid4())
            if starts:
                # A session start's reply names the new session, whether or not its handler reads it.
                if claims.claim(new_session_id) != self.routing_id:
                    await _conflict("This new session's ID is already claimed. Retry it.", scope, receive, send)
                    return
            else:
                # Claimed by the session middleware only if the handler uses it; otherwise no cookie names it.
                scope[SESSION_CLAIM_SCOPE_KEY] = lambda: claims.claim(new_session_id) == self.routing_id
            await self._serve_claimed(claims, new_session_id, scope, receive, send)
            return
        elif names_by_body or starts:
            claimant = claims.claim(session_id)
        elif forwarded:
            # Never forwarded again; served here as a session of this worker only if it still claims it.
            claimant = self.routing_id if claims.mine(session_id) else None
        else:
            claimant = claims.claimant(session_id)

        # 3. Forward it to the claimant, or handle it here.
        if claimant == self.routing_id:
            await self._serve_claimed(claims, session_id, scope, receive, send)
        elif claimant is None and self.kind.cookie_session and not forwarded:
            # A cookie or token whose claim ended, or was never made: the session is claimed here if the handler uses it,
            # so its later requests reach this worker.
            scope[SESSION_ID_SCOPE_KEY] = session_id
            scope[SESSION_CLAIM_SCOPE_KEY] = lambda: claims.claim(session_id) == self.routing_id
            await self._serve_claimed(claims, session_id, scope, receive, send)
        elif claimant is None:
            await self.app(scope, receive, send)
        elif forwarded:
            # The claimant released the ID while the request was forwarded here, and another worker claimed it since.
            await _conflict(
                f"This {self.kind.id_field} moved to another server worker while the request was forwarded. Retry it.",
                scope,
                receive,
                send,
            )
        else:
            await self._forward(claimant, scope, receive, send, body=body)

    async def _serve_claimed(
        self, claims: SessionClaims, session_id: str, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Serve a request for a session this worker claims, or may claim, and record it for the claim's lifetime."""
        status: list[int] = []

        async def send_and_record(message: Message) -> None:
            if message["type"] == "http.response.start":
                status.append(message["status"])
            await send(message)

        with claims.serving(session_id):
            try:
                await self.app(scope, receive, send_and_record)
            finally:
                if self.lifetimes is not None and claims.mine(session_id):
                    self.lifetimes.served(session_id, scope, status[0] if status else None)

    def _session_id_from_headers(self, connection: HTTPConnection) -> Optional[str]:
        cookie = connection.cookies.get(self.session_cookie)
        token = connection.headers.get(self.mcp_token_header) if self.mcp_token_serializer is not None else None
        path = connection.scope["path"]
        is_mcp = path == self.mcp_path or path.startswith(self.mcp_path + "/")
        # MCP tool calls name their session by the token, never by a cookie.
        if token is not None and (is_mcp or cookie is None):
            return mcp_token_session_id(token, serializer=self.mcp_token_serializer)
        session = cookie_session(cookie, signer=self.signer) if cookie is not None else None
        session_id = session.get(self.kind.cookie_key) if session is not None else None
        return session_id if isinstance(session_id, str) else None

    def _client(self, owner: str) -> ClientSession:
        client = self.clients.get(owner)
        if client is None:
            client = self.clients[owner] = ClientSession(
                connector=UnixConnector(
                    worker_socket_path(self.socket_dir, owner), limit=0, keepalive_timeout=_FORWARD_KEEPALIVE_SECONDS
                ),
                cookie_jar=DummyCookieJar(),
                # Pass the reply's bytes through exactly as the claimant encoded them.
                auto_decompress=False,
                # The claimant's handler takes as long as it takes; the caller holds its own timeout.
                timeout=ClientTimeout(total=None),
            )
        return client

    async def _forward(
        self, owner: str, scope: Scope, receive: Receive, send: Send, *, body: Optional[bytes] = None
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
                data=body,
                allow_redirects=False,
                # Send only the caller's headers.
                skip_auto_headers=("User-Agent", "Accept", "Accept-Encoding", "Content-Type"),
            )
        except ClientConnectorError:
            # The claimant's socket is gone or refuses connections: the worker that held this session exited.
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


async def _conflict(detail: str, scope: Scope, receive: Receive, send: Send) -> None:
    await JSONResponse({"detail": detail}, status_code=409)(scope, receive, send)


async def _read_body(receive: Receive) -> Optional[bytes]:
    """Read the request body, or None if the client disconnected."""
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


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


def _session_id(body: bytes, field: str) -> Optional[str]:
    """The session ID in ``field`` of a seed or close body, or None for a body that has no non-empty string one."""
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    value = payload.get(field) if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


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
async def serve_private_socket(
    app: ASGIApp, path: str, *, backlog: int = DEFAULT_UVICORN_BACKLOG
) -> AsyncIterator[None]:
    """Serve *app* on a Unix socket for requests other workers forward here, on the running event loop."""
    config = uvicorn.Config(
        _mark_forwarded(app),
        uds=path,
        # Other workers forward to this socket without a connection limit, so a burst can open thousands at once.
        backlog=backlog,
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


async def _sweep_claims(claims: SessionClaims, lifetimes: SessionLifetimes, interval_seconds: float) -> None:
    """Release the claims whose session has ended, so a claim ends even when no more requests arrive."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            for session_id in lifetimes.sweep():
                claims.release(session_id)
        except Exception:
            # A failed sweep is tried again at the next interval rather than ending every later one.
            LOG.exception("Could not release the claims of ended sessions")


def install_session_routing(
    app: FastAPI,
    *,
    kind: SessionKind,
    socket_dir: Optional[str],
    session_cookie: str,
    secret_key: str,
    mcp_token_header: Optional[str] = None,
    mcp_token_serializer: Optional[itsdangerous.URLSafeSerializer] = None,
    agent_sessions: Optional[AgentSessionHolder] = None,
    close_window_seconds: Optional[float] = None,
    session_starts: tuple[str, ...] = (),
    session_ends: tuple[str, ...] = (),
    close_follows_header: Optional[str] = None,
    backlog: int = DEFAULT_UVICORN_BACKLOG,
) -> Optional[str]:
    """Route this worker's requests by session ID, and serve forwarded requests on a private socket.

    Call once the app's middleware is otherwise in place: the router must sit outside SessionMiddleware.
    Without a ``socket_dir`` (one worker), it only reads the session ID of seeds and closes, and claims nothing.
    With one, it claims sessions in it and returns this worker's ID, also stored in ``app.state.nemo_gym_routing_id``.
    Pass the agent for an agent server, whose own records decide its claims' lifetimes,
    or the close window and the session start and end routes for a resources server (see ``ResourcesSessionLifetimes``).
    """
    route_by = dict(
        kind=kind,
        session_cookie=session_cookie,
        secret_key=secret_key,
        mcp_token_header=mcp_token_header,
        mcp_token_serializer=mcp_token_serializer,
    )
    if socket_dir is None:
        # An agent's own seed handler already refuses a cookie of another session, so it needs nothing here.
        if kind.cookie_session:
            app.add_middleware(SessionRoutingMiddleware, **route_by)
        return None

    routing_id = uuid4().hex
    app.state.nemo_gym_routing_id = routing_id
    clients: dict[str, ClientSession] = {}
    if agent_sessions is not None:
        lifetimes: SessionLifetimes = _AgentSessionLifetimes(agent_sessions)
        window = None
    elif close_window_seconds is not None:
        resources_lifetimes = ResourcesSessionLifetimes(
            close_window_seconds,
            starts=(kind.seed_path, *session_starts),
            ends=(kind.close_path, *session_ends),
            close_follows_header=close_follows_header,
        )
        app.state.nemo_gym_resources_session_lifetimes = resources_lifetimes
        lifetimes = resources_lifetimes
        window = close_window_seconds
    else:
        raise ValueError("install_session_routing needs agent_sessions or close_window_seconds with a socket_dir")
    claims = SessionClaims(socket_dir, routing_id, lifetimes.holds)
    app.state.nemo_gym_session_claims = claims
    if agent_sessions is not None:
        agent_sessions.use_agent_session_claims(claims)
        window = getattr(getattr(agent_sessions, "config", None), "session_close_retry_window_seconds", None)
    sweep_seconds = min(_MAX_SWEEP_SECONDS, max(_MIN_SWEEP_SECONDS, (window or _MAX_SWEEP_SECONDS) / 4))
    app.add_middleware(
        SessionRoutingMiddleware,
        **route_by,
        routing_id=routing_id,
        socket_dir=socket_dir,
        clients=clients,
        claims=claims,
        lifetimes=lifetimes,
    )
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan_with_private_socket(application: FastAPI) -> AsyncIterator[Any]:
        async with original_lifespan(application) as state:
            # Started after the app's own startup, so a forwarded request never reaches a half-started app.
            async with serve_private_socket(application, worker_socket_path(socket_dir, routing_id), backlog=backlog):
                sweeper = asyncio.create_task(_sweep_claims(claims, lifetimes, sweep_seconds))
                try:
                    async with AsyncExitStack() as routed:
                        for lifespan in getattr(application.state, ROUTED_LIFESPANS_STATE, ()):
                            await routed.enter_async_context(lifespan(application))
                        yield state
                finally:
                    sweeper.cancel()
                    await asyncio.gather(sweeper, return_exceptions=True)
                    await asyncio.gather(*(client.close() for client in clients.values()))
                    clients.clear()

    app.router.lifespan_context = lifespan_with_private_socket
    return routing_id
