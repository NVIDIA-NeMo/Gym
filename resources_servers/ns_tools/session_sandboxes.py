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
"""One OpenSandbox sandbox per rollout session, created on the session's first tool
call and deleted as soon as the session ends.

This is the isolation-first alternative to :mod:`sandbox_pool` (shared long-lived pods
with sessions multiplexed by sticky routing). Nothing is shared between rollouts: no
CPU/memory contention from neighbours, no leftover processes or files, and the pod's
requests/limits describe exactly one Python session. The price is one sandbox create
per rollout (seconds) which the create concurrency limit keeps from storming the API.

Lifecycle (``key`` = the nemo-gym session id of the rollout, see ``session_context``):

* ``route(session_id)``   first call for a key creates the sandbox (bounded by
  ``create_concurrency``), later calls reuse it. Concurrent first calls share one
  creation task.
* ``end_session(key)``    called by ``app.py`` when the rollout is verified -> the
  sandbox is deleted. This is the normal end of life.
* ``report_failure(sid)`` after a transport failure the sandbox is health-probed and
  deleted if dead, so the next tool call recreates it instead of failing forever.
* idle sweep             sessions whose rollout never reached ``/verify`` (crash,
  ``skip_verification``) are deleted after ``session_idle_timeout_s``; the provider
  ``ttl_s`` (expireTime) is the cluster-side backstop for a dead server.
* ``aclose()``           deletes every live sandbox at server shutdown.

Only imported when ns_tools selects ``sandbox_type: sandbox_per_session``.
"""

import asyncio
import logging
import shlex
import time
from dataclasses import dataclass, field
from typing import Any

import httpx  # exception types only: the nemo_skills client contract catches httpx errors
from sandbox_pool import sandbox_request
from session_context import current_session_id

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec


LOGGER = logging.getLogger(__name__)

# Requests without any session (no rollout context, no IPython session id) share one
# lazily created sandbox; it follows the same idle/shutdown rules as the others.
STATELESS_KEY = "__stateless__"

# Pod labels every per-session sandbox carries. ``nemo.nvidia.com/resources: custom``
# exempts the pod from the cluster's request-clamping admission policy so the
# ``resource_requests`` below are honoured; config ``metadata`` keys override these.
DEFAULT_METADATA = {
    "purpose": "ns-tools-per-session",
    "nemo.nvidia.com/resources": "custom",
}


def _as_bool(value: Any) -> bool:
    # bool("false") is True; env-fed values arrive as strings.
    return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")


@dataclass
class _Session:
    key: str
    task: "asyncio.Task[AsyncSandbox]"
    created_at: float
    last_used: float
    sandbox: AsyncSandbox | None = None
    ipython_sessions: set[str] = field(default_factory=set)


class SessionSandboxes:
    """One directly created (or pool-claimed) sandbox per rollout session.

    The constructor is pure (validation only). ``start()`` launches the idle sweep;
    ``route()`` lazily starts it as a safety net. The public surface mirrors
    :class:`sandbox_pool.SandboxPool` so :class:`gym_sandbox.GymSandbox` is backend-agnostic.
    """

    def __init__(
        self,
        *,
        provider: dict[str, Any],
        image: str,
        pool_ref: str = "",
        pool_fallback: bool = True,
        port: int = 6000,
        ttl_s: float | None = 7200.0,
        env: dict[str, Any] | None = None,
        entrypoint: list[str] | None = None,
        resources: dict[str, Any] | None = None,
        resource_requests: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        setup_files: dict[str, str] | None = None,
        setup_commands: list[str] | None = None,
        service_command: str | None = None,
        health_path: str = "/health",
        ready_timeout_s: float = 120.0,
        health_budget_s: float = 300.0,
        health_timeout_s: float = 10.0,
        create_concurrency: int = 64,
        delete_concurrency: int = 64,
        stop_timeout_s: float = 60.0,
        session_idle_timeout_s: float = 3600.0,
        session_max_lifetime_s: float | None = None,
        sweep_interval_s: float = 60.0,
        size: int | None = None,
    ) -> None:
        if size is not None:
            raise ValueError(
                "sandbox_per_session creates one sandbox per session; there is no fixed pool — remove 'size' "
                "(NS_SANDBOX_POOL_SIZE) or use sandbox_type: sandbox_pool"
            )
        if not isinstance(provider, dict) or set(provider) != {"opensandbox"}:
            raise ValueError("sandbox_per_session.provider must contain exactly one 'opensandbox' provider")
        provider_config = provider["opensandbox"] or {}
        if not isinstance(provider_config, dict):
            raise TypeError("sandbox_per_session.provider.opensandbox must be a mapping")
        connection = provider_config.get("connection") or {}
        if not connection.get("domain") or not connection.get("api_key"):
            raise ValueError(
                "sandbox_per_session backend selected but the provider connection has an empty "
                "domain or api_key — set OPENSANDBOX_BASE_URL / OPENSANDBOX_API_KEY"
            )
        if not image:
            raise ValueError("sandbox_per_session backend selected but image is empty — set NS_SANDBOX_IMAGE")
        if int(create_concurrency) < 1 or int(delete_concurrency) < 1:
            raise ValueError("create_concurrency and delete_concurrency must be >= 1")
        self._provider = provider
        self._image = image
        self._pool_ref = str(pool_ref or "")
        self._pool_fallback = _as_bool(pool_fallback)
        self._port = int(port)
        self._ttl_s = float(ttl_s) if ttl_s else None
        self._env = dict(env or {})
        self._entrypoint = list(entrypoint) if entrypoint else None
        self._resources = dict(resources or {})
        self._resource_requests = dict(resource_requests or {})
        self._metadata = {**DEFAULT_METADATA, **{str(k): str(v) for k, v in (metadata or {}).items()}}
        self._setup_files = dict(setup_files or {})
        self._setup_commands = list(setup_commands or [])
        self._service_command = service_command
        if (not self._pool_ref or self._pool_fallback) and not (self._entrypoint or self._service_command):
            raise ValueError(
                "sandbox_per_session direct creation requires entrypoint or service_command to start the NS server"
            )
        self._health_path = health_path
        self._ready_timeout_s = float(ready_timeout_s)
        self._health_budget_s = float(health_budget_s)
        self._health_timeout_s = float(health_timeout_s)
        self._create_slots = asyncio.Semaphore(int(create_concurrency))
        self._delete_slots = asyncio.Semaphore(int(delete_concurrency))
        self._stop_timeout_s = float(stop_timeout_s)
        self._session_idle_timeout_s = float(session_idle_timeout_s) if session_idle_timeout_s else None
        self._session_max_lifetime_s = float(session_max_lifetime_s) if session_max_lifetime_s else None
        self._sweep_interval_s = float(sweep_interval_s)

        self._sessions: dict[str, _Session] = {}
        self._ipython_to_key: dict[str, str] = {}
        self._started = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._sweep_task: asyncio.Task[None] | None = None
        self.created = 0
        self.create_failures = 0
        self.deleted = 0

    # ------------------------------------------------------------------ lifecycle

    @property
    def port(self) -> int:
        return self._port

    @property
    def live_count(self) -> int:
        return len(self._sessions)

    # SandboxPool exposes ``ready_count`` for logging; keep the name.
    ready_count = live_count

    async def start(self) -> None:
        """Launch the idle sweep; sandboxes themselves are created on first use."""
        if self._started or self._closed:
            return
        self._started = True
        if self._session_idle_timeout_s or self._session_max_lifetime_s:
            self._sweep_task = asyncio.create_task(self._sweep_loop(), name="osb-session-sweep")

    async def aclose(self) -> None:
        """Delete every live sandbox. Safe to call more than once; survives caller cancellation."""
        if self._close_task is None:
            self._closed = True

            async def cleanup() -> None:
                if self._sweep_task is not None:
                    self._sweep_task.cancel()
                    await asyncio.gather(self._sweep_task, return_exceptions=True)
                    self._sweep_task = None
                keys = list(self._sessions)
                if keys:
                    LOGGER.info("session sandboxes: deleting %d live sandbox(es) at shutdown", len(keys))
                await asyncio.gather(*(self._end(key, reason="shutdown") for key in keys))
                LOGGER.info(
                    "session sandboxes closed: created=%d create_failures=%d deleted=%d",
                    self.created,
                    self.create_failures,
                    self.deleted,
                )

            self._close_task = asyncio.create_task(cleanup())
        cancellation = None
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError as exc:
                if self._close_task.cancelled():
                    raise
                cancellation = exc
        if cancellation is not None:
            raise cancellation

    # ------------------------------------------------------------------ keys

    def _key_for(self, session_id: str | None) -> str:
        rollout = current_session_id.get()
        if rollout:
            return str(rollout)
        if session_id is not None:
            return str(session_id)
        return STATELESS_KEY

    def has_session(self, session_id: str) -> bool:
        return str(session_id) in self._ipython_to_key

    async def sandbox_for(self, session_id: str) -> AsyncSandbox | None:
        """The live sandbox holding an IPython session, or None (never creates one)."""
        key = self._ipython_to_key.get(str(session_id))
        if key is None:
            return None
        entry = self._sessions.get(key)
        if entry is None or not entry.task.done() or entry.task.cancelled() or entry.task.exception() is not None:
            return None
        return entry.task.result()

    def release(self, session_id: str) -> None:
        """Forget an IPython session id (the sandbox itself lives until ``end_session``)."""
        key = self._ipython_to_key.pop(str(session_id), None)
        if key is not None:
            entry = self._sessions.get(key)
            if entry is not None:
                entry.ipython_sessions.discard(str(session_id))

    # ------------------------------------------------------------------ routing

    async def route(self, session_id: str | None) -> AsyncSandbox:
        """Resolve (creating on first use) the sandbox for the current rollout session.

        Raises httpx.TimeoutException when the sandbox cannot be created, which the NS
        client collapses into its timeout contract — a sandbox outage degrades rewards,
        never the server.
        """
        if self._closed:
            raise httpx.TimeoutException("session sandboxes are closed")
        if not self._started:
            await self.start()
        key = self._key_for(session_id)
        now = time.monotonic()
        entry = self._sessions.get(key)
        if entry is None:
            entry = _Session(
                key=key,
                task=asyncio.create_task(self._create(key), name=f"osb-session-create-{key[:8]}"),
                created_at=now,
                last_used=now,
            )
            self._sessions[key] = entry
        entry.last_used = now
        if session_id is not None:
            entry.ipython_sessions.add(str(session_id))
            self._ipython_to_key[str(session_id)] = key
        try:
            # shield: a cancelled tool request must not cancel the creation other
            # requests of the same session are waiting on.
            sandbox = await asyncio.shield(entry.task)
        except asyncio.CancelledError:
            if entry.task.cancelled():
                raise httpx.TimeoutException(f"session sandbox for {key[:8]} was ended during creation")
            raise
        except Exception as exc:
            if self._sessions.get(key) is entry:
                self._forget(entry)
            raise httpx.TimeoutException(f"session sandbox create failed for {key[:8]}: {exc!r}") from exc
        if self._sessions.get(key) is not entry:
            # end_session()/aclose() ran while we were waiting: the sandbox is gone or going.
            raise httpx.TimeoutException(f"session sandbox for {key[:8]} was ended during creation")
        entry.sandbox = sandbox
        return sandbox

    async def report_failure(self, session_id: str | None) -> None:
        """After a transport failure: probe the sandbox and delete it if it is dead, so
        the next tool call gets a fresh one (state is lost; nemo_skills replays history
        unless disable_session_restore)."""
        key = self._key_for(session_id)
        entry = self._sessions.get(key)
        if entry is None or not entry.task.done() or entry.task.cancelled() or entry.task.exception() is not None:
            return
        sandbox = entry.task.result()
        try:
            status, _ = await sandbox_request(
                sandbox, self._port, "GET", self._health_path, timeout_s=self._health_timeout_s
            )
            healthy = status == 200
        except httpx.TimeoutException:
            healthy = False
        if healthy:
            return
        LOGGER.warning("session sandbox %s unhealthy after a transport failure — deleting it", key[:8])
        await self._end(key, reason="unhealthy")

    async def end_session(self, key: str | None) -> None:
        """Delete the sandbox of a finished rollout. Unknown keys are a no-op."""
        if key is None:
            return
        await self._end(str(key), reason="session end")

    # ------------------------------------------------------------------ create / delete

    def _spec(self, claim: bool) -> SandboxSpec:
        if claim:
            return SandboxSpec(
                image=self._image,
                metadata=dict(self._metadata),
                ttl_s=self._ttl_s or 7200.0,
                ready_timeout_s=self._ready_timeout_s,
                provider_options={"extensions": {"poolRef": self._pool_ref}},
            )
        return SandboxSpec(
            image=self._image,
            entrypoint=self._entrypoint,
            env=dict(self._env),
            metadata=dict(self._metadata),
            resources=dict(self._resources),
            provider_options={"resource_requests": dict(self._resource_requests)} if self._resource_requests else {},
            ttl_s=self._ttl_s or 7200.0,
            ready_timeout_s=self._ready_timeout_s,
        )

    async def _acquire_sandbox(self) -> tuple[AsyncSandbox, bool]:
        """Returns (sandbox, from_pool): claim a prewarmed pod when ``pool_ref`` is set,
        otherwise (or on claim failure with ``pool_fallback``) create one directly."""
        if self._pool_ref:
            sandbox = AsyncSandbox(self._provider)
            try:
                await sandbox.start(self._spec(claim=True))
                return sandbox, True
            except Exception as exc:
                if not self._pool_fallback:
                    raise
                LOGGER.warning("pool %r allocation failed (%s); falling back to a direct create", self._pool_ref, exc)
        sandbox = AsyncSandbox(self._provider)
        await sandbox.start(self._spec(claim=False))
        return sandbox, False

    async def _create(self, key: str) -> AsyncSandbox:
        started = time.monotonic()
        async with self._create_slots:
            queued_s = time.monotonic() - started
            try:
                sandbox, from_pool = await self._acquire_sandbox()
            except BaseException:
                self.create_failures += 1
                raise
            try:
                if not from_pool:
                    for target_path, local_path in self._setup_files.items():
                        await sandbox.upload(local_path, target_path)
                    for command in self._setup_commands:
                        execution = await sandbox.exec(command)
                        if execution.return_code != 0:
                            raise RuntimeError(f"setup command failed rc={execution.return_code}: {command!r}")
                    if self._service_command:
                        # Detach from execd command cleanup while retaining its selected shell.
                        execution = await sandbox.exec(f'setsid "$0" -c {shlex.quote(self._service_command)}')
                        if execution.return_code != 0:
                            raise RuntimeError(f"service command failed rc={execution.return_code}")
                await self._wait_healthy(sandbox)
            except BaseException:
                self.create_failures += 1
                await self._stop_sandbox(sandbox, key)
                raise
        self.created += 1
        LOGGER.info(
            "session sandbox ready key=%s create_s=%.1f (queued %.1f) live=%d created=%d deleted=%d",
            key[:8],
            time.monotonic() - started,
            queued_s,
            len(self._sessions),
            self.created,
            self.deleted,
        )
        return sandbox

    async def _wait_healthy(self, sandbox: AsyncSandbox) -> None:
        """Gate admission on the same exec path used by tool requests."""
        deadline = time.monotonic() + self._health_budget_s
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                status, _ = await sandbox_request(
                    sandbox, self._port, "GET", self._health_path, timeout_s=self._health_timeout_s
                )
                if status == 200:
                    return
                last_error = f"HTTP {status}"
            except httpx.TimeoutException as exc:
                last_error = repr(exc)
            await asyncio.sleep(1.0)
        raise RuntimeError(f"sandbox never became healthy through exec: {last_error}")

    async def _stop_sandbox(self, sandbox: AsyncSandbox, key: str) -> None:
        async with self._delete_slots:
            try:
                await asyncio.wait_for(sandbox.stop(), timeout=self._stop_timeout_s)
            except Exception as exc:
                LOGGER.warning("session sandbox %s teardown failed (TTL will reap): %s", key[:8], exc)

    def _forget(self, entry: _Session) -> None:
        self._sessions.pop(entry.key, None)
        for sid in entry.ipython_sessions:
            if self._ipython_to_key.get(sid) == entry.key:
                self._ipython_to_key.pop(sid, None)

    async def _end(self, key: str, *, reason: str) -> None:
        entry = self._sessions.get(key)
        if entry is None:
            return
        self._forget(entry)
        if not entry.task.done():
            # In-flight create: let it finish (bounded) so the sandbox it produces is
            # deleted rather than leaked by a cancelled API call; cancel only if it hangs.
            try:
                sandbox = await asyncio.wait_for(asyncio.shield(entry.task), timeout=self._stop_timeout_s)
            except (asyncio.TimeoutError, TimeoutError):
                entry.task.cancel()
                await asyncio.gather(entry.task, return_exceptions=True)
                return
            except Exception:
                return  # the create failed on its own; nothing to delete
        elif entry.task.cancelled() or entry.task.exception() is not None:
            return
        else:
            sandbox = entry.task.result()
        await self._stop_sandbox(sandbox, key)
        self.deleted += 1
        LOGGER.info(
            "session sandbox deleted key=%s reason=%s lifetime_s=%.0f live=%d",
            key[:8],
            reason,
            time.monotonic() - entry.created_at,
            len(self._sessions),
        )

    # ------------------------------------------------------------------ maintenance

    async def _sweep_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._sweep_interval_s)
            now = time.monotonic()
            stale = [
                entry.key
                for entry in list(self._sessions.values())
                if entry.task.done()
                and (
                    (self._session_idle_timeout_s and now - entry.last_used > self._session_idle_timeout_s)
                    or (self._session_max_lifetime_s and now - entry.created_at > self._session_max_lifetime_s)
                )
            ]
            if stale:
                LOGGER.warning(
                    "session sandboxes: %d session(s) idle past %s s or over max lifetime — deleting "
                    "(rollouts that never reached /verify)",
                    len(stale),
                    self._session_idle_timeout_s,
                )
                await asyncio.gather(*(self._end(key, reason="idle") for key in stale), return_exceptions=True)
