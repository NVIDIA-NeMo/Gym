# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IdeGYM sandbox provider: sandboxes as IdeGYM server pods on Kubernetes.

One sandbox is one IdeGYM *server*: a Kubernetes pod provisioned through the IdeGYM
orchestrator, running the IdeGYM server image whose HTTP API the orchestrator forwards
requests to. This module composes the layers around it — :mod:`.session` (the shared
registered client), :mod:`.spec` (spec translation), :mod:`.errors` (failure
classification). Command context (``cwd``, ``env``, ``user``) and file bytes travel as
fields of IdeGYM's own requests, so nothing here builds shell text.

Capabilities deliberately not claimed: ``endpoint()``, because the orchestrator
forwards API requests rather than routing raw TCP; PTY sessions, because IdeGYM has no
terminal API; and ``serialize()`` / ``connect()``, because a server is only reachable
through the registered client that owns it.
"""

import asyncio
import logging
import math
import posixpath
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.idegym.config import (
    MAX_KEEPALIVE_MINUTES,
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymCreateConfig,
    IdeGymExecConfig,
    IdeGymFilesConfig,
    IdeGymOperationsConfig,
    IdeGymVerifyConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import (
    IdeGymCreateError,
    IdeGymCreateVerificationError,
    IdeGymOperationError,
    IdeGymTransferError,
    is_command_timeout,
    is_deadline_exceeded,
    is_invalid_request,
    is_retryable,
    is_sandbox_gone,
)
from nemo_gym.sandbox.providers.idegym.session import (
    IdeGymServerRef,
    IdeGymSession,
    acquire_session,
    release_session,
)
from nemo_gym.sandbox.providers.idegym.spec import IdeGymProviderOptions, ServerRequestTranslator
from nemo_gym.sandbox.providers.utils import coerce_config


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Returned by exec() when the sandbox runtime failed instead of the command. Same
# sentinel the other Gym providers use, so callers can treat it uniformly.
SANDBOX_RUNTIME_RETURN_CODE = 125

# IdeGYM's bash tool needs a finite, JSON-serializable command timeout, so "no
# timeout" is expressed as a ceiling long enough that no benchmark command reaches
# it before the sandbox itself is torn down.
NO_TIMEOUT_COMMAND_SECONDS = 24 * 60 * 60.0

# Not a start-server field, so the provider takes it out of provider_options itself.
KEEPALIVE_OPTION = "keepalive_minutes"


@dataclass
class _IdeGymSandbox:
    """Provider-owned state for one sandbox, carried in ``SandboxHandle.raw``."""

    server_id: int
    server_name: str
    image: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    # None opts this sandbox out of keepalive holds.
    keepalive_minutes: float | None = None
    # Epoch ms the orchestrator holds the server against its inactivity reaper until.
    held_until_ms: int = 0
    stopped: bool = False

    def __str__(self) -> str:
        return f"{self.server_name} (id={self.server_id})"


def _runtime_failure(message: str, *, error_type: str) -> SandboxExecResult:
    return SandboxExecResult(
        stdout=None,
        stderr=message,
        return_code=SANDBOX_RUNTIME_RETURN_CODE,
        error_type=error_type,
    )


def _now_ms() -> float:
    return time.time() * 1000


def _keepalive_option(value: Any) -> float | None:
    """Validate ``provider_options.keepalive_minutes`` the way ``create.keepalive_minutes`` is."""
    if value is None:
        return None
    minutes = float(value)
    if not 0 < minutes <= MAX_KEEPALIVE_MINUTES:
        raise ValueError(
            f"provider_options.{KEEPALIVE_OPTION} must be > 0 and <= {MAX_KEEPALIVE_MINUTES:g}, got {value!r}"
        )
    return minutes


class IdeGymProvider:
    """Sandbox provider backed by an IdeGYM orchestrator."""

    name = "idegym"

    def __init__(
        self,
        *,
        connection: IdeGymConnectionConfig | Mapping[str, Any] | None = None,
        create: IdeGymCreateConfig | Mapping[str, Any] | None = None,
        exec: IdeGymExecConfig | Mapping[str, Any] | None = None,
        verify: IdeGymVerifyConfig | Mapping[str, Any] | None = None,
        files: IdeGymFilesConfig | Mapping[str, Any] | None = None,
        operations: IdeGymOperationsConfig | Mapping[str, Any] | None = None,
        attribution: IdeGymAttributionConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = coerce_config(connection, IdeGymConnectionConfig)
        self._create = coerce_config(create, IdeGymCreateConfig)
        self._exec = coerce_config(exec, IdeGymExecConfig)
        self._verify = coerce_config(verify, IdeGymVerifyConfig)
        self._files = coerce_config(files, IdeGymFilesConfig)
        self._operations = coerce_config(operations, IdeGymOperationsConfig)
        self._attribution = coerce_config(attribution, IdeGymAttributionConfig)

        self._translator = ServerRequestTranslator(self._create)
        # The session registers a client with the orchestrator, which is async, so
        # it is acquired on first use rather than in this constructor.
        self._session: IdeGymSession | None = None
        self._session_lock = threading.Lock()
        self._closed = False

    # --- session -----------------------------------------------------------

    async def session(self) -> IdeGymSession:
        """Return this provider's shared orchestrator session, acquiring it once.

        A thread lock, not an ``asyncio.Lock``: one provider can be shared by two
        sandboxes, and the sync ``Sandbox`` facade runs each on its own event loop,
        which an ``asyncio.Lock`` cannot span without deadlocking.

        The acquire therefore happens outside the lock, so two callers can race. That
        is safe: ``acquire_session`` hands both of them the same session, and the loser
        gives its spare reference back.
        """
        if self._closed:
            raise IdeGymOperationError("This idegym provider has been closed")
        if self._session is not None:
            return self._session
        session = await acquire_session(self._connection, self._attribution)
        with self._session_lock:
            if self._session is None and not self._closed:
                self._session = session
                return session
            winner = self._session
        await release_session(session)
        if winner is None:
            raise IdeGymOperationError("This idegym provider has been closed")
        return winner

    async def health(self) -> str:
        """Return the orchestrator's health, registering the client if needed.

        Reaching this proves more than the endpoint being up: the session had to
        register a client to exist.
        """
        session = await self.session()
        return await session.health()

    # --- create ------------------------------------------------------------

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start an IdeGYM server for ``spec`` and return it ready to run commands.

        The orchestrator reports a server only once its pod passes the Kubernetes
        readiness probe against the IdeGYM server's own health endpoint, so what is
        left is the keepalive hold and ``spec.workdir``. A failure after provisioning
        tears the server down.
        """
        provider_options = dict(spec.provider_options or {})
        keepalive_minutes = _keepalive_option(provider_options.pop(KEEPALIVE_OPTION, self._create.keepalive_minutes))
        options = IdeGymProviderOptions.from_mapping(provider_options)
        request = self._translator.translate(spec, options, self._attribution.metadata())
        session = await self.session()
        ready_timeout_s = float(spec.ready_timeout_s or self._create.ready_timeout_s)

        ref = await self._start_server(session, request, ready_timeout_s)
        sandbox = _IdeGymSandbox(
            server_id=ref.server_id,
            server_name=ref.server_name,
            image=str(request["image_tag"]),
            workdir=spec.workdir,
            env={str(key): str(value) for key, value in spec.env.items()},
            keepalive_minutes=keepalive_minutes,
        )
        handle = SandboxHandle(sandbox_id=str(ref.server_id), provider_name=self.name, raw=sandbox)
        LOGGER.debug(f"Started IdeGYM sandbox {sandbox} from {sandbox.image!r} with labels {request['labels']}")
        try:
            await self._hold_after_start(handle)
            await self._verify_server(handle)
        except BaseException as e:
            LOGGER.warning(f"Tearing down the IdeGYM sandbox {sandbox}, which create could not finish: {e!r}")
            await self._discard(handle)
            raise
        return handle

    async def _start_server(
        self,
        session: IdeGymSession,
        request: dict[str, Any],
        ready_timeout_s: float,
    ) -> IdeGymServerRef:
        """Issue start-server, retrying failures that look transient.

        ``ready_timeout_s`` bounds the whole call, not each attempt, so a late failure
        cannot multiply the wait the caller configured. Retries keep the server name:
        IdeGYM derives the Kubernetes name from its own server id, so a lost response
        that landed anyway leaves an orphan for the watcher, not a name collision.
        """
        create = self._create
        loop = asyncio.get_running_loop()
        deadline = loop.time() + ready_timeout_s
        delay = create.retry_delay_s
        attempt = 0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise IdeGymCreateError(
                    f"IdeGYM did not start a server for image {request['image_tag']!r} within {ready_timeout_s:g}s"
                )
            try:
                return await session.start_server(request, polling=create.polling, timeout_s=remaining)
            except Exception as e:
                backoff = min(delay, create.retry_max_delay_s)
                # A deadline the SDK ran out of was the whole remaining budget; retrying would
                # multiply ready_timeout_s by the retry count.
                exhausted = is_deadline_exceeded(e) or deadline - loop.time() <= backoff
                if attempt >= create.retries or exhausted or not is_retryable(e):
                    raise IdeGymCreateError(
                        f"IdeGYM could not start a server for image {request['image_tag']!r}: {e}"
                    ) from e
                attempt += 1
                LOGGER.warning(
                    f"start_server attempt {attempt}/{create.retries + 1} for {request['server_name']!r} failed "
                    f"transiently; retrying in {backoff:g}s: {e}"
                )
                await asyncio.sleep(backoff)
                delay = min(delay * 2, create.retry_max_delay_s) if delay > 0 else create.retry_delay_s

    async def _hold_after_start(self, handle: SandboxHandle) -> None:
        """Take the sandbox's keepalive hold; failing to is a create failure."""
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.keepalive_minutes is None:
            return
        try:
            await self._keepalive(sandbox, sandbox.keepalive_minutes)
        except Exception as e:
            raise IdeGymCreateError(
                f"IdeGYM could not hold the new sandbox {sandbox} for {sandbox.keepalive_minutes:g} min against "
                f"its inactivity reaper: {e}"
            ) from e

    async def _verify_server(self, handle: SandboxHandle) -> None:
        """Fail create on an unusable ``spec.workdir``.

        A missing workdir would otherwise show up as every later command failing, which
        reads like a broken agent.
        """
        sandbox: _IdeGymSandbox = handle.raw
        cwd = sandbox.workdir if self._verify.check_workdir else None
        if not cwd:
            return
        result = await self._run(handle, "true", cwd=cwd, env={}, user=None, timeout_s=self._verify.timeout_s)
        stderr = (result.stderr or "").strip()
        if result.error_type == "invalid_request":
            raise IdeGymCreateVerificationError(
                f"The working directory {cwd!r} is not usable in the IdeGYM sandbox {sandbox} "
                f"(image {sandbox.image!r}): {stderr}. Point sandbox_spec.workdir at the image's project "
                f"directory, or leave it unset to use the IdeGYM server's own project directory."
            )
        if result.error_type is not None:
            raise IdeGymCreateVerificationError(
                f"The IdeGYM sandbox {sandbox} (image {sandbox.image!r}) could not run its verification "
                f"command: {stderr}"
            )

    async def _discard(self, handle: SandboxHandle) -> None:
        """Best-effort teardown of a sandbox that never became usable."""
        try:
            await self.close(handle)
        except Exception as e:
            LOGGER.warning(
                f"Failed to stop the half-created IdeGYM sandbox {handle.raw}; it may be left running on the "
                f"cluster until the orchestrator reaps it: {e}"
            )

    # --- keepalive ---------------------------------------------------------

    async def _keepalive(self, sandbox: _IdeGymSandbox, minutes: float) -> None:
        session = await self.session()
        async with asyncio.timeout(self._operations.status_timeout_s):
            until_ms = await session.keepalive(sandbox.server_id, minutes)
        sandbox.held_until_ms = max(sandbox.held_until_ms, until_ms)

    async def _hold_for(self, handle: SandboxHandle, deadline_s: float) -> None:
        """Extend the sandbox's keepalive hold to cover the next ``deadline_s`` seconds.

        IdeGYM's watcher reaps on time since the last completed request, so a command
        longer than the inactivity timeout would be killed mid-run. The hold only ever
        grows, so racing callers cannot shorten it. A failure is not fatal: the command
        may still finish inside the inactivity timeout.
        """
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.keepalive_minutes is None or _now_ms() + deadline_s * 1000 <= sandbox.held_until_ms:
            return
        minutes = min(self._operations.keepalive_minutes_max, math.ceil(deadline_s / 60) + 1)
        try:
            await self._keepalive(sandbox, minutes)
        except Exception as e:
            if is_sandbox_gone(e):
                sandbox.stopped = True
                LOGGER.debug(f"IdeGYM sandbox {sandbox} is gone; further commands will short-circuit: {e}")
                return
            LOGGER.warning(
                f"Could not extend the keepalive hold on IdeGYM sandbox {sandbox} to {minutes:g} min; the command "
                f"runs without it and may be reaped as idle: {e}"
            )

    # --- exec --------------------------------------------------------------

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run ``command`` in the sandbox; never raises for a failing command.

        A request the sandbox cannot honour (a missing ``cwd``, an unknown ``user``, an
        environment name that is not a shell identifier) is reported the same way, so a
        bad model-generated command cannot end a rollout.
        """
        if isinstance(user, int):
            # The server switches users by name (runuser/sudo -u), and a uid has no name to give.
            return _runtime_failure(
                f"The idegym provider runs commands as a named user, not uid {user!r}", error_type="invalid_request"
            )
        sandbox: _IdeGymSandbox = handle.raw
        merged_env = {**sandbox.env, **{str(key): str(value) for key, value in (env or {}).items()}}
        effective_cwd = cwd if cwd is not None else sandbox.workdir
        timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        return await self._run(handle, command, cwd=effective_cwd, env=merged_env, user=user, timeout_s=timeout)

    async def _run(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str],
        user: str | None,
        timeout_s: float | None,
    ) -> SandboxExecResult:
        """Send one command to the sandbox and normalize the outcome."""
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.stopped:
            return _runtime_failure(f"IdeGYM sandbox {sandbox} has been stopped", error_type="sandbox")
        session = await self.session()
        command_timeout = float(timeout_s) if timeout_s is not None else NO_TIMEOUT_COMMAND_SECONDS
        # The sandbox's own timeout has to fire before the client stops waiting,
        # otherwise a timed-out command comes back as a transport error with none
        # of the output the caller needs.
        request_timeout = command_timeout + self._exec.request_overhead_s
        await self._hold_for(handle, request_timeout)
        if sandbox.stopped:
            return _runtime_failure(f"IdeGYM sandbox {sandbox} is no longer available", error_type="sandbox")
        try:
            result = await session.execute_bash(
                sandbox.server_id,
                command,
                cwd=cwd,
                env=env,
                user=user,
                max_output_bytes=self._exec.max_output_bytes,
                command_timeout_s=command_timeout,
                graceful_termination_timeout_s=self._exec.graceful_termination_timeout_s,
                request_timeout_s=request_timeout,
                polling=self._create.polling,
            )
        except Exception as e:
            if is_command_timeout(e):
                return _runtime_failure(
                    f"Command timed out after {command_timeout}s in IdeGYM sandbox {sandbox}: {e}",
                    error_type="timeout",
                )
            if is_sandbox_gone(e):
                sandbox.stopped = True
                LOGGER.debug(f"IdeGYM sandbox {sandbox} is gone; further commands will short-circuit: {e}")
                return _runtime_failure(f"IdeGYM sandbox {sandbox} is no longer available: {e}", error_type="sandbox")
            if is_invalid_request(e):
                return _runtime_failure(
                    f"IdeGYM sandbox {sandbox} rejected the command request: {e}", error_type="invalid_request"
                )
            # Unclassified: the return value keeps the rollout alive but throws the
            # traceback away, so this is the only place it can still be recovered.
            LOGGER.warning(f"IdeGYM sandbox {sandbox} failed to run the command", exc_info=True)
            return _runtime_failure(f"IdeGYM sandbox {sandbox} failed to run the command: {e}", error_type="sandbox")
        return SandboxExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    # --- files -------------------------------------------------------------

    @staticmethod
    def _remote_path(sandbox: _IdeGymSandbox, path: str) -> str:
        # The SDK resolves a relative path against the server's project directory, but
        # the sandbox contract resolves it against spec.workdir.
        if sandbox.workdir and not posixpath.isabs(path):
            return posixpath.join(sandbox.workdir, path)
        return path

    async def _transfer(self, handle: SandboxHandle, what: str, call: Callable[[IdeGymSession], Awaitable[T]]) -> T:
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.stopped:
            raise IdeGymTransferError(f"{what} failed: IdeGYM sandbox {sandbox} has been stopped")
        session = await self.session()
        try:
            return await call(session)
        except Exception as e:
            if is_sandbox_gone(e):
                sandbox.stopped = True
                raise IdeGymTransferError(
                    f"{what} failed: IdeGYM sandbox {sandbox} is no longer available: {e}"
                ) from e
            raise IdeGymTransferError(f"{what} failed: {e}") from e

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into the sandbox, creating its parent directories."""
        sandbox: _IdeGymSandbox = handle.raw
        source = Path(source_path)
        remote = self._remote_path(sandbox, target_path)
        await self._transfer(
            handle,
            f"Uploading {source} to {remote!r} in IdeGYM sandbox {sandbox}",
            lambda session: session.upload_file(
                sandbox.server_id,
                source,
                remote,
                chunk_bytes=self._files.chunk_bytes,
                request_timeout_s=self._files.timeout_s,
                polling=self._create.polling,
            ),
        )

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the local filesystem.

        The bytes land in a temporary sibling first, so a failed download neither leaves
        a truncated file behind nor clobbers one that was already there.
        """
        sandbox: _IdeGymSandbox = handle.raw
        remote = self._remote_path(sandbox, source_path)
        target = Path(target_path)
        polling = self._create.polling
        partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            await self._transfer(
                handle,
                f"Downloading {remote!r} from IdeGYM sandbox {sandbox}",
                lambda session: session.download_file(
                    sandbox.server_id,
                    remote,
                    partial,
                    chunk_bytes=self._files.chunk_bytes,
                    request_timeout_s=self._files.timeout_s,
                    polling=polling,
                ),
            )
            await asyncio.to_thread(partial.replace, target)
        except BaseException:
            await asyncio.to_thread(partial.unlink, missing_ok=True)
            raise

    # --- status and teardown ----------------------------------------------

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Report the sandbox's lifecycle status from the orchestrator's status endpoint.

        Not-usable means terminal (crashed, killed, stopped): IdeGYM never brings such a
        server back, so the sandbox is marked stopped. Any other failure leaves the
        status unknown rather than guessing.
        """
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.stopped:
            return SandboxStatus.STOPPED
        session = await self.session()
        try:
            async with asyncio.timeout(self._operations.status_timeout_s):
                server_status = await session.server_status(sandbox.server_id)
        except Exception as e:
            if is_sandbox_gone(e):
                return SandboxStatus.STOPPED
            LOGGER.debug(f"Could not determine the status of IdeGYM sandbox {sandbox}: {e}")
            return SandboxStatus.UNKNOWN
        if not server_status.usable:
            sandbox.stopped = True
            LOGGER.debug(
                f"IdeGYM sandbox {sandbox} is {server_status.availability}; further commands will short-circuit: "
                f"{server_status.details}"
            )
            return SandboxStatus.STOPPED
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        """Stop the sandbox and delete its Kubernetes resources.

        Idempotent, and a server neither the orchestrator nor the session still has
        counts as success, so cleanup paths need not check first. A stop that fails for
        real raises and leaves the sandbox marked live, so ``status()`` keeps telling
        the truth and a caller can try again.
        """
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.stopped:
            return
        session = await self.session()
        delay = self._operations.retry_delay_s
        attempt = 0
        while True:
            try:
                async with asyncio.timeout(self._operations.close_timeout_s):
                    await session.stop_server(
                        sandbox.server_id,
                        polling=self._create.polling,
                        timeout_s=self._operations.close_timeout_s,
                    )
            except Exception as e:
                if is_sandbox_gone(e):
                    sandbox.stopped = True
                    return
                # A rejected or unauthorized delete, or a stopped session, fails the same way again.
                if attempt >= self._operations.retries or not is_retryable(e):
                    raise IdeGymOperationError(f"Failed to stop IdeGYM sandbox {sandbox}: {e}") from e
                attempt += 1
                backoff = min(delay, self._operations.retry_max_delay_s)
                LOGGER.warning(
                    f"stop_server attempt {attempt}/{self._operations.retries + 1} for {sandbox} failed; "
                    f"retrying in {backoff:g}s: {e}"
                )
                await asyncio.sleep(backoff)
                delay = min(delay * 2, self._operations.retry_max_delay_s) if delay > 0 else 0.0
            else:
                sandbox.stopped = True
                return

    async def aclose(self) -> None:
        """Give back this provider's reference to the shared session.

        The last provider to release it unregisters the IdeGYM client, which also
        terminates any server that was never closed.
        """
        with self._session_lock:
            if self._closed:
                return
            self._closed = True
            session, self._session = self._session, None
        if session is not None:
            await release_session(session)
