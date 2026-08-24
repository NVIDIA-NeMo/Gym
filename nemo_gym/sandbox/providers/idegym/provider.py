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
registered client), :mod:`.spec` (spec translation), :mod:`.shell` (command shaping),
:mod:`.transfer` (files), :mod:`.errors` (failure classification).

Capabilities deliberately not claimed: ``endpoint()``, because the orchestrator
forwards API requests rather than routing raw TCP; PTY sessions, because IdeGYM has no
terminal API; and ``serialize()`` / ``connect()``, because a server is only reachable
through the registered client that owns it.
"""

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.idegym.config import (
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymCreateConfig,
    IdeGymExecConfig,
    IdeGymFilesConfig,
    IdeGymOperationsConfig,
    IdeGymProbeConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import (
    IdeGymCommandTooLongError,
    IdeGymCreateError,
    IdeGymCreateVerificationError,
    IdeGymOperationError,
    IdeGymUnknownServerError,
    is_command_timeout,
    is_retryable,
    is_sandbox_gone,
)
from nemo_gym.sandbox.providers.idegym.session import (
    IdeGymServerRef,
    IdeGymSession,
    acquire_session,
    release_session,
)
from nemo_gym.sandbox.providers.idegym.shell import BashScriptBuilder, directory_exists_script
from nemo_gym.sandbox.providers.idegym.spec import IdeGymProviderOptions, ServerRequestTranslator
from nemo_gym.sandbox.providers.idegym.transfer import Base64BashFileTransfer
from nemo_gym.sandbox.providers.utils import coerce_config


LOGGER = logging.getLogger(__name__)

# Returned by exec() when the sandbox runtime failed instead of the command. Same
# sentinel the other Gym providers use, so callers can treat it uniformly.
SANDBOX_RUNTIME_RETURN_CODE = 125

# IdeGYM's bash tool needs a finite, JSON-serializable command timeout, so "no
# timeout" is expressed as a ceiling long enough that no benchmark command reaches
# it before the sandbox itself is torn down.
NO_TIMEOUT_COMMAND_SECONDS = 24 * 60 * 60.0


@dataclass
class _IdeGymSandbox:
    """Provider-owned state for one sandbox, carried in ``SandboxHandle.raw``."""

    server_id: int
    server_name: str
    namespace: str
    image: str
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
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


class IdeGymProvider:
    """Sandbox provider backed by an IdeGYM orchestrator."""

    name = "idegym"

    def __init__(
        self,
        *,
        connection: IdeGymConnectionConfig | Mapping[str, Any] | None = None,
        create: IdeGymCreateConfig | Mapping[str, Any] | None = None,
        exec: IdeGymExecConfig | Mapping[str, Any] | None = None,
        probe: IdeGymProbeConfig | Mapping[str, Any] | None = None,
        files: IdeGymFilesConfig | Mapping[str, Any] | None = None,
        operations: IdeGymOperationsConfig | Mapping[str, Any] | None = None,
        attribution: IdeGymAttributionConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = coerce_config(connection, IdeGymConnectionConfig)
        self._create = coerce_config(create, IdeGymCreateConfig)
        self._exec = coerce_config(exec, IdeGymExecConfig)
        self._probe = coerce_config(probe, IdeGymProbeConfig)
        self._files = coerce_config(files, IdeGymFilesConfig)
        self._operations = coerce_config(operations, IdeGymOperationsConfig)
        self._attribution = coerce_config(attribution, IdeGymAttributionConfig)

        self._translator = ServerRequestTranslator(self._create)
        self._script = BashScriptBuilder(self._exec)
        # The session registers a client with the orchestrator, which is async, so
        # it is acquired on first use rather than in this constructor.
        self._session: IdeGymSession | None = None
        self._session_lock = asyncio.Lock()
        self._closed = False

    # --- session -----------------------------------------------------------

    async def session(self) -> IdeGymSession:
        """Return this provider's shared orchestrator session, acquiring it once."""
        if self._closed:
            raise IdeGymOperationError("This idegym provider has been closed")
        if self._session is not None:
            return self._session
        async with self._session_lock:
            if self._session is None:
                self._session = await acquire_session(self._connection, self._attribution)
            return self._session

    async def health(self) -> str:
        """Return the orchestrator's health status, registering the client if needed.

        Reaching this point already proves more than the endpoint being up: the
        session had to register a client to exist.
        """
        session = await self.session()
        return await session.health()

    # --- create ------------------------------------------------------------

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start an IdeGYM server for ``spec`` and return it ready to run commands.

        The pod being scheduled is not the same as the sandbox being usable, so a
        readiness probe runs before the handle is returned, and the server is torn
        down if anything after provisioning fails.
        """
        options = IdeGymProviderOptions.from_mapping(spec.provider_options)
        request = self._translator.translate(spec, options)
        session = await self.session()
        ready_timeout_s = float(spec.ready_timeout_s or self._create.ready_timeout_s)

        ref = await self._start_server(session, request, ready_timeout_s)
        sandbox = _IdeGymSandbox(
            server_id=ref.server_id,
            server_name=ref.server_name,
            namespace=ref.namespace,
            image=str(request["image_tag"]),
            workdir=spec.workdir,
            env={str(key): str(value) for key, value in spec.env.items()},
            metadata={str(key): str(value) for key, value in spec.metadata.items()},
        )
        handle = SandboxHandle(sandbox_id=str(ref.server_id), provider_name=self.name, raw=sandbox)
        LOGGER.debug(f"Started IdeGYM sandbox {sandbox} from {sandbox.image!r} with metadata {sandbox.metadata}")
        try:
            await self._verify(handle)
        except BaseException:
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

        ``ready_timeout_s`` is a deadline for the whole call, not per attempt: each
        retry gets only the remaining budget, so a late failure cannot multiply the
        wait the caller configured. Retries keep the same server name: IdeGYM derives
        the Kubernetes name from its own server id, so a lost response that landed
        anyway leaves an orphan the watcher reaps, not a name collision.
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
                # A bare TimeoutError means the SDK spent the budget it was given, so
                # it is never worth another attempt even if the clock disagrees.
                # Transport timeouts are a different exception type and stay retryable.
                exhausted = type(e) is TimeoutError or deadline - loop.time() <= backoff
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

    async def _verify(self, handle: SandboxHandle) -> None:
        """Check that the new sandbox can actually run commands.

        The probe runs first: it is the retrying, deadline-bounded check, so letting
        the single-shot workdir check go first would report a server that is merely
        still warming up as a mis-set working directory.
        """
        await self._verify_probe(handle)
        await self._verify_workdir(handle)

    async def _verify_workdir(self, handle: SandboxHandle) -> None:
        """Fail create when ``spec.workdir`` does not exist in the image.

        Without this check the mismatch shows up as every later command failing on
        ``cd``, which reads like a broken agent rather than a mis-set workdir.
        """
        sandbox: _IdeGymSandbox = handle.raw
        if not self._probe.verify_workdir or not sandbox.workdir:
            return
        # Deliberately no cwd: the point is to test the directory, not enter it.
        script = self._script.build(directory_exists_script(sandbox.workdir))
        result = await self._exec_script(handle, script, timeout_s=self._probe.timeout_s)
        if result.return_code != 0:
            raise IdeGymCreateVerificationError(
                f"The working directory {sandbox.workdir!r} is not usable in the IdeGYM sandbox {sandbox} "
                f"(image {sandbox.image!r}): return code {result.return_code}, "
                f"stderr {(result.stderr or '').strip()!r}. Point sandbox_spec.workdir at the image's project "
                f"directory, or leave it unset to use the IdeGYM server's own project directory."
            )

    async def _verify_probe(self, handle: SandboxHandle) -> None:
        """Poll the readiness command until it passes ``stable_count`` times."""
        probe = self._probe
        if probe.command is None:
            return
        script = self._script.build(probe.command, cwd=handle.raw.workdir)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        detail = "no probe attempt completed"
        while True:
            result = await self._exec_script(handle, script, timeout_s=probe.timeout_s)
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                detail = f"return code {result.return_code}, stderr {(result.stderr or '').strip()!r}"
                if deadline is None:
                    raise IdeGymCreateVerificationError(
                        f"IdeGYM sandbox {handle.raw} failed its readiness probe: {detail}"
                    )
            if deadline is not None and loop.time() >= deadline:
                raise IdeGymCreateVerificationError(
                    f"IdeGYM sandbox {handle.raw} did not pass its readiness probe within "
                    f"{probe.deadline_s:g}s: {detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def _discard(self, handle: SandboxHandle) -> None:
        """Best-effort teardown of a sandbox that never became usable."""
        try:
            await self.close(handle)
        except Exception as e:
            LOGGER.warning(
                f"Failed to stop the half-created IdeGYM sandbox {handle.raw}; it may be left running on the "
                f"cluster until the orchestrator reaps it: {e}"
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

        A command that cannot be delivered at all (too long for the sandbox's
        shell, or an environment name ``export`` cannot carry) is reported the same
        way, so a bad model-generated command cannot end a rollout.
        """
        sandbox: _IdeGymSandbox = handle.raw
        merged_env = {**sandbox.env, **{str(key): str(value) for key, value in (env or {}).items()}}
        effective_cwd = cwd if cwd is not None else sandbox.workdir
        try:
            script = self._script.build(command, cwd=effective_cwd, env=merged_env, user=user)
        except IdeGymCommandTooLongError as e:
            return _runtime_failure(str(e), error_type="command_too_long")
        except ValueError as e:
            return _runtime_failure(str(e), error_type="invalid_request")
        timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        return await self._exec_script(handle, script, timeout_s=timeout)

    async def _exec_script(
        self,
        handle: SandboxHandle,
        script: str,
        *,
        timeout_s: int | float | None,
    ) -> SandboxExecResult:
        """Send one prepared script to the sandbox and normalize the outcome."""
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.stopped:
            return _runtime_failure(f"IdeGYM sandbox {sandbox} has been stopped", error_type="sandbox")
        session = await self.session()
        command_timeout = float(timeout_s) if timeout_s is not None else NO_TIMEOUT_COMMAND_SECONDS
        # The sandbox's own timeout has to fire before the client stops waiting,
        # otherwise a timed-out command comes back as a transport error with none
        # of the output the caller needs.
        request_timeout = command_timeout + self._exec.request_overhead_s
        try:
            result = await session.execute_bash(
                sandbox.server_id,
                script,
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
            if is_sandbox_gone(e) or isinstance(e, IdeGymUnknownServerError):
                sandbox.stopped = True
                return _runtime_failure(f"IdeGYM sandbox {sandbox} is no longer available: {e}", error_type="sandbox")
            return _runtime_failure(f"IdeGYM sandbox {sandbox} failed to run the command: {e}", error_type="sandbox")
        return SandboxExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    # --- files -------------------------------------------------------------

    def _transfer(self, handle: SandboxHandle) -> Base64BashFileTransfer:
        async def run(command: str, *, timeout_s: int | float | None = None) -> SandboxExecResult:
            return await self.exec(handle, command, timeout_s=timeout_s)

        return Base64BashFileTransfer(self._files, run)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into the sandbox."""
        await self._transfer(handle).upload(Path(source_path), target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the local filesystem."""
        await self._transfer(handle).download(source_path, Path(target_path))

    # --- status and teardown ----------------------------------------------

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Report the sandbox's lifecycle status.

        IdeGYM has no server-status endpoint, so the capabilities call stands in:
        it validates the server's database record *and* reaches the pod, which is
        exactly the liveness question being asked. A "gone" answer from the
        orchestrator (unknown or terminal server) is a stopped sandbox; anything
        else that fails leaves the status unknown rather than guessing.
        """
        sandbox: _IdeGymSandbox = handle.raw
        if sandbox.stopped:
            return SandboxStatus.STOPPED
        session = await self.session()
        try:
            async with asyncio.timeout(self._operations.status_timeout_s):
                await session.list_capabilities(sandbox.server_id)
        except Exception as e:
            if is_sandbox_gone(e) or isinstance(e, IdeGymUnknownServerError):
                return SandboxStatus.STOPPED
            LOGGER.debug(f"Could not determine the status of IdeGYM sandbox {sandbox}: {e}")
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        """Stop the sandbox and delete its Kubernetes resources.

        Idempotent, and a server the orchestrator or the session no longer has counts
        as success, so cleanup paths can call this without checking first. A stop that
        fails for real raises and leaves the sandbox marked live, so ``status()``
        keeps telling the truth and a caller can try again.
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
            except IdeGymUnknownServerError:
                # The session only forgets a server after stopping it, so this is a
                # second teardown of the same sandbox.
                sandbox.stopped = True
                return
            except Exception as e:
                if is_sandbox_gone(e):
                    sandbox.stopped = True
                    return
                if attempt >= self._operations.retries:
                    raise IdeGymOperationError(f"Failed to stop IdeGYM sandbox {sandbox}: {e}") from e
                attempt += 1
                await asyncio.sleep(min(delay, self._operations.retry_max_delay_s))
                delay = min(delay * 2, self._operations.retry_max_delay_s) if delay > 0 else 0.0
            else:
                sandbox.stopped = True
                return

    async def aclose(self) -> None:
        """Give back this provider's reference to the shared session.

        The last provider to release it unregisters the IdeGYM client, which also
        terminates any server that was never closed.
        """
        if self._closed:
            return
        self._closed = True
        session, self._session = self._session, None
        if session is not None:
            await release_session(session)
