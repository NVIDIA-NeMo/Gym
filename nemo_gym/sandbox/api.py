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

"""Provider-neutral public sandbox API."""

import asyncio
import tempfile
import threading
import uuid
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, TypeVar

from nemo_gym.sandbox.providers import (
    ConnectableProvider,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxPtyError,
    SandboxPtySession,
    SandboxPtySpec,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxEndpoint,
    SupportsSandboxPty,
    SupportsSandboxPtyAttach,
    create_provider,
)


T = TypeVar("T")
SYNC_OPERATION_TIMEOUT_S = 3600.0
# Matches the providers' non-process exec sentinel (see docker provider).
SANDBOX_PTY_RUNTIME_RETURN_CODE = 125
SYNC_LOOP_CLOSE_TIMEOUT_S = 5.0


def _pty_timeout_result(command: str, timeout_s: float | int | None, *, reusable: bool) -> SandboxExecResult:
    detail = "" if reusable else "; the command is still running and the session should be discarded"
    return SandboxExecResult(
        stdout=None,
        stderr=f"PTY command timed out after {timeout_s}s: {command!r}{detail}",
        return_code=SANDBOX_PTY_RUNTIME_RETURN_CODE,
        error_type="timeout",
    )


async def _scan_for_marker(session: SandboxPtySession, needle: bytes, buffer: bytearray, quiet_s: float) -> bool:
    """Drain whatever the session has buffered into ``buffer``; True once
    ``needle`` is present, False when the stream goes quiet without it."""
    while needle not in buffer:
        try:
            chunk = await session.read(timeout_s=quiet_s)
        except (TimeoutError, asyncio.TimeoutError):
            return False
        if not chunk:
            raise SandboxPtyError("PTY session ended before the command finished")
        buffer.extend(chunk)
    return True


async def _run_in_pty_session(session: SandboxPtySession, command: str) -> SandboxExecResult:
    """Run ``command`` in a live session, delimited by a unique marker."""
    token = f"NGPTY{uuid.uuid4().hex[:12]}"
    # The marker is assembled from two literals so the shell's echo of this
    # line cannot itself match the marker we scan for. The brace group keeps
    # shell state while putting stdin at EOF: the session's stdin never ends,
    # so a stdin-reading command would block forever and eat the marker line.
    await session.write(
        f"{{ {command}\n}} </dev/null\nprintf '%s%s:%s\\n' '{token[:5]}' '{token[5:]}' \"$?\"\n".encode()
    )

    needle = f"{token}:".encode()
    buffer = bytearray()
    while needle not in buffer:
        chunk = await session.read()
        if not chunk:
            raise SandboxPtyError("PTY session ended before the command finished")
        buffer.extend(chunk)

    stdout, _, trailing = bytes(buffer).partition(needle)
    while b"\n" not in trailing:
        # The status digits can straddle the chunk that carried the marker.
        chunk = await session.read()
        if not chunk:
            break
        trailing += chunk
    exit_text = trailing.split(b"\n", 1)[0].strip()
    stderr = bytearray()
    try:
        while chunk := await session.read_stderr(timeout_s=0.05):
            stderr.extend(chunk)
    except (TimeoutError, asyncio.TimeoutError):
        pass
    parsed = exit_text.isdigit()
    return SandboxExecResult(
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace") or None,
        return_code=int(exit_text) if parsed else SANDBOX_PTY_RUNTIME_RETURN_CODE,
        error_type=None if parsed else "pty",
    )


class SandboxPty:
    """PTY namespace of a sandbox: ``await sandbox.pty.create(...)`` for a live
    session, ``await sandbox.pty.exec(...)`` for one-shot run-and-collect."""

    def __init__(self, sandbox: "AsyncSandbox") -> None:
        self._sandbox = sandbox
        # The oldest live default-shell session (create() with no command);
        # exec() reuses it when called without a session.
        self._default_session: SandboxPtySession | None = None
        # Session-mode execs share one output stream per session; serialize
        # them so two concurrent calls cannot consume each other's marker.
        self._session_exec_lock = asyncio.Lock()

    async def create(
        self,
        command: str | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        rows: int = 24,
        cols: int = 80,
        user: str | int | None = None,
        pty: bool = True,
    ) -> SandboxPtySession:
        """Open an interactive terminal; the returned session carries
        ``read``/``read_stderr``/``write``/``resize``/``send_signal``/
        ``wait_exit``/``close`` and is an async context manager.

        ``pty=False`` selects pipe mode: no TTY, stdout/stderr split across
        ``read()``/``read_stderr()``. ``rows``/``cols`` are applied right after
        the terminal connects, because the backend has no spawn-time size, so a
        ``command`` that reads the size in its first moments can still see the
        80x24 default; programs that honor SIGWINCH pick up the real size.
        Close the session before the sandbox stops: a session that outlives its
        sandbox fails subsequent reads with ``SandboxPtyError``. Async-only;
        the sync ``Sandbox`` facade does not mirror it.
        """
        sandbox = self._sandbox
        if not isinstance(sandbox._provider, SupportsSandboxPty):
            provider_name = getattr(sandbox._provider, "name", type(sandbox._provider).__name__)
            raise NotImplementedError(
                f"Sandbox provider {provider_name!r} does not support PTY sessions; use exec() instead"
            )
        session = await sandbox._provider.create_pty(
            sandbox._require_handle(),
            SandboxPtySpec(
                command=command,
                cwd=cwd if cwd is not None else sandbox._spec.workdir if sandbox._spec is not None else None,
                env=env,
                rows=rows,
                cols=cols,
                user=user,
                pty=pty,
            ),
        )
        if command is None and (self._default_session is None or self._default_session.closed):
            self._default_session = session
        return session

    async def attach(
        self,
        session_id: str,
        *,
        takeover: bool = True,
        since: int | None = None,
    ) -> SandboxPtySession:
        """Re-attach to a session opened earlier, here or in another process.

        Sessions outlive the client that opened them, so ``session.session_id``
        is all another process needs. ``takeover`` evicts the current holder,
        whose session then fails with ``SandboxPtyError``; without it,
        attaching to a held session fails. ``since`` replays retained output
        from that byte offset first (``0`` replays all of it).
        """
        sandbox = self._sandbox
        if not isinstance(sandbox._provider, SupportsSandboxPtyAttach):
            provider_name = getattr(sandbox._provider, "name", type(sandbox._provider).__name__)
            raise NotImplementedError(f"Sandbox provider {provider_name!r} does not support re-attaching PTY sessions")
        return await sandbox._provider.attach_pty(
            sandbox._require_handle(), session_id, takeover=takeover, since=since
        )

    async def exec(
        self,
        command: str,
        *,
        session: SandboxPtySession | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = 180,
        user: str | int | None = None,
        rows: int = 24,
        cols: int = 80,
        pty: bool = True,
        detach: bool = False,
        poll_interval_s: float = 15.0,
    ) -> SandboxExecResult:
        """Run one command in a terminal session and collect its output.

        Without ``session``, the sandbox's default-shell session — the oldest
        live one opened by ``create()`` with no ``command`` — is reused,
        provided the call sets none of the session-shaping arguments
        (``cwd``/``env``/``user``, non-default ``rows``/``cols``, or
        ``pty=False``), since those are fixed at ``create()``. Custom-command
        and attached sessions run arbitrary programs, so they are only used
        when passed explicitly. When no default-shell session exists (or
        shaping arguments are given) a private session is opened for the
        command, drained and closed. Session-mode execs are serialized per
        sandbox: concurrent calls into one shared stream would corrupt it.
        With ``session`` the command runs in that live session, which stays open
        and keeps its shell state. In a live session the output also contains
        the shell's echo of the command, ``stderr`` is best-effort (pipe mode
        only), and a command that ends the shell (``exit``) raises
        ``SandboxPtyError``.

        With ``detach=True`` the command runs without holding a connection
        while it works: it starts in a session, the socket is dropped, and the
        session is briefly re-attached every ``poll_interval_s`` to check for
        completion, so a long command occupies a connection for milliseconds
        per poll instead of its whole runtime (completion latency is bounded
        by ``poll_interval_s``). Output is captured to files inside the
        sandbox — the server retains only ~1 MiB of terminal output across a
        detach — and collected when the command finishes, so stdout and
        stderr come back separated in both modes. A detached exec never
        reuses the default-shell session implicitly: without ``session`` it
        opens a private one. With ``session``, the session is detached while
        the command works, must not be used concurrently, and is attached and
        reusable again when this returns.

        PTY mode returns all output on ``stdout`` and ``None`` stderr; pipe mode
        splits the two. A command that outlives ``timeout_s`` returns
        ``error_type="timeout"`` like ``sandbox.exec()`` rather than raising;
        in an explicitly passed session that command keeps running and leaves
        unread output behind, so discard the session rather than reusing it (an
        implicitly reused session is retired automatically).
        """
        if detach:
            return await self._exec_detached(
                command,
                session=session,
                cwd=cwd,
                env=env,
                user=user,
                rows=rows,
                cols=cols,
                pty=pty,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
        implicit = False
        if session is None and cwd is None and env is None and user is None and pty and (rows, cols) == (24, 80):
            if self._default_session is not None and self._default_session.closed:
                self._default_session = None
            session = self._default_session
            implicit = session is not None
        if session is not None:
            if cwd is not None or env is not None or user is not None:
                raise ValueError(
                    "cwd/env/user apply only when pty.exec opens its own session; "
                    "for an existing session they are fixed at pty.create() time"
                )
            try:
                # The timeout covers waiting for the lock too: a caller's
                # budget must not be consumed invisibly by another exec.
                async with asyncio.timeout(timeout_s):
                    async with self._session_exec_lock:
                        return await _run_in_pty_session(session, command)
            except (TimeoutError, asyncio.TimeoutError):
                if implicit:
                    # This call did not select the session explicitly, and its
                    # stream now carries the stray command's output; retire it
                    # so a later implicit exec cannot inherit it.
                    try:
                        await session.close()
                    except Exception:
                        pass
                    if self._default_session is session:
                        self._default_session = None
                return _pty_timeout_result(command, timeout_s, reusable=False)
        session = await self.create(command, cwd=cwd, env=env, rows=rows, cols=cols, user=user, pty=pty)
        try:

            async def drain(read: Callable[[], Awaitable[bytes]]) -> bytes:
                chunks = bytearray()
                while chunk := await read():
                    chunks.extend(chunk)
                return bytes(chunks)

            stdout, stderr, return_code = await asyncio.wait_for(
                asyncio.gather(drain(session.read), drain(session.read_stderr), session.wait_exit()),
                timeout=timeout_s,
            )
        except (TimeoutError, asyncio.TimeoutError):
            return _pty_timeout_result(command, timeout_s, reusable=True)
        finally:
            await session.close()
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=None if pty else stderr.decode(errors="replace"),
            return_code=return_code,
        )

    async def _exec_detached(
        self,
        command: str,
        *,
        session: SandboxPtySession | None,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
        rows: int,
        cols: int,
        pty: bool,
        timeout_s: int | float | None,
        poll_interval_s: float,
    ) -> SandboxExecResult:
        """``exec(detach=True)``: start, drop the socket, poll by re-attach."""
        private = session is None
        if private:
            session = await self.create(cwd=cwd, env=env, user=user, rows=rows, cols=cols, pty=pty)
            if self._default_session is session:
                # Private to this call: an implicit exec() grabbing it would
                # collide with the detach cycle.
                self._default_session = None
        elif cwd is not None or env is not None or user is not None:
            raise ValueError(
                "cwd/env/user apply only when a detached exec opens its own session; "
                "for an existing session they are fixed at pty.create() time"
            )
        token = f"NGPTY{uuid.uuid4().hex[:12]}"
        capture = f"/tmp/.ng_{token}"
        needle = f"{token}:".encode()
        buffer = bytearray()
        try:
            async with asyncio.timeout(timeout_s):
                async with self._session_exec_lock:
                    # Same marker discipline as _run_in_pty_session, plus file
                    # capture: the replay ring cannot carry a chatty command's
                    # output across a detach, the files can.
                    await session.write(
                        f"{{ {command}\n}} </dev/null >{capture}.out 2>{capture}.err\n"
                        f"printf '%s%s:%s\\n' '{token[:5]}' '{token[5:]}' \"$?\"\n".encode()
                    )
                    found = await _scan_for_marker(session, needle, buffer, quiet_s=1.0)
                while not found:
                    await session.detach()
                    await asyncio.sleep(poll_interval_s)
                    async with self._session_exec_lock:
                        await session.reattach()
                        found = await _scan_for_marker(session, needle, buffer, quiet_s=1.0)
                async with self._session_exec_lock:
                    trailing = bytes(buffer).partition(needle)[2]
                    while b"\n" not in trailing:
                        # The status digits can straddle the chunk with the marker.
                        chunk = await session.read()
                        if not chunk:
                            break
                        trailing += chunk
                    exit_text = trailing.split(b"\n", 1)[0].strip()
                    stdout_res = await _run_in_pty_session(session, f"cat {capture}.out")
                    stderr_res = await _run_in_pty_session(
                        session, f"cat {capture}.err; rm -f {capture}.out {capture}.err"
                    )
        except (TimeoutError, asyncio.TimeoutError):
            return _pty_timeout_result(command, timeout_s, reusable=False)
        finally:
            if private:
                await session.close()
        parsed = exit_text.isdigit()
        return SandboxExecResult(
            stdout=stdout_res.stdout,
            stderr=stderr_res.stdout or None,
            return_code=int(exit_text) if parsed else SANDBOX_PTY_RUNTIME_RETURN_CODE,
            error_type=None if parsed else "pty",
        )


class AsyncSandbox:
    """Async sandbox object backed by a runtime provider."""

    def __init__(
        self,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec | None = None,
    ) -> None:
        self._provider = create_provider(provider) if isinstance(provider, Mapping) else provider
        self._spec = spec
        self._handle: SandboxHandle | None = None
        self._stopped = True
        self._closed = False
        self.pty = SandboxPty(self)

    def _require_handle(self) -> SandboxHandle:
        if self._handle is None or self._stopped:
            raise RuntimeError("Sandbox has not been started")
        return self._handle

    async def start(
        self,
        spec: SandboxSpec | None = None,
    ) -> "AsyncSandbox":
        if self._closed:
            raise RuntimeError("Sandbox has been stopped")
        if self._handle is not None and not self._stopped:
            raise RuntimeError("Sandbox is already started")
        requested_spec = spec if spec is not None else self._spec
        if requested_spec is None:
            raise ValueError("Sandbox.start() requires a SandboxSpec")

        handle = await self._provider.create(requested_spec)
        try:
            if requested_spec.files:
                with tempfile.TemporaryDirectory(prefix="nemo-gym-sandbox-upload-") as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    for index, (target_path, contents) in enumerate(requested_spec.files.items()):
                        source_path = tmp_path / f"file-{index}"
                        source_path.write_text(contents, encoding="utf-8")
                        await self._provider.upload_file(handle, source_path, target_path)
        except Exception:
            await self._provider.close(handle)
            await self._provider.aclose()
            self._closed = True
            raise

        self._spec = requested_spec
        self._handle = handle
        self._stopped = False
        return self

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = 180,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        return await self._provider.exec(
            self._require_handle(),
            command,
            cwd=cwd if cwd is not None else self._spec.workdir if self._spec is not None else None,
            env=env,
            timeout_s=timeout_s,
            user=user,
        )

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        await self._provider.upload_file(self._require_handle(), Path(local_path), remote_path)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        await self._provider.download_file(self._require_handle(), remote_path, Path(local_path))

    async def status(self) -> SandboxStatus:
        if self._handle is None:
            return SandboxStatus.UNKNOWN
        if self._stopped:
            return SandboxStatus.STOPPED
        return await self._provider.status(self._handle)

    async def endpoint(self, port: int) -> SandboxEndpoint:
        """Resolve a declared sandbox service port without exposing provider state."""

        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"Sandbox endpoint port must be an integer between 1 and 65535, got {port!r}")
        declared_ports = self._spec.ports if self._spec is not None else ()
        if port not in declared_ports:
            raise ValueError(
                f"Sandbox port {port} was not declared in SandboxSpec.ports; declared ports: {list(declared_ports)!r}"
            )
        if not isinstance(self._provider, SupportsSandboxEndpoint):
            provider_name = getattr(self._provider, "name", type(self._provider).__name__)
            raise NotImplementedError(f"Sandbox provider {provider_name!r} does not support service endpoints")
        resolved = await self._provider.endpoint(self._require_handle(), port)
        if not isinstance(resolved, SandboxEndpoint):
            raise TypeError(f"Sandbox provider endpoint() must return SandboxEndpoint, got {type(resolved).__name__}")
        return resolved

    async def stop(self) -> None:
        if self._closed:
            return
        try:
            if self._handle is not None and not self._stopped:
                self._stopped = True
                await self._provider.close(self._handle)
        finally:
            await self._provider.aclose()
            self._closed = True

    async def serialize(self, *, scope: str | None = None) -> dict[str, Any]:
        """Return a JSON descriptor another process can rebuild this box from.

        Requires a provider that supports the connect capability (the remote
        provider, or an external-control-plane provider such as OpenSandbox). For
        the remote provider, ``scope`` mints a co-lease (``scope="operate"``).
        """
        provider = self._provider
        if not isinstance(provider, ConnectableProvider):
            name = getattr(provider, "name", type(provider).__name__)
            raise RuntimeError(f"provider {name!r} does not support serialize()/connect()")
        descriptor = await provider.serialize_handle(self._require_handle(), scope=scope)
        # Carry the working directory so a reattached sandbox lands in the same
        # place, even for providers whose descriptor does not include it (the
        # remote provider's SandboxRef already has it; e.g. OpenSandbox does not).
        if isinstance(descriptor, dict) and descriptor.get("workdir") is None and self._spec is not None:
            descriptor = {**descriptor, "workdir": self._spec.workdir}
        return descriptor

    @classmethod
    async def connect(cls, descriptor: Mapping[str, Any] | Any, *, provider: SandboxProvider) -> "AsyncSandbox":
        """Rebuild a sandbox in this process from a descriptor produced by
        :meth:`serialize`, using ``provider`` (which must support connect)."""
        if not isinstance(provider, ConnectableProvider):
            name = getattr(provider, "name", type(provider).__name__)
            raise RuntimeError(f"provider {name!r} does not support serialize()/connect()")
        if not isinstance(descriptor, Mapping) and hasattr(descriptor, "to_dict"):
            descriptor = descriptor.to_dict()
        handle = await provider.connect(descriptor)
        workdir = descriptor.get("workdir") if isinstance(descriptor, Mapping) else None
        sandbox = cls(provider, SandboxSpec(workdir=workdir))
        sandbox._handle = handle
        sandbox._stopped = False
        return sandbox

    async def __aenter__(self) -> "AsyncSandbox":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


class _AsyncLoopRunner:
    """Run async sandbox operations for sync callers."""

    def __init__(
        self,
        *,
        wait_timeout_s: float = SYNC_OPERATION_TIMEOUT_S,
        close_timeout_s: float = SYNC_LOOP_CLOSE_TIMEOUT_S,
    ) -> None:
        self._wait_timeout_s = wait_timeout_s
        self._close_timeout_s = close_timeout_s
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, name="nemo-gym-sandbox-sync-loop", daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _ensure_can_block(self, operation: str) -> None:
        if self._closed or self._loop.is_closed():
            raise RuntimeError("Sandbox sync loop is closed")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(f"Sandbox.{operation}() is blocking; use AsyncSandbox in async code instead.")

    def _wait_for_result(self, operation: str, future: Future[T]) -> T:
        try:
            return future.result(timeout=self._wait_timeout_s)
        except FutureTimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"Sandbox.{operation}() timed out waiting for the sync loop after {self._wait_timeout_s:g}s"
            ) from e

    def call(self, operation: str, func: Callable[[], T]) -> T:
        self._ensure_can_block(operation)
        future: Future[T] = Future()

        def invoke() -> None:
            try:
                result = func()
            except BaseException as e:
                if not future.cancelled():
                    future.set_exception(e)
            else:
                if not future.cancelled():
                    future.set_result(result)

        self._loop.call_soon_threadsafe(invoke)
        return self._wait_for_result(operation, future)

    def run(self, operation: str, awaitable_factory: Callable[[], Awaitable[T]]) -> T:
        self._ensure_can_block(operation)
        future = asyncio.run_coroutine_threadsafe(awaitable_factory(), self._loop)
        try:
            return future.result(timeout=self._wait_timeout_s)
        except FutureTimeoutError as e:
            future.cancel()
            raise TimeoutError(
                f"Sandbox.{operation}() timed out waiting for the sync loop after {self._wait_timeout_s:g}s"
            ) from e

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=self._close_timeout_s)
            if self._thread.is_alive():
                return
            self._loop.close()


class Sandbox:
    """Synchronous wrapper around ``AsyncSandbox``.

    ``pty``, ``serialize`` and ``connect`` are async-only; use ``AsyncSandbox``
    for those."""

    def __init__(
        self,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec | None = None,
    ) -> None:
        self._runner = _AsyncLoopRunner()
        try:
            self._async_sandbox = self._runner.call(
                "__init__",
                lambda: AsyncSandbox(provider, spec),
            )
        except BaseException:
            self._runner.close()
            raise
        self._closed = False

    def start(
        self,
        spec: SandboxSpec | None = None,
    ) -> "Sandbox":
        self._runner.run(
            "start",
            lambda: self._async_sandbox.start(spec),
        )
        return self

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = 180,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        return self._runner.run(
            "exec",
            lambda: self._async_sandbox.exec(
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                user=user,
            ),
        )

    def upload(self, local_path: Path | str, remote_path: str) -> None:
        self._runner.run("upload", lambda: self._async_sandbox.upload(local_path, remote_path))

    def download(self, remote_path: str, local_path: Path | str) -> None:
        self._runner.run("download", lambda: self._async_sandbox.download(remote_path, local_path))

    def status(self) -> SandboxStatus:
        if self._closed:
            return SandboxStatus.STOPPED
        return self._runner.run("status", self._async_sandbox.status)

    def endpoint(self, port: int) -> SandboxEndpoint:
        return self._runner.run("endpoint", lambda: self._async_sandbox.endpoint(port))

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run("stop", self._async_sandbox.stop)
        finally:
            self._runner.close()

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __del__(self) -> None:  # pragma: no cover
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.stop()
            except Exception:
                pass
