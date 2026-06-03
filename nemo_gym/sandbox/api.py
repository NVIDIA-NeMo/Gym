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
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TypeVar

from nemo_gym.sandbox.providers import (
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
    create_provider,
)


T = TypeVar("T")


class AsyncSandbox:
    """Async sandbox object backed by a runtime provider."""

    def __init__(
        self,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec | None = None,
        *,
        delete_on_stop: bool = False,
    ) -> None:
        self._provider = create_provider(provider) if isinstance(provider, Mapping) else provider
        self._spec = spec
        self._handle: SandboxHandle | None = None
        self._delete_on_stop = delete_on_stop
        self._stopped = True
        self._closed = False

    def _require_handle(self) -> SandboxHandle:
        if self._handle is None or self._stopped:
            raise RuntimeError("Sandbox has not been started")
        return self._handle

    async def _write_inline_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="nemo-gym-sandbox-upload-") as tmp_dir:
            source_path = Path(tmp_dir) / "contents"
            if isinstance(data, str):
                source_path.write_text(data, encoding="utf-8")
            else:
                source_path.write_bytes(data)
            await self._provider.upload_file(handle, source_path, target_path)

    async def _write_initial_files(self, handle: SandboxHandle, files: dict[str, str]) -> None:
        for target_path, contents in files.items():
            await self._write_inline_file(handle, target_path, contents)

    async def start(
        self,
        spec: SandboxSpec | None = None,
        *,
        delete_on_stop: bool | None = None,
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
            await self._write_initial_files(handle, requested_spec.files)
        except Exception:
            await self._provider.close(handle, delete=True)
            await self._provider.aclose()
            self._closed = True
            raise

        self._spec = requested_spec
        self._handle = handle
        self._delete_on_stop = self._delete_on_stop if delete_on_stop is None else delete_on_stop
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

    async def stop(self, *, delete: bool | None = None) -> None:
        if self._closed:
            return
        try:
            if self._handle is not None and not self._stopped:
                self._stopped = True
                await self._provider.close(
                    self._handle,
                    delete=self._delete_on_stop if delete is None else delete,
                )
        finally:
            await self._provider.aclose()
            self._closed = True

    async def __aenter__(self) -> "AsyncSandbox":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


class _AsyncLoopRunner:
    """Run async sandbox operations for sync callers."""

    def __init__(self) -> None:
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

    def call(self, operation: str, func: Callable[[], T]) -> T:
        self._ensure_can_block(operation)
        future: Future[T] = Future()

        def invoke() -> None:
            try:
                future.set_result(func())
            except BaseException as e:
                future.set_exception(e)

        self._loop.call_soon_threadsafe(invoke)
        return future.result()

    def run(self, operation: str, awaitable_factory: Callable[[], Awaitable[T]]) -> T:
        self._ensure_can_block(operation)
        future = asyncio.run_coroutine_threadsafe(awaitable_factory(), self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()


class Sandbox:
    """Synchronous wrapper around ``AsyncSandbox``."""

    def __init__(
        self,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec | None = None,
        *,
        delete_on_stop: bool = False,
    ) -> None:
        self._runner = _AsyncLoopRunner()
        try:
            self._async_sandbox = self._runner.call(
                "__init__",
                lambda: AsyncSandbox(provider, spec, delete_on_stop=delete_on_stop),
            )
        except BaseException:
            self._runner.close()
            raise
        self._closed = False

    def start(
        self,
        spec: SandboxSpec | None = None,
        *,
        delete_on_stop: bool | None = None,
    ) -> "Sandbox":
        self._runner.run(
            "start",
            lambda: self._async_sandbox.start(
                spec,
                delete_on_stop=delete_on_stop,
            ),
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

    def stop(self, *, delete: bool | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run("stop", lambda: self._async_sandbox.stop(delete=delete))
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
