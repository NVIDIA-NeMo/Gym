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

"""Provider-neutral public sandbox API.

This module is the boundary Gym code should use when it needs a sandbox.
Provider packages implement the lower-level async protocol; callers use
``AsyncSandbox`` in async code and ``Sandbox`` in synchronous integrations.
"""

import asyncio
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
    create_provider,
)


T = TypeVar("T")


def rewrite_image(image: str | None, rewrites: list[dict[str, str]]) -> str | None:
    """Apply ordered image-prefix rewrites used by sandbox configs."""
    if image is None:
        return None
    for rewrite in rewrites:
        from_prefix = rewrite["from"]
        to_prefix = rewrite["to"]
        if image.startswith(from_prefix):
            return to_prefix + image[len(from_prefix) :]
    return image


class AsyncSandbox:
    """Async public facade for provider-backed sandbox operations."""

    def __init__(self, provider: Mapping[str, Any] | SandboxProvider) -> None:
        self._provider = create_provider(provider) if isinstance(provider, Mapping) else provider

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return await self._provider.create(spec)

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        return await self._provider.create_batch(spec, count, allow_partial=allow_partial)

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        return await self._provider.connect(sandbox_id)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        return await self._provider.exec(
            handle,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            user=user,
        )

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        await self._provider.write_file(handle, target_path, data)

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        return await self._provider.read_file(handle, source_path)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await self._provider.upload_file(handle, source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        await self._provider.download_file(handle, source_path, target_path)

    async def close(self, handle: SandboxHandle, *, delete: bool = False) -> None:
        await self._provider.close(handle, delete=delete)

    async def delete(self, handle: SandboxHandle) -> None:
        await self.close(handle, delete=True)

    async def aclose(self) -> None:
        close_provider = getattr(self._provider, "aclose", None)
        if close_provider is not None:
            await close_provider()

    async def shutdown(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> "AsyncSandbox":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def handle_reference(self, handle: SandboxHandle) -> Any:
        make_reference = getattr(self._provider, "handle_reference", None)
        if make_reference is None:
            return handle
        return make_reference(handle)

    async def materialize_handle(self, value: Any) -> SandboxHandle:
        materialize = getattr(self._provider, "materialize_handle", None)
        if materialize is None:
            if isinstance(value, SandboxHandle):
                return value
            raise ValueError(f"Provider {self.provider_name!r} cannot materialize handle references")
        result = materialize(value)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, SandboxHandle):
            raise TypeError(f"materialize_handle must return SandboxHandle, got {type(result).__name__}")
        return result


class _AsyncLoopRunner:
    """Run async sandbox operations for sync integrations on one private loop."""

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
    """Sync public facade for provider-backed sandbox operations."""

    def __init__(self, provider: Mapping[str, Any] | SandboxProvider) -> None:
        self._runner = _AsyncLoopRunner()
        try:
            self._async_sandbox = self._runner.call(
                "__init__",
                lambda: AsyncSandbox(provider),
            )
        except BaseException:
            self._runner.close()
            raise
        self._closed = False

    @property
    def provider_name(self) -> str:
        return self._runner.call("provider_name", lambda: self._async_sandbox.provider_name)

    def create(self, spec: SandboxSpec) -> SandboxHandle:
        return self._runner.run("create", lambda: self._async_sandbox.create(spec))

    def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        return self._runner.run(
            "create_batch",
            lambda: self._async_sandbox.create_batch(spec, count, allow_partial=allow_partial),
        )

    def connect(self, sandbox_id: str) -> SandboxHandle:
        return self._runner.run("connect", lambda: self._async_sandbox.connect(sandbox_id))

    def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        return self._runner.run(
            "exec",
            lambda: self._async_sandbox.exec(
                handle,
                command,
                cwd=cwd,
                env=env,
                timeout_s=timeout_s,
                user=user,
            ),
        )

    def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        self._runner.run("write_file", lambda: self._async_sandbox.write_file(handle, target_path, data))

    def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        return self._runner.run("read_file", lambda: self._async_sandbox.read_file(handle, source_path))

    def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self._runner.run("upload_file", lambda: self._async_sandbox.upload_file(handle, source_path, target_path))

    def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self._runner.run("download_file", lambda: self._async_sandbox.download_file(handle, source_path, target_path))

    def close(self, handle: SandboxHandle, *, delete: bool = False) -> None:
        self._runner.run("close", lambda: self._async_sandbox.close(handle, delete=delete))

    def delete(self, handle: SandboxHandle) -> None:
        self.close(handle, delete=True)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run("shutdown", self._async_sandbox.shutdown)
        finally:
            self._runner.close()

    def handle_reference(self, handle: SandboxHandle) -> Any:
        return self._runner.call("handle_reference", lambda: self._async_sandbox.handle_reference(handle))

    def materialize_handle(self, value: Any) -> SandboxHandle:
        return self._runner.run("materialize_handle", lambda: self._async_sandbox.materialize_handle(value))

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def __del__(self) -> None:  # pragma: no cover
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.shutdown()
            except Exception:
                pass
