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
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from nemo_gym.sandbox.providers import (
    ImageBuildRequest,
    OutsideEndpoint,
    SandboxAddressProvider,
    SandboxAttachProvider,
    SandboxExecResult,
    SandboxHandle,
    SandboxHandleReferenceProvider,
    SandboxImageBuildProvider,
    SandboxInlineFileProvider,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
    SandboxStatusProvider,
    create_provider,
)


T = TypeVar("T")


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if hasattr(value, "__await__"):
        return await value
    return value


class AsyncSandbox:
    """Async public facade for provider-backed sandbox operations."""

    def __init__(self, provider: Mapping[str, Any] | SandboxProvider) -> None:
        self._provider = create_provider(provider) if isinstance(provider, Mapping) else provider

    @property
    def provider_name(self) -> str:
        return self._provider.name

    async def build_images(self, request: ImageBuildRequest) -> list[str]:
        if not isinstance(self._provider, SandboxImageBuildProvider):
            raise NotImplementedError(f"Provider {self.provider_name!r} does not support sandbox image builds")
        return await self._provider.build_images(request)

    async def _resolve_image_build(self, spec: SandboxSpec) -> SandboxSpec:
        if spec.image_build is None:
            return spec
        built_images = await self.build_images(ImageBuildRequest(specs=[spec.image_build]))
        if not built_images:
            raise ValueError("build_images returned no image references")
        return replace(spec, image=spec.image or built_images[0])

    async def _write_initial_files(self, handle: SandboxHandle, files: dict[str, str]) -> None:
        for target_path, contents in files.items():
            await self.write_file(handle, target_path, contents)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        spec = await self._resolve_image_build(spec)
        handle = await self._provider.create(spec)
        try:
            await self._write_initial_files(handle, spec.files)
        except Exception:
            await self.close(handle, delete=True)
            raise
        return handle

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        spec = await self._resolve_image_build(spec)
        handles = await self._provider.create_batch(spec, count, allow_partial=allow_partial)
        try:
            await asyncio.gather(*(self._write_initial_files(handle, spec.files) for handle in handles))
        except Exception:
            await asyncio.gather(*(self.close(handle, delete=True) for handle in handles), return_exceptions=True)
            raise
        return handles

    async def start(
        self,
        spec: SandboxSpec,
        *,
        outside_endpoints: list[OutsideEndpoint] | None = None,
        delete_on_stop: bool = False,
    ) -> "AsyncSandboxInstance":
        if outside_endpoints:
            endpoint_env = {endpoint.env_var: endpoint.url for endpoint in outside_endpoints}
            spec = replace(spec, env={**spec.env, **endpoint_env})
        handle = await self.create(spec)
        return AsyncSandboxInstance(
            sandbox=self,
            spec=spec,
            handle=handle,
            delete_on_stop=delete_on_stop,
        )

    async def attach(self, sandbox_id: str) -> SandboxHandle:
        if isinstance(self._provider, SandboxAttachProvider):
            return await self._provider.attach(sandbox_id)
        connect = getattr(self._provider, "connect", None)
        if connect is None:
            raise NotImplementedError(f"Provider {self.provider_name!r} does not support attaching to sandboxes")
        return await connect(sandbox_id)

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        """Compatibility alias for ``attach``."""
        return await self.attach(sandbox_id)

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
        return await self._provider.exec(
            handle,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            user=user,
        )

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        if isinstance(self._provider, SandboxInlineFileProvider):
            await self._provider.write_file(handle, target_path, data)
            return
        with tempfile.TemporaryDirectory(prefix="nemo-gym-sandbox-upload-") as tmp_dir:
            source_path = Path(tmp_dir) / "contents"
            if isinstance(data, str):
                source_path.write_text(data, encoding="utf-8")
            else:
                source_path.write_bytes(data)
            await self.upload_file(handle, source_path, target_path)

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        if isinstance(self._provider, SandboxInlineFileProvider):
            return await self._provider.read_file(handle, source_path)
        with tempfile.TemporaryDirectory(prefix="nemo-gym-sandbox-download-") as tmp_dir:
            target_path = Path(tmp_dir) / "contents"
            await self.download_file(handle, source_path, target_path)
            return target_path.read_bytes()

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        await self._provider.upload_file(handle, source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        await self._provider.download_file(handle, source_path, target_path)

    async def close(self, handle: SandboxHandle, *, delete: bool = False) -> None:
        await self._provider.close(handle, delete=delete)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        if not isinstance(self._provider, SandboxStatusProvider):
            return SandboxStatus.UNKNOWN
        return await self._provider.status(handle)

    async def container_ip(self, handle: SandboxHandle) -> str | None:
        if not isinstance(self._provider, SandboxAddressProvider):
            return None
        return await self._provider.container_ip(handle)

    async def aclose(self) -> None:
        await self._provider.aclose()

    async def shutdown(self) -> None:
        """Close provider-scoped resources such as SDK clients or warm pools."""
        await self.aclose()

    async def __aenter__(self) -> "AsyncSandbox":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def handle_reference(self, handle: SandboxHandle) -> Any:
        if not isinstance(self._provider, SandboxHandleReferenceProvider):
            return handle
        return await _maybe_await(self._provider.handle_reference(handle))

    async def materialize_handle(self, value: Any) -> SandboxHandle:
        if not isinstance(self._provider, SandboxHandleReferenceProvider):
            if isinstance(value, SandboxHandle):
                return value
            raise ValueError(f"Provider {self.provider_name!r} cannot materialize handle references")
        result = await _maybe_await(self._provider.materialize_handle(value))
        if not isinstance(result, SandboxHandle):
            raise TypeError(f"materialize_handle must return SandboxHandle, got {type(result).__name__}")
        return result


class AsyncSandboxInstance:
    """Evaluator-style async sandbox object returned by ``AsyncSandbox.start``."""

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox,
        spec: SandboxSpec,
        handle: SandboxHandle,
        delete_on_stop: bool,
    ) -> None:
        self._sandbox = sandbox
        self._spec = spec
        self._handle = handle
        self._delete_on_stop = delete_on_stop
        self._stopped = False

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    @property
    def handle(self) -> SandboxHandle:
        return self._handle

    async def exec(
        self,
        command: str,
        timeout_sec: int | float | None = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        return await self._sandbox.exec(
            self._handle,
            command,
            cwd=cwd if cwd is not None else self._spec.workdir,
            env=env,
            timeout_s=timeout_s if timeout_s is not None else timeout_sec,
            user=user,
        )

    async def write_file(self, target_path: str, data: str | bytes) -> None:
        await self._sandbox.write_file(self._handle, target_path, data)

    async def read_file(self, source_path: str) -> bytes:
        return await self._sandbox.read_file(self._handle, source_path)

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        await self._sandbox.upload_file(self._handle, Path(local_path), remote_path)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        await self._sandbox.download_file(self._handle, remote_path, Path(local_path))

    async def status(self) -> SandboxStatus:
        return await self._sandbox.status(self._handle)

    async def is_running(self) -> bool:
        return await self.status() == SandboxStatus.RUNNING

    async def container_ip(self) -> str | None:
        return await self._sandbox.container_ip(self._handle)

    async def stop(self, *, delete: bool | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self._sandbox.close(
            self._handle,
            delete=self._delete_on_stop if delete is None else delete,
        )

    async def close(self, *, delete: bool | None = None) -> None:
        await self.stop(delete=delete)

    async def __aenter__(self) -> "AsyncSandboxInstance":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()


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

    def build_images(self, request: ImageBuildRequest) -> list[str]:
        return self._runner.run("build_images", lambda: self._async_sandbox.build_images(request))

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

    def start(
        self,
        spec: SandboxSpec,
        *,
        outside_endpoints: list[OutsideEndpoint] | None = None,
        delete_on_stop: bool = False,
    ) -> "SandboxInstance":
        async_instance = self._runner.run(
            "start",
            lambda: self._async_sandbox.start(
                spec,
                outside_endpoints=outside_endpoints,
                delete_on_stop=delete_on_stop,
            ),
        )
        return SandboxInstance(self, async_instance)

    def attach(self, sandbox_id: str) -> SandboxHandle:
        return self._runner.run("attach", lambda: self._async_sandbox.attach(sandbox_id))

    def connect(self, sandbox_id: str) -> SandboxHandle:
        """Compatibility alias for ``attach``."""
        return self.attach(sandbox_id)

    def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
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

    def status(self, handle: SandboxHandle) -> SandboxStatus:
        return self._runner.run("status", lambda: self._async_sandbox.status(handle))

    def container_ip(self, handle: SandboxHandle) -> str | None:
        return self._runner.run("container_ip", lambda: self._async_sandbox.container_ip(handle))

    def shutdown(self) -> None:
        """Close provider-scoped resources such as SDK clients or warm pools."""
        if self._closed:
            return
        self._closed = True
        try:
            self._runner.run("shutdown", self._async_sandbox.shutdown)
        finally:
            self._runner.close()

    def handle_reference(self, handle: SandboxHandle) -> Any:
        return self._runner.run("handle_reference", lambda: self._async_sandbox.handle_reference(handle))

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


class SandboxInstance:
    """Evaluator-style sync sandbox object returned by ``Sandbox.start``."""

    def __init__(self, owner: Sandbox, async_instance: AsyncSandboxInstance) -> None:
        self._owner = owner
        self._async_instance = async_instance

    @property
    def spec(self) -> SandboxSpec:
        return self._owner._runner.call("spec", lambda: self._async_instance.spec)

    @property
    def handle(self) -> SandboxHandle:
        return self._owner._runner.call("handle", lambda: self._async_instance.handle)

    def exec(
        self,
        command: str,
        timeout_sec: int | float | None = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
        timeout_s: int | float | None = None,
    ) -> SandboxExecResult:
        return self._owner._runner.run(
            "instance.exec",
            lambda: self._async_instance.exec(
                command,
                timeout_sec=timeout_sec,
                cwd=cwd,
                env=env,
                user=user,
                timeout_s=timeout_s,
            ),
        )

    def write_file(self, target_path: str, data: str | bytes) -> None:
        self._owner._runner.run(
            "instance.write_file",
            lambda: self._async_instance.write_file(target_path, data),
        )

    def read_file(self, source_path: str) -> bytes:
        return self._owner._runner.run("instance.read_file", lambda: self._async_instance.read_file(source_path))

    def upload(self, local_path: Path | str, remote_path: str) -> None:
        self._owner._runner.run(
            "instance.upload",
            lambda: self._async_instance.upload(local_path, remote_path),
        )

    def download(self, remote_path: str, local_path: Path | str) -> None:
        self._owner._runner.run(
            "instance.download",
            lambda: self._async_instance.download(remote_path, local_path),
        )

    def status(self) -> SandboxStatus:
        return self._owner._runner.run("instance.status", self._async_instance.status)

    @property
    def is_running(self) -> bool:
        return self.status() == SandboxStatus.RUNNING

    def container_ip(self) -> str | None:
        return self._owner._runner.run("instance.container_ip", self._async_instance.container_ip)

    def stop(self, *, delete: bool | None = None) -> None:
        self._owner._runner.run("instance.stop", lambda: self._async_instance.stop(delete=delete))

    def close(self, *, delete: bool | None = None) -> None:
        self.stop(delete=delete)

    def __enter__(self) -> "SandboxInstance":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
