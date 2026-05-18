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
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

from nemo_gym.sandbox.config import SandboxProviderConfig
from nemo_gym.sandbox.observability import (
    current_recorder,
    ensure_env_recorder,
    observability_span,
    push_event_context,
    record_event,
    reset_current_recorder,
    reset_event_context,
    set_current_recorder,
)
from nemo_gym.sandbox.observability.diagnostics import (
    aperf_archive_path,
    aperf_config_from_extensions,
    aperf_start_command,
    aperf_stop_command,
)
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

    def __init__(
        self,
        provider: SandboxProviderConfig | SandboxProvider,
        *,
        observability_context: dict[str, Any] | None = None,
    ) -> None:
        self._provider = (
            create_provider(cast(SandboxProviderConfig, provider)) if isinstance(provider, Mapping) else provider
        )
        self._observability_context = dict(observability_context or {})
        self._handle_observability_context: dict[str, dict[str, Any]] = {}
        self._handle_aperf_sessions: dict[str, dict[str, Any]] = {}

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def _spec_observability_context(self, spec: SandboxSpec) -> dict[str, Any]:
        metadata = dict(spec.metadata)
        context: dict[str, Any] = {
            "provider": self.provider_name,
            "environment_type": "sandbox",
        }

        for key in ("benchmark", "harness", "instance_id", "trajectory_id", "trial_name"):
            value = metadata.get(key)
            if value is not None:
                context[key] = value

        if "harness" not in context and metadata.get("nemo_gym_agent") is not None:
            context["harness"] = metadata["nemo_gym_agent"]

        if "trajectory_id" not in context:
            for key in ("instance_id", "trial_name", "environment_name", "harbor_instance_id"):
                value = metadata.get(key)
                if value is not None:
                    context["trajectory_id"] = value
                    break

        return {**self._observability_context, **context}

    def _handle_context(self, handle: SandboxHandle) -> dict[str, Any]:
        return self._handle_observability_context.get(
            handle.sandbox_id,
            {
                **self._observability_context,
                "provider": self.provider_name,
                "environment_type": "sandbox",
                "sandbox_id": handle.sandbox_id,
            },
        )

    @asynccontextmanager
    async def _observed(self, attributes: dict[str, Any]) -> AsyncIterator[None]:
        recorder = current_recorder() or ensure_env_recorder()
        recorder_token = None
        if recorder is not None and current_recorder() is None:
            recorder_token = set_current_recorder(recorder)
        context_token = push_event_context(attributes)
        try:
            yield
        finally:
            reset_event_context(context_token)
            if recorder_token is not None:
                reset_current_recorder(recorder_token)

    def _remember_handle(self, handle: SandboxHandle, context: dict[str, Any]) -> None:
        handle_context = {**context, "sandbox_id": handle.sandbox_id}
        self._handle_observability_context[handle.sandbox_id] = handle_context

    async def _start_diagnostics(self, handle: SandboxHandle, spec: SandboxSpec, context: dict[str, Any]) -> None:
        aperf_config = aperf_config_from_extensions(
            spec.extensions,
            metadata=spec.metadata,
            sandbox_id=handle.sandbox_id,
            timeout_s=spec.timeout_s,
        )
        if aperf_config is None:
            return

        handle_context = {**context, "sandbox_id": handle.sandbox_id}
        async with self._observed(handle_context):
            async with observability_span(
                "sandbox.diagnostic.aperf.start",
                phase="diagnostic",
                attributes={
                    "provider": self.provider_name,
                    "sandbox_id": handle.sandbox_id,
                    "run_name": aperf_config["run_name"],
                    "output_dir": aperf_config.get("output_dir"),
                },
            ):
                try:
                    result = await self._provider.exec(
                        handle,
                        aperf_start_command(aperf_config),
                        cwd="/",
                        timeout_s=120,
                        user="root",
                    )
                except Exception as e:
                    record_event(
                        "error",
                        "sandbox.diagnostic.aperf.start_error",
                        attributes={"error_type": type(e).__name__, "error": str(e)},
                    )
                    return

                if result.return_code != 0:
                    record_event(
                        "error",
                        "sandbox.diagnostic.aperf.start_failed",
                        attributes={
                            "return_code": result.return_code,
                            "stderr": (result.stderr or "")[-2000:],
                            "stdout": (result.stdout or "")[-2000:],
                        },
                    )
                    return

                self._handle_aperf_sessions[handle.sandbox_id] = {"config": aperf_config}
                record_event(
                    "diagnostic",
                    "sandbox.diagnostic.aperf.started",
                    attributes={
                        "run_name": aperf_config["run_name"],
                        "output_dir": aperf_config.get("output_dir"),
                    },
                )

    async def _stop_diagnostics(self, handle: SandboxHandle, context: dict[str, Any]) -> None:
        session = self._handle_aperf_sessions.pop(handle.sandbox_id, None)
        if session is None:
            return

        aperf_config = session["config"]
        async with self._observed(context):
            async with observability_span(
                "sandbox.diagnostic.aperf.stop",
                phase="diagnostic",
                attributes={
                    "provider": self.provider_name,
                    "sandbox_id": handle.sandbox_id,
                    "run_name": aperf_config["run_name"],
                    "output_dir": aperf_config.get("output_dir"),
                },
            ):
                try:
                    result = await self._provider.exec(
                        handle,
                        aperf_stop_command(aperf_config),
                        cwd="/",
                        timeout_s=180,
                        user="root",
                    )
                except Exception as e:
                    record_event(
                        "error",
                        "sandbox.diagnostic.aperf.stop_error",
                        attributes={"error_type": type(e).__name__, "error": str(e)},
                    )
                    return

                record_event(
                    "diagnostic",
                    "sandbox.diagnostic.aperf.stopped",
                    attributes={
                        "return_code": result.return_code,
                        "stderr": (result.stderr or "")[-2000:],
                        "stdout": (result.stdout or "")[-2000:],
                    },
                )
                local_output_dir = aperf_config.get("local_output_dir")
                if local_output_dir:
                    target_path = Path(local_output_dir) / f"{handle.sandbox_id}.aperf_artifacts.tgz"
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        await self._provider.download_file(handle, aperf_archive_path(aperf_config), target_path)
                    except Exception as e:
                        record_event(
                            "error",
                            "sandbox.diagnostic.aperf.download_error",
                            attributes={
                                "error_type": type(e).__name__,
                                "error": str(e),
                                "target_path": str(target_path),
                            },
                        )
                    else:
                        record_event(
                            "diagnostic",
                            "sandbox.diagnostic.aperf.downloaded",
                            attributes={"target_path": str(target_path)},
                        )

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        context = self._spec_observability_context(spec)
        async with self._observed(context):
            async with observability_span(
                "sandbox.start",
                phase="startup",
                attributes={
                    "provider": self.provider_name,
                    "image": spec.image,
                },
            ):
                handle = await self._provider.create(spec)
            self._remember_handle(handle, context)
            await self._start_diagnostics(handle, spec, context)
            return handle

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        context = self._spec_observability_context(spec)
        async with self._observed(context):
            async with observability_span(
                "sandbox.start_batch",
                phase="startup",
                attributes={
                    "provider": self.provider_name,
                    "count": count,
                    "allow_partial": allow_partial,
                    "image": spec.image,
                },
            ):
                handles = await self._provider.create_batch(spec, count, allow_partial=allow_partial)
            for handle in handles:
                self._remember_handle(handle, context)
                await self._start_diagnostics(handle, spec, context)
            return handles

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        context = {
            **self._observability_context,
            "provider": self.provider_name,
            "environment_type": "sandbox",
            "sandbox_id": sandbox_id,
        }
        async with self._observed(context):
            handle = await self._provider.connect(sandbox_id)
            self._remember_handle(handle, context)
            return handle

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
        context = self._handle_context(handle)
        async with self._observed(context):
            async with observability_span(
                "trajectory.tool",
                phase="execution",
                attributes={
                    "provider": self.provider_name,
                    "sandbox_id": handle.sandbox_id,
                    "cwd": cwd,
                    "timeout_s": timeout_s,
                    "user": user,
                    "command": command,
                },
            ):
                return await self._provider.exec(
                    handle,
                    command,
                    cwd=cwd,
                    env=env,
                    timeout_s=timeout_s,
                    user=user,
                )

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        async with self._observed(self._handle_context(handle)):
            await self._provider.write_file(handle, target_path, data)

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        async with self._observed(self._handle_context(handle)):
            return await self._provider.read_file(handle, source_path)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        async with self._observed(self._handle_context(handle)):
            await self._provider.upload_file(handle, source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        async with self._observed(self._handle_context(handle)):
            await self._provider.download_file(handle, source_path, target_path)

    async def close(self, handle: SandboxHandle, *, delete: bool = False) -> None:
        context = self._handle_context(handle)
        async with self._observed(context):
            async with observability_span(
                "sandbox.cleanup",
                phase="cleanup",
                attributes={
                    "provider": self.provider_name,
                    "sandbox_id": handle.sandbox_id,
                    "delete": delete,
                },
            ):
                try:
                    await self._stop_diagnostics(handle, context)
                    await self._provider.close(handle, delete=delete)
                finally:
                    self._handle_observability_context.pop(handle.sandbox_id, None)
                    self._handle_aperf_sessions.pop(handle.sandbox_id, None)

    async def delete(self, handle: SandboxHandle) -> None:
        await self.close(handle, delete=True)

    async def aclose(self) -> None:
        self._handle_observability_context.clear()
        self._handle_aperf_sessions.clear()
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

    def __init__(
        self,
        provider: SandboxProviderConfig | SandboxProvider,
        *,
        observability_context: dict[str, Any] | None = None,
    ) -> None:
        self._runner = _AsyncLoopRunner()
        try:
            self._async_sandbox = self._runner.call(
                "__init__",
                lambda: AsyncSandbox(provider, observability_context=observability_context),
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

    def __del__(self) -> None:
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.shutdown()
            except Exception:
                pass
