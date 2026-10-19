# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CoreWeave Sandbox provider using the SDK's async gRPC operations.

Requires the optional ``nemo-gym[cwsandbox]`` extra. Images must contain bash;
non-root exec additionally requires su (and getent for numeric user IDs).
Harbor directory transfers are handled by NemoGymSandboxEnvironment.
"""

import asyncio
import logging
import math
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.utils import coerce_config


if TYPE_CHECKING:
    from cwsandbox import Sandbox

LOGGER = logging.getLogger(__name__)


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class CWSandboxConnectionConfig:
    """Control-plane connection; absent credentials use SDK authentication."""

    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    request_timeout_s: float = 3700

    def __post_init__(self) -> None:
        _positive("connection.request_timeout_s", self.request_timeout_s)


@dataclass(frozen=True)
class CWSandboxCreateConfig:
    """Default lifecycle budgets and CoreWeave placement selectors."""

    ttl_s: float = 21600
    ready_timeout_s: float = 900
    runner_ids: list[str] | None = None
    strict_resources: bool = False

    def __post_init__(self) -> None:
        _positive("create.ttl_s", self.ttl_s)
        _positive("create.ready_timeout_s", self.ready_timeout_s)


class CWSandboxProvider:
    """Run prebuilt OCI images on CoreWeave; no Dockerfile or Compose builds.

    CPU/memory requests default to their limits. ``provider_options`` accepts
    ``resource_requests`` (cpu and memory_mib) and ``deny_egress``. Disk quotas
    are unsupported: warn, or reject with ``create.strict_resources``.
    Native port endpoints and interactive PTY sessions are not advertised.
    """

    name = "cwsandbox"

    def __init__(
        self,
        *,
        connection: CWSandboxConnectionConfig | Mapping[str, object] | None = None,
        create: CWSandboxCreateConfig | Mapping[str, object] | None = None,
        default_exec_timeout_s: float = 1800,
    ) -> None:
        self.connection = coerce_config(connection, CWSandboxConnectionConfig)
        self.create_config = coerce_config(create, CWSandboxCreateConfig)
        _positive("default_exec_timeout_s", default_exec_timeout_s)
        self.default_exec_timeout_s = default_exec_timeout_s
        try:
            self._sdk = import_module("cwsandbox")
        except ModuleNotFoundError as exc:
            if exc.name != "cwsandbox":
                raise
            raise ModuleNotFoundError("Install nemo-gym[cwsandbox] to use the cwsandbox provider") from exc
        self._auth = (
            self._sdk.AuthHeaders(headers={"Authorization": f"Bearer {self.connection.api_key}"}, strategy="api_key")
            if self.connection.api_key
            else None
        )

    def _sandbox(self, handle: SandboxHandle) -> "Sandbox":
        if handle.provider_name != self.name:
            raise ValueError(f"Expected a cwsandbox handle, got {handle.provider_name!r}")
        return handle.raw

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Wait for readiness and an exec probe; reclaim failed allocations."""
        if not spec.image:
            raise ValueError("cwsandbox requires a prebuilt SandboxSpec.image")
        if spec.ports:
            raise ValueError("cwsandbox does not support SandboxSpec.ports yet")
        options = spec.provider_options
        unknown = options.keys() - {"resource_requests", "deny_egress"}
        if unknown:
            raise ValueError(f"Unknown cwsandbox provider_options: {sorted(unknown)}")
        deny_egress = options.get("deny_egress", False)
        if not isinstance(deny_egress, bool):
            raise ValueError("deny_egress must be a boolean")
        resources = spec.resources
        if resources.disk_gib is not None:
            message = "cwsandbox cannot enforce disk_gib; storage is determined by the CoreWeave runner"
            if self.create_config.strict_resources:
                raise ValueError(message)
            LOGGER.warning(message)
        limits = {}
        if resources.cpu is not None:
            _positive("resources.cpu", resources.cpu)
            limits["cpu"] = str(resources.cpu)
        if resources.memory_mib is not None:
            _positive("resources.memory_mib", resources.memory_mib)
            limits["memory"] = f"{resources.memory_mib}Mi"
        requests = dict(limits)
        overrides = options.get("resource_requests", {})
        if not isinstance(overrides, Mapping) or overrides.keys() - {"cpu", "memory_mib"}:
            raise ValueError("resource_requests must contain only cpu and memory_mib")
        for key, value in overrides.items():
            _positive(f"resource_requests.{key}", value)
            limit = resources.cpu if key == "cpu" else resources.memory_mib
            if limit is not None and value > limit:
                raise ValueError(f"resource_requests.{key} exceeds the resource limit")
            requests["cpu" if key == "cpu" else "memory"] = str(value) if key == "cpu" else f"{value}Mi"
        sdk_resources = {"requests": requests or None, "limits": limits or None}
        if resources.gpu is not None and (isinstance(resources.gpu, bool) or resources.gpu < 0):
            raise ValueError("resources.gpu must be nonnegative")
        if resources.gpu:
            gpu = {"count": resources.gpu}
            if resources.gpu_type:
                gpu["type"] = resources.gpu_type
            sdk_resources["gpu"] = gpu
        elif resources.gpu_type:
            raise ValueError("resources.gpu_type requires a positive gpu count")
        ttl = spec.ttl_s if spec.ttl_s is not None else self.create_config.ttl_s
        ready_timeout = (
            spec.ready_timeout_s if spec.ready_timeout_s is not None else self.create_config.ready_timeout_s
        )
        _positive("ttl_s", ttl)
        _positive("ready_timeout_s", ready_timeout)
        if spec.entrypoint is not None and not spec.entrypoint:
            raise ValueError("entrypoint must be nonempty when supplied")
        sandbox = self._sdk.Sandbox(
            container_image=spec.image.removeprefix("docker://"),
            base_url=self.connection.base_url,
            auth=self._auth,
            request_timeout_seconds=self.connection.request_timeout_s,
            max_lifetime_seconds=ttl,
            runner_ids=self.create_config.runner_ids,
            resources=sdk_resources if requests or limits or resources.gpu else None,
            environment_variables=dict(spec.env),
            working_dir=spec.workdir,
            annotations=dict(spec.metadata),
            network=self._sdk.NetworkOptions(deny_egress=deny_egress),
            security_context=self._sdk.SecurityContext(run_as_user=0, run_as_group=0, privileged=False),
            command=spec.entrypoint[0] if spec.entrypoint else None,
            args=spec.entrypoint[1:] if spec.entrypoint else None,
        )
        start_task = asyncio.ensure_future(sandbox.start())
        try:
            async with asyncio.timeout(ready_timeout):
                # Retain start on cancellation so its allocated ID can be reclaimed.
                await asyncio.shield(start_task)
                await asyncio.to_thread(sandbox.wait, timeout=ready_timeout)
                if str(sandbox.status) != "running":
                    raise SandboxCreateError(f"cwsandbox reached {sandbox.status}, expected running")
                result = await sandbox.exec(["bash", "-c", "printf nemo-gym-ready"], timeout_seconds=30)
                if result.returncode != 0 or result.stdout != "nemo-gym-ready":
                    raise SandboxCreateError("cwsandbox readiness exec probe failed")
        except BaseException:
            try:
                await asyncio.wait_for(asyncio.shield(start_task), timeout=30)
            except BaseException:
                start_task.cancel()
            if sandbox.sandbox_id:
                try:
                    await self.close(SandboxHandle(sandbox.sandbox_id, self.name, sandbox))
                except Exception:
                    LOGGER.exception("Failed to clean up cwsandbox %s after startup failure", sandbox.sandbox_id)
            raise
        return SandboxHandle(sandbox.sandbox_id, self.name, sandbox)

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
        """Execute once; never retry commands that may have side effects."""
        sandbox = self._sandbox(handle)
        timeout = timeout_s if timeout_s is not None else self.default_exec_timeout_s
        _positive("timeout_s", timeout)
        if env:
            command = (
                f"env {shlex.join([f'{key}={value}' for key, value in env.items()])} bash -c {shlex.quote(command)}"
            )
        if user is not None and str(user) not in {"root", "0"}:
            username = str(user)
            if username.isdecimal():
                lookup = await sandbox.exec(["getent", "passwd", username], timeout_seconds=30)
                if lookup.returncode != 0 or not lookup.stdout.strip():
                    raise ValueError(f"UID {user} not found in sandbox /etc/passwd")
                username = lookup.stdout.split(":", 1)[0].strip()
            command = f"su {shlex.quote(username)} -s /bin/bash -c {shlex.quote(command)}"
        try:
            result = await sandbox.exec(["bash", "-c", command], cwd=cwd, timeout_seconds=timeout)
        except self._sdk.SandboxTimeoutError:
            return SandboxExecResult(None, f"cwsandbox command timed out after {timeout:g}s", 124, "timeout")
        return SandboxExecResult(result.stdout, result.stderr, result.returncode)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Transfer binary contents using the SDK file API."""
        data = await asyncio.to_thread(source_path.read_bytes)
        await self._sandbox(handle).write_file(target_path, data, timeout_seconds=self.connection.request_timeout_s)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download a binary file and create its local parent directory."""
        data = await self._sandbox(handle).read_file(source_path, timeout_seconds=self.connection.request_timeout_s)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target_path.write_bytes, data)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Translate SDK lifecycle states into Gym states."""
        try:
            state = str(await asyncio.to_thread(self._sandbox(handle).get_status))
        except self._sdk.SandboxNotFoundError:
            return SandboxStatus.STOPPED
        return {
            "pending": SandboxStatus.STARTING,
            "creating": SandboxStatus.STARTING,
            "preparing": SandboxStatus.STARTING,
            "running": SandboxStatus.RUNNING,
            "paused": SandboxStatus.PAUSED,
            "terminating": SandboxStatus.STOPPED,
            "completed": SandboxStatus.STOPPED,
            "terminated": SandboxStatus.STOPPED,
            "failed": SandboxStatus.ERROR,
        }.get(state, SandboxStatus.UNKNOWN)

    async def close(self, handle: SandboxHandle) -> None:
        """Stop and delete, attempting deletion even if graceful stop fails."""
        try:
            await self._sandbox(handle).stop(missing_ok=True, wait_for_ready=False)
        finally:
            await self._sdk.Sandbox.delete(
                handle.sandbox_id,
                base_url=self.connection.base_url,
                auth=self._auth,
                timeout_seconds=self.connection.request_timeout_s,
                missing_ok=True,
            )

    async def aclose(self) -> None:
        """The SDK owns its shared event loop; per-sandbox channels close on stop."""
