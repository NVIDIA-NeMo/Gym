# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CoreWeave Sandbox provider using the SDK's async gRPC operations.

Requires the optional ``nemo-gym[cwsandbox]`` extra. Images must contain bash;
non-root exec additionally requires su (and getent for numeric user IDs).
Directory transfers are handled by the calling harness.
"""

import asyncio
import logging
import math
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxEndpoint,
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
    resource requests, network policy, services, volumes, and snapshots. Disk quotas
    are unsupported: warn, or reject with ``create.strict_resources``.
    Declared ports expose HTTPS services with sandbox-scoped share tokens.
    Native interactive PTY sessions are not advertised.
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
        self._attached: list["Sandbox"] = []
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

    @staticmethod
    def _mapping_list(value: object, name: str) -> list[dict[str, Any]]:
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, Mapping) for item in value):
            raise TypeError(f"{name} must be a list of mappings")
        return [dict(item) for item in value]

    def _services(self, spec: SandboxSpec) -> list[object]:
        overrides = {}
        for service in self._mapping_list(spec.provider_options.get("services", []), "services"):
            port = service.get("port")
            if isinstance(port, bool) or not isinstance(port, int) or port not in spec.ports:
                raise ValueError("Every services port must be declared in SandboxSpec.ports")
            if port in overrides:
                raise ValueError(f"Duplicate services port: {port}")
            overrides[port] = service
        return [
            self._sdk.Service(
                **overrides.get(
                    port,
                    {
                        "port": port,
                        "visibility": "public",
                        "endpoint": {"kind": "https", "auth": "share_token"},
                    },
                )
            )
            for port in spec.ports
        ]

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Wait for readiness and an exec probe; reclaim failed allocations."""
        if not spec.image:
            raise ValueError("cwsandbox requires a prebuilt SandboxSpec.image")
        if spec.sidecars:
            raise ValueError("cwsandbox does not support SandboxSpec.sidecars yet")
        options = spec.provider_options
        unknown = options.keys() - {
            "resource_requests",
            "deny_egress",
            "network_policy",
            "services",
            "volumes",
            "file_system_snapshot",
        }
        if unknown:
            raise ValueError(f"Unknown cwsandbox provider_options: {sorted(unknown)}")
        deny_egress = options.get("deny_egress", False)
        if not isinstance(deny_egress, bool):
            raise ValueError("deny_egress must be a boolean")
        network_policy = options.get("network_policy", {})
        if not isinstance(network_policy, Mapping):
            raise TypeError("network_policy must be a mapping of CoreWeave NetworkOptions")
        network_kwargs = dict(network_policy)
        if "deny_egress" in options:
            if "deny_egress" in network_kwargs and network_kwargs["deny_egress"] != deny_egress:
                raise ValueError("deny_egress conflicts with network_policy.deny_egress")
            network_kwargs["deny_egress"] = deny_egress
        elif not network_kwargs:
            network_kwargs["deny_egress"] = False
        services = self._services(spec)
        volumes = []
        for volume in self._mapping_list(options.get("volumes", []), "volumes"):
            volume_type = (
                self._sdk.RegisteredVolumeOptions if "volume_id" in volume else self._sdk.ScratchVolumeOptions
            )
            volumes.append(volume_type(**volume))
        snapshot_options = options.get("file_system_snapshot")
        if snapshot_options is not None:
            if not isinstance(snapshot_options, Mapping):
                raise TypeError("file_system_snapshot must be a mapping")
            snapshot_options = self._sdk.FileSystemSnapshotOptions(**snapshot_options)
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
            network=self._sdk.NetworkOptions(**network_kwargs),
            services=services or None,
            volumes=volumes or None,
            file_system_snapshot=snapshot_options,
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

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a credential-free descriptor; CoreWeave has no lease scopes.

        The owner's create-only HTTPS share token is deliberately excluded.
        Reconnected clients need their own control-plane credentials.
        """
        self._sandbox(handle)
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Attach to a running sandbox without assuming lifecycle ownership."""
        sandbox_id = descriptor.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not sandbox_id.strip():
            raise ValueError("A cwsandbox descriptor requires a nonempty sandbox_id")
        sandbox = await self._sdk.Sandbox.from_id(
            sandbox_id,
            base_url=self.connection.base_url,
            auth=self._auth,
            timeout_seconds=self.connection.request_timeout_s,
        )
        if str(sandbox.status) != "running":
            await asyncio.to_thread(sandbox._cleanup_channels)
            raise RuntimeError(f"Cannot connect to cwsandbox {sandbox_id!r}: status is {sandbox.status}")
        self._attached.append(sandbox)
        return SandboxHandle(sandbox_id, self.name, sandbox)

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve an HTTPS service, without forwarding the control-plane key.

        Share-token endpoints require the create handle: the SDK's from_id()
        response does not include the owner-only token. The owner can pass the
        resolved SandboxEndpoint to a borrower separately if needed.
        """
        sandbox = self._sandbox(handle)
        await asyncio.to_thread(sandbox.get_status)
        for endpoint in sandbox.service_endpoints:
            if endpoint.port != port or not endpoint.url:
                continue
            headers = {}
            if str(endpoint.auth) == "share_token":
                token = sandbox.endpoint_share_token
                if token is None:
                    raise ValueError("Reconnected endpoints require the owner's create-only share token")
                headers = token.as_headers()
            elif str(endpoint.auth) != "open":
                raise ValueError(f"Unsupported cwsandbox endpoint authentication: {endpoint.auth}")
            return SandboxEndpoint(endpoint.url, headers)
        raise ValueError(f"No HTTPS endpoint is available for sandbox {handle.sandbox_id!r} port {port}")

    async def snapshot(self, handle: SandboxHandle) -> str:
        """Capture the configured filesystem mount and wait for a ready snapshot.

        Snapshot storage must be enabled for the organization. Snapshots outlive
        their sandbox and must be removed explicitly with delete_snapshot().
        """
        return await self._sandbox(handle).snapshot(wait_for_ready=True)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete one caller-owned snapshot; an already-deleted id is harmless."""
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot_id must be a nonempty string")
        await self._sdk.Sandbox.delete_snapshot(
            snapshot_id,
            base_url=self.connection.base_url,
            auth=self._auth,
            timeout_seconds=self.connection.request_timeout_s,
            missing_ok=True,
        )

    async def aclose(self) -> None:
        """Release attached clients without stopping the owner's sandboxes."""
        for sandbox in self._attached:
            # The SDK has no public detach operation. Its synchronous cleanup
            # helper schedules channel closure on the SDK's own event loop.
            await asyncio.to_thread(sandbox._cleanup_channels)
        self._attached.clear()
