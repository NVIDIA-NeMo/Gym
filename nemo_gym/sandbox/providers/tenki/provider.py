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

"""Tenki provider implementation."""

import asyncio
import logging
import math
import posixpath
import shlex
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.attribution import RUN_KEY, log_attribution_once, resolve_attribution, resolve_run_id
from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_TRANSFER_ROOT = "/home/tenki/.nemo-gym-transfers"
SANDBOX_RUNTIME_RETURN_CODE = 125


class TenkiCreateError(SandboxCreateError):
    """Raised when Tenki cannot create a sandbox."""


class TenkiCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created Tenki sandbox cannot execute a probe command."""


def _require_tenki_sdk() -> tuple[Any, Any, Any, Any]:
    try:
        from tenki import AsyncClient, CommandTimeoutError, SessionNotFoundError, SessionTerminatedError
    except ImportError as exc:
        raise ModuleNotFoundError(
            "The Tenki SDK is required for the tenki sandbox provider. Install nemo-gym[sandbox] "
            "before using env.sandbox.provider.name=tenki."
        ) from exc
    return AsyncClient, CommandTimeoutError, SessionNotFoundError, SessionTerminatedError


class TenkiConfigBase:
    """Shared strict construction for Tenki provider config blocks."""

    @classmethod
    def from_mapping(cls, value: Any) -> Any:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"{cls.__name__} must be a mapping or {cls.__name__} instance")
        allowed = {item.name for item in fields(cls) if item.init}
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(f"Unsupported Tenki {cls.__name__} settings: {', '.join(unknown)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class TenkiConnectionConfig(TenkiConfigBase):
    """Tenki API client connection settings."""

    auth_token: str | None = None
    base_url: str | None = None
    gateway_url: str | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_s is not None and (not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("connection.timeout_s must be > 0")


@dataclass(frozen=True)
class TenkiCreateConfig(TenkiConfigBase):
    """Tenki sandbox creation and readiness settings."""

    ready_timeout_s: float = 300.0
    max_duration_s: float = 3600.0
    probe_command: str | None = "printf nemo-gym-tenki-ready"
    probe_expected_stdout: str | None = "nemo-gym-tenki-ready"
    probe_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.ready_timeout_s) or self.ready_timeout_s <= 0:
            raise ValueError("create.ready_timeout_s must be > 0")
        if not math.isfinite(self.max_duration_s) or self.max_duration_s <= 0:
            raise ValueError("create.max_duration_s must be > 0")
        if self.probe_command is not None and not isinstance(self.probe_command, str):
            raise TypeError("create.probe_command must be a string or null")
        if self.probe_expected_stdout is not None and not isinstance(self.probe_expected_stdout, str):
            raise TypeError("create.probe_expected_stdout must be a string or null")
        if self.probe_command is not None and (not math.isfinite(self.probe_timeout_s) or self.probe_timeout_s <= 0):
            raise ValueError("create.probe_timeout_s must be > 0")


@dataclass(frozen=True)
class TenkiOperationsConfig(TenkiConfigBase):
    """Tenki post-create operation settings."""

    close_timeout_s: float = 120.0
    close_poll_interval_s: float = 0.5
    transfer_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.close_timeout_s) or self.close_timeout_s <= 0:
            raise ValueError("operations.close_timeout_s must be > 0")
        if not math.isfinite(self.close_poll_interval_s) or self.close_poll_interval_s <= 0:
            raise ValueError("operations.close_poll_interval_s must be > 0")
        if not math.isfinite(self.transfer_timeout_s) or self.transfer_timeout_s <= 0:
            raise ValueError("operations.transfer_timeout_s must be > 0")


@dataclass(frozen=True)
class TenkiProviderOptions:
    """Recognized per-sandbox options carried in ``SandboxSpec.provider_options``."""

    workspace_id: str | None = None
    name: str | None = None
    allow_inbound: bool = True
    allow_outbound: bool = True
    idle_timeout_minutes: int | None = None
    template: str | None = None
    snapshot_id: str | None = None
    volumes: tuple[Mapping[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    sticky: bool = False
    wait_for_runtime: bool = False

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> "TenkiProviderOptions":
        if options is None:
            return cls()
        if not isinstance(options, Mapping):
            raise TypeError("Tenki provider_options must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(str(key) for key in options if key not in allowed)
        if unknown:
            raise ValueError(f"Unknown Tenki provider_options keys: {unknown}. Allowed keys: {sorted(allowed)}")

        volumes = options.get("volumes", ())
        if not isinstance(volumes, (list, tuple)) or not all(isinstance(item, Mapping) for item in volumes):
            raise TypeError("provider_options['volumes'] must be a list of mappings")
        tags = options.get("tags", ())
        if isinstance(tags, str):
            tags = (tags,)
        if not isinstance(tags, (list, tuple)) or not all(isinstance(item, str) for item in tags):
            raise TypeError("provider_options['tags'] must be a string or list of strings")

        normalized: dict[str, Any] = dict(options)
        normalized["volumes"] = tuple(dict(item) for item in volumes)
        normalized["tags"] = tuple(tags)
        return cls(**normalized)

    def __post_init__(self) -> None:
        if isinstance(self.idle_timeout_minutes, bool):
            raise TypeError("provider_options.idle_timeout_minutes must be an integer")
        if self.idle_timeout_minutes is not None and self.idle_timeout_minutes <= 0:
            raise ValueError("provider_options.idle_timeout_minutes must be > 0")
        for name in ("workspace_id", "name", "template", "snapshot_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"provider_options.{name} must be a non-empty string")
        for name in ("allow_inbound", "allow_outbound", "sticky", "wait_for_runtime"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"provider_options.{name} must be a boolean")


@dataclass
class _TenkiSandbox:
    sdk: Any
    workdir: str | None = None
    endpoints: dict[int, SandboxEndpoint] = field(default_factory=dict)


def _to_sandbox_status(value: Any) -> SandboxStatus:
    state = str(value or "").upper()
    if state in {"RUNNING", "READY"}:
        return SandboxStatus.RUNNING
    if state in {"CREATING", "PENDING", "PROVISIONING", "STARTING"}:
        return SandboxStatus.STARTING
    if state in {"PAUSED", "TERMINATING", "TERMINATED", "STOPPED"}:
        return SandboxStatus.STOPPED
    if state in {"ERROR", "FAILED", "UNHEALTHY"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


def _resource_kwargs(resources: SandboxResources) -> dict[str, int]:
    if resources.gpu is not None or resources.gpu_type is not None:
        raise ValueError("The Tenki SDK does not expose GPU selection; resources.gpu and gpu_type are unsupported")
    result: dict[str, int] = {}
    if resources.cpu is not None:
        if resources.cpu <= 0:
            raise ValueError("resources.cpu must be > 0")
        result["cpu_cores"] = math.ceil(resources.cpu)
    if resources.memory_mib is not None:
        if resources.memory_mib <= 0:
            raise ValueError("resources.memory_mib must be > 0")
        result["memory_mb"] = resources.memory_mib
    if resources.disk_gib is not None:
        if resources.disk_gib <= 0:
            raise ValueError("resources.disk_gib must be > 0")
        result["disk_size_gb"] = resources.disk_gib
    return result


def _effective_remote_path(state: _TenkiSandbox, path: str) -> str:
    if posixpath.isabs(path):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(state.workdir or "/home/tenki", path))


def _is_home_path(path: str) -> bool:
    normalized = posixpath.normpath(path)
    return normalized == "/home/tenki" or normalized.startswith("/home/tenki/")


class TenkiProvider:
    """Run Gym sandboxes with the asynchronous Tenki SDK."""

    name = "tenki"

    def __init__(
        self,
        connection: TenkiConnectionConfig | Mapping[str, Any] | None = None,
        create: TenkiCreateConfig | Mapping[str, Any] | None = None,
        operations: TenkiOperationsConfig | Mapping[str, Any] | None = None,
        attribution: Mapping[str, str | None] | None = None,
    ) -> None:
        self._connection = TenkiConnectionConfig.from_mapping(connection)
        self._create = TenkiCreateConfig.from_mapping(create)
        self._operations = TenkiOperationsConfig.from_mapping(operations)
        if attribution is not None and not isinstance(attribution, Mapping):
            raise TypeError("attribution must be a mapping")
        allowed_attribution = {"team", "user", "workload", "run"}
        unknown_attribution = sorted(str(key) for key in (attribution or {}) if key not in allowed_attribution)
        if unknown_attribution:
            raise ValueError(f"Unsupported Tenki attribution settings: {', '.join(unknown_attribution)}")
        self._attribution = dict(attribution or {})
        self._client_instance: Any | None = None

    def _client(self) -> Any:
        if self._client_instance is None:
            AsyncClient, _, _, _ = _require_tenki_sdk()
            kwargs = {
                "auth_token": self._connection.auth_token,
                "base_url": self._connection.base_url,
                "gateway_url": self._connection.gateway_url,
                "timeout": self._connection.timeout_s,
            }
            self._client_instance = AsyncClient(**{key: value for key, value in kwargs.items() if value is not None})
        return self._client_instance

    def _metadata(self, metadata: Mapping[str, Any]) -> dict[str, str]:
        resolved = resolve_attribution(
            team=self._attribution.get("team"),
            user=self._attribution.get("user"),
            workload=self._attribution.get("workload"),
        )
        resolved[RUN_KEY] = resolve_run_id(self._attribution.get("run"))
        resolved.update({str(key): str(value) for key, value in metadata.items()})
        log_attribution_once(resolved)
        return resolved

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a descriptor for reconnecting to a Tenki sandbox from another process."""
        state: _TenkiSandbox = handle.raw
        return {"sandbox_id": handle.sandbox_id, "workdir": state.workdir}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Rebuild a live handle from a Tenki sandbox id."""
        if not isinstance(descriptor, Mapping):
            raise TypeError("Tenki sandbox descriptor must be a mapping")
        sandbox_id = descriptor.get("sandbox_id")
        if not isinstance(sandbox_id, str) or not sandbox_id.strip():
            raise ValueError("Tenki sandbox descriptor requires a non-empty sandbox_id")
        workdir = descriptor.get("workdir")
        if workdir is not None and not isinstance(workdir, str):
            raise TypeError("Tenki sandbox descriptor workdir must be a string or null")
        sandbox = await self._client().get(sandbox_id.strip())
        state = _TenkiSandbox(sdk=sandbox, workdir=workdir)
        return SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=state)

    async def _terminate_partial(self, sandbox: Any, sandbox_id: str) -> None:
        _, _, SessionNotFoundError, SessionTerminatedError = _require_tenki_sdk()
        try:
            await asyncio.wait_for(sandbox.close_if_open(), timeout=self._operations.close_timeout_s)
        except (SessionNotFoundError, SessionTerminatedError):
            return
        except Exception as cleanup_error:
            LOGGER.warning(
                "Failed to terminate Tenki sandbox after create failure; sandbox_id=%s; error=%r",
                sandbox_id,
                cleanup_error,
            )
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._operations.close_timeout_s
        while True:
            try:
                info = await sandbox.refresh()
            except (SessionNotFoundError, SessionTerminatedError):
                return
            except Exception as cleanup_error:
                LOGGER.warning(
                    "Failed to verify Tenki sandbox termination after create failure; sandbox_id=%s; error=%r",
                    sandbox_id,
                    cleanup_error,
                )
                return
            if _to_sandbox_status(info.state) is SandboxStatus.STOPPED:
                return
            if loop.time() >= deadline:
                LOGGER.warning(
                    "Tenki sandbox did not terminate after create failure; sandbox_id=%s; last_state=%r",
                    sandbox_id,
                    info.state,
                )
                return
            await asyncio.sleep(self._operations.close_poll_interval_s)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        if self._create.probe_command is None:
            return
        result = await self.exec(handle, self._create.probe_command, timeout_s=self._create.probe_timeout_s)
        if result.return_code != 0 or (
            self._create.probe_expected_stdout is not None
            and self._create.probe_expected_stdout not in (result.stdout or "")
        ):
            raise TenkiCreateVerificationError(
                f"Tenki sandbox {handle.sandbox_id!r} failed readiness probe: "
                f"return_code={result.return_code}, stderr={(result.stderr or '')[:200]!r}"
            )

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a ready Tenki sandbox and clean up every admitted failure."""
        if spec.entrypoint is not None:
            raise ValueError("The Tenki provider does not support SandboxSpec.entrypoint")
        if isinstance(spec.ttl_s, bool) or isinstance(spec.ready_timeout_s, bool):
            raise ValueError("ttl_s and ready_timeout_s must be numeric durations, not booleans")
        options = TenkiProviderOptions.from_mapping(spec.provider_options)
        source_count = sum(value is not None for value in (spec.image, options.template, options.snapshot_id))
        if source_count > 1:
            raise ValueError("Set only one of SandboxSpec.image, provider_options.template, or snapshot_id")
        ready_timeout_s = float(
            spec.ready_timeout_s if spec.ready_timeout_s is not None else self._create.ready_timeout_s
        )
        if not math.isfinite(ready_timeout_s) or ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be > 0")
        max_duration_s = float(spec.ttl_s if spec.ttl_s is not None else self._create.max_duration_s)
        if not math.isfinite(max_duration_s) or max_duration_s <= 0:
            raise ValueError("ttl_s must be > 0")

        kwargs: dict[str, Any] = {
            "workspace_id": options.workspace_id,
            "name": options.name or f"nemo-gym-{uuid.uuid4().hex[:20]}",
            "wait": True,
            "timeout": ready_timeout_s,
            "allow_inbound": options.allow_inbound,
            "allow_outbound": options.allow_outbound,
            "max_duration": max_duration_s,
            "idle_timeout_minutes": options.idle_timeout_minutes,
            "metadata": self._metadata(spec.metadata),
            "tags": list(options.tags) or None,
            "env": {str(key): str(value) for key, value in spec.env.items()} or None,
            "image": spec.image,
            "from_template_spec": options.template,
            "snapshot_id": options.snapshot_id,
            "volumes": [dict(item) for item in options.volumes] or None,
            "sticky": options.sticky,
            "wait_for_runtime": options.wait_for_runtime,
            **_resource_kwargs(spec.resources),
        }
        create_task = asyncio.create_task(self._client().create(**kwargs))
        try:
            # asyncio.wait does not cancel its child task when this caller is cancelled.
            # That lets us observe an admitted sandbox and terminate it before propagating cancellation.
            await asyncio.wait({create_task})
            sandbox = create_task.result()
        except asyncio.CancelledError:
            try:
                sandbox = await create_task
            except Exception as exc:
                partial = getattr(exc, "sandbox", None)
                if partial is not None:
                    await self._terminate_partial(partial, str(getattr(partial, "id", "<unknown>")))
            else:
                await self._terminate_partial(sandbox, str(sandbox.id))
            raise
        except Exception as exc:
            partial = getattr(exc, "sandbox", None)
            if partial is not None:
                await self._terminate_partial(partial, str(getattr(partial, "id", "<unknown>")))
            raise TenkiCreateError(
                f"Failed to create Tenki sandbox within {ready_timeout_s:g}s; image={spec.image!r}"
            ) from exc

        state = _TenkiSandbox(sdk=sandbox, workdir=spec.workdir)
        handle = SandboxHandle(sandbox_id=str(sandbox.id), provider_name=self.name, raw=state)
        try:
            await self._verify_created_handle(handle)
            for port in spec.ports:
                exposed = await sandbox.expose_port(port, ttl=max_duration_s)
                state.endpoints[port] = SandboxEndpoint(endpoint=str(exposed.url))
        except BaseException:
            await self._terminate_partial(sandbox, handle.sandbox_id)
            raise
        return handle

    @staticmethod
    def _privileged(user: str | int | None) -> bool:
        if isinstance(user, bool):
            raise NotImplementedError("The Tenki provider does not accept a boolean command user")
        if user in (None, "tenki", 1000):
            return False
        if user in ("root", 0):
            return True
        raise NotImplementedError(
            "The Tenki provider supports the default tenki user and root only; use user='root' for privileged exec"
        )

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
        """Run a shell command inside a Tenki sandbox."""
        state: _TenkiSandbox = handle.raw
        _, CommandTimeoutError, SessionNotFoundError, SessionTerminatedError = _require_tenki_sdk()
        try:
            result = await state.sdk.exec(
                "bash",
                "-lc",
                command,
                cwd=cwd if cwd is not None else state.workdir,
                env={str(key): str(value) for key, value in (env or {}).items()} or None,
                timeout=timeout_s,
                privileged=self._privileged(user),
            )
        except CommandTimeoutError as exc:
            return SandboxExecResult(stdout=None, stderr=str(exc), return_code=124, error_type="timeout")
        except (SessionNotFoundError, SessionTerminatedError) as exc:
            return SandboxExecResult(
                stdout=None,
                stderr=str(exc),
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )
        return SandboxExecResult(
            stdout=bytes(result.stdout).decode("utf-8", errors="replace") if result.stdout is not None else None,
            stderr=bytes(result.stderr).decode("utf-8", errors="replace") if result.stderr is not None else None,
            return_code=int(result.exit_code),
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file, including to paths that require privileged access."""
        state: _TenkiSandbox = handle.raw
        destination = _effective_remote_path(state, target_path)
        data = await asyncio.to_thread(source_path.read_bytes)
        if _is_home_path(destination):
            parent = posixpath.dirname(destination)
            await asyncio.wait_for(state.sdk.fs.mkdir(parent), timeout=self._operations.transfer_timeout_s)
            await asyncio.wait_for(
                state.sdk.fs.write_bytes(destination, data), timeout=self._operations.transfer_timeout_s
            )
            return

        staging = f"{DEFAULT_TRANSFER_ROOT}/{uuid.uuid4().hex}"
        await asyncio.wait_for(state.sdk.fs.mkdir(DEFAULT_TRANSFER_ROOT), timeout=self._operations.transfer_timeout_s)
        await asyncio.wait_for(state.sdk.fs.write_bytes(staging, data), timeout=self._operations.transfer_timeout_s)
        parent = posixpath.dirname(destination)
        command = f"mkdir -p -- {shlex.quote(parent)} && cp -- {shlex.quote(staging)} {shlex.quote(destination)}"
        try:
            result = await self.exec(handle, command, timeout_s=self._operations.transfer_timeout_s, user="root")
            if result.return_code != 0:
                raise RuntimeError(
                    f"Tenki upload to {target_path!r} failed with code {result.return_code}: "
                    f"{(result.stderr or '').strip()}"
                )
        finally:
            try:
                await state.sdk.fs.remove(staging)
            except Exception as cleanup_error:
                LOGGER.debug("Failed to remove Tenki upload staging file %s: %r", staging, cleanup_error)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file, including paths that require privileged access."""
        state: _TenkiSandbox = handle.raw
        source = _effective_remote_path(state, source_path)
        if _is_home_path(source):
            data = await asyncio.wait_for(state.sdk.fs.read_bytes(source), timeout=self._operations.transfer_timeout_s)
        else:
            result = await state.sdk.exec(
                "cat",
                "--",
                source,
                timeout=self._operations.transfer_timeout_s,
                privileged=True,
            )
            if result.exit_code != 0:
                stderr = bytes(result.stderr).decode("utf-8", errors="replace")
                raise RuntimeError(f"Tenki download from {source_path!r} failed: {stderr.strip()}")
            data = bytes(result.stdout)
        await asyncio.to_thread(target_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(target_path.write_bytes, data)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        state: _TenkiSandbox = handle.raw
        _, _, SessionNotFoundError, SessionTerminatedError = _require_tenki_sdk()
        try:
            info = await state.sdk.refresh()
        except (SessionNotFoundError, SessionTerminatedError):
            return SandboxStatus.STOPPED
        return _to_sandbox_status(info.state)

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        state: _TenkiSandbox = handle.raw
        endpoint = state.endpoints.get(port)
        if endpoint is not None:
            return endpoint
        for exposed in await state.sdk.list_exposed_ports():
            if int(exposed.port) == port:
                endpoint = SandboxEndpoint(endpoint=str(exposed.url))
                state.endpoints[port] = endpoint
                return endpoint
        raise ValueError(f"Tenki sandbox {handle.sandbox_id!r} has no exposed port {port}")

    async def close(self, handle: SandboxHandle) -> None:
        """Terminate the sandbox and wait until Tenki reports it stopped."""
        state: _TenkiSandbox = handle.raw
        _, _, SessionNotFoundError, SessionTerminatedError = _require_tenki_sdk()
        try:
            await asyncio.wait_for(state.sdk.close_if_open(), timeout=self._operations.close_timeout_s)
        except (SessionNotFoundError, SessionTerminatedError):
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._operations.close_timeout_s
        while True:
            try:
                info = await state.sdk.refresh()
            except (SessionNotFoundError, SessionTerminatedError):
                return
            if _to_sandbox_status(info.state) is SandboxStatus.STOPPED:
                return
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"Tenki sandbox {handle.sandbox_id!r} did not terminate within "
                    f"{self._operations.close_timeout_s:g}s; last state={info.state!r}"
                )
            await asyncio.sleep(self._operations.close_poll_interval_s)

    async def aclose(self) -> None:
        """Close the provider-scoped Tenki SDK client."""
        if self._client_instance is None:
            return
        client, self._client_instance = self._client_instance, None
        await client.close()
