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

"""OpenShell sandbox provider: sandboxes managed by an OpenShell gateway (github.com/NVIDIA/OpenShell).

The provider talks to the gateway's gRPC control plane through the synchronous ``openshell``
SDK; blocking SDK calls run on a provider-owned thread pool bounded by ``exec.concurrency``.
The SDK has no file-transfer API, so uploads stream bytes through ``exec`` stdin and downloads
round-trip through ``base64`` on the sandbox's stdout.
"""

import asyncio
import base64
import binascii
import contextlib
import functools
import logging
import math
import posixpath
import shlex
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

SANDBOX_NAME_PREFIX = "nemo-gym-"
SANDBOX_LABEL = "nemo-gym.sandbox"
READY_PROBE_COMMAND = "printf openshell-sandbox-ready"
READY_PROBE_EXPECTED = "openshell-sandbox-ready"
SANDBOX_RUNTIME_RETURN_CODE = 125

# SandboxPhase values from OpenShell's public proto (proto/openshell.proto).
PHASE_UNSPECIFIED = 0
PHASE_PROVISIONING = 1
PHASE_READY = 2
PHASE_ERROR = 3
PHASE_DELETING = 4
PHASE_UNKNOWN = 5

_PHASE_TO_STATUS = {
    PHASE_UNSPECIFIED: SandboxStatus.UNKNOWN,
    PHASE_PROVISIONING: SandboxStatus.STARTING,
    PHASE_READY: SandboxStatus.RUNNING,
    PHASE_ERROR: SandboxStatus.ERROR,
    PHASE_DELETING: SandboxStatus.STOPPED,
    PHASE_UNKNOWN: SandboxStatus.UNKNOWN,
}


class OpenShellCreateError(SandboxCreateError):
    """Raised when the OpenShell gateway cannot create a sandbox."""


class OpenShellCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a new sandbox fails its readiness probe."""


def _require_openshell() -> None:
    try:
        import openshell  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The 'openshell' package is required for the openshell sandbox provider. Install the "
            "sandbox extra (pip install 'nemo-gym[sandbox]') or `pip install openshell` before "
            "selecting the openshell provider."
        ) from e


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def _coerce_str_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise OpenShellCreateError(f"provider_options[{what!r}] must be a string or list, got {type(value).__name__}")


def _normalize_image(image: str) -> str:
    prefix = "docker://"
    return image[len(prefix) :] if image.startswith(prefix) else image


def _grpc_status_code(exc: BaseException) -> Any | None:
    """The ``grpc.StatusCode`` of an RPC error, else None."""
    import grpc

    if not isinstance(exc, grpc.RpcError):
        return None
    code = getattr(exc, "code", None)
    if not callable(code):
        return None
    try:
        return code()
    except Exception:
        return None


def _is_grpc_error(exc: BaseException) -> bool:
    import grpc

    return isinstance(exc, grpc.RpcError)


def _is_not_found(exc: BaseException) -> bool:
    import grpc

    return _grpc_status_code(exc) == grpc.StatusCode.NOT_FOUND


def _is_grpc_timeout(exc: BaseException) -> bool:
    import grpc

    return _grpc_status_code(exc) == grpc.StatusCode.DEADLINE_EXCEEDED


def _is_sdk_error(exc: BaseException) -> bool:
    from openshell import SandboxError

    return isinstance(exc, SandboxError)


def _is_runtime_failure(exc: BaseException) -> bool:
    return _is_grpc_error(exc) or _is_sdk_error(exc)


@dataclass(frozen=True)
class OpenShellConnectionConfig:
    """Gateway connection settings. Defaults target a local plaintext gateway (deploy/docker compose)."""

    endpoint: str = "localhost:8080"
    bearer_token: str | None = None
    tls_ca_path: str | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    request_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("connection.endpoint must be a non-empty host:port")
        if self.request_timeout_s <= 0:
            raise ValueError("connection.request_timeout_s must be > 0")
        if bool(self.tls_cert_path) != bool(self.tls_key_path):
            raise ValueError("connection.tls_cert_path and connection.tls_key_path must be set together")


@dataclass(frozen=True)
class OpenShellCreateConfig:
    ready_timeout_s: float = 300
    poll_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.ready_timeout_s <= 0:
            raise ValueError("create.ready_timeout_s must be > 0")
        if self.poll_interval_s <= 0:
            raise ValueError("create.poll_interval_s must be > 0")


@dataclass(frozen=True)
class OpenShellExecConfig:
    default_timeout_s: float | None = 180
    concurrency: int = 32
    exec_shell: str = "/bin/sh"

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")
        if not self.exec_shell:
            raise ValueError("exec.exec_shell must be a non-empty shell name/path")


@dataclass(frozen=True)
class OpenShellProbeConfig:
    command: str | None = READY_PROBE_COMMAND
    expected_stdout: str | None = READY_PROBE_EXPECTED
    timeout_s: int = 30
    deadline_s: float | None = 60
    stable_count: int = 1
    stable_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        if self.stable_delay_s < 0:
            raise ValueError("probe.stable_delay_s must be >= 0")


@dataclass(frozen=True)
class OpenShellOperationsConfig:
    close_wait_deleted: bool = True
    close_timeout_s: float = 60
    poll_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if self.close_timeout_s <= 0:
            raise ValueError("operations.close_timeout_s must be > 0")
        if self.poll_interval_s <= 0:
            raise ValueError("operations.poll_interval_s must be > 0")


@dataclass
class _OpenShellSandbox:
    name: str
    sandbox_id: str
    image: str | None
    env: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None


class OpenShellProvider:
    """Sandbox provider backed by an OpenShell gateway's gRPC control plane."""

    name = "openshell"

    def __init__(
        self,
        *,
        connection: OpenShellConnectionConfig | Mapping[str, Any] | None = None,
        create: OpenShellCreateConfig | Mapping[str, Any] | None = None,
        exec: OpenShellExecConfig | Mapping[str, Any] | None = None,
        probe: OpenShellProbeConfig | Mapping[str, Any] | None = None,
        operations: OpenShellOperationsConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _coerce_config(connection, OpenShellConnectionConfig)
        self._create_config = _coerce_config(create, OpenShellCreateConfig)
        self._exec_config = _coerce_config(exec, OpenShellExecConfig)
        self._probe = _coerce_config(probe, OpenShellProbeConfig)
        self._operations = _coerce_config(operations, OpenShellOperationsConfig)
        _require_openshell()
        self._client = self._build_client()
        # The sync SDK blocks a thread per in-flight RPC, so the pool bounds gateway concurrency.
        self._executor = ThreadPoolExecutor(
            max_workers=self._exec_config.concurrency, thread_name_prefix="openshell-sandbox"
        )
        self._closed = False

    def _build_client(self) -> Any:
        from openshell import SandboxClient, TlsConfig

        conn = self._connection
        tls = None
        if conn.tls_ca_path or conn.tls_cert_path:
            tls = TlsConfig(
                ca_path=Path(conn.tls_ca_path) if conn.tls_ca_path else None,
                cert_path=Path(conn.tls_cert_path) if conn.tls_cert_path else None,
                key_path=Path(conn.tls_key_path) if conn.tls_key_path else None,
            )
        return SandboxClient(
            conn.endpoint,
            tls=tls,
            bearer_token=conn.bearer_token,
            timeout=conn.request_timeout_s,
        )

    async def _call(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

    def _build_sandbox_spec(self, spec: SandboxSpec, image: str | None, providers: list[str]) -> Any:
        from openshell._proto import openshell_pb2

        kwargs: dict[str, Any] = {"environment": {str(k): str(v) for k, v in spec.env.items()}}
        if image:
            kwargs["template"] = openshell_pb2.SandboxTemplate(image=image)
        if providers:
            kwargs["providers"] = providers
        resources = spec.resources
        if resources.gpu:
            kwargs["resource_requirements"] = openshell_pb2.ResourceRequirements(
                gpu=openshell_pb2.GpuResourceRequirements(count=resources.gpu)
            )
        ignored = [
            key
            for key, value in (
                ("cpu", resources.cpu),
                ("memory_mib", resources.memory_mib),
                ("disk_gib", resources.disk_gib),
                ("gpu_type", resources.gpu_type),
            )
            if value is not None
        ]
        if ignored:
            LOGGER.warning(
                "The OpenShell gateway API has no request fields for %s; these resource requests are ignored.",
                ", ".join(ignored),
            )
        return openshell_pb2.SandboxSpec(**kwargs)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox through the gateway, wait for the READY phase, then probe exec readiness.

        ``spec.image`` is optional (the gateway's configured default image is used when unset).
        ``spec.ttl_s`` is not enforced (OpenShell sandboxes live until deleted) and only logs a
        warning. ``spec.entrypoint`` is unsupported: the OpenShell supervisor owns the sandbox
        entrypoint. ``spec.provider_options`` may carry ``providers`` (OpenShell credential-provider
        names attached to the sandbox). A half-created sandbox is deleted on any failure.
        """
        if spec.entrypoint:
            raise OpenShellCreateError(
                "spec.entrypoint is not supported by the openshell provider; the OpenShell "
                "supervisor owns the sandbox entrypoint"
            )
        if spec.ttl_s is not None:
            LOGGER.warning("ttl_s is not enforced by the openshell provider; sandboxes live until close().")

        image = _normalize_image(spec.image) if spec.image else None
        providers = _coerce_str_list(spec.provider_options.get("providers"), "providers")
        pb_spec = self._build_sandbox_spec(spec, image, providers)
        name = SANDBOX_NAME_PREFIX + uuid.uuid4().hex
        labels = {SANDBOX_LABEL: "1", **{str(k): str(v) for k, v in spec.metadata.items()}}

        try:
            ref = await self._call(self._client.create, spec=pb_spec, name=name, labels=labels)
        except Exception as e:
            raise OpenShellCreateError(f"CreateSandbox failed for image={image!r}: {e}") from e

        handle = SandboxHandle(
            sandbox_id=ref.id,
            provider_name=self.name,
            raw=_OpenShellSandbox(
                name=ref.name, sandbox_id=ref.id, image=image, env=dict(spec.env), workdir=spec.workdir
            ),
        )
        try:
            ready_timeout_s = spec.ready_timeout_s or self._create_config.ready_timeout_s
            await self._wait_ready(handle, timeout_s=ready_timeout_s)
            await self._verify_created_handle(handle)
        except Exception:
            await self._cleanup_failed_create_handle(handle)
            raise
        return handle

    async def _wait_ready(self, handle: SandboxHandle, *, timeout_s: int | float) -> None:
        """Poll GetSandbox until the READY phase, raising on the ERROR phase or the deadline."""
        inst = handle.raw
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last_phase: int | None = None
        while True:
            try:
                ref = await self._call(self._client.get, inst.name)
                last_phase = ref.phase
            except Exception as e:
                if not _is_runtime_failure(e):
                    raise
                LOGGER.debug(f"GetSandbox failed while waiting for {inst.name!r} to become ready: {e}")
            else:
                if ref.phase == PHASE_READY:
                    return
                if ref.phase == PHASE_ERROR:
                    raise OpenShellCreateError(f"sandbox {inst.name!r} entered the ERROR phase while provisioning")
            if loop.time() >= deadline:
                raise OpenShellCreateError(
                    f"sandbox {inst.name!r} was not READY within {timeout_s:g}s (last phase={last_phase})"
                )
            await asyncio.sleep(self._create_config.poll_interval_s)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        """Poll the readiness probe until it passes ``stable_count`` times or the deadline elapses."""
        probe = self._probe
        if probe.command is None:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"
        while True:
            result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = f"return_code={result.return_code}, stderr={(result.stderr or '').strip()!r}"
                if deadline is None:
                    raise OpenShellCreateVerificationError(
                        f"sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )
            if deadline is not None and loop.time() >= deadline:
                raise OpenShellCreateVerificationError(
                    f"sandbox {handle.sandbox_id!r} did not pass readiness probe within "
                    f"{probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        with contextlib.suppress(Exception):
            await self._call(self._client.delete, handle.raw.name)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        """Run ``<shell> -c <command>`` through the gateway's streaming exec; never raises for command failure.

        The timeout is enforced by the gateway (``timeout_seconds``); the SDK extends its gRPC
        deadline past it. ``user`` is ignored with a warning: the OpenShell exec API has no user
        field, so commands run as the sandbox's default user.
        """
        inst = handle.raw
        if user is not None:
            LOGGER.warning(
                f"The openshell provider cannot run commands as user={user!r}; the OpenShell exec API "
                "has no user field. Running as the sandbox default user."
            )
        merged_env = dict(inst.env)
        if env:
            merged_env.update(env)
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s
        timeout_seconds = max(1, math.ceil(effective_timeout)) if effective_timeout is not None else None
        workdir = cwd if cwd is not None else inst.workdir
        try:
            result = await self._call(
                self._client.exec,
                inst.sandbox_id,
                [self._exec_config.exec_shell, "-c", command],
                workdir=workdir,
                env=merged_env or None,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            if _is_grpc_timeout(e):
                return SandboxExecResult(
                    stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
                )
            if _is_runtime_failure(e):
                return SandboxExecResult(
                    stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="sandbox"
                )
            raise
        return SandboxExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file by streaming its bytes through exec stdin (creates the parent dir)."""
        data = await asyncio.to_thread(Path(source_path).read_bytes)
        command = f"cat > {shlex.quote(target_path)}"
        parent = posixpath.dirname(target_path)
        if parent:
            command = f"mkdir -p {shlex.quote(parent)} && {command}"
        result = await self.exec(handle, command, stdin=data)
        if result.return_code != 0:
            raise RuntimeError(
                f"openshell upload to {target_path!r} failed (code={result.return_code}): "
                f"{(result.stderr or '').strip()}"
            )

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file via a base64 round-trip (binary-safe over the text exec stream)."""
        result = await self.exec(handle, f"base64 {shlex.quote(source_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"openshell download from {source_path!r} failed (code={result.return_code}): "
                f"{(result.stderr or '').strip()}"
            )
        try:
            data = base64.b64decode("".join((result.stdout or "").split()), validate=True)
        except (binascii.Error, ValueError) as e:
            raise RuntimeError(f"openshell download from {source_path!r} returned invalid base64: {e}") from e
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target_path.write_bytes, data)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Sandbox phase via GetSandbox (missing -> STOPPED; RPC failure -> UNKNOWN)."""
        inst = handle.raw
        try:
            ref = await self._call(self._client.get, inst.name)
        except Exception as e:
            if _is_not_found(e):
                return SandboxStatus.STOPPED
            if _is_runtime_failure(e):
                return SandboxStatus.UNKNOWN
            raise
        return _PHASE_TO_STATUS.get(ref.phase, SandboxStatus.UNKNOWN)

    async def close(self, handle: SandboxHandle) -> None:
        """Delete the sandbox (already-gone counts as success), then wait until it is fully gone."""
        inst = handle.raw
        try:
            await self._call(self._client.delete, inst.name)
        except Exception as e:
            if _is_not_found(e):
                return
            if _is_runtime_failure(e):
                raise RuntimeError(f"openshell delete failed for {inst.name!r}: {e}") from e
            raise
        if self._operations.close_wait_deleted:
            await self._wait_deleted(inst.name)

    async def _wait_deleted(self, name: str) -> None:
        """Poll GetSandbox until NOT_FOUND (transient RPC failures keep polling until the deadline)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._operations.close_timeout_s
        while True:
            try:
                await self._call(self._client.get, name)
            except Exception as e:
                if _is_not_found(e):
                    return
                if not _is_runtime_failure(e):
                    raise
                LOGGER.debug(f"GetSandbox failed while waiting for {name!r} to be deleted: {e}")
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"openshell sandbox {name!r} was not deleted within {self._operations.close_timeout_s:g}s"
                )
            await asyncio.sleep(self._operations.poll_interval_s)

    async def aclose(self) -> None:
        """Close the gRPC channel and shut down the worker thread pool (idempotent)."""
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._client.close)
        self._executor.shutdown(wait=False)
