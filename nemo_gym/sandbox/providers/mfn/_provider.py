# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MFN sandbox provider using the vendored gRPC protocol."""

import asyncio
import logging
import math
import os
import shlex
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.duration_pb2 import Duration

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxPtySession,
    SandboxPtySpec,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.mfn.protos import mfn_sandbox_pb2 as pb
from nemo_gym.sandbox.providers.mfn.pty import MFNPtySession
from nemo_gym.sandbox.providers.mfn.rpc import MFNSandboxStub
from nemo_gym.sandbox.providers.utils import coerce_config


LOGGER = logging.getLogger(__name__)
RUNTIME_RETURN_CODE = 125
TRANSIENT_CODES = {
    grpc.StatusCode.ABORTED,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.UNAVAILABLE,
}


class MFNCreateError(SandboxCreateError):
    """MFN sandbox creation failed."""


class MFNCreateVerificationError(SandboxCreateVerificationError):
    """A created MFN sandbox failed its command probe."""


@dataclass(frozen=True)
class MFNConnectionConfig:
    address: str = "localhost:10128"
    caller: str = "nemo-gym"
    username: str = "nemo-gym"
    keepalive_time_s: float = 60.0
    keepalive_timeout_s: float = 20.0
    max_receive_message_mib: int = 1024

    def __post_init__(self) -> None:
        if not self.address.strip():
            raise ValueError("connection.address must be a non-empty host:port")
        if not self.caller.strip() or not self.username.strip():
            raise ValueError("connection.caller and connection.username must be non-empty")
        if self.keepalive_time_s <= 0 or self.keepalive_timeout_s <= 0:
            raise ValueError("connection keepalive values must be > 0")
        if self.max_receive_message_mib < 1:
            raise ValueError("connection.max_receive_message_mib must be >= 1")


@dataclass(frozen=True)
class MFNCreateConfig:
    request_timeout_s: float = 60.0
    ready_timeout_s: float = 300.0
    poll_initial_delay_s: float = 5.0
    poll_interval_s: float = 2.0
    poll_max_interval_s: float = 10.0
    retries: int = 2
    retry_delay_s: float = 1.0

    def __post_init__(self) -> None:
        numeric = (
            self.request_timeout_s,
            self.ready_timeout_s,
            self.poll_interval_s,
            self.poll_max_interval_s,
        )
        if any(value <= 0 for value in numeric) or self.poll_initial_delay_s < 0:
            raise ValueError("create timeouts and poll intervals must be positive")
        if self.retries < 0 or self.retry_delay_s < 0:
            raise ValueError("create retry values must be non-negative")


@dataclass(frozen=True)
class MFNResourceDefaults:
    cpu: float = 1.0
    memory_mib: int = 1024
    disk_gib: int | None = None

    def __post_init__(self) -> None:
        if self.cpu <= 0 or self.memory_mib <= 0:
            raise ValueError("resource defaults cpu and memory_mib must be > 0")
        if self.disk_gib is not None and self.disk_gib <= 0:
            raise ValueError("resource defaults disk_gib must be > 0 or null")


@dataclass(frozen=True)
class MFNOperationConfig:
    default_exec_timeout_s: float | None = 300.0
    file_timeout_s: float = 600.0
    status_timeout_s: float = 30.0
    close_timeout_s: float = 60.0
    exec_shell: str = "/bin/sh"
    upload_chunk_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if self.default_exec_timeout_s is not None and self.default_exec_timeout_s <= 0:
            raise ValueError("operations.default_exec_timeout_s must be > 0 or null")
        if min(self.file_timeout_s, self.status_timeout_s, self.close_timeout_s) <= 0:
            raise ValueError("operation timeouts must be > 0")
        if not self.exec_shell or self.upload_chunk_bytes < 1:
            raise ValueError("operations.exec_shell must be set and upload_chunk_bytes must be >= 1")


@dataclass(frozen=True)
class MFNProbeConfig:
    command: str | None = "printf mfn-sandbox-ready"
    expected_stdout: str | None = "mfn-sandbox-ready"
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")


@dataclass(frozen=True)
class MFNProviderOptions:
    snapshot_id: str | None = None
    require_vm: bool = False
    beta_no_ttl_cap: bool = False
    shard_pin: str | None = None
    network_mode: str | None = None
    allowed_cidrs: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    blocked_cidrs: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    network_name: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MFNProviderOptions":
        values = dict(value or {})
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown MFN provider_options keys: {', '.join(sorted(unknown))}")
        for key in ("allowed_cidrs", "allowed_domains", "blocked_cidrs", "blocked_domains"):
            values[key] = tuple(str(item) for item in values.get(key, ()))
        result = cls(**values)
        if result.network_mode not in (None, "allow", "block", "allow_all"):
            raise ValueError("provider_options.network_mode must be allow, block, allow_all, or null")
        rules = result.allowed_cidrs + result.allowed_domains + result.blocked_cidrs + result.blocked_domains
        if result.network_mode is None and rules:
            raise ValueError("network rules require provider_options.network_mode")
        if result.network_mode == "allow_all" and rules:
            raise ValueError("network_mode=allow_all cannot be combined with network rules")
        if result.network_mode == "allow" and (result.blocked_cidrs or result.blocked_domains):
            raise ValueError("blocked network rules require network_mode=block")
        if result.network_mode == "block" and (result.allowed_cidrs or result.allowed_domains):
            raise ValueError("allowed network rules cannot be used with network_mode=block")
        if result.network_mode == "block" and not (result.blocked_cidrs or result.blocked_domains):
            raise ValueError("network_mode=block requires at least one blocked rule")
        return result


@dataclass(frozen=True)
class _MFNHandle:
    sandbox_id: str


def _duration(seconds: int | float) -> Any:
    whole = math.floor(seconds)
    return Duration(seconds=whole, nanos=round((seconds - whole) * 1_000_000_000))


def _quantity(value: float | int) -> str:
    return str(int(value)) if isinstance(value, int) or value.is_integer() else str(value)


def _rpc_code(error: BaseException) -> Any | None:
    return error.code() if isinstance(error, grpc.RpcError) else None


def _rpc_detail(error: BaseException) -> str:
    details = getattr(error, "details", None)
    return str(details()) if callable(details) else str(error)


class MFNProvider:
    """Gym provider for MFN's SandboxService."""

    name = "mfn"

    def __init__(
        self,
        *,
        connection: MFNConnectionConfig | Mapping[str, Any] | None = None,
        create: MFNCreateConfig | Mapping[str, Any] | None = None,
        resources: MFNResourceDefaults | Mapping[str, Any] | None = None,
        operations: MFNOperationConfig | Mapping[str, Any] | None = None,
        probe: MFNProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = coerce_config(connection, MFNConnectionConfig)
        self._create = coerce_config(create, MFNCreateConfig)
        self._resource_defaults = coerce_config(resources, MFNResourceDefaults)
        self._operations = coerce_config(operations, MFNOperationConfig)
        self._probe = coerce_config(probe, MFNProbeConfig)
        options = [
            ("grpc.max_receive_message_length", self._connection.max_receive_message_mib * 1024 * 1024),
            ("grpc.keepalive_time_ms", round(self._connection.keepalive_time_s * 1000)),
            ("grpc.keepalive_timeout_ms", round(self._connection.keepalive_timeout_s * 1000)),
        ]
        self._channel = grpc.aio.insecure_channel(self._connection.address, options=options)
        self._stub: Any = MFNSandboxStub(self._channel)
        self._closed = False

    def _create_request(self, spec: SandboxSpec, idempotency_key: str) -> Any:
        options = MFNProviderOptions.from_mapping(spec.provider_options)
        if spec.entrypoint is not None:
            raise ValueError("SandboxSpec.entrypoint is not supported by MFN")
        if bool(spec.image) == bool(options.snapshot_id):
            raise ValueError("Set exactly one of SandboxSpec.image and provider_options.snapshot_id")

        resources = spec.resources
        resource = pb.ResourceRequest()
        cpu = resources.cpu if resources.cpu is not None else self._resource_defaults.cpu
        memory_mib = resources.memory_mib if resources.memory_mib is not None else self._resource_defaults.memory_mib
        disk_gib = resources.disk_gib if resources.disk_gib is not None else self._resource_defaults.disk_gib
        resource.cpu_request = _quantity(cpu)
        resource.memory_request = resource.memory_limit = f"{memory_mib}Mi"
        if disk_gib is not None:
            resource.storage_request = resource.storage_limit = f"{disk_gib}Gi"
        if resources.gpu is not None:
            resource.gpu.count = resources.gpu
            if resources.gpu_type:
                resource.gpu.type_preferences.append(resources.gpu_type)
        elif resources.gpu_type is not None:
            raise ValueError("SandboxResources.gpu_type requires a non-null gpu count")

        specs = pb.SandboxSpecs(
            resource_request=resource,
            env_vars={str(key): str(value) for key, value in spec.env.items()},
            require_vm=options.require_vm,
            beta_no_ttl_cap_use_only_when_really_needed=options.beta_no_ttl_cap,
        )
        if spec.image:
            specs.image = spec.image.removeprefix("docker://")
        else:
            specs.snapshot_id = options.snapshot_id
        if spec.ttl_s is not None:
            specs.sandbox_ttl.CopyFrom(_duration(spec.ttl_s))

        request = pb.CreateOptions(
            specs=specs,
            attributes=pb.SandboxAttributes(attributes={str(k): str(v) for k, v in spec.metadata.items()}),
            ports=[pb.ContainerPort(name=f"tcp-{port}", port=port) for port in spec.ports],
            idempotency_key=idempotency_key,
            client_info=pb.ClientInfo(caller=self._connection.caller, username=self._connection.username),
        )
        if options.shard_pin:
            request.shard_pin = options.shard_pin
        # An omitted NetworkConfig preserves MFN's default-open posture.
        if options.network_mode not in (None, "allow_all"):
            mode = pb.NetworkConfig.ALLOW if options.network_mode == "allow" else pb.NetworkConfig.BLOCK
            request.network_config.CopyFrom(
                pb.NetworkConfig(
                    mode=mode,
                    allowed_cidrs=options.allowed_cidrs,
                    allowed_domains=options.allowed_domains,
                    blocked_cidrs=options.blocked_cidrs,
                    blocked_domains=options.blocked_domains,
                    name=options.network_name,
                )
            )
        return request

    async def _wait_ready(self, sandbox_id: str, timeout_s: float) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        await asyncio.sleep(min(self._create.poll_initial_delay_s, timeout_s))
        interval = self._create.poll_interval_s
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            try:
                response = await self._stub.Get(
                    pb.GetRequest(sandbox_id=sandbox_id),
                    timeout=min(self._operations.status_timeout_s, remaining),
                )
                if response.status.is_ready:
                    return
                if response.status.phase.lower() == "failed":
                    raise MFNCreateError(f"MFN sandbox {sandbox_id!r} entered Failed phase")
            except grpc.aio.AioRpcError as error:
                if _rpc_code(error) not in TRANSIENT_CODES:
                    raise MFNCreateError(f"MFN readiness check failed: {_rpc_detail(error)}") from error
            await asyncio.sleep(min(interval, max(0.0, deadline - loop.time())))
            interval = min(interval * 1.5, self._create.poll_max_interval_s)
        raise MFNCreateError(f"MFN sandbox {sandbox_id!r} was not ready within {timeout_s:g}s")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        request = self._create_request(spec, uuid.uuid4().hex)
        response: Any | None = None
        delay = self._create.retry_delay_s
        for attempt in range(self._create.retries + 1):
            try:
                response = await self._stub.Create(request, timeout=self._create.request_timeout_s)
                break
            except grpc.aio.AioRpcError as error:
                if attempt == self._create.retries or _rpc_code(error) not in TRANSIENT_CODES:
                    raise MFNCreateError(f"MFN Create failed: {_rpc_detail(error)}") from error
                LOGGER.warning("Retrying MFN Create with the same idempotency key: %s", _rpc_detail(error))
                await asyncio.sleep(delay)
                delay = min(max(delay * 2, 0.1), 30.0)
        if response is None:
            raise MFNCreateError("MFN Create retry loop did not run")

        handle = SandboxHandle(response.sandbox_id, self.name, _MFNHandle(response.sandbox_id))
        try:
            if not response.is_ready:
                await self._wait_ready(
                    response.sandbox_id, float(spec.ready_timeout_s or self._create.ready_timeout_s)
                )
            if self._probe.command is not None:
                result = await self.exec(handle, self._probe.command, timeout_s=self._probe.timeout_s)
                if result.return_code != 0 or (
                    self._probe.expected_stdout is not None
                    and self._probe.expected_stdout not in (result.stdout or "")
                ):
                    raise MFNCreateVerificationError(
                        f"MFN sandbox {response.sandbox_id!r} failed its exec probe: {result}"
                    )
        except BaseException:
            try:
                await self.close(handle)
            except BaseException:
                LOGGER.warning("Failed to clean up MFN sandbox %s after create failure", response.sandbox_id)
            raise
        return handle

    def _exec_request(
        self,
        handle: SandboxHandle,
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
    ) -> Any:
        if user not in (None, "root", 0):
            command = (
                f"su -s {shlex.quote(self._operations.exec_shell)} -c {shlex.quote(command)} {shlex.quote(str(user))}"
            )
        return pb.ExecRequest(
            sandbox_id=handle.sandbox_id,
            command=[self._operations.exec_shell, "-c", command],
            cwd=cwd or "",
            env={str(key): str(value) for key, value in (env or {}).items()},
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
        timeout = self._operations.default_exec_timeout_s if timeout_s is None else float(timeout_s)
        stdout = bytearray()
        stderr = bytearray()
        complete: Any | None = None

        async def requests() -> Any:
            yield pb.ExecStreamRequest(request=self._exec_request(handle, command, cwd, env, user))

        try:
            # Ending this one-message request iterator half-closes stdin. This is
            # important for MFN's local backend, whose pipe-mode process does
            # not complete while the separate Exec stdin channel remains open.
            stream = self._stub.ExecStream(requests(), timeout=timeout)
            async for response in stream:
                if response.HasField("output"):
                    target = stdout if response.output.stream == pb.ExecOutput.STDOUT else stderr
                    target.extend(response.output.data)
                elif response.HasField("complete"):
                    complete = response.complete
        except grpc.aio.AioRpcError as error:
            error_type = "timeout" if _rpc_code(error) == grpc.StatusCode.DEADLINE_EXCEEDED else "sandbox"
            return SandboxExecResult(None, _rpc_detail(error), RUNTIME_RETURN_CODE, error_type)
        if complete is None:
            return SandboxExecResult(
                stdout.decode(errors="replace") or None,
                "MFN ExecStream ended without ExecComplete",
                RUNTIME_RETURN_CODE,
                "sandbox",
            )
        extra = "\n".join(value for value in (complete.error, complete.termination_detail) if value)
        if extra:
            stderr.extend((f"\n{extra}" if stderr else extra).encode())
        return SandboxExecResult(
            stdout.decode(errors="replace") or None,
            stderr.decode(errors="replace") or None,
            complete.exit_code,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        async def chunks() -> Any:
            with Path(source_path).open("rb") as source:
                while True:
                    chunk = await asyncio.to_thread(source.read, self._operations.upload_chunk_bytes)
                    if not chunk:
                        return
                    yield pb.AddFileRequest(content=chunk)

        metadata = (("sandbox-id", handle.sandbox_id), ("dest-path", target_path))
        await self._stub.AddFile(chunks(), metadata=metadata, timeout=self._operations.file_timeout_s)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target = Path(target_path)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        try:
            with open(temporary, "wb") as destination:
                stream = self._stub.ReadFile(
                    pb.ReadFileRequest(sandbox_id=handle.sandbox_id, path=source_path),
                    timeout=self._operations.file_timeout_s,
                )
                async for response in stream:
                    await asyncio.to_thread(destination.write, response.content)
            await asyncio.to_thread(os.replace, temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        try:
            response = await self._stub.Get(
                pb.GetRequest(sandbox_id=handle.sandbox_id), timeout=self._operations.status_timeout_s
            )
        except grpc.aio.AioRpcError as error:
            if _rpc_code(error) == grpc.StatusCode.NOT_FOUND:
                return SandboxStatus.STOPPED
            return SandboxStatus.UNKNOWN
        phase = response.status.phase.lower()
        if response.status.is_ready and phase == "running":
            return SandboxStatus.RUNNING
        return {
            "pending": SandboxStatus.STARTING,
            "running": SandboxStatus.STARTING,
            "succeeded": SandboxStatus.STOPPED,
            "failed": SandboxStatus.ERROR,
        }.get(phase, SandboxStatus.UNKNOWN)

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        response = await self._stub.GetHost(
            pb.GetHostRequest(sandbox_id=handle.sandbox_id, port=port),
            timeout=self._operations.status_timeout_s,
        )
        uri = response.uri if "://" in response.uri else f"http://{response.uri}"
        return SandboxEndpoint(uri)

    async def create_pty(self, handle: SandboxHandle, spec: SandboxPtySpec) -> SandboxPtySession:
        return await MFNPtySession(self._stub, handle.sandbox_id, spec, shell=self._operations.exec_shell).start()

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        del scope
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        handle = SandboxHandle(sandbox_id, self.name, _MFNHandle(sandbox_id))
        current = await self.status(handle)
        if current in (SandboxStatus.STOPPED, SandboxStatus.ERROR):
            raise RuntimeError(f"Cannot connect to MFN sandbox {sandbox_id!r}: status={current.value}")
        return handle

    async def close(self, handle: SandboxHandle) -> None:
        try:
            await self._stub.Shutdown(
                pb.ShutdownOptions(sandbox_id=handle.sandbox_id), timeout=self._operations.close_timeout_s
            )
        except grpc.aio.AioRpcError as error:
            if _rpc_code(error) != grpc.StatusCode.NOT_FOUND:
                raise RuntimeError(f"MFN Shutdown failed: {_rpc_detail(error)}") from error

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._channel.close()
