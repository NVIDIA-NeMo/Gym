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

"""Provider-facing sandbox protocol."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SandboxStatus(str, Enum):
    """Provider-neutral sandbox lifecycle status."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SandboxEndpoint:
    """Provider-neutral route to a long-lived service inside a sandbox.

    ``endpoint`` is an absolute URL. ``headers`` carries provider-required
    authentication or routing headers without exposing the provider's opaque
    handle to callers.
    """

    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise ValueError("Sandbox endpoint must be a non-empty absolute URL")
        endpoint = self.endpoint.strip()
        parsed = urlsplit(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Sandbox endpoint must be a non-empty absolute URL")
        if not isinstance(self.headers, Mapping):
            raise TypeError("Sandbox endpoint headers must be a mapping")
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(
            self,
            "headers",
            {str(key): str(value) for key, value in self.headers.items()},
        )


_SANDBOX_RESOURCES_POSITIONAL_FIELDS = (
    "cpu",
    "memory_mib",
    "disk_gib",
    "gpu",
    "gpu_type",
)


class SandboxResources(BaseModel):
    """Provider-neutral resource quantities."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    cpu: int | float | None = None
    memory_mib: int | None = None
    disk_gib: int | None = None
    gpu: int | None = None
    gpu_type: str | None = None

    def __init__(self, *args: Any, **data: Any) -> None:
        if len(args) > len(_SANDBOX_RESOURCES_POSITIONAL_FIELDS):
            raise TypeError(
                f"SandboxResources() accepts at most {len(_SANDBOX_RESOURCES_POSITIONAL_FIELDS)} "
                f"positional arguments, got {len(args)}"
            )

        positional_data = dict(zip(_SANDBOX_RESOURCES_POSITIONAL_FIELDS, args, strict=False))
        for field_name in positional_data:
            if field_name in data:
                raise TypeError(f"SandboxResources() got multiple values for argument {field_name!r}")

        super().__init__(**positional_data, **data)

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_fields(cls, resources: Any) -> Any:
        if isinstance(resources, Mapping):
            unknown_keys = set(resources) - set(cls.model_fields)
            if unknown_keys:
                unknown = ", ".join(sorted(unknown_keys))
                allowed = ", ".join(sorted(cls.model_fields))
                raise ValueError(f"Unknown sandbox resource keys: {unknown}. Expected keys: {allowed}")
        return resources

    @field_validator("cpu", mode="before")
    @classmethod
    def coerce_cpu(cls, cpu: Any) -> int | float | None:
        if cpu is None or isinstance(cpu, int):
            return cpu
        return float(cpu)

    @field_validator("memory_mib", "disk_gib", "gpu", mode="before")
    @classmethod
    def coerce_integer_resource(cls, value: Any) -> int | None:
        return int(value) if value is not None else None

    @field_validator("gpu_type", mode="before")
    @classmethod
    def coerce_gpu_type(cls, gpu_type: Any) -> str | None:
        return str(gpu_type) if gpu_type is not None else None

    @classmethod
    def from_mapping(cls, resources: Mapping[str, Any] | Self | None) -> Self:
        if resources is None:
            return cls()
        if isinstance(resources, cls):
            return resources
        values = dict(resources)
        if values.get("cpu") is not None:
            values["cpu"] = float(values["cpu"])
        return cls.model_validate(values)


_SANDBOX_SPEC_POSITIONAL_FIELDS = (
    "image",
    "ttl_s",
    "ready_timeout_s",
    "workdir",
    "env",
    "files",
    "metadata",
    "resources",
    "entrypoint",
    "provider_options",
    "ports",
)


class SandboxSpec(BaseModel):
    """Sandbox creation request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str | None = None
    ttl_s: int | float | None = None
    ready_timeout_s: int | float | None = None
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    resources: SandboxResources = Field(default_factory=SandboxResources)
    resource_requests: SandboxResources | None = None
    entrypoint: list[str] | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    ports: tuple[int, ...] = Field(default_factory=tuple)

    def __init__(self, *args: Any, **data: Any) -> None:
        if len(args) > len(_SANDBOX_SPEC_POSITIONAL_FIELDS):
            raise TypeError(
                f"SandboxSpec() accepts at most {len(_SANDBOX_SPEC_POSITIONAL_FIELDS)} "
                f"positional arguments, got {len(args)}"
            )

        positional_data = dict(zip(_SANDBOX_SPEC_POSITIONAL_FIELDS, args, strict=False))
        for field_name in positional_data:
            if field_name in data:
                raise TypeError(f"SandboxSpec() got multiple values for argument {field_name!r}")

        super().__init__(**positional_data, **data)

    @field_validator("ports", mode="before")
    @classmethod
    def normalize_ports(cls, ports: Any) -> tuple[int, ...]:
        if not isinstance(ports, (list, tuple)):
            raise TypeError("Sandbox ports must be a list or tuple of TCP port numbers")
        normalized_ports: list[int] = []
        for raw_port in ports:
            if isinstance(raw_port, bool):
                raise ValueError(f"Invalid sandbox TCP port: {raw_port!r}")
            if not isinstance(raw_port, (int, str)):
                raise ValueError(f"Invalid sandbox TCP port: {raw_port!r}")
            try:
                port = int(raw_port)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid sandbox TCP port: {raw_port!r}") from exc
            if port < 1 or port > 65535:
                raise ValueError(f"Sandbox TCP port must be between 1 and 65535, got {port}")
            if port in normalized_ports:
                raise ValueError(f"Duplicate sandbox TCP port: {port}")
            normalized_ports.append(port)
        return tuple(normalized_ports)

    @field_validator("resources", mode="before")
    @classmethod
    def default_resources(cls, resources: Any) -> Any:
        """Keep accepting an explicit null resource block as an empty request."""
        if resources is None or isinstance(resources, (Mapping, SandboxResources)):
            return SandboxResources.from_mapping(resources)
        return resources

    @field_validator("resource_requests", mode="before")
    @classmethod
    def normalize_resource_requests(cls, resource_requests: Any) -> Any:
        if isinstance(resource_requests, (Mapping, SandboxResources)):
            return SandboxResources.from_mapping(resource_requests)
        return resource_requests

    @field_validator("provider_options", mode="before")
    @classmethod
    def normalize_provider_options(cls, provider_options: Any) -> Any:
        if provider_options is None:
            return {}
        if isinstance(provider_options, BaseModel):
            return provider_options.model_dump(mode="python")
        return provider_options

    @model_validator(mode="after")
    def validate_resource_requests(self) -> Self:
        if self.resource_requests is None:
            return self

        for field_name in ("cpu", "memory_mib", "disk_gib", "gpu"):
            request = getattr(self.resource_requests, field_name)
            limit = getattr(self.resources, field_name)
            if request is not None and limit is not None and request > limit:
                raise ValueError(
                    f"resource_requests.{field_name} ({request}) cannot exceed resources.{field_name} ({limit})"
                )
        return self


@dataclass
class SandboxHandle:
    """Provider-neutral handle to a created sandbox.

    ``raw`` is provider-owned opaque state. Public code should pass it back to
    the provider through this handle rather than inspecting or mutating it
    directly.
    """

    sandbox_id: str
    provider_name: str
    raw: Any


@dataclass(frozen=True)
class SandboxExecResult:
    """Provider-neutral process execution result.

    ``return_code`` is the process exit code when the sandbox actually ran the
    command. Providers may use a non-process sentinel with ``error_type`` set
    when the sandbox runtime reports an execution failure without a process
    exit code.
    """

    stdout: str | None
    stderr: str | None
    return_code: int
    error_type: str | None = None


ExecResult = SandboxExecResult


class SandboxCreateError(RuntimeError):
    """Raised when a provider cannot create a sandbox."""


class SandboxCreateVerificationError(SandboxCreateError):
    """Raised when a newly-created sandbox fails provider readiness checks."""


class SandboxProvider(Protocol):
    """Runtime/infra provider contract used by the public sandbox API."""

    name: str

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a ready sandbox and return a provider-neutral handle.

        Providers must return only after the sandbox is healthy enough to run
        commands and transfer files. If the sandbox cannot become ready before
        the configured timeout, providers should raise ``SandboxCreateError``
        or a provider-specific subclass.
        """
        ...

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
        """Run a command inside a sandbox."""
        ...

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into a sandbox."""
        ...

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the local filesystem."""
        ...

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the current sandbox lifecycle status."""
        ...

    async def close(self, handle: SandboxHandle) -> None:
        """End the sandbox lifecycle and close provider resources for it."""
        ...

    async def aclose(self) -> None:
        """Close provider-scoped resources such as SDK clients."""
        ...


@runtime_checkable
class SupportsSandboxEndpoint(Protocol):
    """Optional provider capability for resolving declared service ports."""

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a declared service port to a caller-reachable endpoint."""
        ...


@runtime_checkable
class ConnectableProvider(Protocol):
    """Optional capability: rebuild a handle in another process from a descriptor.

    Providers whose sandboxes are reachable by id (external control plane, e.g.
    OpenSandbox and Fargate, and the sandbox server's remote provider) implement
    this. A provider that does not implement it can only be shared by fronting it
    with a sandbox server. Membership is checked with ``isinstance`` because the
    protocol is ``runtime_checkable``.
    """

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a JSON-serializable descriptor that ``connect`` can rebuild a
        handle from. ``scope`` is honored by providers that mint leases (the
        remote provider) and ignored by the rest."""
        ...

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Rebuild a live handle in this process from a descriptor."""
        ...
