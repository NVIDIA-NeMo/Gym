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
from typing import Any, Protocol
from urllib.parse import urlsplit


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


@dataclass(frozen=True)
class SandboxResources:
    """Provider-neutral resource request."""

    cpu: float | None = None
    memory_mib: int | None = None
    disk_gib: int | None = None
    gpu: int | None = None
    gpu_type: str | None = None

    @classmethod
    def from_mapping(cls, resources: Mapping[str, Any] | None) -> "SandboxResources":
        if resources is None:
            return cls()
        allowed_keys = set(cls.__dataclass_fields__)
        unknown_keys = set(resources) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            allowed = ", ".join(sorted(allowed_keys))
            raise ValueError(f"Unknown sandbox resource keys: {unknown}. Expected keys: {allowed}")
        return cls(
            cpu=float(resources["cpu"]) if resources.get("cpu") is not None else None,
            memory_mib=int(resources["memory_mib"]) if resources.get("memory_mib") is not None else None,
            disk_gib=int(resources["disk_gib"]) if resources.get("disk_gib") is not None else None,
            gpu=int(resources["gpu"]) if resources.get("gpu") is not None else None,
            gpu_type=str(resources["gpu_type"]) if resources.get("gpu_type") is not None else None,
        )


@dataclass(frozen=True)
class SandboxSpec:
    """Sandbox creation request."""

    image: str | None = None
    ttl_s: int | float | None = None
    ready_timeout_s: int | float | None = None
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    resources: SandboxResources | Mapping[str, Any] = field(default_factory=SandboxResources)
    entrypoint: list[str] | None = None
    ports: tuple[int, ...] | list[int] = field(default_factory=tuple)
    provider_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resources, SandboxResources):
            object.__setattr__(self, "resources", SandboxResources.from_mapping(self.resources))
        if not isinstance(self.ports, (list, tuple)):
            raise TypeError("Sandbox ports must be a list or tuple of TCP port numbers")
        normalized_ports: list[int] = []
        for raw_port in self.ports:
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
        object.__setattr__(self, "ports", tuple(normalized_ports))


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

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a declared service port to a caller-reachable endpoint.

        Providers without service networking may omit this optional capability;
        the public API raises ``NotImplementedError`` in that case.
        """
        ...

    async def close(self, handle: SandboxHandle) -> None:
        """End the sandbox lifecycle and close provider resources for it."""
        ...

    async def aclose(self) -> None:
        """Close provider-scoped resources such as SDK clients."""
        ...
