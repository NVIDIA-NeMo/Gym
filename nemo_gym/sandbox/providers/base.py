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
from typing import Any, Protocol, runtime_checkable


class SandboxStatus(str, Enum):
    """Provider-neutral sandbox lifecycle status."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


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
    provider_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resources, SandboxResources):
            object.__setattr__(self, "resources", SandboxResources.from_mapping(self.resources))


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
