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

"""Provider-facing sandbox protocol.

Providers are the only layer that talks to runtime and infrastructure APIs.
Gym agents and external harnesses consume the public ``nemo_gym.sandbox`` API
instead of importing provider-specific modules.
"""

from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SandboxSpec:
    """Provider-neutral sandbox creation request."""

    image: str | None = None
    snapshot_id: str | None = None
    timeout_s: int | None = None
    ready_timeout_s: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)
    entrypoint: list[str] | None = None
    extensions: dict[str, str] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxHandle:
    """Provider-neutral handle to a created sandbox.

    ``raw`` is provider-owned opaque state, such as an SDK sandbox object,
    transport session, or lightweight provider reference. Public Gym code
    should pass it back to the provider through this handle rather than
    inspecting or mutating it directly.
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


class SandboxCreateError(RuntimeError):
    """Raised when a provider cannot create a sandbox."""


class SandboxBatchCreateError(RuntimeError):
    """Raised when a provider cannot complete sandbox batch creation."""


class SandboxCreateVerificationError(SandboxCreateError):
    """Raised when a newly-created sandbox fails provider readiness checks."""


class SandboxProvider(Protocol):
    """Runtime/infra provider contract used by the public sandbox API."""

    name: str

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Create a sandbox and return a provider-neutral handle."""
        ...

    async def create_batch(
        self,
        spec: SandboxSpec,
        count: int,
        *,
        allow_partial: bool = False,
    ) -> list[SandboxHandle]:
        """Create several equivalent sandboxes.

        Providers that have a native bulk-allocation primitive or warm-pool
        implementation should use it. Providers without one may fall back to
        calling ``create`` repeatedly. Long-lived pools are provider-owned and
        configured through provider config or ``SandboxSpec.provider_options``,
        rather than through a separate public pool handle.

        When ``allow_partial`` is true, providers may return a smaller
        contiguous prefix of successfully created handles instead of failing the
        whole batch.
        """
        ...

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        """Connect to an existing sandbox."""
        ...

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
        """Run a command inside a sandbox."""
        ...

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        """Write a file into a sandbox."""
        ...

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        """Read a file from a sandbox."""
        ...

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one local file into a sandbox."""
        ...

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the local filesystem."""
        ...

    async def close(self, handle: SandboxHandle, *, delete: bool = False) -> None:
        """Close provider resources and optionally delete the sandbox."""
        ...

    async def aclose(self) -> None:
        """Close provider-scoped resources such as SDK clients or warm pools."""
        ...


@runtime_checkable
class SandboxHandleReferenceProvider(Protocol):
    """Optional provider trait for loop-safe sandbox handle references."""

    def handle_reference(self, handle: SandboxHandle) -> Any | Awaitable[Any]:
        """Return a serializable or loop-safe reference for ``handle``."""
        ...

    def materialize_handle(self, value: Any) -> SandboxHandle | Awaitable[SandboxHandle]:
        """Convert a value from ``handle_reference`` back into a local handle."""
        ...
