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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


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
    platform: dict[str, Any] | None = None
    volumes: list[dict[str, Any]] | None = None
    skip_health_check: bool | None = None


@dataclass(frozen=True)
class SandboxHandle:
    """Provider-neutral handle to a created sandbox."""

    sandbox_id: str
    provider_name: str
    raw: Any


@dataclass(frozen=True)
class SandboxExecResult:
    """Provider-neutral process execution result."""

    stdout: str | None
    stderr: str | None
    return_code: int


class SandboxCreateError(RuntimeError):
    """Raised when a provider cannot create a sandbox."""


class SandboxBatchCreateError(SandboxCreateError):
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

        Providers that have a native bulk-allocation primitive should use it.
        Providers without one may fall back to calling ``create`` repeatedly.
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

    async def close(self, handle: SandboxHandle, *, delete: bool) -> None:
        """Close provider resources and optionally delete the sandbox."""
        ...
