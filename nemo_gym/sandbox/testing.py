# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fake sandbox provider for tests of code that owns or borrows sandboxes.

Tests drive a real :class:`~nemo_gym.sandbox.api.AsyncSandbox` over :class:`FakeSandboxProvider`, so what the
sandbox guarantees, such as a stop that survives cancellation and is bounded in time, holds in the test as it does in
production. A test scripts the provider, not the sandbox: a slow, failing or hanging stop is a side effect on
``provider.on_close``.
"""

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus


class FakeSandboxProvider:
    """An in-memory provider: sandboxes start at once, commands succeed with no output, and nothing leaves the process.

    Attributes:
        on_close: Awaited, with no arguments, every time a sandbox is closed. Give it a side effect to make the
            provider's close slow, fail or hang: an async function, or a plain one that raises or returns an
            awaitable, which the close awaits too.
        closed: The ids of the sandboxes whose close succeeded, in order.
        client_closed: Whether the provider's client was released (``aclose``).
    """

    name = "fake"

    def __init__(self) -> None:
        self.on_close = AsyncMock()
        self.closed: list[str] = []
        self.client_closed = False
        self._created = 0

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self._created += 1
        return SandboxHandle(sandbox_id=f"fake-{self._created}", provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.STOPPED if handle.sandbox_id in self.closed else SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        outcome = await self.on_close()
        if inspect.isawaitable(outcome):
            await outcome
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        self.client_closed = True

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id, **({"scope": scope} if scope is not None else {})}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)


def fake_sandbox(
    sandbox_id: str = "fake-1",
    *,
    provider: FakeSandboxProvider | None = None,
    spec: SandboxSpec | None = None,
    owns_provider: bool = False,
) -> AsyncSandbox:
    """A started sandbox on ``provider`` (a new :class:`FakeSandboxProvider` by default), built without awaiting.

    It is what ``await AsyncSandbox(provider, spec).start()`` returns, for tests that need one in a synchronous
    fixture, and ``sandbox.provider`` is the provider to script. By default the sandbox does not own the provider,
    so stopping it leaves the provider's client open; with ``owns_provider`` a successful stop closes it too.
    """
    sandbox = AsyncSandbox(provider or FakeSandboxProvider(), spec or SandboxSpec(), owns_provider=owns_provider)
    sandbox._handle = SandboxHandle(sandbox_id=sandbox_id, provider_name=FakeSandboxProvider.name, raw=None)
    sandbox._stopped = False
    return sandbox
