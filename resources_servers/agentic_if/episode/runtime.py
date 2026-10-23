# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution in the borrowed task sandbox as the task identity."""

import math
from collections.abc import Mapping
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxExecResult


class BoundSandbox:
    """Execute as the task identity; staging transport remains explicitly separate.

    This enforces the episode's configuration, not an OS security boundary. Trusted
    staging may use ``transport`` for root-owned operations; work done on the
    actor's behalf (workspace staging, baseline, prefix replay) must use this bound
    interface. Lifecycle methods deliberately are not forwarded: the task backend
    owns the sandbox.
    """

    def __init__(
        self,
        transport: AsyncSandbox,
        *,
        user: str | int | None,
        workdir: str | None,
        timeout_s: float,
        execution_env: Mapping[str, str] | None = None,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("execution timeout must be finite and positive")
        self.transport = transport
        self.user = user
        self.workdir = workdir
        self.timeout_s = timeout_s
        self.execution_env = dict(execution_env or {})

    async def exec(
        self,
        *,
        command: str,
        timeout_s: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxExecResult:
        """Preserve explicit image-default identity and cap per-command timeouts."""
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("command timeout must be finite and positive")
        return await self.transport.exec(
            command=command,
            user=self.user,
            cwd=self.workdir if cwd is None else cwd,
            timeout_s=min(self.timeout_s, timeout_s) if timeout_s is not None else self.timeout_s,
            env=(self.execution_env | dict(env or {})) if self.execution_env or env is not None else None,
        )

    async def upload(self, source: Path, destination: str) -> None:
        """Stage an artifact through the provider; this is not an agent shell call."""
        await self.transport.upload(source, destination)

    async def download(self, source: str, destination: Path) -> None:
        """Collect an artifact without changing the actor identity."""
        await self.transport.download(source, destination)
