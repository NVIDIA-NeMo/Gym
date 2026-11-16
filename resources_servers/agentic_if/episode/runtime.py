# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution in the borrowed task sandbox as the task identity."""

import logging
import tempfile
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from shlex import quote
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxExecResult


LOG = logging.getLogger(__name__)


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
        self.transport = transport
        self.user = user
        self.workdir = workdir
        self.timeout_s = timeout_s
        self.execution_env = dict(execution_env or {})

    async def exec(self, *, command: str, timeout_s: float, cwd: str | None = None) -> SandboxExecResult:
        """Preserve explicit image-default identity and cap per-command timeouts."""
        return await self.transport.exec(
            command=command,
            user=self.user,
            cwd=self.workdir if cwd is None else cwd,
            timeout_s=min(self.timeout_s, timeout_s),
            env=dict(self.execution_env) or None,
        )

    async def upload(self, source: Path, destination: str) -> None:
        """Stage an artifact through the provider; this is not an agent shell call."""
        await self.transport.upload(source, destination)


@asynccontextmanager
async def private_staging(sandbox: BoundSandbox, name: str) -> AsyncIterator[str]:
    """Yield a directory outside the repository that only the task identity can read, and remove it afterwards."""
    staging = f"/tmp/nemo-gym-agentic-if-{name}-{uuid4().hex}"
    created = await sandbox.exec(command=f"umask 077; mkdir -p {quote(staging)}", timeout_s=30)
    if created.return_code or created.error_type:
        raise RuntimeError(f"could not prepare a private {name} staging directory")
    try:
        yield staging
    finally:
        # Only the task identity can read the directory, so a failed removal is logged, and an error raised inside
        # stays the one reported.
        try:
            removed = await sandbox.transport.exec(command=f"rm -rf {quote(staging)}", user="root", timeout_s=30)
        except Exception:
            LOG.warning("could not remove %s from the task sandbox", staging, exc_info=True)
        else:
            if removed.return_code or removed.error_type:
                LOG.warning("could not remove %s from the task sandbox", staging)


async def upload_text(sandbox: BoundSandbox, path: str, text: str) -> None:
    """Upload ``text`` as UTF-8 to ``path`` in the task sandbox, readable by the task identity."""
    with tempfile.TemporaryDirectory(prefix="agentic-if-staging-") as temporary:
        local = Path(temporary) / "payload"
        local.write_text(text, encoding="utf-8")
        await sandbox.upload(local, path)
    mode = await sandbox.transport.exec(command=f"chmod 0644 {quote(path)}", user="root", timeout_s=30)
    if mode.return_code != 0 or mode.error_type:
        raise RuntimeError(f"cannot stage {path} for the task identity")
