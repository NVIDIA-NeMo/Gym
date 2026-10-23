# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode's use of the IF OpenCode plugin bundle: stage it at seed, read what it recorded at verify.

``opencode_bundle`` owns the bundle's files, compilation and formats; this module adapts the borrowed sandbox to its
``SandboxIO``.
"""

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from shlex import quote

from pydantic import JsonValue

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if import opencode_bundle
from resources_servers.agentic_if.opencode_bundle.bundle import BundleOutputs
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


class BundleSandboxIO:
    """``opencode_bundle.SandboxIO`` over the borrowed sandbox's transport; ``user=None`` keeps its default."""

    def __init__(self, transport: AsyncSandbox) -> None:
        self.transport = transport

    async def exec(self, command: str, *, timeout: float, user: str | int | None = None) -> SandboxExecResult:
        """Run one command in the task sandbox."""
        return await self.transport.exec(command, timeout_s=timeout, user=user)

    async def upload_bytes(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path`` in the task sandbox."""
        with tempfile.TemporaryDirectory(prefix="agentic-if-bundle-") as temporary:
            local = Path(temporary) / "payload"
            local.write_bytes(data)
            await self.transport.upload(local, path)

    async def download_bytes(self, path: str) -> bytes:
        """Read ``path`` from the task sandbox."""
        with tempfile.TemporaryDirectory(prefix="agentic-if-bundle-") as temporary:
            local = Path(temporary) / "payload"
            await self.transport.download(path, local)
            return local.read_bytes()


async def stage(
    transport: AsyncSandbox, *, spec: RuntimeSpec, agent_user: str | int | None
) -> opencode_bundle.BundleReceipt:
    """Stage the plugins and their compiled configuration, readable and writable by ``agent_user``."""
    return await opencode_bundle.stage_bundle(BundleSandboxIO(transport), spec=spec, agent_user=agent_user)


async def collect(transport: AsyncSandbox, *, expected: Mapping[str, JsonValue]) -> BundleOutputs:
    """Read the OpenCode exports, hook receipts, plant firings and boundary call records the bundle holds.

    ``expected`` is the staging receipt seed kept: the plugins' load receipts must match it.
    """
    return await opencode_bundle.collect_bundle(BundleSandboxIO(transport), expected=expected)


async def stage_runtime_files(transport: AsyncSandbox, paths: Sequence[Path]) -> list[str]:
    """Copy host files into the bundle directory by name, readable by every identity; return their sandbox paths."""
    staged = []
    for path in paths:
        target = f"{opencode_bundle.BUNDLE_ROOT}/{path.name}"
        await transport.upload(path, target)
        mode = await transport.exec(f"chmod 0644 {quote(target)}", user="root", timeout_s=30)
        if mode.return_code != 0 or mode.error_type:
            raise RuntimeError(f"cannot stage {path.name} for the agent")
        staged.append(target)
    return staged


async def stage_session_import(transport: AsyncSandbox, prefix_export: Mapping[str, JsonValue]) -> str:
    """Write a continuation's recorded prefix where the harness imports it, readable by every identity."""
    path = opencode_bundle.SESSION_IMPORT_PATH
    await BundleSandboxIO(transport).upload_bytes(path, json.dumps(prefix_export).encode())
    mode = await transport.exec(f"chmod 0644 {quote(path)}", user="root", timeout_s=30)
    if mode.return_code != 0 or mode.error_type:
        raise RuntimeError("cannot stage the recorded prefix for import")
    return path
