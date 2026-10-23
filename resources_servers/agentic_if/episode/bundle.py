# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode's use of the IF OpenCode plugin bundle: stage it at seed, read what it recorded at verify.

``opencode_bundle`` owns the bundle's files, compilation and formats; this module adapts the borrowed sandbox to its
``SandboxIO`` and converts its outputs to the inputs of the evidence builder, in one place.
"""

import dataclasses
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import BaseModel, JsonValue

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxExecResult
from resources_servers.agentic_if import opencode_bundle
from resources_servers.agentic_if.if_grading.contracts import SessionInfo
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


@dataclass(frozen=True)
class StagedBundle:
    """What seed staged: the profile hash the plugin's load receipt must carry, and the staging receipt."""

    profile_sha256: str
    receipt: dict[str, JsonValue]


@dataclass(frozen=True)
class CollectedBundle:
    """What the plugins and the harness left in the bundle, in the shapes the evidence builder consumes."""

    exports: dict[str, dict]
    # The harness's session inventory; a session listed without a readable export is a gap, never an empty one.
    sessions: list[SessionInfo]
    hook_receipts: list[dict[str, JsonValue]]
    event_records: list[dict[str, JsonValue]]
    call_records: list[dict[str, JsonValue]]
    # Exception type names of model calls the boundary could not forward.
    boundary_issues: list[str]
    gaps: list[str]
    # The raw bytes of every file read, by path relative to the bundle root, kept as the episode's artifacts.
    files: dict[str, bytes]
    # Where the boundary's records show the model was not given the row's literal system prompt.
    presentation_gaps: list[str]


def _jsonable(value: object) -> dict[str, JsonValue]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json.loads(json.dumps(dataclasses.asdict(value), default=str))
    if isinstance(value, Mapping):
        return json.loads(json.dumps(dict(value), default=str))
    raise TypeError(f"unsupported bundle receipt type {type(value).__name__}")


async def stage(transport: AsyncSandbox, *, spec: RuntimeSpec, agent_user: str | int | None) -> StagedBundle:
    """Stage the plugins and their compiled configuration, readable and writable by ``agent_user``."""
    receipt = await opencode_bundle.stage_bundle(BundleSandboxIO(transport), spec=spec, agent_user=agent_user)
    profile = getattr(receipt, "profile_sha256", None)
    if not isinstance(profile, str) or not profile:
        raise RuntimeError("the plugin bundle was staged without a profile hash")
    return StagedBundle(profile_sha256=profile, receipt=_jsonable(receipt))


async def collect(
    transport: AsyncSandbox, *, expected: Mapping[str, JsonValue] | None, system: str | None
) -> CollectedBundle:
    """Read the OpenCode exports, hook receipts, plant firings and boundary call records the bundle holds.

    ``expected`` is what seed staged (``StagedBundle.receipt``): the plugins' load receipts must match it. ``system``
    is the text of the row's system message, or None: every model call must have carried it as its system prompt.
    """
    outputs = await opencode_bundle.collect_bundle(BundleSandboxIO(transport), expected=expected)
    return CollectedBundle(
        exports={str(key): dict(value) for key, value in outputs.exports.items()},
        sessions=[SessionInfo.model_validate(session, from_attributes=True) for session in outputs.sessions],
        hook_receipts=[dict(row) for row in outputs.hook_receipts],
        event_records=[dict(row) for row in outputs.event_records],
        call_records=[dict(row) for row in outputs.call_records],
        boundary_issues=[str(issue) for issue in outputs.boundary_issues],
        gaps=[str(gap) for gap in outputs.gaps],
        files={str(path): bytes(data) for path, data in outputs.files.items()},
        presentation_gaps=[str(gap) for gap in opencode_bundle.presentation_gaps(outputs, system=system)],
    )


async def stage_session_import(transport: AsyncSandbox, prefix_export: Mapping[str, JsonValue]) -> str:
    """Write a continuation's recorded prefix where the harness imports it, readable by every identity."""
    path = opencode_bundle.SESSION_IMPORT_PATH
    await BundleSandboxIO(transport).upload_bytes(path, json.dumps(prefix_export).encode())
    mode = await transport.exec(f"chmod 0644 {quote(path)}", user="root", timeout_s=30)
    if mode.return_code != 0 or mode.error_type:
        raise RuntimeError("cannot stage the recorded prefix for import")
    return path
