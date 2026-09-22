# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex-specific borrowed-sandbox execution; Resources retains sandbox ownership."""

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxPtySession


class CodexSandboxResult(BaseModel):
    """Require an explicit cleanup acknowledgement, not just process exit."""

    model_config = ConfigDict(extra="forbid", strict=True)
    return_code: int
    timed_out: bool
    cleanup_confirmed: bool
    error: str | None
    hostname: str
    pid: int


@dataclass
class CodexSandboxSession:
    """Worker-local session with retryable, fail-closed runner teardown."""

    seed: AgentSeedSessionRequest
    sandbox: AsyncSandbox
    directory: str
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    runner: SandboxPtySession | None = None
    exit_task: asyncio.Task[int] | None = None
    result: CodexSandboxResult | None = None
    observations: AgentObservationBundle | None = None
    activated: bool = False
    closing: bool = False
    launch_started: bool = False
    runner_closed: bool = False
    closed: bool = False
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Upload adapter-owned data beneath this session's directory."""
        with tempfile.TemporaryDirectory(prefix="codex-session-upload-") as directory:
            path = Path(directory) / "payload.json"
            path.write_text(json.dumps(payload))
            await self.sandbox.upload(path, f"{self.directory}/{name}")

    async def upload_text(self, name: str, text: str) -> None:
        """Upload adapter-owned configuration without shell interpolation."""
        with tempfile.TemporaryDirectory(prefix="codex-session-upload-") as directory:
            path = Path(directory) / "payload"
            path.write_text(text)
            await self.sandbox.upload(path, f"{self.directory}/{name}")

    async def read_text(self, name: str) -> str:
        """Read an adapter-owned result without interpreting it as a shell command."""
        with tempfile.TemporaryDirectory(prefix="codex-session-download-") as directory:
            path = Path(directory) / "payload"
            await self.sandbox.download(f"{self.directory}/{name}", path)
            return path.read_text(errors="replace")

    async def stop_runner(self, timeout: float) -> None:
        """Wait for the supervisor's cleanup receipt; retain handles on any failure."""
        if not self.launch_started:
            return
        if self.runner is None or self.exit_task is None:
            raise RuntimeError("Codex launch outcome is unknown; cannot authorize verification")
        if not self.exit_task.done():
            await self.runner.send_signal("SIGTERM")
        await asyncio.wait_for(asyncio.shield(self.exit_task), timeout=timeout)
        if self.result is None:
            self.result = CodexSandboxResult.model_validate_json(await self.read_text("result.json"))
        if not self.result.cleanup_confirmed:
            raise RuntimeError(f"Codex sandbox cleanup was not confirmed: {self.result.error}")
        if not self.runner_closed:
            await self.runner.close()
            self.runner_closed = True
        # Keep the successful receipt for retries if file cleanup/disconnect fails.

    async def close(self, timeout: float) -> None:
        """Stop only Codex-owned work and detach; never call sandbox.stop()."""
        async with self.close_lock:
            if self.closed:
                return
            self.closing = True
            if self.task is not None:
                if not self.task.done() and not self.task.cancelling():
                    self.task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
                except asyncio.CancelledError:
                    if not self.task.cancelled():
                        raise
                except Exception:
                    if not self.task.done():
                        raise
                    # A response error does not establish cleanup; stop_runner below must.
            await self.stop_runner(timeout)
            result = await self.sandbox.exec(f"rm -rf -- {quote(self.directory)}", timeout_s=timeout)
            if result.return_code != 0 or result.error_type:
                raise RuntimeError(f"Could not remove Codex session files: {result}")
            await self.sandbox.disconnect()
            self.closed = True

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Start the supervisor and Codex inside the borrowed task sandbox."""
        await self.upload_json("input.json", payload)
        self.launch_started = True
        self.runner = await self.sandbox.pty.create(
            command=f"exec python3 -I {quote(self.directory + '/sandbox_runner.py')} {quote(self.directory + '/input.json')}",
            cwd=self.seed.sandbox_access.workdir,
            pty=False,
        )
        self.exit_task = asyncio.create_task(self.runner.wait_exit())
        try:
            # The supervisor owns the execution timeout; this bounds missing receipts/provider failures too.
            await asyncio.wait_for(asyncio.shield(self.exit_task), timeout=timeout + close_timeout * 3)
            self.result = CodexSandboxResult.model_validate_json(await self.read_text("result.json"))
            return await self.read_text("events.jsonl")
        finally:
            await self.stop_runner(close_timeout)
