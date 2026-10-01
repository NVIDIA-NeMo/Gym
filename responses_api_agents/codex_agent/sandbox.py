# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex-specific borrowed-sandbox execution; Resources retains sandbox ownership."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult


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
    exec_task: asyncio.Task[SandboxExecResult] | None = None
    result: CodexSandboxResult | None = None
    observations: AgentObservationBundle | None = None
    activated: bool = False
    closing: bool = False
    launch_started: bool = False
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
        """Fence delayed launches and require a receipt before allowing verification."""
        if not self.launch_started or (self.result is not None and self.result.cleanup_confirmed):
            return
        receipt_path = f"{self.directory}/result.json"
        try:
            result = CodexSandboxResult.model_validate_json(await self.read_text("result.json"))
        except Exception:
            result = None
        if result is None:
            # The atomic claim decides whether launch or close won. A stopped
            # claim can never start a harness, even if exec arrives much later.
            stop_path = quote(f"{self.directory}/runner.stop")
            claim_path = quote(f"{self.directory}/launch.claim")
            temporary = quote(f"{receipt_path}.{uuid4().hex}.tmp")
            stopped = quote(
                json.dumps(
                    {
                        "return_code": 1,
                        "timed_out": False,
                        "cleanup_confirmed": True,
                        "error": "Closed before runner launch",
                        "hostname": "",
                        "pid": 0,
                    }
                )
            )
            script = (
                f"[ -f {quote(receipt_path)} ] && exit 0; "
                f"touch {stop_path} || exit 1; "
                f"ln -s stop {claim_path} 2>/dev/null || true; "
                f'if [ "$(readlink {claim_path})" = stop ]; then '
                f"printf '%s' {stopped} > {temporary} && mv {temporary} {quote(receipt_path)}; exit $?; fi; "
                f"for _ in $(seq 1 {max(1, int(timeout))}); do "
                f"[ -f {quote(receipt_path)} ] && exit 0; sleep 1; done; exit 1"
            )
            await self.sandbox.exec(script, cwd=self.seed.sandbox_access.workdir, timeout_s=timeout + 5)
            try:
                result = CodexSandboxResult.model_validate_json(await self.read_text("result.json"))
            except Exception as error:
                raise RuntimeError("Codex launch outcome is unknown; cannot authorize verification") from error
        if not result.cleanup_confirmed:
            raise RuntimeError(f"Codex sandbox cleanup was not confirmed: {result.error}")
        # Cache only a successful receipt; failed cleanup remains retryable.
        self.result = result

    async def close(self, timeout: float) -> None:
        """Stop only Codex-owned work and detach; never call sandbox.stop()."""
        async with self.close_lock:
            if self.closed:
                return
            self.closing = True
            # Cancelling provider exec can kill the supervisor. Obtain its
            # descendant-cleanup receipt before cancelling the response task.
            await self.stop_runner(timeout)
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
            if self.exec_task is not None:
                # A confirmed receipt makes it safe to cancel a stuck provider
                # response; transport completion is not another cleanup gate.
                if not self.exec_task.done():
                    self.exec_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self.exec_task), timeout=timeout)
                except asyncio.CancelledError:
                    if not self.exec_task.cancelled():
                        raise
                except Exception:
                    if not self.exec_task.done():
                        raise
                    # Transport failure is not cleanup failure once the receipt is confirmed.
            retired = f"{self.directory}.closed"
            result = await self.sandbox.exec(
                f"if [ -d {quote(self.directory)} ]; then "
                f"mv {quote(self.directory)} {quote(retired)} || exit 1; fi; rm -rf -- {quote(retired)}",
                timeout_s=timeout,
            )
            if result.return_code != 0 or getattr(result, "error_type", None):
                raise RuntimeError(f"Could not remove Codex session files: {result.stderr}")
            await self.sandbox.disconnect()
            self.closed = True

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Run the supervisor through provider-neutral exec, without a PTY."""
        await self.upload_json("input.json", payload)
        command = (
            f"trap '' TERM; ln -s launch {quote(self.directory + '/launch.claim')} 2>/dev/null || exit 0; "
            f"exec python3 -I {quote(self.directory + '/sandbox_runner.py')} {quote(self.directory + '/input.json')} "
            f">{quote(self.directory + '/runner.log')} 2>&1"
        )
        self.launch_started = True
        try:
            # The runner enforces its own deadline and reaps descendants.
            # Leave extra time for cleanup and transport before provider timeout.
            self.exec_task = asyncio.create_task(
                self.sandbox.exec(
                    command, cwd=self.seed.sandbox_access.workdir, timeout_s=timeout + close_timeout * 3 + 30
                )
            )
            # HTTP cancellation must not propagate into provider exec before
            # the supervisor has stopped and reaped the harness descendants.
            launched = await asyncio.shield(self.exec_task)
            if getattr(launched, "error_type", None) == "timeout":
                raise TimeoutError("Codex sandbox supervisor exceeded its execution deadline")
            if launched.return_code != 0 or getattr(launched, "error_type", None):
                raise RuntimeError(f"Codex sandbox supervisor failed: {launched.stderr}")
        except BaseException:
            try:
                await self.stop_runner(close_timeout)
            except Exception:
                logging.getLogger(__name__).exception("Codex cleanup remains unconfirmed; close must retry")
            raise
        else:
            await self.stop_runner(close_timeout)
        return await self.read_text("events.jsonl")
