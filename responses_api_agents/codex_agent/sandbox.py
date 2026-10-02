# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Codex-specific borrowed-sandbox execution; Resources retains sandbox ownership."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, process_supervisor
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
class CodexSandboxSession(AgentSessionState):
    """Worker-local session with retryable, fail-closed runner teardown."""

    sandbox: AsyncSandbox
    directory: str
    runtime: str
    task: asyncio.Task[NeMoGymResponse] | None = None
    exec_task: asyncio.Task[SandboxExecResult] | None = None
    result: CodexSandboxResult | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    closing: bool = False
    launch_started: bool = False
    cleanup: dict[str, JsonValue] | None = None
    closed: bool = False

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Upload adapter-owned data beneath this session's directory."""
        with tempfile.TemporaryDirectory(prefix="codex-session-upload-") as directory:
            path = Path(directory) / "payload.json"
            path.write_text(json.dumps(payload))
            await self.sandbox.upload(path, f"{self.directory}/{name}")

    async def upload_text(self, name: str, text: str) -> None:
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
        """Fence a delayed launch or require the shared supervisor's cleanup receipt."""
        if not self.launch_started or self.cleanup is not None:
            return
        receipt_path = f"{self.directory}/cleanup.json"
        try:
            receipt = json.loads(await self.read_text("cleanup.json"))
        except Exception:
            receipt = {}
        if receipt.get("cleanup_confirmed") is not True:
            pid_path = quote(f"{self.directory}/runner.pid")
            stop_path = quote(f"{self.directory}/runner.stop")
            claim_path = quote(f"{self.directory}/launch.claim")
            temporary = quote(f"{receipt_path}.{uuid4().hex}.tmp")
            stopped = quote(json.dumps({"cleanup_confirmed": True, "error": None}))
            script = (
                f"[ -f {quote(receipt_path)} ] && exit 0; "
                f"touch {stop_path} || exit 1; "
                f"ln -s stop {claim_path} 2>/dev/null || true; "
                f'if [ "$(readlink {claim_path})" = stop ]; then '
                f"printf '%s' {stopped} > {temporary} && mv {temporary} {quote(receipt_path)}; exit $?; fi; "
                f'if [ -s {pid_path} ]; then kill -TERM "$(cat {pid_path})" 2>/dev/null || true; fi; '
                f"for _ in $(seq 1 {max(1, int(timeout))}); do "
                f"[ -f {quote(receipt_path)} ] && exit 0; sleep 1; done; exit 1"
            )
            await self.sandbox.exec(script, cwd=self.request.sandbox_access.workdir, timeout_s=timeout + 5)
            try:
                receipt = json.loads(await self.read_text("cleanup.json"))
            except Exception as error:
                raise RuntimeError("Codex launch outcome is unknown; cannot confirm termination") from error
            if receipt.get("cleanup_confirmed") is not True:
                raise RuntimeError(f"Codex sandbox cleanup was not confirmed: {receipt.get('error')}")
        self.cleanup = receipt

    async def _release_exec(self) -> None:
        # Only release the provider transport after remote cleanup is acknowledged.
        if self.cleanup is not None and self.exec_task is not None:
            if not self.exec_task.done():
                self.exec_task.cancel()
            await asyncio.gather(self.exec_task, return_exceptions=True)

    async def close(self, timeout: float) -> None:
        """Stop only Codex-owned work and detach; never call sandbox.stop()."""
        if self.closed:
            return
        self.closing = True
        # Provider cancellation can kill its exec process group, including the
        # supervisor. Let the supervisor reap Codex and acknowledge cleanup first.
        await self.stop_runner(timeout)
        await self._release_exec()
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
                # The supervisor has already acknowledged cleanup above.
        retired = f"{self.directory}.closed"
        # Keep the claim intact until its parent path is retired, fencing delayed execs.
        result = await self.sandbox.exec(
            f"if [ -d {quote(self.directory)} ]; then "
            f"mv {quote(self.directory)} {quote(retired)} || exit 1; fi; rm -rf -- {quote(retired)}",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError("Could not remove Codex session files")
        await self.sandbox.disconnect()
        self.closed = True

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Start the supervisor and Codex inside the borrowed task sandbox."""
        await self.upload_json("input.json", payload)
        cleanup_timeout = close_timeout / 3
        command = (
            f"trap '' TERM; ln -s launch {quote(self.directory + '/launch.claim')} 2>/dev/null || exit 0; "
            f"echo $$ > {quote(self.directory + '/runner.pid')} && "
            f"exec python3 -I {quote(self.directory + '/process_supervisor.py')} "
            f"--timeout {timeout} --cleanup-timeout {cleanup_timeout} "
            f"--stop-file {quote(self.directory + '/runner.stop')} "
            f"--receipt {quote(self.directory + '/cleanup.json')} -- "
            f"python3 -I {quote(self.directory + '/sandbox_runner.py')} {quote(self.directory + '/input.json')} "
            f">{quote(self.directory + '/runner.log')} 2>&1"
        )
        self.launch_started = True
        self.exec_task = asyncio.create_task(
            self.sandbox.exec(
                command,
                cwd=self.request.sandbox_access.workdir,
                timeout_s=process_supervisor.exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout),
            )
        )
        try:
            launched = await asyncio.shield(self.exec_task)
            if launched.error_type == "timeout":
                raise TimeoutError("Codex sandbox supervisor did not finish within its execution deadline")
        except BaseException:
            try:
                await self.stop_runner(close_timeout)
            except Exception:
                logging.getLogger(__name__).exception("Codex cleanup remains unconfirmed; session close must retry")
            raise
        else:
            await self.stop_runner(close_timeout)
        finally:
            await self._release_exec()
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
            self.result = CodexSandboxResult.model_validate({**self.cleanup, **runtime})
            return await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.sandbox.exec(f"cat {quote(self.directory + '/runner.log')}", timeout_s=30)
            raise RuntimeError(f"Codex sandbox runner returned no valid result: {logs.stdout or ''}") from error
