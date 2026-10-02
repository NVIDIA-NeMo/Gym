# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi execution in borrowed or agent-owned sandboxes."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue

from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, process_supervisor


class PiSandboxResult(BaseModel):
    """Require an explicit cleanup acknowledgement, not just process exit."""

    model_config = ConfigDict(extra="forbid", strict=True)
    return_code: int
    timed_out: bool
    cleanup_confirmed: bool
    error: str | None
    hostname: str
    pid: int


@dataclass
class PiSandboxSession(AgentSessionState):
    """Worker-local session with retryable, fail-closed runner teardown."""

    sandbox: AsyncSandbox
    directory: str
    runtime: str
    workdir: str = field(kw_only=True)
    owns_sandbox: bool = field(default=False, kw_only=True)
    sandbox_stopped: bool = False
    task: asyncio.Task[NeMoGymResponse] | None = None
    result: PiSandboxResult | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    closing: bool = False
    launch_started: bool = False
    cleanup: dict[str, JsonValue] | None = None
    closed: bool = False

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Upload adapter-owned data beneath this session's directory."""
        with tempfile.TemporaryDirectory(prefix="pi-session-upload-") as directory:
            path = Path(directory) / "payload.json"
            path.write_text(json.dumps(payload))
            await self.sandbox.upload(path, f"{self.directory}/{name}")

    async def read_text(self, name: str) -> str:
        """Read an adapter-owned result without interpreting it as a shell command."""
        with tempfile.TemporaryDirectory(prefix="pi-session-download-") as directory:
            path = Path(directory) / "payload"
            await self.sandbox.download(f"{self.directory}/{name}", path)
            return path.read_text(errors="replace")

    async def stop_runner(self, timeout: float) -> None:
        """Fence a delayed launch or require the shared supervisor's cleanup receipt."""
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
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
            await self.sandbox.exec(script, cwd=self.workdir, timeout_s=timeout + 5)
            try:
                receipt = json.loads(await self.read_text("cleanup.json"))
            except Exception as error:
                raise RuntimeError("Pi launch outcome is unknown; cannot confirm termination") from error
            if receipt.get("cleanup_confirmed") is not True:
                raise RuntimeError(f"Pi sandbox cleanup was not confirmed: {receipt.get('error')}")
        self.cleanup = receipt

    async def close(self, timeout: float) -> None:
        """Stop owned sandboxes; only stop harness work and disconnect borrowed ones."""
        if self.closed:
            return
        self.closing = True
        # Provider cancellation can kill its exec process group, including the
        # supervisor. Let the supervisor reap Pi and acknowledge cleanup first.
        if self.owns_sandbox:
            # The provider is the cleanup authority for an agent-owned sandbox.
            # Keep the handle retryable if stop fails or times out.
            if not self.sandbox_stopped:
                await asyncio.wait_for(self.sandbox.stop(), timeout=timeout)
                self.sandbox_stopped = True
        else:
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
                # The supervisor has already acknowledged cleanup above.
        if self.owns_sandbox:
            self.closed = True
            return
        retired = f"{self.directory}.closed"
        # Keep the claim intact until its parent path is retired, fencing delayed execs.
        result = await self.sandbox.exec(
            f"if [ -d {quote(self.directory)} ]; then "
            f"mv {quote(self.directory)} {quote(retired)} || exit 1; fi; rm -rf -- {quote(retired)}",
            timeout_s=timeout,
        )
        if result.return_code != 0:
            raise RuntimeError("Could not remove Pi session files")
        await self.sandbox.disconnect()
        self.closed = True

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Start the supervisor and Pi inside the session sandbox."""
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
        try:
            launched = await self.sandbox.exec(
                command,
                cwd=self.workdir,
                timeout_s=process_supervisor.exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout),
            )
            if launched.error_type == "timeout":
                raise TimeoutError("Pi sandbox supervisor did not finish within its execution deadline")
        except BaseException:
            try:
                await self.stop_runner(close_timeout)
            except Exception:
                logging.getLogger(__name__).exception("Pi cleanup remains unconfirmed; session close must retry")
            raise
        else:
            await self.stop_runner(close_timeout)
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
            self.result = PiSandboxResult.model_validate({**self.cleanup, **runtime})
            return await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.sandbox.exec(f"cat {quote(self.directory + '/runner.log')}", timeout_s=30)
            raise RuntimeError(f"Pi sandbox runner returned no valid result: {logs.stdout or ''}") from error
