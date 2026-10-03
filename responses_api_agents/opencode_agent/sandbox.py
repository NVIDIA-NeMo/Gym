# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode execution in borrowed or agent-owned sandboxes."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from shlex import quote

from pydantic import JsonValue

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, process_supervisor
from nemo_gym.sandbox.process_supervisor import CleanupReceipt
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.runner import (
    RunnerRuntimeInfo,
    confirm_runner_cleanup,
    read_text,
    supervisor_command,
    upload_text,
)


@dataclass
class OpenCodeSandboxSession:
    """Worker-local session with retryable, fail-closed runner teardown."""

    seed: AgentSeedSessionRequest
    sandbox: AsyncSandbox
    directory: str
    runtime: str
    workdir: str = field(kw_only=True)
    owns_sandbox: bool = field(default=False, kw_only=True)
    sandbox_stopped: bool = False
    task: asyncio.Task[NeMoGymResponse] | None = None
    exec_task: asyncio.Task[SandboxExecResult] | None = None
    cleanup: CleanupReceipt | None = None
    runtime_info: RunnerRuntimeInfo | None = None
    observations: AgentObservationBundle | None = None
    activated: bool = False
    closing: bool = False
    launch_started: bool = False
    closed: bool = False
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Upload adapter-owned data beneath this session's directory."""
        await upload_text(self.sandbox, path=f"{self.directory}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read adapter-owned output independently of cleanup confirmation."""
        return await read_text(self.sandbox, path=f"{self.directory}/{name}")

    async def stop_runner(self, timeout: float) -> None:
        """Fence a delayed launch or require the shared supervisor's cleanup receipt."""
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
            return
        self.cleanup = await confirm_runner_cleanup(
            self.sandbox, directory=self.directory, workdir=self.workdir, timeout=timeout, harness="OpenCode"
        )

    async def close(self, timeout: float) -> None:
        """Stop owned sandboxes; only stop harness work and disconnect borrowed ones."""
        async with self.close_lock:
            if self.closed:
                return
            self.closing = True
            # Cancelling provider exec can kill the supervisor. Obtain its
            # descendant-cleanup receipt before cancelling the response task.
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
            if self.owns_sandbox:
                self.closed = True
                return
            retired = f"{self.directory}.closed"
            result = await self.sandbox.exec(
                f"if [ -d {quote(self.directory)} ]; then "
                f"mv {quote(self.directory)} {quote(retired)} || exit 1; fi; rm -rf -- {quote(retired)}",
                timeout_s=timeout,
            )
            if result.return_code != 0 or getattr(result, "error_type", None):
                raise RuntimeError(f"Could not remove OpenCode session files: {result.stderr}")
            await self.sandbox.disconnect()
            self.closed = True

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Run the supervisor through provider-neutral exec, without a PTY."""
        await self.upload_json("input.json", payload)
        cleanup_timeout = close_timeout / 3
        command = supervisor_command(
            directory=self.directory,
            command=["python3", "-I", f"{self.directory}/sandbox_runner.py", f"{self.directory}/input.json"],
            timeout=timeout,
            cleanup_timeout=cleanup_timeout,
        )
        self.launch_started = True
        try:
            # The runner enforces its own deadline and reaps descendants.
            # Leave extra time for cleanup and transport before provider timeout.
            self.exec_task = asyncio.create_task(
                self.sandbox.exec(
                    command,
                    cwd=self.workdir,
                    timeout_s=process_supervisor.exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout),
                )
            )
            # HTTP cancellation must not propagate into provider exec before
            # the supervisor has stopped and reaped the harness descendants.
            launched = await asyncio.shield(self.exec_task)
            if getattr(launched, "error_type", None) == "timeout":
                raise TimeoutError("OpenCode sandbox supervisor exceeded its execution deadline")
            if launched.return_code != 0 or getattr(launched, "error_type", None):
                raise RuntimeError(f"OpenCode sandbox supervisor failed: {launched.stderr}")
        except BaseException:
            try:
                await self.stop_runner(close_timeout)
                # Preserve interrupted transcripts before the caller's close
                # retires session files; never snapshot without cleanup evidence.
                if self.cleanup is not None and self.cleanup["return_code"] is not None:
                    await self.snapshot(close_timeout)
            except Exception:
                logging.getLogger(__name__).exception("OpenCode cleanup or interrupted transcript capture failed")
            raise
        else:
            await self.stop_runner(close_timeout)
        # A fenced launch can be safely closed without ever running a worker.
        if self.cleanup is None or self.cleanup["return_code"] is None:
            raise RuntimeError("OpenCode sandbox runner has no worker exit code")
        self.runtime_info = RunnerRuntimeInfo.model_validate_json(await self.read_text("runtime.json"))
        await self.snapshot(close_timeout)
        return await self.read_text("export.json")

    async def snapshot(self, timeout: float) -> None:
        """Capture output only after confirmed cleanup, without changing its receipt."""
        if self.cleanup is None or not self.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        # Snapshot only after the supervisor has reaped all database writers.
        captured = await self.sandbox.exec(
            f"python3 -I {quote(self.directory + '/sandbox_runner.py')} --snapshot {quote(self.directory)}",
            cwd=self.workdir,
            timeout_s=timeout,
        )
        if captured.return_code != 0 or getattr(captured, "error_type", None):
            raise RuntimeError(f"OpenCode transcript capture failed: {captured.stderr}")
