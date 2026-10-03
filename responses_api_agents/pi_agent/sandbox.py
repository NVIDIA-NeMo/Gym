# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi execution in borrowed or agent-owned sandboxes."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from shlex import quote

from pydantic import JsonValue

from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, process_supervisor
from nemo_gym.sandbox.process_supervisor import CleanupReceipt
from nemo_gym.sandbox.runner import (
    RunnerRuntimeInfo,
    confirm_runner_cleanup,
    read_text,
    supervisor_command,
    upload_text,
)


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
    runtime_info: RunnerRuntimeInfo | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    closing: bool = False
    launch_started: bool = False
    cleanup: CleanupReceipt | None = None
    closed: bool = False

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Upload adapter-owned data beneath this session's directory."""
        await upload_text(self.sandbox, path=f"{self.directory}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read an adapter-owned result without interpreting it as a shell command."""
        return await read_text(self.sandbox, path=f"{self.directory}/{name}")

    async def stop_runner(self, timeout: float) -> None:
        """Fence a delayed launch or require the shared supervisor's cleanup receipt."""
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
            return
        self.cleanup = await confirm_runner_cleanup(
            self.sandbox, directory=self.directory, workdir=self.workdir, timeout=timeout, harness="Pi"
        )

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
        command = supervisor_command(
            directory=self.directory,
            command=["python3", "-I", f"{self.directory}/sandbox_runner.py", f"{self.directory}/input.json"],
            timeout=timeout,
            cleanup_timeout=cleanup_timeout,
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
            # Cleanup alone can acknowledge a fenced launch that never ran a worker.
            if self.cleanup is None or self.cleanup["return_code"] is None:
                raise RuntimeError("Sandbox runner has no worker exit code")
            self.runtime_info = RunnerRuntimeInfo.model_validate_json(await self.read_text("runtime.json"))
            return await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.sandbox.exec(f"cat {quote(self.directory + '/runner.log')}", timeout_s=30)
            raise RuntimeError(f"Pi sandbox runner returned no valid result: {logs.stdout or ''}") from error
