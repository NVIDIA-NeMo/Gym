# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenClaw execution in borrowed or agent-owned sandboxes."""

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
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.runner import (
    RunnerRuntimeInfo,
    confirm_runner_cleanup,
    read_text,
    supervisor_command,
    upload_text,
)


# Executed by sandbox Python before uploading any adapter-owned files. Resolve
# symlinks and '..' on the sandbox filesystem, not on the agent-server host.
_SANDBOX_PATH_CHECK = """
from pathlib import Path
import sys
workdir, session_root, runtime_root = (Path(value).resolve() for value in sys.argv[1:])
if not workdir.is_dir():
    raise SystemExit("OpenClaw task workdir is missing or is not a directory: " + str(workdir))
for owned in (session_root, runtime_root):
    if workdir == owned or workdir in owned.parents or owned in workdir.parents:
        raise SystemExit("OpenClaw task workdir overlaps adapter-owned storage: " + str(owned))
"""


@dataclass
class OpenClawSandboxSession(AgentSessionState):
    """Worker-local session with retryable, fail-closed runner teardown."""

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
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
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
            self.sandbox, directory=self.directory, workdir=self.workdir, timeout=timeout, harness="OpenClaw"
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
                raise RuntimeError(f"Could not remove OpenClaw session files: {result.stderr}")
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
                raise TimeoutError("OpenClaw sandbox supervisor exceeded its execution deadline")
            if launched.return_code != 0 or getattr(launched, "error_type", None):
                raise RuntimeError(f"OpenClaw sandbox supervisor failed: {launched.stderr}")
        except BaseException:
            try:
                await self.stop_runner(close_timeout)
            except Exception:
                logging.getLogger(__name__).exception("OpenClaw cleanup remains unconfirmed; close must retry")
            raise
        else:
            await self.stop_runner(close_timeout)
        # A fenced launch can be safely closed without ever running a worker.
        if self.cleanup is None or self.cleanup["return_code"] is None:
            raise RuntimeError("OpenClaw sandbox runner has no worker exit code")
        self.runtime_info = RunnerRuntimeInfo.model_validate_json(await self.read_text("runtime.json"))
        return await self.read_text("stdout.log")
