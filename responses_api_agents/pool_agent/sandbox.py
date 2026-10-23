# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool runtime and artifact transfer over Gym's shared sandbox lifecycle."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolArtifacts:
    """Artifacts copied before releasing the sandbox, including incomplete trajectories."""

    trajectory: str
    stdout: str
    stderr: str
    exit_code: int | None
    trajectory_error: str | None = None


@dataclass
class PoolSession(AgentSessionState):
    """Per-episode state; SandboxSession owns termination and connection release."""

    session: SandboxSession[PoolArtifacts]
    provider_name: str
    sandbox_id: str | None
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    failure: str | None = None
    installation_task: asyncio.Task[None] | None = None

    async def install(self, *, version: str, timeout: float) -> None:
        """Retain installation when a seed caller disconnects; close joins it before releasing the sandbox."""
        if self.installation_task is None:
            self.installation_task = asyncio.create_task(self._install(version=version, timeout=timeout))
        await asyncio.shield(self.installation_task)

    async def _install(self, *, version: str, timeout: float) -> None:
        directory = self.session.session_dir
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=timeout)
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError("Cannot create Pool session directory")
        for filename in ("install_pool.sh", "sandbox_runner.py"):
            await sandbox.upload(Path(__file__).with_name(filename), f"{directory}/{filename}")
        result = await sandbox.exec(
            f"sh {quote(directory + '/install_pool.sh')} {quote(directory)} {quote(version)}",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(
                f"Pool installation failed: {result.error_type or result.return_code}: "
                f"{(result.stderr or result.stdout or '')[-4096:]}"
            )
        # Resolve symlinks too: session cleanup must never remove a task workdir.
        validation = (
            "from pathlib import Path; import sys; "
            "a,b=map(lambda p: Path(p).resolve(),sys.argv[1:]); "
            "assert b.is_dir() and a != b and a not in b.parents and b not in a.parents"
        )
        result = await sandbox.exec(
            f"python3 -I -c {quote(validation)} {quote(directory)} {quote(self.session.workdir)}",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError("Pool session storage must be disjoint from an existing task workdir")

    async def stage(self, *, prompt: str, config: str) -> SandboxCommand:
        """Transfer inputs as files, never as shell interpolation."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=prompt)
        await upload_text(self.session.sandbox, path=f"{directory}/agent_config.json", text=config)
        access = self.request.sandbox_access
        await upload_text(
            self.session.sandbox, path=f"{directory}/task_env.json", text=json.dumps(access.env if access else {})
        )
        return SandboxCommand(
            python="python3",
            argv=["python3", "-I", f"{directory}/sandbox_runner.py", directory, self.session.workdir],
        )

    async def collect(self) -> PoolArtifacts:
        """Read partial output even when a terminated runner could not record its exit."""
        directory = self.session.session_dir

        async def optional(name: str) -> str:
            try:
                return await read_text(self.session.sandbox, path=f"{directory}/{name}")
            except Exception:
                LOG.warning("Pool artifact unavailable: %s", name, exc_info=True)
                return ""

        stdout = await optional("stdout.jsonl")
        stderr = await optional("stderr.log")
        try:
            exit_code = json.loads(await optional("exit.json"))
        except ValueError:
            exit_code = None
        if type(exit_code) is not int:
            exit_code = None
        trajectory, trajectory_error = "", None
        try:
            result = await self.session.sandbox.exec(
                f"python3 -I {quote(directory + '/sandbox_runner.py')} --snapshot {quote(directory)}",
                cwd="/",
                timeout_s=30,
            )
            if result.return_code != 0 or result.error_type:
                raise RuntimeError("Pool root trajectory snapshot failed")
            trajectory = await read_text(self.session.sandbox, path=f"{directory}/trajectory.ndjson")
        except Exception:
            trajectory_error = "trajectory_unavailable"
            LOG.warning("Pool trajectory unavailable after cleanup", exc_info=True)
        return PoolArtifacts(trajectory, stdout, stderr, exit_code, trajectory_error)

    async def close(self, *, timeout: float) -> None:
        """Capture artifacts and release the sandbox before cancelling the shared HTTP task."""
        if self.installation_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.installation_task), timeout=timeout)
            except Exception:
                if not self.installation_task.done():
                    # Retain state so a later close can retry after installation finishes.
                    raise
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done():
            self.task.cancel()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
