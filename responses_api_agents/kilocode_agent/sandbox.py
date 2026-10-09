# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kilo-specific runtime and artifacts for the shared task-sandbox lifecycle."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from time import perf_counter

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.utils import read_text, upload_text
from responses_api_agents.kilocode_agent.observability import read_kilo_observations


LOG = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class KiloArtifacts:
    """CLI events and diagnostics retained before releasing the sandbox."""

    stdout: str
    stderr: str
    exit_code: int | None
    observations: AgentObservationBundle | None = None
    usage: NeMoGymResponseUsage | None = None
    wall_time_s: float | None = None


@dataclass
class KiloSandboxSession(AgentSessionState):
    """Kilo input/output state; SandboxSession owns process cleanup and release."""

    session: SandboxSession[KiloArtifacts]
    provider_name: str | None = None
    started_at: float | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None

    @property
    def kilo_argv(self) -> list[str]:
        """Invoke the CLI through the private Node, so the task's PATH stays untouched."""
        runtime = f"{self.session.session_dir}/runtime"
        return [f"{runtime}/node/bin/node", f"{runtime}/kilo/node_modules/@kilocode/cli/bin/kilo"]

    async def install_runtime(self, *, version: str, timeout: float) -> None:
        """Install pinned Node and Kilo into the session directory with the bundled installer."""
        directory = self.session.session_dir
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=30)
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(f"Cannot create Kilo session directory: {prepared.stderr or prepared.stdout}")
        installer = f"{directory}/install_kilo_runtime.sh"
        await sandbox.upload(Path(__file__).with_name("install_kilo_runtime.sh"), installer)
        installed = await sandbox.exec(
            f"bash {quote(installer)} {quote(directory + '/runtime')} {quote(version)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if installed.return_code != 0 or installed.error_type:
            # Background execution can leave only a generic message in stderr; keep both streams.
            details = "\n".join(part for part in (installed.stderr, installed.stdout) if part)
            raise RuntimeError(
                f"Kilo sandbox installation failed (exit {installed.return_code}, "
                f"error_type={installed.error_type}): {details[-16000:]}"
            )
        await sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/sandbox_runner.py")

    async def stage_activation(self, *, command: list[str], config: str) -> SandboxCommand:
        """Stage the isolated provider config and CLI argv without altering task files."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/kilo.json", text=config)
        await upload_text(self.session.sandbox, path=f"{directory}/input.json", text=json.dumps({"command": command}))
        return SandboxCommand(
            argv=["python3", "-I", f"{directory}/sandbox_runner.py", directory, self.session.workdir],
            python="python3",
        )

    async def collect_artifacts(self) -> KiloArtifacts:
        """Retain flushed CLI output, including partial events after a supervised stop."""
        wall_time_s = perf_counter() - self.started_at if self.started_at is not None else None
        directory = self.session.session_dir
        try:
            stdout = await read_text(self.session.sandbox, path=f"{directory}/stdout.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Kilo runner did not produce readable artifacts: {logs}") from error
        try:
            stderr = await read_text(self.session.sandbox, path=f"{directory}/stderr.log")
        except Exception:
            stderr = await self.session.read_output_log()
        try:
            raw = json.loads(await read_text(self.session.sandbox, path=f"{directory}/exit.json"))
        except Exception:
            # A killed runner cannot record a CLI exit. Cleanup proof comes from the supervisor.
            LOG.warning("Kilo exit diagnostics unavailable; retaining partial CLI output", exc_info=True)
            raw = None
        exit_code = raw if type(raw) is int else None
        observations, usage = None, None
        try:
            snapshot = await self.session.sandbox.exec(
                f"python3 -I {quote(directory + '/sandbox_runner.py')} --snapshot {quote(directory)}",
                cwd=self.session.workdir,
                timeout_s=30,
            )
            if snapshot.error_type or snapshot.return_code != 0:
                raise RuntimeError(f"Kilo database snapshot failed: {snapshot.stderr}")
            with TemporaryDirectory(prefix="nemo-gym-kilo-observations-") as temporary:
                database = Path(temporary) / "observations.db"
                await self.session.sandbox.download(f"{directory}/observations.db", database)
                observations, usage = await asyncio.to_thread(
                    read_kilo_observations, database, fallback_invocation_id=self.request.agent_session_id
                )
        except Exception:
            # Partial stdout remains usable even if a killed worker never initialized its database.
            LOG.warning("Kilo persisted observations unavailable; retaining CLI output", exc_info=True)
        if stderr and (exit_code != 0 or (self.session.cleanup and self.session.cleanup["timed_out"])):
            LOG.warning(
                "Kilo diagnostics: episode=%s session=%s stderr=%s",
                self.request.episode_id.capture_key,
                self.request.agent_session_id,
                stderr[-4096:],
            )
        return KiloArtifacts(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            observations=observations,
            usage=usage,
            wall_time_s=wall_time_s,
        )

    async def close(self, *, timeout: float) -> None:
        """Confirm remote cleanup and capture output before cancelling activation waiters."""
        await self.session.close(timeout=timeout)
        if self.task is not None:
            if not self.task.done():
                self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
