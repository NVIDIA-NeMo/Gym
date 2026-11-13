# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of Terminus-2 execution in borrowed or owned task sandboxes."""

import asyncio
import functools
import importlib.metadata
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import JsonValue

from nemo_gym.agents import (
    AgentSeedRejectedError,
    AgentSessionState,
    SandboxCommand,
    SandboxPythonRuntime,
    SandboxSession,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)

# Harbor's dependencies that Terminus-2 never imports: its sandbox backends, dataset tools and CLI. Overriding them
# with a marker that never matches leaves them out of the sandbox runtime.
_UNUSED_HARBOR_DEPENDENCIES = (
    "claude-agent-sdk",
    "datasets",
    "daytona",
    "dirhash",
    "dockerfile-parse",
    "e2b",
    "fastapi",
    "kubernetes",
    "modal",
    "runloop-api-client",
    "supabase",
    "typer",
    "uvicorn",
)
# The files the runner imports, uploaded beside it for each session.
_HARNESS_FILES = ("sandbox_runner.py", "terminus.py", "llm.py")
# Harbor's tmux launcher and its command runner need bash, and the launcher runs util-linux `script` to give tmux a
# terminal.
_TERMINAL_PROBE = (
    "command -v bash >/dev/null || { echo 'missing bash'; exit 3; }; "
    "command -v script >/dev/null || { echo 'missing script (util-linux)'; exit 3; }; tmux -V"
)


def _stopped_without_output() -> dict[str, JsonValue]:
    """The result of a run its deadline or a rollout stop ended before the runner saved anything; graded as such."""
    return {
        "stop_reason": "wall_time",
        "trajectory": {"steps": []},
        "finished_naturally": False,
        "response_ids": [],
        "calls_without_usage": 0,
        "compactions": [],
    }


def _sandbox_harbor_requirement() -> str:
    """Return the requirement the sandbox runtime is built from: the Harbor installed with this server.

    A git install is fetched as a GitHub archive, so building it does not need git.
    """
    distribution = importlib.metadata.distribution("harbor")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    if commit is None:
        return f"harbor=={distribution.version}"
    url = str(direct_url.get("url") or "").removesuffix(".git")
    if not url.startswith("https://github.com/"):
        raise RuntimeError(f"Cannot build a sandbox install URL for harbor installed from {url!r}")
    return f"harbor @ {url}/archive/{commit}.tar.gz"


@functools.cache
def terminus_runtime() -> SandboxPythonRuntime:
    """The private Python runtime with this server's Harbor, built on first use so the host path never needs it."""
    return SandboxPythonRuntime(
        harness="terminus-2",
        requirements=(_sandbox_harbor_requirement(),),
        import_check=(
            "import os; os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'; import httpx, tenacity; "
            "from harbor.agents.terminus_2.terminus_2 import Terminus2"
        ),
        overrides=tuple(f"{name}; sys_platform == 'never'" for name in _UNUSED_HARBOR_DEPENDENCIES),
    )


@dataclass
class Terminus2SandboxSession(AgentSessionState):
    """Terminus-2 transport and cleanup; HTTP retry bookkeeping stays in the agent base."""

    session: SandboxSession[dict[str, JsonValue]]
    sandbox_id: str | None = None
    provider: str | None = None
    # The activation's invocation; the runner sends it as x-session-id, which model-call capture joins on.
    invocation_id: str | None = None
    observations: AgentObservationBundle | None = None
    provider_wait_failure: NgFailureEvent | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    installation_task: asyncio.Task[None] | None = None

    @property
    def harness_dir(self) -> str:
        return f"{self.session.session_dir}/harness"

    async def install_runtime(self, *, install_timeout: float) -> None:
        """Prepare the sandbox once; a seed caller that disconnects leaves it running for close to join."""
        if self.installation_task is None:
            self.installation_task = asyncio.create_task(self._install_runtime(install_timeout=install_timeout))
        await asyncio.shield(self.installation_task)

    async def _install_runtime(self, *, install_timeout: float) -> None:
        """Reuse or install the pinned runtime, stage the harness files and make sure tmux runs.

        The runtime directory is keyed by its inputs, so a runtime an earlier session left in this sandbox is reused
        when it passes the import check. The harness files go into this session's directory, so concurrent sessions
        never rewrite each other's copy.
        """
        sandbox = self.session.sandbox
        runtime = terminus_runtime()
        directories = [self.harness_dir, f"{self.session.session_dir}/logs"]
        if self.session.owns_sandbox:
            # Providers need not create the workdir; a borrowed sandbox's task directory belongs to Resources.
            directories.append(self.session.workdir)
        prepared = await sandbox.exec(f"mkdir -p {' '.join(map(quote, directories))}", cwd="/", timeout_s=30)
        if prepared.return_code != 0 or prepared.error_type is not None:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Terminus 2 sandbox paths")
        if not await runtime.is_installed(sandbox):
            await runtime.install(sandbox, timeout_s=install_timeout)
        for name in _HARNESS_FILES:
            await sandbox.upload(Path(__file__).with_name(name), f"{self.harness_dir}/{name}")

        probe = await sandbox.exec(_TERMINAL_PROBE, cwd="/", timeout_s=30)
        if probe.return_code == 3:
            raise AgentSeedRejectedError(f"Terminus 2 cannot run in this sandbox: {(probe.stdout or '').strip()}")
        if probe.return_code == 0 and probe.error_type is None:
            return
        # The same installer Terminus runs at its first command, moved to setup. It uses the image's package
        # manager, so a sandbox without tmux needs network access to its package index.
        installed = await sandbox.exec(
            f"{quote(runtime.python)} -E -s {quote(self.harness_dir + '/sandbox_runner.py')} --ensure-tmux",
            cwd="/",
            timeout_s=install_timeout,
        )
        if installed.return_code != 0 or installed.error_type is not None:
            raise RuntimeError(
                f"Installing tmux for Terminus 2 failed with {installed.error_type or installed.return_code}: "
                f"{(installed.stderr or '')[-2000:]}{(installed.stdout or '')[-2000:]}"
            )

    async def close(self, timeout: float) -> None:
        """Finish installation, then sandbox capture and release, before cancelling the HTTP activation."""
        if self.installation_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.installation_task), timeout=timeout)
            except Exception:
                if not self.installation_task.done():
                    # Retain the state so a later close can retry once installation finishes.
                    raise
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done() and not self.task.cancelling():
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
                # The agent base replays the activation error independently of close.

    async def execute(
        self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> dict[str, JsonValue]:
        """Use the common lifecycle; the adapter reads the runner's result."""
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage this activation's input and describe the runner process."""
        directory = self.session.session_dir
        await upload_text(
            self.session.sandbox,
            path=f"{directory}/input.json",
            text=json.dumps(
                {
                    **payload,
                    "logs_dir": f"{directory}/logs",
                    "stop_request_path": self.session.stop_request_path,
                    "workdir": self.session.workdir,
                }
            ),
        )
        return SandboxCommand(
            argv=[
                terminus_runtime().python,
                # Ignore task Python settings without changing the environment inherited by tool subprocesses.
                "-E",
                "-s",
                f"{self.harness_dir}/sandbox_runner.py",
                f"{directory}/input.json",
                f"{directory}/output.json",
            ],
        )

    async def _output_absent(self) -> bool:
        """Whether the runner left no output.json, as opposed to a read that failed."""
        path = quote(f"{self.session.session_dir}/output.json")
        try:
            probe = await self.session.sandbox.exec(f"[ -e {path} ] || echo absent", cwd="/", timeout_s=10)
        except Exception:
            return False
        return probe.return_code == 0 and probe.error_type is None and (probe.stdout or "").strip() == "absent"

    async def collect_artifacts(self) -> dict[str, JsonValue]:
        """Copy the runner's result before close removes files, including an interrupted run's partial result."""
        path = f"{self.session.session_dir}/output.json"
        if self.session.expired_before_launch:
            # A rollout stop arrived before launch, so Terminus never ran: an empty run that ended at its limit.
            return _stopped_without_output()
        try:
            output = json.loads(await read_text(self.session.sandbox, path=path))
        except Exception as error:
            cleanup = self.session.cleanup
            if cleanup is not None and cleanup["timed_out"] and await self._output_absent():
                # The supervisor's deadline or a rollout stop ended the runner before it could save anything: it was
                # still importing Harbor, before its signal handler was installed.
                LOG.warning("Terminus 2 saved no result before its deadline or a rollout stop; reporting an empty run")
                return _stopped_without_output()
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Terminus 2 sandbox runner exited without output: {logs[-4000:]}") from error
        if not isinstance(output, dict):
            raise TypeError(f"Terminus 2 sandbox output at {path} is not an object")
        return output
