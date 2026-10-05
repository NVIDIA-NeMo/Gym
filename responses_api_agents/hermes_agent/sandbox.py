# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of Hermes execution in borrowed or owned task sandboxes."""

import asyncio
import importlib.metadata
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import JsonValue

from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, process_supervisor
from nemo_gym.sandbox.process_supervisor import CleanupReceipt
from nemo_gym.sandbox.supervisor_client import (
    HarnessProcessInfo,
    remove_session_directory,
    stop_and_confirm_cleanup,
    supervised_launch_command,
)


LOG = logging.getLogger(__name__)


def _sandbox_hermes_install() -> tuple[str, str]:
    """Return the requirement the sandbox installs and the key that names its runtime directory.

    Both come from the Hermes installed with this server, so ``requirements.txt`` is the only version pin and
    the sandbox runs the same Hermes as the host. A git install is fetched as a GitHub archive, so the sandbox
    does not need git. The ``mcp`` extra carries Hermes' MCP client, which episode tool grants use.
    """
    distribution = importlib.metadata.distribution("hermes-agent")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    if commit is None:
        return f"hermes-agent[mcp]=={distribution.version}", distribution.version
    url = str(direct_url.get("url") or "").removesuffix(".git")
    if not url.startswith("https://github.com/"):
        raise RuntimeError(f"Cannot build a sandbox install URL for hermes-agent installed from {url!r}")
    return f"hermes-agent[mcp] @ {url}/archive/{commit}.tar.gz", commit[:12]


_HERMES_REQUIREMENT, _HERMES_RUNTIME_KEY = _sandbox_hermes_install()
_SANDBOX_RUNTIME_DIR = f"/tmp/nemo-gym-hermes-runtime-{_HERMES_RUNTIME_KEY}"
_SANDBOX_UV = f"{_SANDBOX_RUNTIME_DIR}/uv"
_SANDBOX_PYTHON = f"{_SANDBOX_RUNTIME_DIR}/venv/bin/python"
_SANDBOX_RUNNER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_runner.py"
_SANDBOX_SUPERVISOR = f"{_SANDBOX_RUNTIME_DIR}/process_supervisor.py"
_SANDBOX_OBSERVER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_observer.py"
_SANDBOX_MODEL_KWARGS = f"{_SANDBOX_RUNTIME_DIR}/model_kwargs.py"


@dataclass
class HermesSandboxSession(AgentSessionState):
    """Hermes transport and cleanup; HTTP retry bookkeeping stays in the agent base."""

    sandbox: AsyncSandbox
    workdir: str | None
    directory: str
    owns_sandbox: bool = False
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    launch_started: bool = False
    cleanup: CleanupReceipt | None = None
    sandbox_stopped: bool = False
    closed: bool = False

    async def prepare(self, *, install_timeout: float) -> None:
        """Reuse or install the pinned runtime and stage the Hermes worker files."""
        prepared = await self.sandbox.exec(
            f"mkdir -p {quote(_SANDBOX_RUNTIME_DIR)} {quote(self.directory)}",
            cwd=self.workdir,
            timeout_s=30,
        )
        if prepared.return_code != 0:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Hermes sandbox paths")
        if not await self._runtime_installed():
            await self._install_runtime(install_timeout)
        await self.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), _SANDBOX_RUNNER)
        await self.sandbox.upload(Path(__file__).with_name("sandbox_observer.py"), _SANDBOX_OBSERVER)
        await self.sandbox.upload(Path(__file__).with_name("model_kwargs.py"), _SANDBOX_MODEL_KWARGS)
        await self.sandbox.upload(Path(process_supervisor.__file__), _SANDBOX_SUPERVISOR)

    async def _runtime_installed(self) -> bool:
        """Whether the pinned Hermes and its MCP client import from its runtime path.

        The path is keyed by the pinned commit, so a runtime baked into the image or left by an earlier
        session in this sandbox is reused.
        """
        check = await self.sandbox.exec(
            f"{quote(_SANDBOX_PYTHON)} -c 'import run_agent, mcp'",
            cwd=self.workdir,
            timeout_s=120,
        )
        return check.return_code == 0

    async def _install_runtime(self, timeout: float) -> None:
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise RuntimeError("Hermes agent server requires uv to install the sandbox runtime")
        await self.sandbox.upload(uv_path, _SANDBOX_UV)
        venv = quote(_SANDBOX_RUNTIME_DIR + "/venv")
        # A runtime that failed the import check is incomplete, so rebuild it rather than reuse it.
        install = await self.sandbox.exec(
            f"chmod 755 {quote(_SANDBOX_UV)} && rm -rf {venv} && "
            f"{quote(_SANDBOX_UV)} venv {venv} --python 3.13 && "
            f"{quote(_SANDBOX_UV)} pip install --python {quote(_SANDBOX_PYTHON)} {quote(_HERMES_REQUIREMENT)}",
            cwd=self.workdir,
            timeout_s=timeout,
        )
        if install.return_code != 0 or not await self._runtime_installed():
            raise RuntimeError(install.stderr or install.stdout or "Hermes sandbox installation failed")

    async def upload_json(self, name: str, payload: dict[str, JsonValue]) -> None:
        """Write a Hermes input under the session directory using file transfer."""
        await self.sandbox.upload_text(f"{self.directory}/{name}", text=json.dumps(payload))

    async def read_json(self, name: str) -> dict[str, JsonValue]:
        """Read a Hermes output object without mixing it with the cleanup contract."""
        path = f"{self.directory}/{name}"
        payload = json.loads(await self.sandbox.read_text(path))
        if not isinstance(payload, dict):
            raise TypeError(f"Hermes sandbox payload at {path} is not an object")
        return payload

    async def stop_runner(self, timeout: float) -> None:
        """Fence a delayed launch or require the supervisor's cleanup receipt."""
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
            return
        self.cleanup = await stop_and_confirm_cleanup(
            self.sandbox, directory=self.directory, workdir=self.workdir, timeout=timeout, harness="Hermes"
        )

    async def close(self, timeout: float) -> None:
        """Stop owned sandboxes; clean only harness work in borrowed sandboxes."""
        if self.closed:
            return
        if not self.owns_sandbox:
            # Provider cancellation may kill its process group, including the
            # supervisor. Require descendant cleanup before cancelling exec.
            await self.stop_runner(timeout)
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()
        if self.owns_sandbox and not self.sandbox_stopped:
            # Container teardown is the authority, even without a runner receipt.
            # Stop before waiting for an activation that might be stuck in cleanup.
            await self.sandbox.stop()
            self.sandbox_stopped = True
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
                # The activation error is replayed by the agent base. Cleanup
                # is independent and must still release the borrowed connection.
        if not self.owns_sandbox:
            await remove_session_directory(
                self.sandbox, directory=self.directory, workdir=self.workdir, timeout=timeout, harness="Hermes"
            )
            await self.sandbox.disconnect()
        self.closed = True

    async def execute(
        self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> dict[str, JsonValue]:
        """Run Hermes under supervision and read output only after confirmed cleanup."""
        await self.upload_json("input.json", payload)
        cleanup_timeout = close_timeout / 3
        command = supervised_launch_command(
            directory=self.directory,
            command=[
                _SANDBOX_PYTHON,
                _SANDBOX_RUNNER,
                f"{self.directory}/input.json",
                f"{self.directory}/output.json",
            ],
            timeout=timeout,
            cleanup_timeout=cleanup_timeout,
            python=_SANDBOX_PYTHON,
            supervisor_path=_SANDBOX_SUPERVISOR,
        )
        self.launch_started = True
        try:
            await self.sandbox.exec(
                command,
                cwd=self.workdir,
                timeout_s=process_supervisor.exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout),
            )
        except BaseException:
            try:
                await self.stop_runner(close_timeout)
            except Exception:
                LOG.exception("Hermes cleanup remains unconfirmed; close must retry before verification")
            raise
        else:
            await self.stop_runner(close_timeout)
        try:
            output = await self.read_json("output.json")
        except Exception as error:
            logs = await self.sandbox.exec(
                f"cat {quote(self.directory + '/runner.log')} 2>/dev/null || true",
                cwd=self.workdir,
                timeout_s=30,
            )
            raise RuntimeError(f"Hermes sandbox runner exited without output: {logs.stdout or ''}") from error
        if output.get("error") is not None:
            raise RuntimeError(f"Hermes sandbox runner failed: {output['error']}\n{output.get('traceback', '')}")
        try:
            self.runtime_info = HarnessProcessInfo.model_validate(output.get("runtime"))
        except ValueError as error:
            raise RuntimeError("Hermes sandbox runner returned invalid runtime metadata") from error
        return output
