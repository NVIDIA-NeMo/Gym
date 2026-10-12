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

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.utils import read_text, upload_text


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
_SANDBOX_OBSERVER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_observer.py"
_SANDBOX_MODEL_KWARGS = f"{_SANDBOX_RUNTIME_DIR}/model_kwargs.py"


class HarnessProcessInfo(BaseModel):
    """Optional identity of the Hermes harness process, separate from cleanup evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read Hermes diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Hermes runtime metadata is missing or malformed")
        return None


@dataclass
class HermesSandboxSession(AgentSessionState):
    """Hermes transport and cleanup; HTTP retry bookkeeping stays in the agent base."""

    session: SandboxSession[dict[str, JsonValue]]
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None

    async def install_runtime(self, *, install_timeout: float) -> None:
        """Reuse or install the pinned runtime and stage the Hermes harness files."""
        prepared = await self.session.sandbox.exec(
            f"mkdir -p {quote(_SANDBOX_RUNTIME_DIR)} {quote(self.session.session_dir)}",
            cwd=self.session.workdir,
            timeout_s=30,
        )
        if prepared.return_code != 0:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Hermes sandbox paths")
        if not await self._runtime_installed():
            await self._install_runtime(install_timeout)
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), _SANDBOX_RUNNER)
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_observer.py"), _SANDBOX_OBSERVER)
        await self.session.sandbox.upload(Path(__file__).with_name("model_kwargs.py"), _SANDBOX_MODEL_KWARGS)

    async def _runtime_installed(self) -> bool:
        """Whether the pinned Hermes and its MCP client import from its runtime path.

        The path is keyed by the pinned commit, so a runtime baked into the image or left by an earlier
        session in this sandbox is reused.
        """
        check = await self.session.sandbox.exec(
            f"{quote(_SANDBOX_PYTHON)} -c 'import run_agent, mcp'",
            cwd=self.session.workdir,
            timeout_s=120,
        )
        return check.return_code == 0

    async def _install_runtime(self, timeout: float) -> None:
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise RuntimeError("Hermes agent server requires uv to install the sandbox runtime")
        await self.session.sandbox.upload(uv_path, _SANDBOX_UV)
        venv = quote(_SANDBOX_RUNTIME_DIR + "/venv")
        # A runtime that failed the import check is incomplete, so rebuild it rather than reuse it.
        install = await self.session.sandbox.exec(
            f"chmod 755 {quote(_SANDBOX_UV)} && rm -rf {venv} && "
            f"{quote(_SANDBOX_UV)} venv {venv} --python 3.13 && "
            f"{quote(_SANDBOX_UV)} pip install --python {quote(_SANDBOX_PYTHON)} {quote(_HERMES_REQUIREMENT)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if install.return_code != 0 or not await self._runtime_installed():
            raise RuntimeError(install.stderr or install.stdout or "Hermes sandbox installation failed")

    async def upload_json(self, name: str, payload: dict[str, JsonValue]) -> None:
        """Write a Hermes input under the session directory using file transfer."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_json(self, name: str) -> dict[str, JsonValue]:
        """Read a Hermes output object without mixing it with the cleanup contract."""
        path = f"{self.session.session_dir}/{name}"
        payload = json.loads(await read_text(self.session.sandbox, path=path))
        if not isinstance(payload, dict):
            raise TypeError(f"Hermes sandbox payload at {path} is not an object")
        return payload

    async def close(self, timeout: float) -> None:
        """Finish sandbox capture/release before cancelling the HTTP activation."""
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
        """Use the common lifecycle, then check the Hermes-specific result."""
        output = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if output.get("error") is not None:
            raise RuntimeError(f"Hermes sandbox runner failed: {output['error']}\n{output.get('traceback', '')}")
        return output

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage this activation's input and describe its harness process."""
        await self.upload_json("input.json", {**payload, "stop_request_path": self.session.stop_request_path})
        return SandboxCommand(
            argv=[
                _SANDBOX_PYTHON,
                _SANDBOX_RUNNER,
                f"{self.session.session_dir}/input.json",
                f"{self.session.session_dir}/output.json",
            ],
            python=_SANDBOX_PYTHON,
        )

    async def collect_artifacts(self) -> dict[str, JsonValue]:
        """Copy Hermes output before close removes files, including interrupted output."""
        try:
            output = await self.read_json("output.json")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Hermes sandbox runner exited without output: {logs}") from error
        self.runtime_info = parse_runtime_info(output.get("runtime"))
        return output
