# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of Stirrup execution in a borrowed task sandbox."""

import asyncio
import hashlib
import importlib.metadata
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.utils import read_text, upload_text
from nemo_gym.tool_access import DirectHTTPToolAccess


# The sandbox runs the Python and Stirrup versions installed with this server.
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
# NOTE(martas): transformers was removed from deps because it was a dead dependency
# it's only used when stirrup gets model id (it's use to count tokens)
STIRRUP_REQUIREMENT = f"stirrup=={importlib.metadata.version('stirrup')}"
_RUNTIME_KEY = hashlib.sha256(f"{PYTHON_VERSION} {STIRRUP_REQUIREMENT}".encode()).hexdigest()[:12]
# Outside the working directory, so the model's files stay its own.
RUNTIME_DIR = f"/tmp/nemo-gym-stirrup-runtime-{_RUNTIME_KEY}"
SANDBOX_PYTHON = f"{RUNTIME_DIR}/venv/bin/python"
_PACKAGE_DIR = f"{RUNTIME_DIR}/harness/responses_api_agents/stirrup_agent"
# The runner and the modules it imports; the rest of this package needs Gym, which the sandbox lacks.
HARNESS_FILES = ("nemo_agent.py", "nemo_client.py", "stirrup_utils.py", "sandbox_runner.py")


@dataclass
class StirrupSessionState(AgentSessionState):
    session: SandboxSession[dict[str, Any]]
    tool_access: DirectHTTPToolAccess
    resources_cookies: dict[str, str]
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = field(default=None)

    async def install_runtime(self, *, timeout: float) -> None:
        """Reuse or install the Stirrup runtime and stage the harness files."""
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(
            f"mkdir -p {quote(_PACKAGE_DIR)} {quote(self.session.session_dir)}", timeout_s=30
        )
        if prepared.return_code != 0:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Stirrup sandbox paths")
        if not await self._runtime_installed():
            await self._install_runtime(timeout)
        for name in HARNESS_FILES:
            await sandbox.upload(Path(__file__).with_name(name), f"{_PACKAGE_DIR}/{name}")
        for package in (_PACKAGE_DIR, str(Path(_PACKAGE_DIR).parent)):
            await upload_text(sandbox, path=f"{package}/__init__.py", text="")

    async def _runtime_installed(self) -> bool:
        check = await self.session.sandbox.exec(f"{quote(SANDBOX_PYTHON)} -c 'import stirrup'", timeout_s=120)
        return check.return_code == 0

    async def _install_runtime(self, timeout: float) -> None:
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise RuntimeError("Stirrup agent server requires uv to install the sandbox runtime")
        uv = f"{RUNTIME_DIR}/uv"
        await self.session.sandbox.upload(uv_path, uv)
        venv = quote(f"{RUNTIME_DIR}/venv")
        # uv downloads the server's Python version if the image lacks it.
        install = await self.session.sandbox.exec(
            f"export UV_CACHE_DIR={quote(f'{RUNTIME_DIR}/uv-cache')} "
            f"UV_PYTHON_INSTALL_DIR={quote(f'{RUNTIME_DIR}/python')} && "
            f"chmod 755 {quote(uv)} && rm -rf {venv} && "
            f"{quote(uv)} venv {venv} --python {PYTHON_VERSION} && "
            f"{quote(uv)} pip install --python {quote(SANDBOX_PYTHON)} {quote(STIRRUP_REQUIREMENT)}",
            timeout_s=timeout,
        )
        if install.return_code != 0 or not await self._runtime_installed():
            raise RuntimeError(install.stderr or install.stdout or "Stirrup sandbox installation failed")

    async def execute(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Run the harness once and return its output, raising the runner's error."""
        input_path = f"{self.session.session_dir}/input.json"
        output_path = f"{self.session.session_dir}/output.json"

        async def stage() -> SandboxCommand:
            await upload_text(self.session.sandbox, path=input_path, text=json.dumps(payload))
            return SandboxCommand(
                argv=[SANDBOX_PYTHON, f"{_PACKAGE_DIR}/sandbox_runner.py", input_path, output_path],
                python=SANDBOX_PYTHON,
            )

        async def collect() -> dict[str, Any]:
            try:
                return json.loads(await read_text(self.session.sandbox, path=output_path))
            except Exception as error:
                logs = await self.session.read_output_log()
                raise RuntimeError(f"Stirrup sandbox runner exited without output: {logs}") from error

        output = await self.session.execute(
            stage_activation=stage, collect=collect, timeout=timeout, close_timeout=close_timeout
        )
        if output.get("error") is not None:
            raise RuntimeError(f"Stirrup sandbox runner failed: {output['error']}\n{output.get('traceback', '')}")
        self.resources_cookies = output["resources_cookies"]
        return output

    async def close(self, timeout: float) -> None:
        """Stop the harness and release the sandbox before cancelling the HTTP activation."""
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except (asyncio.CancelledError, Exception):
                if not self.task.done():
                    raise
