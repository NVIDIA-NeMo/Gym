# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of Stirrup execution in a borrowed task sandbox."""

import asyncio
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any

from nemo_gym.agent_utils.sandbox_python import SandboxPythonRuntime
from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.utils import read_text, upload_text
from nemo_gym.tool_access import DirectHTTPToolAccess


# The sandbox runs the Stirrup version installed with this server, with its dependencies at this server's versions.
# Those versions resolve only with the overrides this server's environment is installed with.
STIRRUP_REQUIREMENT = f"stirrup=={importlib.metadata.version('stirrup')}"
RUNTIME = SandboxPythonRuntime(
    harness="stirrup",
    requirements=(STIRRUP_REQUIREMENT,),
    import_check="import stirrup",
    overrides=tuple(
        line
        for line in Path(__file__).with_name("overrides.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ),
)
# Outside the working directory, so the model's files stay its own.
RUNTIME_DIR = RUNTIME.directory
SANDBOX_PYTHON = RUNTIME.python
_PACKAGE_DIR = f"{RUNTIME_DIR}/harness/responses_api_agents/stirrup_agent"
# The runner and the modules it imports; the rest of this package needs Gym, which the sandbox lacks.
HARNESS_FILES = ("nemo_agent.py", "nemo_client.py", "stirrup_utils.py", "sandbox_runner.py")


@dataclass
class StirrupSessionState(AgentSessionState):
    session: SandboxSession[dict[str, Any]]
    tool_access: DirectHTTPToolAccess | None
    resources_cookies: dict[str, str]
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = field(default=None)

    async def install_runtime(self, *, timeout: float) -> None:
        """Reuse or install the Stirrup runtime and stage the harness files."""
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(
            f"mkdir -p {quote(_PACKAGE_DIR)} {quote(self.session.session_dir)}", cwd="/", timeout_s=30
        )
        if prepared.return_code != 0:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Stirrup sandbox paths")
        if not await RUNTIME.is_installed(sandbox):
            await RUNTIME.install(sandbox, timeout_s=timeout)
        for name in HARNESS_FILES:
            await sandbox.upload(Path(__file__).with_name(name), f"{_PACKAGE_DIR}/{name}")
        for package in (_PACKAGE_DIR, str(Path(_PACKAGE_DIR).parent)):
            await upload_text(sandbox, path=f"{package}/__init__.py", text="")

    async def execute(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Run the harness once and return its output, raising the runner's error."""
        input_path = f"{self.session.session_dir}/input.json"
        output_path = f"{self.session.session_dir}/output.json"

        async def stage() -> SandboxCommand:
            await upload_text(self.session.sandbox, path=input_path, text=json.dumps(payload))
            # -s keeps the task's user site-packages out of the harness, as its virtual environment did.
            return SandboxCommand(
                argv=[SANDBOX_PYTHON, "-s", f"{_PACKAGE_DIR}/sandbox_runner.py", input_path, output_path],
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
