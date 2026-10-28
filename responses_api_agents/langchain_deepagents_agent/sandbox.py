# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepAgents runtime staging and artifacts over Gym's shared sandbox lifecycle."""

import asyncio
import hashlib
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agent_utils.uv_bootstrap import uv_for_sandbox
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)
REQUIREMENTS = tuple(
    line
    for line in Path(__file__).with_name("runtime_requirements.txt").read_text().splitlines()
    if line and not line.startswith("#")
)
RUNTIME_KEY = hashlib.sha256(" ".join(REQUIREMENTS).encode()).hexdigest()[:16]
RUNTIME_DIR = f"/tmp/nemo-gym-deepagents-runtime-{RUNTIME_KEY}"
RUNTIME_PYTHON = f"{RUNTIME_DIR}/venv/bin/python"
RUNTIME_EXECUTABLE = f"{RUNTIME_DIR}/venv/bin/deepagents-runtime"
RUNNER_NAME = "sandbox_runner"
RUNTIME_FILES = ("sandbox_runner.py", "runtime_model.py", "runtime_messages.py", "runtime_observer.py")


@dataclass
class DeepAgentsSandboxSession(AgentSessionState):
    """One activation, with retained observations and cleanup owned by SandboxSession."""

    session: SandboxSession[dict[str, Any]]
    sandbox_id: str | None
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None

    @property
    def package_dir(self) -> str:
        return f"{self.session.session_dir}/harness/responses_api_agents/langchain_deepagents_agent"

    async def install_runtime(self, *, timeout: float) -> None:
        """Install the pinned SDK in the actual sandbox and stage this adapter's source."""
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(
            f"mkdir -p {shlex.quote(RUNTIME_DIR)} {shlex.quote(self.package_dir)} && "
            f"chmod 700 {shlex.quote(self.session.session_dir)}",
            timeout_s=30,
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(f"DeepAgents runtime directory setup failed: {prepared.stderr or prepared.stdout}")
        versions = {name: version for name, version in (r.split("==") for r in REQUIREMENTS if "==" in r)}
        check_code = (
            "import deepagents, aiohttp; from importlib.metadata import version; "
            f"assert all(version(k) == v for k, v in {versions!r}.items())"
        )
        check = f"{shlex.quote(RUNTIME_PYTHON)} -c {shlex.quote(check_code)}"
        ready = await sandbox.exec(check, timeout_s=60)
        if ready.return_code != 0 or ready.error_type:
            uv = f"{self.session.session_dir}/uv"
            await sandbox.upload(await uv_for_sandbox(sandbox, self.session.workdir), uv)
            # A version-keyed runtime is reusable; uv serializes writes to its environment and cache.
            command = (
                f"chmod 755 {shlex.quote(uv)} && "
                f"UV_PYTHON_INSTALL_DIR={shlex.quote(RUNTIME_DIR + '/python')} "
                f"{shlex.quote(uv)} venv --allow-existing {shlex.quote(RUNTIME_DIR + '/venv')} --python 3.13 && "
                f"{shlex.quote(uv)} pip install --python {shlex.quote(RUNTIME_PYTHON)} "
                + " ".join(shlex.quote(r) for r in REQUIREMENTS)
            )
            installed = await sandbox.exec(command, timeout_s=timeout)
            if installed.return_code != 0 or installed.error_type:
                raise RuntimeError(f"DeepAgents runtime installation failed: {installed.stderr or installed.stdout}")
            ready = await sandbox.exec(check, timeout_s=60)
            if ready.return_code != 0 or ready.error_type:
                raise RuntimeError(f"DeepAgents runtime import check failed: {ready.stderr or ready.stdout}")
        # Keep the venv layout (and SDK imports), but avoid matching task cleanup commands
        # such as pkill python3 / pkill -f '\.py'. This is not a stdlib-only runner.
        neutral = await sandbox.exec(
            f"cp -L {shlex.quote(RUNTIME_PYTHON)} {shlex.quote(RUNTIME_EXECUTABLE)} && "
            f"chmod 755 {shlex.quote(RUNTIME_EXECUTABLE)}",
            timeout_s=30,
        )
        if neutral.return_code != 0 or neutral.error_type:
            raise RuntimeError("Could not prepare the DeepAgents runtime executable")
        for name in RUNTIME_FILES:
            target = RUNNER_NAME if name == "sandbox_runner.py" else name
            await sandbox.upload(Path(__file__).with_name(name), f"{self.package_dir}/{target}")
        for package in (self.package_dir, str(Path(self.package_dir).parent)):
            await upload_text(sandbox, path=f"{package}/__init__.py", text="")

    async def execute(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Stage input, run under the supervisor, then collect only after termination."""

        async def stage() -> SandboxCommand:
            await upload_text(
                self.session.sandbox, path=f"{self.session.session_dir}/input.json", text=json.dumps(payload)
            )
            return SandboxCommand(
                argv=[
                    RUNTIME_EXECUTABLE,
                    "-I",
                    f"{self.package_dir}/{RUNNER_NAME}",
                    f"{self.session.session_dir}/input.json",
                    f"{self.session.session_dir}/output.json",
                ],
            )

        async def collect() -> dict[str, Any]:
            try:
                result = json.loads(
                    await read_text(self.session.sandbox, path=f"{self.session.session_dir}/output.json")
                )
            except Exception as error:
                raise RuntimeError(
                    f"DeepAgents returned no valid artifact: {await self.session.read_output_log()}"
                ) from error
            LOG.info("DeepAgents sandbox runtime: %s", result.get("runtime"))
            return result

        return await self.session.execute(
            stage_activation=stage, collect=collect, timeout=timeout, close_timeout=close_timeout
        )

    async def close(self, *, timeout: float) -> None:
        """Confirm cleanup and release resources before cancelling an activation waiter."""
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
