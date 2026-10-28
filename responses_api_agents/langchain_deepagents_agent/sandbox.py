# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepAgents runtime staging and artifacts over Gym's shared sandbox lifecycle."""

import asyncio
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agent_utils.uv_bootstrap import uv_for_sandbox
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.utils import read_text, upload_text


REQUIREMENTS = (
    "deepagents==0.6.12",
    "langchain==1.4.7",
    "langchain-core==1.7.0",
    "langgraph==1.2.16",
    "aiohttp>=3.13,<4",
)
RUNTIME_KEY = hashlib.sha256(" ".join(REQUIREMENTS).encode()).hexdigest()[:16]
RUNTIME_DIR = f"/tmp/nemo-gym-deepagents-runtime-{RUNTIME_KEY}"
RUNTIME_PYTHON = f"{RUNTIME_DIR}/venv/bin/python"
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
            f"mkdir -p {shlex.quote(RUNTIME_DIR)} {shlex.quote(self.package_dir)}", timeout_s=30
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(f"DeepAgents runtime directory setup failed: {prepared.stderr or prepared.stdout}")
        check = f'{shlex.quote(RUNTIME_PYTHON)} -c \'import deepagents, aiohttp; from importlib.metadata import version; assert version("deepagents") == "0.6.12"\''
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
        for name in RUNTIME_FILES:
            await sandbox.upload(Path(__file__).with_name(name), f"{self.package_dir}/{name}")
        for package in (self.package_dir, str(Path(self.package_dir).parent)):
            await upload_text(sandbox, path=f"{package}/__init__.py", text="")

    async def execute(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Stage input, run under the supervisor, then collect only after termination."""

        async def stage() -> SandboxCommand:
            await upload_text(
                self.session.sandbox, path=f"{self.session.session_dir}/input.json", text=json.dumps(payload)
            )
            return SandboxCommand(
                python=RUNTIME_PYTHON,
                argv=[
                    RUNTIME_PYTHON,
                    "-I",
                    f"{self.package_dir}/sandbox_runner.py",
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
