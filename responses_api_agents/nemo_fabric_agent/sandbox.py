# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric invocation state over Gym's shared sandbox supervision and cleanup."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from nemo_gym.agents import HARNESS_RUNNER_FILE, AgentSessionState, SandboxCommand, SandboxSession, shell_assignments
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


@dataclass
class FabricSandboxSession(AgentSessionState):
    """Keep one idempotent activation, its artifacts, and its borrowed connection."""

    session: SandboxSession[dict[str, JsonValue]]
    python: str = ""
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None

    async def close(self, timeout: float) -> None:
        """Terminate and collect before releasing; then join response parsing."""
        await self.session.close(timeout=timeout)
        if self.task is not None:
            if self.session.artifacts is None and not self.task.done():
                self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage worker input and launch with the private Python, preserving task env."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/input.json", text=json.dumps(payload))
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), f"{directory}/fabric_runner")
        access = self.request.sandbox_access
        params = {
            "cwd": self.session.workdir,
            "command": [self.python, "-I", "-B", f"{directory}/fabric_runner", directory],
            "env": access.env if access else {},
            "stdin": "/dev/null",
            "stdout": f"{directory}/stdout.log",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect_artifacts(self) -> dict[str, JsonValue]:
        """Read the atomic result only after the supervisor confirms termination."""
        try:
            text = await read_text(self.session.sandbox, path=f"{self.session.session_dir}/result.json")
        except Exception as error:
            # A timeout can kill the worker before its final result. It is not a successful empty rollout.
            try:
                diagnostic = (await read_text(self.session.sandbox, path=f"{self.session.session_dir}/stderr.log"))[
                    -8000:
                ]
            except Exception:
                diagnostic = "stderr unavailable"
            raise RuntimeError(f"Fabric result unavailable in {self.session.session_dir}: {diagnostic}") from error
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Fabric worker result must be a JSON object")
        return value
