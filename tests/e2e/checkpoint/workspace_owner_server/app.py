# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e harness scenarios: it owns each session's sandbox and lends it.

The agent's harness writes one line per step to ``notes`` in the sandbox's working directory. Verification captures
the file once (``capture_once``), releases the sandbox, and rewards 1.0 when the file holds exactly the expected
steps. ``verify_delay_s`` holds the grading after the capture, so a checkpoint can fall inside it.
"""

import asyncio
import sys
from pathlib import Path
from typing import ClassVar

from fastapi import Request


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_sandbox"))
import local_sandbox_provider  # noqa: E402, F401  (registers the local_dir providers)

from nemo_gym._checkpoint.steps import StepType  # noqa: E402
from nemo_gym.base_resources_server import (  # noqa: E402
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec  # noqa: E402
from nemo_gym.sandbox_owner import SandboxOwnerConfig, SandboxOwnerResourcesServer, SessionSetup  # noqa: E402
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402


class WorkspaceOwnerConfig(SandboxOwnerConfig):
    verify_delay_s: float = 0.0


class WorkspaceVerifyRequest(BaseVerifyRequest):
    steps: int


class WorkspaceOwnerResourcesServer(SandboxOwnerResourcesServer):
    ray_enabled = False
    config: WorkspaceOwnerConfig
    checkpoint_verify: ClassVar[StepType] = "replay"

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> object:
        return await self.seed_task_sandbox_session(request, body, self._set_up)

    async def _set_up(self, session_id: str, body: ResourcesSeedSessionRequest) -> None:
        async def setup(sandbox: AsyncSandbox) -> SessionSetup:
            result = await sandbox.exec(": > notes && pwd")
            if result.return_code != 0:
                raise RuntimeError(f"setup failed: {result.stderr}")
            return SessionSetup(workdir=(result.stdout or "").strip())

        await self.start_session_sandbox(session_id, SandboxSpec(image="workspace:1", ttl_s=3600), setup)

    async def verify(self, request: Request, body: WorkspaceVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        async def read_notes() -> list[str]:
            sandbox = self.require_session_sandbox(session_id)
            result = await sandbox.exec(f"cat {self.session_workdir(session_id)}/notes")
            return (result.stdout or "").splitlines()

        notes = await self.capture_once(session_id, "notes", read_notes, release_sandbox_after=True)
        await asyncio.sleep(self.config.verify_delay_s)
        expected = [f"step-{index}" for index in range(body.steps)]
        return BaseVerifyResponse(**body.model_dump(), reward=float(notes == expected))


if __name__ == "__main__":
    WorkspaceOwnerResourcesServer.run_webserver()
