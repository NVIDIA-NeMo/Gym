# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e harness scenarios: each episode session lends the agent a task sandbox.

The agent's harness writes one line per step to ``notes`` in the sandbox's working directory. Verification captures
the file once, stops the sandbox, and rewards 1.0 when the file holds exactly the expected steps. ``verify_delay_s``
holds the grading after the capture, so a checkpoint can fall inside it.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, ClassVar

from fastapi import Request
from pydantic import BaseModel, ConfigDict


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_sandbox"))
import local_sandbox_provider  # noqa: E402, F401  (registers the local_dir providers)

from nemo_gym.client.server_client import SESSION_ID_KEY  # noqa: E402
from nemo_gym.sandbox.api import AsyncSandbox  # noqa: E402
from nemo_gym.sandbox.providers.base import SandboxSpec  # noqa: E402
from nemo_gym.server_runtime._checkpoint.steps import StepType  # noqa: E402
from nemo_gym.server_runtime.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    lost_session_verification,
)
from resources_servers.swebench.sandbox_session_checkpoints import (  # noqa: E402
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)


class WorkspaceOwnerConfig(SandboxCheckpointFields, BaseResourcesServerConfig):
    sandbox_provider: str
    sandbox_config: dict[str, Any] = {}
    verify_delay_s: float = 0.0


class WorkspaceTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    steps: int


class WorkspaceVerifyRequest(BaseVerifyRequest):
    steps: int


class WorkspaceOwnerResourcesServer(CheckpointedSandboxSessionResourcesServer):
    ray_enabled = False
    config: WorkspaceOwnerConfig
    task_model: ClassVar[type[BaseModel]] = WorkspaceTask
    # Verification grades the notes it captured once, so a checkpoint may replay it.
    checkpoint_verify: ClassVar[StepType] = "replay"

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        return await self.seed_task_sandbox_session(request, body, WorkspaceTask)

    async def _start_task_sandbox(self, session_id: str, task: BaseModel) -> str:
        spec = SandboxSpec(image="workspace:1", ttl_s=3600)
        workdir = ""

        async def setup(sandbox: AsyncSandbox) -> None:
            nonlocal workdir
            result = await sandbox.exec(": > notes && pwd")
            if result.return_code != 0:
                raise RuntimeError(f"setup failed: {result.stderr}")
            workdir = (result.stdout or "").strip()

        await self.new_task_sandbox(session_id, spec).start_with_setup(spec, setup)
        return workdir

    async def verify(self, request: Request, body: WorkspaceVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        async def grade() -> BaseVerifyResponse:
            notes = await self.capture_task_sandbox_patch(session_id, lambda: self._read_notes(session_id))
            await self.stop_task_sandbox_after_capture(session_id)
            await asyncio.sleep(self.config.verify_delay_s)
            expected = [f"step-{index}" for index in range(body.steps)]
            return BaseVerifyResponse(**body.model_dump(), reward=float(notes == expected))

        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=grade,
            lost_response=lambda reason, failure_kind: BaseVerifyResponse.model_validate(
                body.model_dump() | lost_session_verification(reason, failure_kind=failure_kind)
            ),
        )

    async def _read_notes(self, session_id: str) -> list[str]:
        workdir = self._task_sessions[session_id].workdir
        result = await self._session_id_to_sandbox[session_id].exec(f"cat {workdir}/notes")
        return (result.stdout or "").splitlines()


if __name__ == "__main__":
    WorkspaceOwnerResourcesServer.run_webserver()
