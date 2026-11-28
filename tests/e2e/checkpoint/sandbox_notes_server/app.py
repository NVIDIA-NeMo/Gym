# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e suite whose sessions each own a task sandbox, with sandbox checkpoints.

The ``append_note`` tool appends a line to a file inside the session's task sandbox, and verification captures the
file once and rewards 1.0 when it holds exactly the expected lines. The sandboxes live in the suite's fake
sandbox backend, which outlives a Gym crash. The agent calls the tool over direct HTTP, so this server runs every
command itself and lends the agent no sandbox.
"""

import sys
from pathlib import Path
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict


sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_sandbox_provider  # noqa: E402, F401  (registers the fake_remote providers)

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
    ResourcesSessionConflictError,
    lost_session_verification,
)
from resources_servers.swebench.sandbox_session_checkpoints import (  # noqa: E402
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)
from resources_servers.swebench.sandbox_sessions import _TaskSession  # noqa: E402


class SandboxNotesConfig(SandboxCheckpointFields, BaseResourcesServerConfig):
    sandbox_provider: str
    sandbox_config: dict[str, Any] = {}


class NotesTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_notes: list[str]


class AppendNoteRequest(BaseModel):
    line: str


class AppendNoteResponse(BaseModel):
    success: bool


class SandboxNotesVerifyRequest(BaseVerifyRequest):
    expected_notes: list[str]


class SandboxNotesResourcesServer(CheckpointedSandboxSessionResourcesServer):
    ray_enabled = False
    config: SandboxNotesConfig
    task_model: ClassVar[type[BaseModel]] = NotesTask
    # Verification grades the notes it captured once, so a checkpoint may replay it.
    checkpoint_verify: ClassVar[StepType] = "replay"

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/append_note")(self.append_note)
        return app

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        return await self.seed_task_sandbox_session(request, body, NotesTask)

    async def _start_task_sandbox(self, session_id: str, task: BaseModel) -> str:
        spec = SandboxSpec(image="notes:1", workdir="/work")

        async def setup(sandbox: AsyncSandbox) -> None:
            result = await sandbox.exec("append seeded yes")
            if result.return_code != 0:
                raise RuntimeError(f"setup failed: {result.stderr}")

        await self.new_task_sandbox(session_id, spec).start_with_setup(spec, setup)
        return "/work"

    async def _task_sandbox_response(self, session_id: str, record: _TaskSession) -> ResourcesSeedSessionResponse:
        # The agent reaches the sandbox only through this server's tool.
        if session_id not in self._session_id_to_sandbox:
            raise ResourcesSessionConflictError(f"The task sandbox of {session_id} is no longer available")
        return ResourcesSeedSessionResponse(resources_session_id=session_id)

    async def append_note(self, request: Request, body: AppendNoteRequest) -> AppendNoteResponse:
        sandbox = self._session_id_to_sandbox.get(request.session[SESSION_ID_KEY])
        if sandbox is None:
            return AppendNoteResponse(success=False)
        result = await sandbox.exec(f"append notes {body.line}")
        return AppendNoteResponse(success=result.return_code == 0)

    async def verify(self, request: Request, body: SandboxNotesVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        async def grade() -> BaseVerifyResponse:
            notes = await self.capture_task_sandbox_patch(session_id, lambda: self._read_notes(session_id))
            await self.stop_task_sandbox_after_capture(session_id)
            return BaseVerifyResponse(**body.model_dump(), reward=float(notes == body.expected_notes))

        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=grade,
            lost_response=lambda reason, failure_kind: BaseVerifyResponse.model_validate(
                body.model_dump() | lost_session_verification(reason, failure_kind=failure_kind)
            ),
        )

    async def _read_notes(self, session_id: str) -> list[str]:
        result = await self._session_id_to_sandbox[session_id].exec("read notes")
        return (result.stdout or "").splitlines()


if __name__ == "__main__":
    SandboxNotesResourcesServer.run_webserver()
