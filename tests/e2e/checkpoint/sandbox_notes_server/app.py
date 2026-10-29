# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e suite whose sessions each own a sandbox, on the sandbox owner base.

The ``append_note`` tool appends a line to a file inside the session's sandbox, and verification captures the
file once and rewards 1.0 when it holds exactly the expected lines. The sandboxes live in the suite's fake
sandbox backend, which outlives a Gym crash. The agent calls the tool over direct HTTP, so this server runs every
command itself and lends the agent no sandbox.
"""

import sys
from pathlib import Path
from typing import ClassVar

from fastapi import FastAPI, Request
from pydantic import BaseModel


sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_sandbox_provider  # noqa: E402, F401  (registers the fake_remote providers)

from nemo_gym._checkpoint.steps import StepType  # noqa: E402
from nemo_gym.base_resources_server import (  # noqa: E402
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec  # noqa: E402
from nemo_gym.sandbox.access import SandboxAccess  # noqa: E402
from nemo_gym.sandbox_owner import SandboxOwnerConfig, SandboxOwnerResourcesServer, SessionSetup  # noqa: E402
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402


class AppendNoteRequest(BaseModel):
    line: str


class AppendNoteResponse(BaseModel):
    success: bool


class SandboxNotesVerifyRequest(BaseVerifyRequest):
    expected_notes: list[str]


class SandboxNotesResourcesServer(SandboxOwnerResourcesServer):
    ray_enabled = False
    config: SandboxOwnerConfig
    # Verification grades the notes it captured once, so a checkpoint may replay it.
    checkpoint_verify: ClassVar[StepType] = "replay"

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/append_note")(self.append_note)
        return app

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> object:
        return await self.seed_task_sandbox_session(request, body, self._set_up)

    async def _set_up(self, session_id: str, body: ResourcesSeedSessionRequest) -> None:
        async def setup(sandbox: AsyncSandbox) -> SessionSetup:
            result = await sandbox.exec("append seeded yes")
            if result.return_code != 0:
                raise RuntimeError(f"setup failed: {result.stderr}")
            return SessionSetup(workdir="/work", data={"expected": body.task_data.get("expected_notes")})

        await self.start_session_sandbox(session_id, SandboxSpec(image="notes:1", workdir="/work"), setup)

    async def current_sandbox_access(self, session_id: str) -> SandboxAccess | None:
        # The agent reaches the sandbox only through this server's tool.
        return None

    async def append_note(self, request: Request, body: AppendNoteRequest) -> AppendNoteResponse:
        sandbox = self.session_sandbox(request.session[SESSION_ID_KEY])
        if sandbox is None:
            return AppendNoteResponse(success=False)
        result = await sandbox.exec(f"append notes {body.line}")
        return AppendNoteResponse(success=result.return_code == 0)

    async def verify(self, request: Request, body: SandboxNotesVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        async def read_notes() -> list[str]:
            result = await self.require_session_sandbox(session_id).exec("read notes")
            return (result.stdout or "").splitlines()

        notes = await self.capture_once(session_id, "notes", read_notes, release_sandbox_after=True)
        return BaseVerifyResponse(**body.model_dump(), reward=float(notes == body.expected_notes))


if __name__ == "__main__":
    SandboxNotesResourcesServer.run_webserver()
