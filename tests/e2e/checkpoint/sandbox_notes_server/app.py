# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e suite whose sessions each own a sandbox.

The ``append_note`` tool appends a line to a file inside the session's sandbox, and verification reads the
file back and rewards 1.0 when it holds exactly the expected lines. The sandboxes live in the suite's fake
sandbox backend, which outlives a Gym crash, and the server owns them through ``SandboxOwnerResourcesServer``,
as the SWE-bench server does, so the suite's process-level scenarios exercise that base.

An agent that runs its own harness in the session's sandbox gets the sandbox's ID in the seed reply, as from the
SWE-bench server, and asks ``/sandbox_access`` for the current one after a restore, which forks the sandbox
under a new ID.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel


sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_sandbox_provider  # noqa: E402, F401  (registers the fake_remote provider)

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.sandbox.owner import SandboxOwnerConfig, SandboxOwnerResourcesServer  # noqa: E402
from nemo_gym.sandbox.providers.base import SandboxSpec  # noqa: E402
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402


class SandboxNotesConfig(SandboxOwnerConfig):
    # ``sandbox_provider`` names the top-level sandbox block the provider is resolved from, as a production
    # server's does. The suite's block selects ``fake_remote`` and, for the lease scenario, its lease.
    pass


class AppendNoteRequest(BaseModel):
    line: str


class AppendNoteResponse(BaseModel):
    success: bool


class SandboxNotesVerifyRequest(BaseVerifyRequest):
    expected_notes: list[str]


class SandboxNotesSeedSessionResponse(BaseSeedSessionResponse):
    # For an agent that runs its harness inside the session's sandbox, as the SWE-bench server's agent does.
    sandbox_handle: str


class SandboxNotesResourcesServer(SandboxOwnerResourcesServer):
    ray_enabled = False
    config: SandboxNotesConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/append_note")(self.append_note)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> SandboxNotesSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox = self.session_sandbox(session_id)
        if sandbox is None:
            sandbox = await self.start_session_sandbox(
                session_id, SandboxSpec(image="notes:1", workdir="/work"), workdir="/work"
            )
        return SandboxNotesSeedSessionResponse(sandbox_handle=sandbox.handle.sandbox_id)

    async def append_note(self, request: Request, body: AppendNoteRequest) -> AppendNoteResponse:
        sandbox = self.session_sandbox(request.session[SESSION_ID_KEY])
        if sandbox is None:
            return AppendNoteResponse(success=False)
        result = await sandbox.exec(f"append notes {body.line}")
        return AppendNoteResponse(success=result.return_code == 0)

    async def verify(self, request: Request, body: SandboxNotesVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        reward = 0.0
        sandbox = self.session_sandbox(session_id)
        if sandbox is not None:
            result = await sandbox.exec("read notes")
            reward = float((result.stdout or "").splitlines() == body.expected_notes)
            # The episode is over: free its sandbox. A checkpoint taken before may still name its snapshots,
            # so only a release deletes them.
            await self.end_session_sandbox(session_id)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    SandboxNotesResourcesServer.run_webserver()
