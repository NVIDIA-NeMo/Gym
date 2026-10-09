# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server for the checkpoint e2e suite whose sessions each own a sandbox.

The ``append_note`` tool appends a line to a file inside the session's sandbox, and verification reads the
file back and rewards 1.0 when it holds exactly the expected lines. The sandboxes live in the suite's fake
sandbox backend, which outlives a Gym crash, and are checkpointed through ``SandboxSessionCheckpointer``.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel, JsonValue, PrivateAttr


sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_sandbox_provider import RemoteFakeSandboxProvider  # noqa: E402  (registers the provider)

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox.access import SandboxAccess  # noqa: E402
from nemo_gym.sandbox.checkpoint import SandboxSessionCheckpointer  # noqa: E402
from nemo_gym.sandbox.providers.base import SandboxSpec  # noqa: E402
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402


class SandboxNotesConfig(BaseResourcesServerConfig):
    sandbox_backend_url: str
    # The top-level sandbox block a borrower resolves the handed-out access through.
    sandbox_provider_ref: str = "sandbox"


class AppendNoteRequest(BaseModel):
    line: str


class AppendNoteResponse(BaseModel):
    success: bool


class SandboxNotesVerifyRequest(BaseVerifyRequest):
    expected_notes: list[str]


class SandboxNotesSeedSessionResponse(BaseSeedSessionResponse):
    # For a legacy /run agent that runs its harness inside the session's sandbox, as the SWE-bench server's does.
    sandbox_handle: str


class SandboxNotesResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    checkpoint_mode = "exported"
    config: SandboxNotesConfig
    _sandboxes: SandboxSessionCheckpointer = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        provider = RemoteFakeSandboxProvider(self.config.sandbox_backend_url)
        self._sandboxes = SandboxSessionCheckpointer(provider, parallelism=8)
        app = super().setup_webserver()
        app.post("/append_note")(self.append_note)
        return app

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | BaseSeedSessionRequest
    ) -> ResourcesSeedSessionResponse | SandboxNotesSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._sandboxes:
            await self._sandboxes.create(session_id, SandboxSpec(image="notes:1", workdir="/work"))
        if isinstance(body, ResourcesSeedSessionRequest):
            # An Environment Server episode: the agent may run tools in this sandbox itself.
            return ResourcesSeedSessionResponse(
                resources_session_id=body.resources_session_id,
                sandbox_access=await self.current_sandbox_access(session_id),
            )
        sandbox = await self._sandboxes.ensure_running(session_id)
        return SandboxNotesSeedSessionResponse(sandbox_handle=sandbox.handle.sandbox_id)

    async def current_sandbox_access(self, session_id: str) -> SandboxAccess | None:
        if session_id not in self._sandboxes:
            return None
        return await self._sandboxes.access(
            session_id, provider_config_ref=self.config.sandbox_provider_ref, workdir="/work"
        )

    async def append_note(self, request: Request, body: AppendNoteRequest) -> AppendNoteResponse:
        sandbox = await self._sandboxes.ensure_running(request.session[SESSION_ID_KEY])
        result = await sandbox.exec(f"append notes {body.line}")
        return AppendNoteResponse(success=result.return_code == 0)

    async def verify(self, request: Request, body: SandboxNotesVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        reward = 0.0
        if session_id in self._sandboxes:
            sandbox = await self._sandboxes.ensure_running(session_id)
            result = await sandbox.exec("read notes")
            reward = float((result.stdout or "").splitlines() == body.expected_notes)
            # The episode is over: free its sandbox. Its snapshots stay for the checkpoints that name them.
            await self._sandboxes.stop(session_id)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return await self._sandboxes.export(session_ids)

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        await self._sandboxes.restore(states)

    async def retire_session_state(self, session_id: str) -> None:
        await self._sandboxes.stop(session_id)

    async def resume_session_states(self, session_ids: list[str]) -> None:
        await self._sandboxes.resume_paused(session_ids)


if __name__ == "__main__":
    SandboxNotesResourcesServer.run_webserver()
