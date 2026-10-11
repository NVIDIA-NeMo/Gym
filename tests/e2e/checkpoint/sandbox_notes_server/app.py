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
import fake_sandbox_provider  # noqa: E402, F401  (registers the fake_remote provider)

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import create_provider  # noqa: E402
from nemo_gym.sandbox.checkpoint import SandboxSessionCheckpointer, checkpoint_state_dir  # noqa: E402
from nemo_gym.sandbox.config import resolve_provider_config  # noqa: E402
from nemo_gym.sandbox.providers.base import SandboxSpec  # noqa: E402
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402


class SandboxNotesConfig(BaseResourcesServerConfig):
    # The top-level sandbox block the provider is resolved from, as a production server's ``sandbox_provider``
    # names one. The suite's block selects ``fake_remote`` and, for the lease scenario, its lease.
    sandbox_provider: str


class AppendNoteRequest(BaseModel):
    line: str


class AppendNoteResponse(BaseModel):
    success: bool


class SandboxNotesVerifyRequest(BaseVerifyRequest):
    expected_notes: list[str]


class SandboxNotesResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    checkpoint_mode = "exported"
    config: SandboxNotesConfig
    _sandboxes: SandboxSessionCheckpointer = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        global_config = self.server_client.global_config_dict
        provider = create_provider(resolve_provider_config(self.config.sandbox_provider, global_config))
        state_dir = checkpoint_state_dir(global_config)
        self._sandboxes = SandboxSessionCheckpointer(
            provider, state_dir=state_dir, server_name=self.config.name, parallelism=8
        )
        app = super().setup_webserver()
        app.post("/append_note")(self.append_note)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self._sandboxes:
            await self._sandboxes.create(session_id, SandboxSpec(image="notes:1", workdir="/work"))
        return BaseSeedSessionResponse()

    async def append_note(self, request: Request, body: AppendNoteRequest) -> AppendNoteResponse:
        sandbox = self._sandboxes.get(request.session[SESSION_ID_KEY])
        if sandbox is None:
            return AppendNoteResponse(success=False)
        result = await sandbox.exec(f"append notes {body.line}")
        return AppendNoteResponse(success=result.return_code == 0)

    async def verify(self, request: Request, body: SandboxNotesVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        reward = 0.0
        if session_id in self._sandboxes:
            result = await self._sandboxes.get(session_id).exec("read notes")
            reward = float((result.stdout or "").splitlines() == body.expected_notes)
            # The episode is over: free its sandbox. A checkpoint taken before may still name its snapshots,
            # so only a release deletes them.
            await self._sandboxes.stop(session_id)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        return await self._sandboxes.export(session_ids, checkpoint_id=checkpoint_id)

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        await self._sandboxes.restore(states)

    async def retire_session_state(self, session_id: str) -> None:
        await self._sandboxes.stop(session_id)

    async def park_session_states(self, session_ids: list[str]) -> None:
        await self._sandboxes.park(session_ids)

    async def release_external_state(self, kept_checkpoint_ids: list[str]) -> None:
        await self._sandboxes.release(kept_checkpoint_ids)


if __name__ == "__main__":
    SandboxNotesResourcesServer.run_webserver()
