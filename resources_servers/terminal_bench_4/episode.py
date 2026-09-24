# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 resources implementing the shared single-agent episode protocol."""

import asyncio

from fastapi import FastAPI, HTTPException, Request

from nemo_gym.agent_context import AgentTaskContext
from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.single_agent_episode_types import ResponsesResourcesVerifyRequest
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.app import TerminalBench4Config, TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    TerminalBench4RunRequest,
)


class TerminalBench4EpisodeConfig(TerminalBench4Config):
    """Name the provider configuration available to sandbox borrowers."""

    sandbox_provider_ref: str
    sandbox_provider_ref_cpu: str | None = None
    sandbox_provider_ref_gpu: str | None = None


class TerminalBench4EpisodeResourcesServer(TerminalBench4ResourcesServer):
    """Provision and grade TB4 without selecting or invoking a harness."""

    config: TerminalBench4EpisodeConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close_session")(self.close_session)
        return app

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        if body.task_id.task_id != body.task_data.get("task_name"):
            raise HTTPException(422, "TaskId does not match the TB4 task_name")
        legacy = TerminalBench4RunRequest.model_validate(
            body.task_data
            | {
                "rollout_id": body.episode_id.capture_key,
                "_ng_rollout_id": body.episode_id.capture_key,
                "client_session_id": None,
                "responses_create_params": {"input": []},
                "episode_id": body.episode_id.model_dump(),
                "native_task_id": body.task_id.model_dump(),
            }
        )
        seed = await super().seed_session(request, legacy)
        session = self._session(request, seed.session_id)
        request.session["tb4_resources_session_id"] = seed.session_id
        if seed.termination is not None:
            raise HTTPException(503, seed.termination.detail or "TB4 provisioning failed")
        if session.phase != "ready":
            raise HTTPException(409, "TB4 session is already finalized")
        provider_ref = self.config.sandbox_provider_ref
        if session.environment.pool == "cpu":
            provider_ref = self.config.sandbox_provider_ref_cpu or provider_ref
        elif session.environment.pool == "gpu":
            provider_ref = self.config.sandbox_provider_ref_gpu or provider_ref
        try:
            return ResourcesSeedSessionResponse(
                resources_session_id=seed.session_id,
                sandbox_access=SandboxAccess(
                    connection=DirectSandboxConnection(
                        provider_config_ref=provider_ref,
                        descriptor=seed.sandbox_descriptor,
                    ),
                    workdir=await session.environment.agent_workdir(),
                ),
                agent_context=AgentTaskContext(
                    instruction=seed.instruction,
                    timeout_sec=seed.agent_timeout_sec,
                    user=seed.user,
                    mcp_servers=seed.mcp_servers,
                    skills_dir=seed.skills_dir,
                ),
            )
        except BaseException:
            # Resources owns rollback when its handoff cannot be serialized or
            # the working directory cannot be resolved after provisioning.
            await self.close_session(
                request,
                ResourcesCloseSessionRequest(
                    resources_session_id=session.session_id,
                    episode_id=body.episode_id,
                ),
            )
            raise

    def _episode_session(
        self, request: Request, episode_id: EpisodeId, task_id: TaskId | None = None
    ) -> lifecycle.Session:
        session_id = request.session.get("tb4_resources_session_id")
        if not session_id:
            raise HTTPException(404, "Missing TB4 resource session")
        session = self._session(request, session_id)
        if session.request.episode_id != episode_id.model_dump():
            raise HTTPException(409, "Episode identity does not match the seeded session")
        if task_id is not None and session.request.native_task_id != task_id.model_dump():
            raise HTTPException(409, "Task identity does not match the seeded session")
        return session

    async def verify(self, request: Request, body: ResponsesResourcesVerifyRequest) -> SandboxedVerifyResponse:
        session = self._episode_session(request, body.episode_id, body.task_id)
        if session.phase in {"cleaning", "closed"} and session.verify_body is None:
            raise HTTPException(410, "TB4 session was closed without verification")
        response = body.verification_input.response
        metadata = response.metadata or {}
        reason = metadata.get("termination_reason")
        if reason not in {"completed", "timeout", "nonzero_exit", "cancelled", "infrastructure_error"}:
            reason = (
                "infrastructure_error"
                if response.status == "failed"
                else "nonzero_exit"
                if response.status == "incomplete"
                else "completed"
            )
        return await super().verify(
            request,
            SandboxedVerifyRequest(
                session_id=session.session_id,
                **body.verification_input.model_dump(),
                termination=AgentTermination(reason=reason, detail=metadata.get("termination_detail")),
                agent_started=metadata.get("agent_started", "true") == "true",
            ),
        )

    async def close_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session = self._episode_session(request, body.episode_id)
        if body.resources_session_id != session.session_id:
            raise HTTPException(409, "Resource session identity does not match the cookie")
        if session.execution is not None:
            await asyncio.shield(session.execution)
        if session.phase != "closed" and session.finalization is None:
            if session.expiry_task is not None:
                session.expiry_task.cancel()
            session.phase = "cleaning"
            session.finalization = asyncio.create_task(lifecycle.finalize_session(session, grade=False))
        if session.finalization is not None:
            await asyncio.shield(session.finalization)
        return ResourcesCloseSessionResponse(resources_session_id=session.session_id)


if __name__ == "__main__":
    TerminalBench4EpisodeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4EpisodeResourcesServer.run_webserver()
