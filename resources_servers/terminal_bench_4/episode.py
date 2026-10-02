# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 resources implementing the shared single-agent episode protocol."""

import asyncio
import hashlib

from fastapi import HTTPException, Request
from pydantic import ConfigDict

from nemo_gym.agent_context import AgentTaskContext
from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.app import TerminalBench4Config, TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    TerminalBench4RunRequest,
)


class TerminalBench4NativeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class TerminalBench4EpisodeConfig(TerminalBench4Config):
    """Name the provider configuration available to sandbox borrowers."""

    sandbox_provider_ref: str
    sandbox_provider_ref_cpu: str | None = None
    sandbox_provider_ref_gpu: str | None = None


class TerminalBench4EpisodeResourcesServer(TerminalBench4ResourcesServer):
    """Provision and grade TB4 without selecting or invoking a harness."""

    config: TerminalBench4EpisodeConfig

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        # The merged base class binds /close_session to this method. Registering
        # a second route would leave its default no-op handler first in the router.
        return await self.close_session(request, body)

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._native_requests: dict[str, ResourcesSeedSessionRequest] = {}
        self._native_closed: set[str] = set()
        self._native_close_locks: dict[str, asyncio.Lock] = {}

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        if session_id in self._native_closed:
            raise HTTPException(410, "Resources session is closed")
        previous = self._native_requests.setdefault(session_id, body.model_copy(deep=True))
        if previous != body:
            raise HTTPException(409, "Resources session ID is already bound to a different request")
        if body.task_id.task_id != body.task_data.get("task_name"):
            raise HTTPException(422, "TaskId does not match the TB4 task_name")
        legacy = TerminalBench4RunRequest.model_validate(
            body.task_data
            | {
                "rollout_id": body.episode_id.capture_key,
                "_ng_rollout_id": body.episode_id.capture_key,
                "client_session_id": body.resources_session_id,
                "responses_create_params": {"input": []},
                "episode_id": body.episode_id.model_dump(),
                "native_task_id": body.task_id.model_dump(),
            }
        )
        seed = await super().seed_session(request, legacy)
        session = self._session(request, seed.session_id)
        if body.resources_session_id in self._native_closed:
            raise HTTPException(410, "Resources session is closed")
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
                resources_session_id=body.resources_session_id,
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
                    resources_session_id=body.resources_session_id,
                    episode_id=body.episode_id,
                ),
            )
            raise

    async def verify(self, request: Request, body: TerminalBench4NativeVerifyRequest) -> SandboxedVerifyResponse:
        session_id = request.session.get("tb4_resources_session_id")
        if not session_id:
            raise HTTPException(404, "Missing TB4 resource session")
        session = self._session(request, session_id)
        task_data = body.model_dump()
        for key in ("task_name", "task_ref", "dataset_ref"):
            if task_data.get(key) != getattr(session.request, key):
                raise HTTPException(409, f"Verification {key} differs from the seeded task")
        if session.phase in {"cleaning", "closed"} and session.verify_body is None:
            raise HTTPException(410, "TB4 session was closed without verification")
        response = body.response
        metadata = response.metadata or {}
        reason = metadata.get("termination_reason")
        if reason not in {"completed", "timeout", "nonzero_exit", "cancelled", "infrastructure_error"}:
            reason = "infrastructure_error" if response.status == "failed" else "completed"
        return await super().verify(
            request,
            SandboxedVerifyRequest(
                **(
                    body.model_dump()
                    | {
                        "session_id": session.session_id,
                        "termination": AgentTermination(reason=reason, detail=metadata.get("termination_detail")),
                        "agent_started": metadata.get("agent_started", "true") == "true",
                    }
                ),
            ),
        )

    async def close_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        async with self._native_close_locks.setdefault(body.resources_session_id, asyncio.Lock()):
            return await self._close_native_session(request, body)

    async def _close_native_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        native = self._native_requests.get(body.resources_session_id)
        if native is None:
            self._native_closed.add(body.resources_session_id)
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)
        if native.episode_id != body.episode_id:
            raise HTTPException(409, "Episode identity does not match the resources session")
        request.session["tb4_client_session_id"] = body.resources_session_id
        identity = hashlib.sha256(f"{self._owner(request)}:{body.episode_id.capture_key}".encode()).hexdigest()
        session_id = self._by_identity.get(identity)
        if session_id is None:
            self._native_closed.add(body.resources_session_id)
            return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)
        session = self._session(request, session_id)
        self._native_closed.add(body.resources_session_id)
        if session.execution is not None:
            await asyncio.shield(session.execution)
        if session.phase != "closed" and session.finalization is None:
            if session.expiry_task is not None:
                session.expiry_task.cancel()
            session.phase = "cleaning"
            session.finalization = asyncio.create_task(lifecycle.finalize_session(session, grade=False))
        if session.finalization is not None:
            await asyncio.shield(session.finalization)
        environments = (session.environment, session.verifier_environment, session.shared_logs)
        if any(env is not None and not env.closed for env in environments):
            # Legacy finalization records deletion errors and marks the episode
            # closed. A native close must retry and confirm owner cleanup.
            await lifecycle.cleanup(session)
        if any(env is not None and not env.closed for env in environments):
            raise HTTPException(503, "TB4 resource cleanup is incomplete; retry close_session")
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)


if __name__ == "__main__":
    TerminalBench4EpisodeResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4EpisodeResourcesServer.run_webserver()
