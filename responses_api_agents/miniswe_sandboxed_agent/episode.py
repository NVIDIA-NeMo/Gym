# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE sessions operating on any compatible resource-owned sandbox."""

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException, Request

from nemo_gym.agent_context import AgentTaskContext
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from responses_api_agents.miniswe_sandboxed_agent.app import MiniSWESandboxedAgent
from responses_api_agents.miniswe_sandboxed_agent.models import AgentExecutionResult, SeedSessionResponse


@dataclass
class AgentSession:
    """One deduplicated activation and its owner, retained for close retries."""

    request: AgentSeedSessionRequest
    owner: str
    params: NeMoGymResponseCreateParamsNonStreaming | None = None
    worker: asyncio.Task[AgentExecutionResult] | None = None
    closed: bool = False


class MiniSWEEpisodeAgent(MiniSWESandboxedAgent):
    """Borrow execution state; the environment server owns orchestration."""

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._agent_sessions: dict[str, AgentSession] = {}
        self._session_identities: dict[tuple[str, str], str] = {}

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        if body.sandbox_access is None:
            raise HTTPException(422, "mini-SWE requires sandbox_access")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "mini-SWE requires task tools reachable from its sandbox")
        owner = request.session[SESSION_ID_KEY]
        identity = (owner, body.episode_id.capture_key)
        session_id = self._session_identities.get(identity)
        if session_id is None:
            session_id = "miniswe-" + uuid4().hex
            self._agent_sessions[session_id] = AgentSession(body.model_copy(deep=True), owner)
            self._session_identities[identity] = session_id
        state = self._agent_sessions[session_id]
        if state.request != body or state.closed:
            raise HTTPException(409, "Episode is already bound to a different or closed session")
        request.session["miniswe_agent_session_id"] = session_id
        return AgentSeedSessionResponse(agent_session_id=session_id)

    def _agent_session(self, request: Request) -> tuple[str, AgentSession]:
        session_id = request.session.get("miniswe_agent_session_id")
        state = self._agent_sessions.get(session_id)
        if state is None or state.owner != request.session[SESSION_ID_KEY]:
            raise HTTPException(404, "Unknown mini-SWE agent session")
        return session_id, state

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        session_id, state = self._agent_session(request)
        if state.closed:
            raise HTTPException(410, "Agent session is closed")
        rollout_id = request.path_params.get("rollout_id")
        if rollout_id != state.request.episode_id.capture_key:
            raise HTTPException(409, "Rollout route does not match the agent session")
        if state.params is not None and state.params != body:
            raise HTTPException(409, "Agent session already has a different activation")
        if state.worker is None:
            context = state.request.agent_context or AgentTaskContext()
            access = state.request.sandbox_access
            connection = access.connection
            instruction = context.instruction
            if instruction is None:
                if isinstance(body.input, str):
                    instruction = body.input
                else:
                    # Keep role labels when a benchmark supplies more than one message.
                    messages = []
                    for item in body.input:
                        if not hasattr(item, "role") or not hasattr(item, "content"):
                            raise HTTPException(422, "mini-SWE requires text task messages")
                        content = item.content
                        if not isinstance(content, str):
                            if any(
                                getattr(part, "type", None) not in {"input_text", "output_text"} for part in content
                            ):
                                raise HTTPException(422, "mini-SWE requires text task messages")
                            content = "\n".join(part.text for part in content)
                        messages.append(f"{item.role}: {content}" if len(body.input) > 1 else content)
                    instruction = "\n\n".join(messages)
            seed = SeedSessionResponse(
                session_id=session_id,
                task_id=state.request.task_id.task_id,
                sandbox_descriptor=connection.descriptor,
                sandbox_provider=resolve_provider_config(
                    connection.provider_config_ref, self.server_client.global_config_dict
                ),
                instruction=instruction,
                workdir=access.workdir,
                user=context.user,
                agent_timeout_sec=context.timeout_sec or 28800,
                mcp_servers=[server.model_dump() for server in context.mcp_servers],
                skills_dir=context.skills_dir,
            )
            state.params = body.model_copy(deep=True)
            state.worker = asyncio.create_task(
                self.execute(
                    seed,
                    body.model_copy(deep=True),
                    rollout_id=rollout_id,
                    capture_model_calls=True,
                    cookies=dict(request.cookies),
                    quiesce=True,
                )
            )
            state.worker.add_done_callback(self._observe_background_task)
        result = await asyncio.shield(state.worker)
        response = result.response.model_copy(deep=True)
        response.status = (
            "failed"
            if result.termination.reason == "infrastructure_error"
            else ("completed" if result.termination.reason == "completed" else "incomplete")
        )
        response.metadata = (response.metadata or {}) | {
            "termination_reason": result.termination.reason,
            "termination_detail": (result.termination.detail or "")[:512],
            "agent_started": str(result.agent_started).lower(),
        }
        return response

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        session_id, state = self._agent_session(request)
        if body.agent_session_id != session_id or body.episode_id != state.request.episode_id:
            raise HTTPException(409, "Close identity does not match the agent session")
        state.closed = True
        observations = None
        if state.worker is not None:
            if not state.worker.done() and not state.worker.cancelling():
                state.worker.cancel()
            done, _ = await asyncio.wait({state.worker}, timeout=self.config.shutdown_timeout_sec)
            if not done:
                raise HTTPException(503, "Agent execution has not stopped yet")
            if not state.worker.cancelled():
                result = state.worker.result()
                raw = result.harness_metadata.get("ng_agent_observations")
                if raw is not None:
                    observations = AgentObservationBundle.model_validate(raw)
        return AgentCloseSessionResponse(agent_session_id=session_id, agent_observations=observations)

    async def shutdown(self) -> None:
        workers = [state.worker for state in self._agent_sessions.values() if state.worker and not state.worker.done()]
        self._closing = True
        for worker in workers:
            if not worker.cancelling():
                worker.cancel()
        if workers:
            await asyncio.wait(workers, timeout=self.config.shutdown_timeout_sec)
        await super().shutdown()


if __name__ == "__main__":
    MiniSWEEpisodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MiniSWEEpisodeAgent.run_webserver()
