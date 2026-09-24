# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native mini-SWE sessions borrowing resource-owned task sandboxes."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from fastapi import HTTPException, Request

from nemo_gym.agent_context import AgentTaskContext
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, create_provider, resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
    rollout_path_prefix,
)
from responses_api_agents.miniswe_sandboxed_agent.app import LOGGER, MiniSWESandboxedAgent
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEHarness


@dataclass
class AgentSession:
    """Process-local state; successful close receipts expire after a bounded window."""

    request: AgentSeedSessionRequest
    owner: str
    provider: SandboxProvider
    harness: MiniSWEHarness
    params: NeMoGymResponseCreateParamsNonStreaming | None = None
    worker: asyncio.Task | None = None
    closing: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    close_result: AgentCloseSessionResponse | None = None
    closed_at: float | None = None


def task_instruction(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Normalize text-only task inputs without dropping roles or request options."""
    supported = {
        "input",
        "instructions",
        "max_output_tokens",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "metadata",
        "model",
        "reasoning",
        "tool_choice",
        "stream",
        "store",
    }
    unsupported = {key for key in body.model_fields_set - supported if getattr(body, key) is not None}
    if unsupported or body.tool_choice not in (None, "auto", "required"):
        raise HTTPException(422, f"Unsupported mini-SWE request options: {sorted(unsupported) or ['tool_choice']}")
    if isinstance(body.input, str):
        return body.input
    messages = []
    for item in body.input:
        if getattr(item, "role", None) not in {"system", "developer", "user", "assistant"}:
            raise HTTPException(422, "mini-SWE requires text task messages")
        content = item.content
        if not isinstance(content, str):
            if any(getattr(part, "type", None) not in {"input_text", "output_text"} for part in content):
                raise HTTPException(422, "mini-SWE requires text task messages")
            content = "\n".join(part.text for part in content)
        messages.append(f"{item.role}: {content}" if len(body.input) > 1 else content)
    return "\n\n".join(messages)


class MiniSWEEpisodeAgent(MiniSWESandboxedAgent):
    """Install at seed, activate once, and confirm cleanup before owner verification."""

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._agent_sessions: dict[str, AgentSession] = {}
        self._session_identities: dict[tuple[str, str], str] = {}
        self._seed_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def _expire_closed(self) -> None:
        for session_id, state in list(self._agent_sessions.items()):
            if (
                state.closed_at is not None
                and monotonic() - state.closed_at > self.config.closed_session_retention_sec
            ):
                del self._agent_sessions[session_id]
                identity = (state.owner, state.request.episode_id.capture_key)
                self._session_identities.pop(identity, None)
                self._seed_locks.pop(identity, None)

    async def seed_agent_session(self, request: Request, body: AgentSeedSessionRequest) -> AgentSeedSessionResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        if body.sandbox_access is None:
            raise HTTPException(422, "mini-SWE requires sandbox_access")
        if any(access.required for access in self.effective_tool_accesses(body)):
            raise HTTPException(422, "mini-SWE requires task tools reachable from its sandbox")
        self._expire_closed()
        owner = request.session[SESSION_ID_KEY]
        identity = (owner, body.episode_id.capture_key)
        async with self._seed_locks.setdefault(identity, asyncio.Lock()):
            session_id = self._session_identities.get(identity)
            if session_id is not None:
                state = self._agent_sessions[session_id]
                if state.request != body or state.closing:
                    raise HTTPException(409, "Episode is already bound to a different or closed session")
            else:
                session_id = "miniswe-" + uuid4().hex
                access = body.sandbox_access
                context = body.agent_context or AgentTaskContext()
                config = self.server_client.global_config_dict
                provider = create_provider(resolve_provider_config(access.connection.provider_config_ref, config))
                harness = None
                try:
                    async with asyncio.timeout(self.config.setup_timeout_sec):
                        sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)

                        async def query(params):
                            prefix = rollout_path_prefix(
                                body.episode_id.capture_key, token_capture=self._token_id_capture_enabled()
                            )
                            response = await self.server_client.post(
                                server_name=self.config.model_server.name,
                                url_path=prefix + "/v1/responses",
                                json=params,
                                headers={"x-session-id": session_id},
                                cookies=dict(request.cookies),
                            )
                            await raise_for_status(response)
                            return NeMoGymResponse.model_validate(await get_response_json(response))

                        harness = MiniSWEHarness(
                            sandbox=sandbox,
                            context=HarnessContext(
                                session_id=session_id,
                                task_id=body.task_id.task_id,
                                rollout_id=body.episode_id.capture_key,
                                instruction=context.instruction or "",
                                user=context.user,
                                workdir=access.workdir,
                                setup_timeout_sec=self.config.setup_timeout_sec,
                                mcp_servers=[server.model_dump() for server in context.mcp_servers],
                                skills_dir=context.skills_dir,
                            ),
                            config=self.config.harness,
                            params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                            query=query,
                            model_name=self.config.model_server.name,
                            directory=self.config.artifacts_dir / session_id,
                            observability_enabled=isinstance(config, Mapping)
                            and bool(config.get(OBSERVABILITY_ENABLED_KEY_NAME, False)),
                        )
                        await harness.setup()
                except BaseException as error:
                    try:
                        if harness is not None:
                            await harness.close()
                            await harness.dispose()
                    except Exception:
                        LOGGER.exception("Failed to remove partially installed mini-SWE runtime")
                    finally:
                        await provider.aclose()
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    raise HTTPException(
                        422, f"mini-SWE session setup failed: {type(error).__name__}: {error}"
                    ) from error
                self._agent_sessions[session_id] = AgentSession(body.model_copy(deep=True), owner, provider, harness)
                self._session_identities[identity] = session_id
            # Do not publish the session before runtime setup has completed.
            request.session["miniswe_agent_session_id"] = session_id
            return AgentSeedSessionResponse(agent_session_id=session_id)

    def _agent_session(self, request: Request) -> tuple[str, AgentSession]:
        self._expire_closed()
        session_id = request.session.get("miniswe_agent_session_id")
        state = self._agent_sessions.get(session_id)
        if state is None or state.owner != request.session[SESSION_ID_KEY]:
            raise HTTPException(404, "Unknown mini-SWE agent session")
        return session_id, state

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        _, state = self._agent_session(request)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Rollout route does not match the agent session")
        instruction = task_instruction(body)
        async with state.lock:
            if state.closing:
                raise HTTPException(410, "Agent session is closed or closing")
            if state.worker is not None:
                raise HTTPException(409, "Agent session permits one activation")
            state.params = body.model_copy(deep=True)
            state.harness.params = body.model_copy(deep=True)
            context = state.request.agent_context or AgentTaskContext()
            state.harness.context.instruction = instruction or context.instruction or ""
            budget = min(context.timeout_sec or 28800, self.config.agent_max_timeout_sec or float("inf"))
            state.worker = asyncio.create_task(state.harness.execute(budget))
            state.worker.add_done_callback(self._observe_background_task)
        response, outcome, extra = await asyncio.shield(state.worker)
        response = response.model_copy(deep=True)
        response.status = (
            "failed"
            if outcome.reason == "infrastructure_error"
            else ("completed" if outcome.reason == "completed" else "incomplete")
        )
        runtime = extra.get("runtime") or {}
        response.metadata = (response.metadata or {}) | {
            "termination_reason": outcome.reason,
            "termination_detail": (outcome.detail or "")[:512],
            "agent_started": "true",
            "harness_execution": "sandbox",
            "harness_hostname": str(runtime.get("hostname", "")),
            "harness_pid": str(runtime.get("pid", "")),
            "harness_version": str(extra.get("harness_version", "")),
        }
        return response

    async def close_agent_session(self, request: Request, body: AgentCloseSessionRequest) -> AgentCloseSessionResponse:
        session_id, state = self._agent_session(request)
        if body.agent_session_id != session_id or body.episode_id != state.request.episode_id:
            raise HTTPException(409, "Close identity does not match the agent session")
        return await self._close_state(session_id, state)

    async def _close_state(self, session_id: str, state: AgentSession) -> AgentCloseSessionResponse:
        async with state.lock:
            if state.close_result is not None:
                return state.close_result
            state.closing = True
            if state.worker is not None:
                if not state.worker.done() and not state.worker.cancelling():
                    state.worker.cancel()
                done, _ = await asyncio.wait({state.worker}, timeout=self.config.shutdown_timeout_sec)
                if not done:
                    raise HTTPException(503, "Agent execution has not stopped yet")
            # A failed launch or a cleanup error cannot be converted into successful close.
            # Retain all state until a later close can obtain positive cleanup evidence.
            await state.harness.close()
            await state.harness.dispose()
            await state.provider.aclose()
            observations = None
            if state.harness.result is not None:
                raw = state.harness.result[2].get("ng_agent_observations")
                if raw is not None:
                    observations = AgentObservationBundle.model_validate(raw)
            state.close_result = AgentCloseSessionResponse(
                agent_session_id=session_id, agent_observations=observations
            )
            state.closed_at = monotonic()
            return state.close_result

    async def shutdown(self) -> None:
        self._closing = True
        pending = {
            asyncio.create_task(self._close_state(session_id, state))
            for session_id, state in self._agent_sessions.items()
            if state.close_result is None
        }
        if pending:
            done, unfinished = await asyncio.wait(pending, timeout=self.config.shutdown_timeout_sec)
            for task in done:
                self._observe_background_task(task)
            for task in unfinished:
                task.cancel()
                task.add_done_callback(self._observe_background_task)
        await super().shutdown()


if __name__ == "__main__":
    MiniSWEEpisodeAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MiniSWEEpisodeAgent.run_webserver()
