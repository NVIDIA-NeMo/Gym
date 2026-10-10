# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import ConfigDict

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent.config import NOOAAgentConfig
from responses_api_agents.nooa_agent.result import finalize_run_result, is_transient_infrastructure_error
from responses_api_agents.nooa_agent.runner import (
    InProcessNOOARunner,
    NOOARunFailure,
    NOOARunner,
    NOOARunRequest,
    NOOARunResult,
)
from responses_api_agents.nooa_agent.sandbox_runner import SandboxNOOARunner
from responses_api_agents.nooa_agent.sandbox_runtime import prepare_nooa_runtime


@dataclass
class NOOASessionState(AgentSessionState):
    tool_access: DirectHTTPToolAccess | None = None
    resources_cookies: dict[str, str] = field(default_factory=dict)
    model_cookies: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False
    activation_key: tuple[NeMoGymResponseCreateParamsNonStreaming, str] | None = None
    execution: asyncio.Task[NeMoGymResponse] | None = None
    result: NOOARunResult | None = None
    runner: NOOARunner | None = None


class NOOAAgent(SimpleResponsesAPIAgent):
    """Run one NOOA activation per Environment Server-owned episode."""

    ray_enabled = False
    config: NOOAAgentConfig
    runner: Any = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:
        if self.config.nooa.execution_mode == "embedded":
            self.runner = InProcessNOOARunner(
                invocation=self.config.nooa,
                server_client=self.server_client,
                model_server_name=self.config.model_server.name,
                max_policy_calls=self.config.max_policy_calls,
            )
        super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.router.routes = [route for route in app.router.routes if getattr(route, "path", None) != "/run"]
        app.post("/v1/agent_sessions/finish")(self.finish_agent_session)
        return app

    async def run(self, body: object) -> None:
        raise NotImplementedError("Use the nooa_single_agent_turn Environment Server")

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> NOOASessionState:
        accesses = self.effective_tool_accesses(body)
        if any(access.required and not isinstance(access, DirectHTTPToolAccess) for access in accesses):
            raise HTTPException(422, "NOOA supports direct HTTP tool grants only")
        direct = [access for access in accesses if isinstance(access, DirectHTTPToolAccess)]
        if len(direct) > 1:
            raise HTTPException(422, "NOOA supports at most one direct HTTP tool grant")
        access = direct[0] if direct else None
        state = NOOASessionState(
            request=body,
            tool_access=access,
            resources_cookies=dict(access.cookies) if access else {},
        )
        if self.config.nooa.execution_mode == "sandboxed":
            if body.sandbox_access is None:
                raise HTTPException(422, "Sandboxed NOOA requires Resources-provided sandbox_access")
            model_base_url = self.server_client._resolve_base_url(self.config.model_server.name)
            connection = body.sandbox_access.connection
            provider = create_provider(
                resolve_provider_config(connection.provider_config_ref, self.server_client.global_config_dict)
            )
            try:
                sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
            runner = SandboxNOOARunner(
                sandbox=sandbox,
                workdir=body.sandbox_access.workdir,
                python="",
                invocation=self.config.nooa,
                model_base_url=model_base_url,
                model_server_name=self.config.model_server.name,
                max_policy_calls=self.config.max_policy_calls,
            )
            state.runner = runner
            try:
                runner.python = await prepare_nooa_runtime(sandbox)
                await runner.prepare()
            except BaseException as error:
                # Main retains this state for close while rejecting activation
                # and seed retries against an incompletely initialized runtime.
                raise AgentSessionSetupError(state, error=error) from error
        return state

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "NOOA requires an agent session")
        state = self._require_agent_session(session_id)
        assert isinstance(state, NOOASessionState)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Agent session does not match the rollout route")
        if body.tools and state.tool_access is None:
            raise HTTPException(422, "Resource tools require a direct HTTP grant")
        route = self.url_path_for_request("/v1/responses", request)
        async with state.lock:
            if state.closing:
                raise HTTPException(409, "Agent session is closing")
            key = (body.model_copy(deep=True), route)
            if state.activation_key is not None and state.activation_key != key:
                raise HTTPException(409, "Session already has another activation")
            if state.execution is None:
                state.activation_key = key
                state.execution = asyncio.create_task(self._execute_session(state, body, route))
            execution = state.execution
        return await asyncio.shield(execution)

    async def _execute_session(
        self, state: NOOASessionState, body: NeMoGymResponseCreateParamsNonStreaming, route: str
    ) -> NeMoGymResponse:
        try:
            with rollout_context(state.request.episode_id.capture_key):
                state.result = await (state.runner or self.runner).run(
                    NOOARunRequest(
                        responses_create_params=body,
                        model_url_path=route,
                        model_cookies=state.model_cookies,
                        resource_cookies=state.resources_cookies,
                        tool_access=state.tool_access,
                        task_id=state.request.task_id.task_id,
                        rollout_id=state.request.episode_id.capture_key,
                    )
                )
        except asyncio.CancelledError as error:
            state.result = getattr(error, "nooa_result", None)
            if state.result is not None:
                state.result.termination_reason = "cancelled"
            raise
        except Exception as error:
            if isinstance(error, NOOARunFailure):
                state.result = error.result
                state.result.termination_reason = "infrastructure_error"
                state.result.termination_error = str(error)
            if is_transient_infrastructure_error(error):
                raise HTTPException(503, "NOOA dependency request failed") from error
            raise
        return finalize_run_result(state.result)[0]

    async def finish_agent_session(
        self, request: Request, body: AgentCloseSessionRequest
    ) -> AgentCloseSessionResponse:
        """Freeze completed execution and collect evidence without stopping task services."""
        current = self._agent_session_id_from_request(request)
        if current is not None and current != body.agent_session_id:
            raise HTTPException(409, "agent_session_id does not match the session cookie")
        async with self._locked_agent_session(body.agent_session_id) as record:
            if record.episode_id != body.episode_id:
                raise HTTPException(409, "episode_id does not match the seeded agent session")
            if record.close_response is not None:
                return record.close_response.model_copy(deep=True)
            state = record.state
            if record.closing or not isinstance(state, NOOASessionState):
                raise HTTPException(409, "Agent session is closing or unavailable")
            async with state.lock:
                if state.execution is None or not state.execution.done():
                    raise HTTPException(409, "Agent execution has not finished")
                # A failed execution must take the normal bounded close path.
                state.execution.result()
                state.closing = True
            return self._session_evidence(state)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, NOOASessionState)
        async with state.lock:
            state.closing = True
            execution = state.execution
        if execution is not None:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        if isinstance(state.runner, SandboxNOOARunner):
            await state.runner.close()
            if state.runner.artifact is not None and state.runner.artifact.response is not None:
                state.result = state.runner.artifact.run_result()
        return self._session_evidence(state)

    def _session_evidence(self, state: NOOASessionState) -> AgentCloseSessionResponse:
        observations = state.runner.observations if isinstance(state.runner, SandboxNOOARunner) else None
        if state.result is not None:
            _, observations = finalize_run_result(state.result)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=observations,
            resources_cookies=dict(
                state.result.resource_cookies if state.result is not None else state.resources_cookies
            )
            if state.tool_access is not None
            else None,
        )


if __name__ == "__main__":
    NOOAAgent.run_webserver()
