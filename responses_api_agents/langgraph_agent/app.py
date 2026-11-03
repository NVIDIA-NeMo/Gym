# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym adapter for LangGraph workflows executed inside task sandboxes."""

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import ClassVar
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field, JsonValue

from nemo_gym.agents import (
    DEFAULT_UV_CACHE_DIR,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SandboxSession,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
    resolve_agent_timeout,
)
from nemo_gym.client import get_response_json, raise_for_status, trace_context_path_segment
from nemo_gym.config import ModelServerRef, ResourcesServerRef, get_global_config_dict
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
    resolve_provider_metadata,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.trajectory import AgentObservationBundle, ObservationGap
from responses_api_agents.langgraph_agent.sandbox import LangGraphSandboxSession


LOG = logging.getLogger(__name__)


class LangGraphAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef | None = None
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(
        default_factory=lambda: {"image": "ubuntu:24.04", "workdir": "/workspace"}
    )
    sandbox_install_timeout_seconds: float = Field(default=900, gt=0, allow_inf_nan=False)
    # A ceiling, not a budget: the episode's agent_timeout_sec, which is usually shorter, decides when a graph stops.
    sandbox_runner_timeout_seconds: float = Field(default=21600, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    uv_cache_dir: str = DEFAULT_UV_CACHE_DIR


class LangGraphRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class LangGraphVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class LangGraphAgentAdapter(SimpleResponsesAPIAgent):
    """One complete sandboxed graph per session, with exact activation replay."""

    config: LangGraphAgentConfig
    workflow: ClassVar[str]
    calls_model_from_sandbox: ClassVar[bool] = True

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[object]:
            try:
                async with parent_lifespan(application) as state:
                    yield state
            finally:
                for session_id, record in list(self._session_records.items()):
                    if record.state is None:
                        continue
                    try:
                        await self.close_agent_session(
                            Request({"type": "http", "session": {}}),
                            AgentCloseSessionRequest(agent_session_id=session_id, episode_id=record.episode_id),
                        )
                    except Exception:
                        LOG.exception("Could not release LangGraph session %s during shutdown", session_id)

        app.router.lifespan_context = lifespan
        return app

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> LangGraphSandboxSession:
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(
                f"LangGraph workflows do not support required tool access: {', '.join(required)}"
            )
        return await self._initialize_agent_session_state(body.agent_session_id, body)

    async def _initialize_agent_session_state(
        self, agent_session_id: str, body: AgentSeedSessionRequest
    ) -> LangGraphSandboxSession:
        access = body.sandbox_access
        owns_sandbox = access is None
        provider_ref = self.config.sandbox_provider if access is None else access.connection.provider_config_ref
        if provider_ref is None:
            raise AgentSeedRejectedError("LangGraph requires sandbox_access or a configured sandbox_provider")
        global_config = get_global_config_dict()
        provider_config = resolve_provider_config(provider_ref, global_config)
        assert_model_url_reachable_from_sandbox(
            self.resolve_model_base_url(self.config.model_server.name), provider_name=next(iter(provider_config), None)
        )
        provider = create_provider(provider_config)
        if access is None:
            sandbox = AsyncSandbox(provider)
        else:
            try:
                sandbox = await AsyncSandbox.connect(access.connection.descriptor, provider=provider)
            except BaseException:
                await provider.aclose()
                raise
        session_dir = f"/tmp/nemo-gym-langgraph-sessions/{uuid4().hex}"
        state = LangGraphSandboxSession(
            request=body,
            session=SandboxSession(
                sandbox=sandbox,
                workdir=access.workdir
                if access is not None
                else str(self.config.sandbox_config.get("workdir") or "/workspace"),
                session_dir=session_dir,
                owns_sandbox=owns_sandbox,
                harness="LangGraph",
                env=dict(access.env) if access is not None else None,
                user=access.user if access is not None else None,
            ),
        )
        try:
            if owns_sandbox:
                config = dict(self.config.sandbox_config)
                config.setdefault(
                    "ttl_s",
                    self.config.sandbox_install_timeout_seconds + self.config.sandbox_runner_timeout_seconds + 600,
                )
                config["metadata"] = {
                    **resolve_provider_metadata(provider_ref, global_config),
                    **(config.get("metadata") or {}),
                    "harness": "langgraph",
                    "agent-session": agent_session_id[:63],
                }
                await sandbox.start(SandboxSpec(**config))
            await state.install_runtime(
                install_timeout=self.config.sandbox_install_timeout_seconds, uv_cache_dir=self.config.uv_cache_dir
            )
        except BaseException as error:
            try:
                await state.close(self.config.session_close_timeout_seconds)
            except BaseException:
                LOG.exception(
                    "LangGraph setup cleanup failed for %s; retaining the session for close", agent_session_id
                )
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    def _rollout_sandbox_session(self, state: AgentSessionState) -> SandboxSession | None:
        return state.session if isinstance(state, LangGraphSandboxSession) else None

    def _require_agent_session(self, agent_session_id: str) -> LangGraphSandboxSession:
        state = super()._require_agent_session(agent_session_id)
        if not isinstance(state, LangGraphSandboxSession):
            raise TypeError("Expected LangGraph sandbox session state")
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, LangGraphSandboxSession):
            raise TypeError("Expected LangGraph sandbox session state")
        await state.close(self.config.session_close_timeout_seconds)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=AgentObservationBundle(
                source=self.config.name,
                gaps=[
                    ObservationGap(
                        code="observation_capture_failed",
                        detail="LangGraph does not yet record native invocation or turn evidence.",
                    )
                ],
            ),
        )

    async def _run_sandbox_graph(
        self, state: LangGraphSandboxSession, body: NeMoGymResponseCreateParamsNonStreaming, *, model_url_path: str
    ) -> NeMoGymResponse:
        # Resolve the server root independently of request prefixes, then preserve the exact inbound capture path.
        root = self.resolve_model_base_url(self.config.model_server.name, trace_context=False).removesuffix("/v1")
        # resolve_model_base_url can apply configured training capture even without a rollout.
        root = root.removesuffix("/training-token-capture")
        # The graph runs in a process Gym does not instrument, so its parent span travels in the URL, after the
        # rollout prefix as apply_rollout_prefix places it.
        rollout_prefix = model_url_path.removesuffix("/v1/responses")
        graph_fields = type(self.config).model_fields.keys() - LangGraphAgentConfig.model_fields.keys()
        payload = {
            "workflow": self.workflow,
            "config": self.config.model_dump(mode="json", include=set(graph_fields)),
            # Unset fields stay unset, so the Model Server's own defaults (such as extra_body sampling) still apply.
            "request": body.model_dump(mode="json", exclude_unset=True),
            "model_url": f"{root}{rollout_prefix}{trace_context_path_segment()}/v1/responses",
        }
        timeout = resolve_agent_timeout(body.metadata, ceiling=self.config.sandbox_runner_timeout_seconds)
        if state.request.agent_timeout_sec is not None:
            timeout = min(timeout, state.request.agent_timeout_sec)
        try:
            output = await state.execute(
                payload, timeout=timeout, close_timeout=self.config.session_close_timeout_seconds
            )
        except RuntimeError:
            # The supervisor stopped the graph at its deadline (or a rollout stop): an agent limit the episode
            # verifies as incomplete, not a dependency failure to retry. The graph leaves no partial answer.
            if not (state.session.cleanup and state.session.cleanup["timed_out"]):
                raise
            LOG.warning(
                "LangGraph graph reached its %ss activation deadline; returning an incomplete response", timeout
            )
            response = NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=time.time(),
                model=body.model or "",
                object="response",
                output=[],
                parallel_tool_calls=body.parallel_tool_calls,
                tool_choice=body.tool_choice,
                tools=body.tools,
                status="incomplete",
                metadata={"harness_execution": "sandbox"},
            )
            return self.report_rollout_stop(response, state.session)
        response = NeMoGymResponse.model_validate(output["response"])
        runtime = output.get("runtime") or {}
        response.metadata = {
            **(response.metadata or {}),
            "harness_execution": "sandbox",
            "harness_hostname": str(runtime.get("hostname", "")),
            "harness_pid": str(runtime.get("pid", "")),
            "harness_python": str(runtime.get("python", "")),
        }
        return self.report_rollout_stop(response, state.session)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        model_url_path = self.url_path_for_request("/v1/responses", request)
        if session_id is not None:
            state = self._require_agent_session(session_id)
            if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
                raise HTTPException(409, "Activation requires the seeded session's rollout path")
            if state.activation_path is not None and state.activation_path != request.url.path:
                raise HTTPException(409, "Activation path does not match the session's first activation")
            if state.activation_request is not None and state.activation_request != body:
                raise HTTPException(409, "LangGraph sessions run one graph; retry the identical request")
            if state.task is None:
                state.activation_path = request.url.path
                state.activation_request = body.model_copy(deep=True)
                state.task = asyncio.create_task(
                    self._run_sandbox_graph(state, state.activation_request, model_url_path=model_url_path)
                )
            return (await asyncio.shield(state.task)).model_copy(deep=True)

        # Compatibility requests still run the whole graph in a temporary owned sandbox.
        identifier = uuid4().hex
        rollout_id = request.path_params.get("rollout_id") or identifier
        seed = AgentSeedSessionRequest(
            agent_session_id=identifier,
            episode_id=EpisodeId.from_capture_key(rollout_id),
            task_id=TaskId(taskset="langgraph-direct", task_id=identifier),
        )
        # Use the same retained lifecycle as explicit sessions without changing the caller's cookie.
        transient_request = Request({**request.scope, "session": {}})
        try:
            await self.seed_agent_session(transient_request, seed)
            state = self._require_agent_session(identifier)
            state.task = asyncio.create_task(self._run_sandbox_graph(state, body, model_url_path=model_url_path))
            return await asyncio.shield(state.task)
        finally:
            activation_error = sys.exception()
            try:
                await self.close_agent_session(
                    transient_request,
                    AgentCloseSessionRequest(agent_session_id=identifier, episode_id=seed.episode_id),
                )
            except BaseException:
                # No caller has this temporary ID: retain failed release for expiry retry and shutdown.
                async with self._locked_agent_session(identifier) as record:
                    if record.state is not None:
                        self._keep_failed_seed(identifier, record)
                if activation_error is None:
                    raise
                LOG.exception("LangGraph temporary session cleanup failed; retaining %s for retry", identifier)

    async def run(self, request: Request, body: LangGraphRunRequest) -> LangGraphVerifyResponse:
        if self.config.resources_server is None:
            raise HTTPException(422, "Legacy /run requires resources_server; use agent sessions for composition")
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed)
        resp = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
        )
        await raise_for_status(resp)
        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": await get_response_json(resp)},
            cookies=seed.cookies,
        )
        await raise_for_status(verify)
        return LangGraphVerifyResponse.model_validate(await get_response_json(verify))
