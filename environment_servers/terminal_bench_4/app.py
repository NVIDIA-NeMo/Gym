# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a session-based agent against the existing TB4 resources lifecycle."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import ClassVar, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field

from nemo_gym.agent_outcome import OutcomeAction, agent_outcome_policy
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    dependency_failure,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import AgentServerRef, AggregateMetrics, AggregateMetricsRequest, ResourcesServerRef
from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    EpisodeId,
    MaterializedTask,
    TaskId,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
    rollout_path_prefix,
)
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    SeedSessionResponse,
    TerminalBench4RunRequest,
)


class TerminalBench4EnvironmentConfig(BaseEnvironmentServerConfig):
    """Bind TB4 Resources to an agent implementing sandbox sessions."""

    num_workers: Literal[1] = 1
    resources_server: ResourcesServerRef
    agent_server: AgentServerRef
    cleanup_timeout_seconds: float = Field(default=300, gt=0)
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    agent_timeout_grace_sec: float = Field(default=120, gt=0)


EpisodeRequest = BaseEpisodeRequest[TerminalBench4RunRequest]
EpisodeResponse = BaseEpisodeResponse[SandboxedVerifyResponse]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TerminalBench4EnvironmentServer(BaseEnvironmentServer[EpisodeRequest, EpisodeResponse]):
    """Keep task setup and grading in Resources, and harness execution in the agent."""

    config: TerminalBench4EnvironmentConfig
    request_model: ClassVar = EpisodeRequest
    response_model: ClassVar = EpisodeResponse

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        self._runs: dict[tuple[str, str], tuple[dict, asyncio.Task]] = {}
        self._closing = False

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app):
            try:
                yield
            finally:
                self._closing = True
                tasks = [task for _, task in self._runs.values() if not task.done()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        app = FastAPI(lifespan=lifespan)
        self.setup_session_middleware(app)
        self.setup_environment_checkpoint(app)
        app.post("/run")(self.run_legacy)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        reply = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(reply)
        return AggregateMetrics.model_validate(await get_response_json(reply))

    async def run_legacy(self, request: Request, body: dict) -> dict:
        """Accept existing prepared TB4 rows and share in-flight collector retries."""
        if self._closing:
            raise HTTPException(503, "TB4 environment is shutting down")
        payload = dict(body)
        rollout = maybe_rollout_id_from_run_body(payload) or payload.get("rollout_id") or uuid4().hex
        owner = request.session[SESSION_ID_KEY]
        payload.update(rollout_id=rollout, client_session_id=owner)
        key = (owner, rollout)
        if key not in self._runs:
            task_input = TerminalBench4RunRequest.model_validate(payload)
            episode = EpisodeRequest(
                episode_id=EpisodeId.from_capture_key(rollout),
                task=MaterializedTask(
                    task_id=TaskId(taskset=task_input.dataset_ref, task_id=task_input.task_name),
                    task_input=task_input,
                ),
            )
            self._runs[key] = (payload, asyncio.create_task(self.run_request(episode)))
        original, worker = self._runs[key]
        if original != payload:
            raise HTTPException(409, "Rollout identity is already bound to another request")
        response = await asyncio.shield(worker)
        if response.failure is not None:
            return {
                "_ng_failure_class": "environment_server_failed",
                "_ng_failure_terminal": response.failure.terminal,
                "_ng_failure_message": response.failure.failure_reason,
                "_ng_failure_stage": response.failure.stage,
            }
        return response.result.model_dump(mode="json") | {"agent_ref": {"name": self.config.agent_server.name}}

    async def run(self, request: EpisodeRequest, cleanup: CleanupContext) -> EpisodeResponse:
        payload = request.task.task_input.model_dump(mode="json")
        resources_cookies: dict[str, str] = {}
        agent_cookies: dict[str, str] = {}
        agent_id = f"tb4-{request.episode_id.capture_key}-{uuid4().hex}"
        close_result: AgentCloseSessionResponse | None = None

        async def post(server: str, path: str, body: dict, cookies: dict):
            stage = (
                "cleanup"
                if path.endswith("close") or path.endswith("close_session")
                else "verify"
                if path == "/verify"
                else "seed"
                if path == "/seed_session"
                else "agent"
            )
            try:
                reply = await self.server_client.post(server_name=server, url_path=path, json=body, cookies=cookies)
                await raise_for_status(reply)
            except Exception as error:
                raise dependency_failure(error, failure_model=EpisodeFailure, stage=stage) from error
            cookies.update({key: value.value for key, value in reply.cookies.items()})
            return await get_response_json(reply)

        async def close_resources() -> None:
            await post(self.config.resources_server.name, "/close_session", payload, resources_cookies)

        async def close_agent() -> None:
            nonlocal close_result
            value = await post(
                self.config.agent_server.name,
                "/v1/agent_sessions/close",
                {"agent_session_id": agent_id, "episode_id": request.episode_id.model_dump()},
                agent_cookies,
            )
            close_result = AgentCloseSessionResponse.model_validate(value)
            if close_result.agent_session_id != agent_id:
                raise ValueError("Agent close returned another session")

        # Register before seed: a lost seed reply must still release the allocated resources.
        cleanup.register_cleanup("TB4 resources", close_resources)
        seed = SeedSessionResponse.model_validate(
            await post(self.config.resources_server.name, "/seed_session", payload, resources_cookies)
        )
        if seed.verified_response is not None:
            return EpisodeResponse(
                episode_id=request.episode_id, task_id=request.task.task_id, result=seed.verified_response
            )
        params = request.task.task_input.responses_create_params.model_copy(deep=True)
        params.input = seed.instruction
        budget = min(seed.agent_timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
        params.metadata = {**(params.metadata or {}), "agent_timeout_sec": str(budget)}
        response = NeMoGymResponse(
            id="",
            created_at=0,
            model="",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        termination = seed.termination or AgentTermination(reason="infrastructure_error", detail="Agent did not start")
        agent_started = False
        timings = {}
        evidence = {}
        if seed.termination is None:
            agent_cleanup = cleanup.register_cleanup("TB4 agent", close_agent)
            try:
                if seed.sandbox_access is None:
                    raise ValueError("TB4 session agents require named sandbox providers")
                timings["agent_setup"] = {"started_at": now()}
                agent_seed = AgentSeedSessionResponse.model_validate(
                    await post(
                        self.config.agent_server.name,
                        "/v1/agent_sessions",
                        AgentSeedSessionRequest(
                            agent_session_id=agent_id,
                            episode_id=request.episode_id,
                            task_id=request.task.task_id,
                            sandbox_access=seed.sandbox_access,
                            agent_timeout_sec=budget,
                        ).model_dump(mode="json"),
                        agent_cookies,
                    )
                )
                if agent_seed.agent_session_id != agent_id:
                    raise ValueError("Agent seed returned another session")
                if (seed.mcp_servers or seed.skills_dir) and not agent_seed.supports_task_tool_context:
                    raise ValueError("This agent does not support the task's required MCP/skills context")
                timings["agent_setup"]["finished_at"] = now()
                timings["agent_execution"] = {"started_at": now()}
                agent_started = True
                async with asyncio.timeout(budget + self.config.agent_timeout_grace_sec):
                    response = NeMoGymResponse.model_validate(
                        await post(
                            self.config.agent_server.name,
                            rollout_path_prefix(request.episode_id.capture_key) + "/v1/responses",
                            params.model_dump(mode="json"),
                            agent_cookies,
                        )
                    )
                if agent_outcome_policy(response) != OutcomeAction.VERIFY or response.status == "failed":
                    termination = AgentTermination(
                        reason="infrastructure_error", detail=str(response.error or response.metadata)
                    )
                else:
                    termination = AgentTermination(
                        reason="timeout" if response.status == "incomplete" else "completed"
                    )
            except Exception as exc:
                termination = AgentTermination(reason="infrastructure_error", detail=f"{type(exc).__name__}: {exc}")
            finally:
                for timing in timings.values():
                    timing.setdefault("finished_at", now())
            # A failed close propagates through the base lifecycle; Resources is then destroyed without grading.
            await agent_cleanup.close()
            if close_result.cleanup_confirmed is not True:
                raise RuntimeError("Agent did not confirm termination; refusing TB4 artifact collection")
            if close_result.agent_observations is not None:
                evidence["ng_agent_observations"] = close_result.agent_observations.model_dump(mode="json")
            if close_result.trajectory is not None:
                evidence["ng_trajectory"] = close_result.trajectory.model_dump(mode="json")
        verify = SandboxedVerifyRequest.model_validate(
            payload
            | {
                "session_id": seed.session_id,
                "responses_create_params": params.model_dump(mode="json"),
                "response": response.model_dump(mode="json"),
                "termination": termination.model_dump(),
                "agent_started": agent_started and termination.reason != "infrastructure_error",
                "agent_timings": timings,
                "harness_metadata": {"agent_server": self.config.agent_server.name},
            }
        )
        result = SandboxedVerifyResponse.model_validate(
            await post(
                self.config.resources_server.name,
                "/verify",
                verify.model_dump(mode="json"),
                resources_cookies,
            )
        )
        result = result.model_copy(update=evidence)
        return EpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=result)


if __name__ == "__main__":
    TerminalBench4EnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4EnvironmentServer.run_webserver()
