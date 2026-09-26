# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE rollout orchestration using resource-owned sandboxes."""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
    rollout_path_prefix,
)
from responses_api_agents.miniswe_sandboxed_agent.harness import (
    HarnessContext,
    HarnessOutcome,
    MiniSWEConfig,
    MiniSWEHarness,
)
from responses_api_agents.miniswe_sandboxed_agent.models import (
    AgentExecutionResult,
    MiniSWERunRequest,
    MiniSWEVerifyResponse,
    SandboxedVerifyRequest,
    SeedSessionResponse,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class MiniSWESession:
    """Execution state shared by /run and responses for one borrowed sandbox."""

    sandbox: AsyncSandbox
    resource_session_id: str
    harness: MiniSWEHarness
    original_params: NeMoGymResponseCreateParamsNonStreaming
    budget: float
    result: tuple[NeMoGymResponse, HarnessOutcome, dict] | None = None
    worker: asyncio.Task | None = None


class MiniSWESandboxedConfig(BaseResponsesAPIAgentConfig):
    num_workers: Literal[1] = 1
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    harness: MiniSWEConfig = Field(default_factory=MiniSWEConfig)
    artifacts_dir: Path = Path("results/miniswe_sandboxed_agent")
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    setup_timeout_sec: float = Field(default=360, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_response(params: NeMoGymResponseCreateParamsNonStreaming, model: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_" + uuid4().hex,
        created_at=int(time()),
        model=model,
        object="response",
        output=[],
        tool_choice=params.tool_choice,
        tools=params.tools,
        parallel_tool_calls=params.parallel_tool_calls,
    )


class MiniSWESandboxedAgent(SimpleResponsesAPIAgent):
    config: MiniSWESandboxedConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._runs = {}
        self._sessions: dict[str, MiniSWESession] = {}
        self._finalizers = set()
        self._closing = False
        self._shutdown_deadline: float | None = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def shutdown(self) -> None:
        self._closing = True
        if self._shutdown_deadline is None:
            self._shutdown_deadline = monotonic() + self.config.shutdown_timeout_sec
        workers = [worker for _, worker in self._runs.values() if not worker.done()]
        for worker in workers:
            if not worker.cancelling():
                worker.cancel()
        if workers:
            # One budget covers seed completion, worker joining, and verification.
            await asyncio.wait(workers, timeout=max(0, self._shutdown_deadline - monotonic()))
        finalizers = set(self._finalizers)
        if finalizers:
            _, pending = await asyncio.wait(finalizers, timeout=max(0, self._shutdown_deadline - monotonic()))
            for task in pending:
                task.cancel()
        # Do not gather unfinished work without a timeout. In particular a lost
        # seed response must not keep shutdown alive; resources owns its expiry.

    @staticmethod
    def _observe_background_task(task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            LOGGER.error("mini-SWE background operation failed", exc_info=(type(error), error, error.__traceback__))

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        key = request.session[SESSION_ID_KEY]
        state = self._sessions.get(key)
        if state is None:
            raise HTTPException(409, "No seeded mini-SWE sandbox for this session")
        if body != state.original_params:
            raise HTTPException(409, "Agent session is already bound to another request")
        if state.worker is None:
            state.worker = asyncio.create_task(state.harness.execute(state.budget))
        try:
            state.result = await asyncio.shield(state.worker)
        except asyncio.CancelledError:
            state.worker.cancel()
            # The sandbox runner must finish cleanup before /run can verify.
            state.result = await state.worker
        return state.result[0]

    async def run(self, request: Request, body: MiniSWERunRequest) -> MiniSWEVerifyResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        payload = body.model_dump(mode="json")
        rollout_id = self.rollout_id_from_run(body)
        payload["rollout_id"] = rollout_id or body.capture_rollout_id or payload.get("rollout_id") or uuid4().hex
        payload["client_session_id"] = request.session[SESSION_ID_KEY]
        if rollout_id:
            payload["_ng_rollout_id"] = rollout_id
        key = (payload["client_session_id"], payload["rollout_id"])
        if key not in self._runs:
            worker = asyncio.create_task(self._run(request, payload, dict(request.cookies), bool(rollout_id)))
            worker.add_done_callback(self._observe_background_task)
            self._runs[key] = (payload, worker)
        saved, worker = self._runs[key]
        if saved != payload:
            raise HTTPException(409, "Rollout identity is already bound to another request")
        # A retried/disconnected collector must never create a second model loop.
        return await asyncio.shield(worker)

    async def _run(
        self, request: Request, payload: dict, cookies: dict, capture_model_calls: bool
    ) -> MiniSWEVerifyResponse:
        # Keep provisioning alive long enough to obtain a session ID for cleanup if cancelled.
        seed_task = asyncio.create_task(
            self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=payload,
                cookies=cookies,
            )
        )
        seed_task.add_done_callback(self._observe_background_task)
        cancelled = False
        try:
            seed_response = await asyncio.shield(seed_task)
        except asyncio.CancelledError:
            cancelled = True
            deadline = self._shutdown_deadline
            if deadline is None:
                deadline = monotonic() + self.config.shutdown_timeout_sec
            done, _ = await asyncio.wait({seed_task}, timeout=max(0, deadline - monotonic()))
            if not done:
                seed_task.cancel()
                raise
            seed_response = seed_task.result()
        await raise_for_status(seed_response)
        cookies = cookies | seed_response.cookies
        seed = SeedSessionResponse.model_validate(await get_response_json(seed_response))
        if seed.verified_response is not None:
            return seed.verified_response
        params = MiniSWERunRequest.model_validate(payload).responses_create_params
        if cancelled:
            result = AgentExecutionResult(
                responses_create_params=params,
                response=empty_response(params, self.config.model_server.name),
                termination=HarnessOutcome(reason="cancelled"),
            )
        else:
            result = await self._run_agent(
                request,
                seed,
                params,
                rollout_id=payload.get("_ng_rollout_id") or payload["rollout_id"],
                capture_model_calls=capture_model_calls,
                cookies=cookies,
                artifact_directory=Path(payload["artifact_directory"]) if payload.get("artifact_directory") else None,
            )
        verify_body = SandboxedVerifyRequest(session_id=seed.session_id, **result.model_dump())
        finalizer = asyncio.create_task(self._verify(verify_body, cookies))
        self._finalizers.add(finalizer)
        finalizer.add_done_callback(self._finalizers.discard)
        finalizer.add_done_callback(self._observe_background_task)
        return await asyncio.shield(finalizer)

    async def _run_agent(
        self,
        request: Request,
        seed: SeedSessionResponse,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
        capture_model_calls: bool,
        cookies: dict,
        artifact_directory: Path | None = None,
    ) -> AgentExecutionResult:
        """Set up session state, invoke responses, and release the borrowed transport."""
        response = empty_response(params, self.config.model_server.name)
        termination = seed.termination or HarnessOutcome(
            reason="infrastructure_error", detail="Setup did not complete"
        )
        extra, timings = {}, {}
        agent_started = False
        provider = None
        harness = None
        state = None
        key = request.session[SESSION_ID_KEY]
        original_params = params.model_copy(deep=True)
        with rollout_context(rollout_id if capture_model_calls else None):
            try:
                if seed.termination is None:
                    timings["agent_setup"] = {"started_at": now()}
                    async with asyncio.timeout(self.config.setup_timeout_sec):
                        provider = create_provider(resolve_provider_config(seed.sandbox_provider))
                        sandbox = await AsyncSandbox.connect(seed.sandbox_descriptor, provider=provider)
                        cwd = await sandbox.exec("pwd", timeout_s=30, user=seed.user)
                        if cwd.return_code:
                            raise RuntimeError("Unable to determine the task working directory")
                        context = HarnessContext(
                            session_id=seed.session_id,
                            task_id=seed.task_id,
                            rollout_id=rollout_id,
                            instruction=seed.instruction,
                            user=seed.user,
                            workdir=cwd.stdout.strip(),
                            setup_timeout_sec=self.config.setup_timeout_sec,
                            mcp_servers=seed.mcp_servers,
                            skills_dir=seed.skills_dir,
                        )
                        params.input = [NeMoGymEasyInputMessage(role="user", content=context.instruction)]

                        async def query(model_params):
                            prefix = rollout_path_prefix(
                                rollout_id if capture_model_calls else None,
                                token_capture=self._token_id_capture_enabled(),
                            )
                            model_response = await self.server_client.post(
                                server_name=self.config.model_server.name,
                                url_path=prefix + "/v1/responses",
                                json=model_params,
                                headers={"x-session-id": seed.session_id},
                                cookies=cookies,
                            )
                            await raise_for_status(model_response)
                            return NeMoGymResponse.model_validate(await get_response_json(model_response))

                        global_config = getattr(self.server_client, "global_config_dict", None)
                        harness = MiniSWEHarness(
                            sandbox=sandbox,
                            context=context,
                            config=self.config.harness,
                            observability_enabled=isinstance(global_config, Mapping)
                            and bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False)),
                            params=params,
                            query=query,
                            model_name=self.config.model_server.name,
                            directory=artifact_directory or self.config.artifacts_dir / seed.session_id,
                        )
                        await harness.setup()
                    timings["agent_setup"]["finished_at"] = now()
                    timings["agent_execution"] = {"started_at": now()}
                    budget = min(seed.agent_timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
                    deadline = monotonic() + budget
                    state = MiniSWESession(
                        sandbox=sandbox,
                        resource_session_id=seed.session_id,
                        harness=harness,
                        original_params=original_params,
                        budget=max(0, deadline - monotonic()),
                    )
                    if key in self._sessions:
                        raise HTTPException(409, "Agent session already has an active mini-SWE sandbox")
                    self._sessions[key] = state
                    agent_started = True
                    # In full swap, the environment server will replace this call
                    # with an HTTP request to the agent's /v1/responses endpoint.
                    response = await self.responses(request, original_params)
                    _, termination, extra = state.result
                    if monotonic() >= deadline:
                        termination.reason = "timeout"
            except asyncio.CancelledError:
                termination = HarnessOutcome(reason="cancelled")
            except Exception as exc:
                termination = HarnessOutcome(
                    reason="timeout"
                    if isinstance(exc, TimeoutError) and not agent_started
                    else "infrastructure_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if state is not None and self._sessions.get(key) is state:
                    self._sessions.pop(key, None)
                if harness is not None and not agent_started:
                    try:
                        await harness.close()
                    except Exception:
                        LOGGER.exception("Failed to clean up mini-SWE setup")
                for timing in timings.values():
                    timing.setdefault("finished_at", now())
                if provider is not None:
                    try:
                        # Drop this process's transport; resources retains sandbox/Compose ownership.
                        await provider.aclose()
                    except Exception:
                        LOGGER.exception("Failed to close the mini-SWE sandbox transport")
        return AgentExecutionResult(
            responses_create_params=params,
            response=response,
            termination=termination,
            agent_started=agent_started,
            agent_timings=timings,
            harness_metadata=extra,
        )

    async def _verify(self, body: SandboxedVerifyRequest, cookies: dict) -> MiniSWEVerifyResponse:
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_for_status(response)
        return MiniSWEVerifyResponse.model_validate(await get_response_json(response))


if __name__ == "__main__":
    MiniSWESandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MiniSWESandboxedAgent.run_webserver()
