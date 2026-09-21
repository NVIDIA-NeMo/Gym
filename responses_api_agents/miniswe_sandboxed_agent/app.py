# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE rollout orchestration using resource-owned sandboxes."""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
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
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    SeedSessionResponse,
)
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig, MiniSWEHarness


LOGGER = logging.getLogger(__name__)


class MiniSWESandboxedConfig(BaseResponsesAPIAgentConfig):
    num_workers: Literal[1] = 1
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    harness: MiniSWEConfig = Field(default_factory=MiniSWEConfig)
    artifacts_dir: Path = Path("results/terminal_bench_4/agent")
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    setup_timeout_sec: float = Field(default=360, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)


class MiniSWERunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MiniSWEVerifyResponse(SandboxedVerifyResponse):
    model_config = ConfigDict(extra="allow")


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
        self._finalizers = set()
        self._closing = False

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
        workers = [worker for _, worker in self._runs.values() if not worker.done()]
        for worker in workers:
            worker.cancel()
        if workers:
            # The harness joins its worker before /verify can collect artifacts.
            await asyncio.wait(workers, timeout=self.config.shutdown_timeout_sec)
        if self._finalizers:
            _, pending = await asyncio.wait(self._finalizers, timeout=self.config.shutdown_timeout_sec)
            for task in pending:
                task.cancel()
            await asyncio.gather(*self._finalizers, return_exceptions=True)
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        raise NotImplementedError("This agent requires /run")

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
            worker = asyncio.create_task(self._run(payload, dict(request.cookies), bool(rollout_id)))
            self._runs[key] = (payload, worker)
        saved, worker = self._runs[key]
        if saved != payload:
            raise HTTPException(409, "Rollout identity is already bound to another request")
        # A retried/disconnected collector must never create a second model loop.
        return await asyncio.shield(worker)

    async def _run(self, payload: dict, cookies: dict, capture_model_calls: bool) -> MiniSWEVerifyResponse:
        # Keep provisioning alive long enough to obtain a session ID for cleanup if cancelled.
        seed_task = asyncio.create_task(
            self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=payload,
                cookies=cookies,
            )
        )
        cancelled = False
        try:
            seed_response = await asyncio.shield(seed_task)
        except asyncio.CancelledError:
            cancelled = True
            seed_response = await seed_task
        await raise_for_status(seed_response)
        cookies = cookies | seed_response.cookies
        seed = SeedSessionResponse.model_validate(await get_response_json(seed_response))
        if seed.verified_response is not None:
            return MiniSWEVerifyResponse.model_validate(seed.verified_response.model_dump())
        params = MiniSWERunRequest.model_validate(payload).responses_create_params
        response = empty_response(params, self.config.model_server.name)
        termination = seed.termination or AgentTermination(
            reason="infrastructure_error", detail="Setup did not complete"
        )
        extra, timings = {}, {}
        agent_started = False
        provider = None
        with rollout_context(payload.get("_ng_rollout_id")):
            try:
                if cancelled:
                    raise asyncio.CancelledError
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
                            task_id=payload.get("task_name"),
                            rollout_id=payload.get("_ng_rollout_id") or payload["rollout_id"],
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
                                payload.get("_ng_rollout_id") if capture_model_calls else None,
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
                            directory=Path(payload["artifact_directory"])
                            if payload.get("artifact_directory")
                            else self.config.artifacts_dir / seed.session_id,
                        )
                        await harness.setup()
                    timings["agent_setup"]["finished_at"] = now()
                    timings["agent_execution"] = {"started_at": now()}
                    budget = min(seed.agent_timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
                    deadline = monotonic() + budget
                    agent_started = True
                    response, outcome, extra = await harness.execute(max(0, deadline - monotonic()))
                    termination = AgentTermination.model_validate(outcome.model_dump())
                    if monotonic() >= deadline:
                        termination.reason = "timeout"
            except asyncio.CancelledError:
                termination = AgentTermination(reason="cancelled")
            except Exception as exc:
                termination = AgentTermination(
                    reason="timeout"
                    if isinstance(exc, TimeoutError) and not agent_started
                    else "infrastructure_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            finally:
                for timing in timings.values():
                    timing.setdefault("finished_at", now())
                if provider is not None:
                    try:
                        # Drop this process's transport; resources retains sandbox/Compose ownership.
                        await provider.aclose()
                    except Exception:
                        LOGGER.exception("Failed to close the mini-SWE sandbox transport")
                verify_body = SandboxedVerifyRequest(
                    responses_create_params=params,
                    session_id=seed.session_id,
                    response=response,
                    termination=termination,
                    agent_started=agent_started,
                    agent_timings=timings,
                    harness_metadata=extra,
                )
                finalizer = asyncio.create_task(self._verify(verify_body, cookies))
                self._finalizers.add(finalizer)
                finalizer.add_done_callback(self._finalizers.discard)
            return await asyncio.shield(finalizer)

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
