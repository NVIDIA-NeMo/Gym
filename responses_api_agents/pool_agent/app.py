# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool CLI adapter for Environment Server episodes and supervised task sandboxes."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from shlex import quote
from time import time
from typing import ClassVar
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, JsonValue, PrivateAttr

from nemo_gym.agent_utils.sandbox_session import SandboxSession
from nemo_gym.agent_utils.task_timeout import AGENT_TIMEOUT_METADATA_KEY, resolve_agent_timeout
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, SandboxObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.pool_agent.configuration import pool_config
from responses_api_agents.pool_agent.sandbox import PoolSession
from responses_api_agents.pool_agent.trajectory import parse_trajectory


LOG = logging.getLogger(__name__)


class PoolAgentConfig(BaseResponsesAPIAgentConfig):
    """Pool runtime pin, model and sandbox settings; benchmark wiring belongs to the environment."""

    model_server: ModelServerRef
    model: str
    pool_version: str = Field(default="v1.0.16", pattern=r"^v\d+\.\d+\.\d+$")
    accept_pool_eula: bool = False
    context_window: int = Field(default=32768, gt=0)
    setup_timeout: float = Field(default=900, gt=0, allow_inf_nan=False)
    timeout: float = Field(default=900, gt=0, allow_inf_nan=False)
    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    failure_artifacts_dir: Path | None = Path("results/pool-agent-failures")
    sandbox_provider: str | None = None
    sandbox_config: dict[str, JsonValue] = Field(default_factory=dict)


def prompt_from_request(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Accept one text instruction and an optional system/developer prefix without dropping controls."""
    params = body.model_dump(exclude_defaults=True, exclude_none=True)
    metadata = params.get("metadata")
    if isinstance(metadata, dict) and set(metadata) <= {AGENT_TIMEOUT_METADATA_KEY}:
        params.pop("metadata")
    unsupported = params.keys() - {"input", "instructions", "model"}
    if unsupported:
        raise HTTPException(422, "Unsupported Pool request fields: " + ", ".join(sorted(unsupported)))
    parts = [body.instructions] if body.instructions else []
    if isinstance(body.input, str):
        parts.append(body.input)
    else:
        roles = [getattr(item, "role", None) for item in body.input]
        if roles not in (["user"], ["system", "user"], ["developer", "user"]):
            raise HTTPException(422, "Pool accepts one user message with an optional system/developer prefix")
        for item in body.input:
            content = item.content
            if isinstance(content, str):
                parts.append(content)
            else:
                for part in content:
                    value = part if isinstance(part, dict) else part.model_dump()
                    if value.get("type") != "input_text":
                        raise HTTPException(422, "Pool accepts text input only")
                    parts.append(value["text"])
    prompt = "\n\n".join(parts)
    if not prompt.strip():
        raise HTTPException(422, "Pool requires a nonempty prompt")
    return prompt


class PoolAgent(SimpleResponsesAPIAgent):
    """Run one Pool activation per session, with replay-safe requests and explicit sandbox ownership."""

    config: PoolAgentConfig
    calls_model_from_sandbox: ClassVar[bool] = True

    _shutting_down: bool = PrivateAttr(default=False)
    _seed_tasks: set[asyncio.Task] = PrivateAttr(default_factory=set)

    def setup_webserver(self) -> FastAPI:
        """Drain sessions before the inherited application lifespan exits."""
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[object]:
            async with parent_lifespan(application) as state:
                try:
                    yield state
                finally:
                    await self._shutdown_sessions()

        app.router.lifespan_context = lifespan
        return app

    async def _close_for_shutdown(self, seed: AgentSeedSessionRequest) -> None:
        try:
            async with asyncio.timeout(self.config.session_close_timeout_seconds):
                await self.close_agent_session(
                    Request({"type": "http", "session": {}}),
                    AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id),
                )
        except Exception:
            # The shared close endpoint retains failed cleanup for retry.
            LOG.exception("Pool shutdown could not close session %s", seed.agent_session_id)

    async def _shutdown_sessions(self) -> None:
        self._shutting_down = True
        if self._seed_tasks:
            _, pending = await asyncio.wait(self._seed_tasks, timeout=self.config.setup_timeout)
            for task in pending:
                task.cancel()
            if pending:
                _, pending = await asyncio.wait(pending, timeout=self.config.session_close_timeout_seconds)
                if pending:
                    LOG.error("Pool shutdown still has %s pending session setups", len(pending))
        # Use shared bookkeeping and the public close path, including failed seeds.
        seeds = [record.state.request for record in self._session_records.values() if record.state is not None]
        await asyncio.gather(*(self._close_for_shutdown(seed) for seed in seeds))

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> PoolSession:
        if self._shutting_down:
            raise HTTPException(503, "Pool server is shutting down")
        if not self.config.accept_pool_eula:
            raise HTTPException(422, "Set accept_pool_eula=true to accept https://poolside.ai/eula before using Pool")
        task = asyncio.current_task()
        assert task is not None
        self._seed_tasks.add(task)
        try:
            if any(access.required for access in self.effective_tool_accesses(body)):
                raise HTTPException(422, "Pool does not yet enforce required external tool grant initialization")
            if body.sandbox_access is not None and body.sandbox_access.user is not None:
                raise HTTPException(422, "Pool does not yet support a SandboxAccess user override")
            owns_sandbox = body.sandbox_access is None
            if owns_sandbox:
                if self.config.sandbox_provider is None:
                    raise HTTPException(422, "Pool requires SandboxAccess or a configured sandbox_provider")
                provider_ref = self.config.sandbox_provider
                workdir = self.config.sandbox_config.get("workdir") or "/app"
            else:
                if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                    raise HTTPException(422, "Pool requires direct SandboxAccess")
                provider_ref = body.sandbox_access.connection.provider_config_ref
                workdir = body.sandbox_access.workdir
            root = "/tmp/nemo-gym-pool-sessions"
            if (
                not isinstance(workdir, str)
                or not workdir.startswith("/")
                or ".." in workdir.split("/")
                or workdir.rstrip("/") == ""
                or root.startswith(workdir.rstrip("/") + "/")
                or workdir == root
                or workdir.startswith(root + "/")
            ):
                raise HTTPException(422, "Pool task workdir must be absolute and disjoint from session storage")
            global_config = get_global_config_dict()
            provider = create_provider(resolve_provider_config(provider_ref, global_config))
            if owns_sandbox:
                sandbox = AsyncSandbox(provider)
            else:
                try:
                    sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
                except BaseException:
                    await provider.aclose()
                    raise
            state = PoolSession(
                request=body,
                provider_name=provider.name,
                sandbox_id=sandbox.sandbox_id,
                session=SandboxSession(
                    sandbox=sandbox,
                    session_dir=f"{root}/{uuid4().hex}",
                    workdir=workdir,
                    harness="Pool",
                    owns_sandbox=owns_sandbox,
                ),
            )
            try:
                if owns_sandbox:
                    settings = self.config.sandbox_config.copy()
                    metadata = settings.pop("metadata", {})
                    if not isinstance(metadata, dict):
                        raise ValueError("Pool sandbox metadata must be a mapping")
                    settings["metadata"] = {
                        **resolve_provider_metadata(provider_ref, global_config),
                        **metadata,
                        "agent_session_id": body.agent_session_id[:63],
                    }
                    settings.setdefault("workdir", workdir)
                    settings.setdefault("ttl_s", self.config.setup_timeout + self.config.timeout + 600)
                    await sandbox.start(SandboxSpec(**settings))
                    state.sandbox_id = sandbox.sandbox_id
                    result = await sandbox.exec(f"mkdir -p -- {quote(workdir)}", cwd="/", timeout_s=30)
                    if result.return_code != 0 or result.error_type:
                        raise RuntimeError("Cannot prepare Pool task workdir")
                await state.install(
                    version=self.config.pool_version,
                    accept_eula=self.config.accept_pool_eula,
                    timeout=self.config.setup_timeout,
                )
                if self._shutting_down:
                    raise HTTPException(503, "Pool server shut down during session setup")
            except BaseException as error:
                try:
                    await state.close(timeout=self.config.session_close_timeout_seconds)
                except BaseException:
                    LOG.exception("Pool setup cleanup failed; retaining session for close")
                    raise AgentSessionSetupError(state, error=error) from error
                raise
            return state
        finally:
            self._seed_tasks.discard(task)

    def _observations(self, state: PoolSession) -> AgentObservationBundle:
        artifacts = state.session.artifacts
        gaps = [
            "model_call_ownership_unavailable",
            "usage_unavailable",
            "harness_tool_timing_unavailable",
            "child_invocations_unavailable",
        ]
        gaps.extend("optional_tool_grant_skipped" for access in self.effective_tool_accesses(state.request))
        status = "unknown"
        conversation = []
        if state.activation_request is not None:
            conversation.append(
                NeMoGymEasyInputMessage(role="user", content=prompt_from_request(state.activation_request))
            )
        if artifacts is not None:
            try:
                trajectory = parse_trajectory(artifacts.trajectory)
                gaps.extend(trajectory.gaps)
                conversation.extend(trajectory.output)
                status = "completed" if trajectory.exit_reason == "exit_tool_called" else "incomplete"
                if trajectory.errors:
                    status = "failed"
                elif trajectory.gaps or (state.session.cleanup and state.session.cleanup["timed_out"]):
                    status = "incomplete"
            except (ValueError, KeyError, TypeError):
                gaps.append("trajectory_invalid")
                status = "failed"
            if artifacts.trajectory_error:
                gaps.append(artifacts.trajectory_error)
        if state.failure:
            status = "failed"
        return AgentObservationBundle(
            source="pool",
            records=[
                AgentInvocation(
                    invocation_id=state.request.agent_session_id,
                    status=status,
                    error_type=state.failure,
                    conversation=conversation,
                ),
                SandboxObservation(role="agent", provider=state.provider_name, sandbox_id=state.sandbox_id),
            ],
            gaps=[ObservationGap(code=code) for code in sorted(set(gaps))],
        )

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, PoolSession):
            raise TypeError("Expected PoolSession")
        await state.close(timeout=self.config.session_close_timeout_seconds)
        await self._preserve_failure(state)
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=self._observations(state),
            cleanup_confirmed=state.session.cleanup_confirmed,
        )

    async def _preserve_failure(self, state: PoolSession) -> None:
        if state.failure and state.session.artifacts is not None and self.config.failure_artifacts_dir is not None:
            try:
                if state.failure_artifact_task is None:
                    state.failure_artifact_task = asyncio.create_task(
                        asyncio.to_thread(state.save_failure_artifacts, self.config.failure_artifacts_dir)
                    )
                await asyncio.wait_for(
                    asyncio.shield(state.failure_artifact_task), timeout=self.config.session_close_timeout_seconds
                )
            except Exception:
                if state.failure_artifact_task.done():
                    state.failure_artifact_task = None
                LOG.exception("Could not preserve Pool failure artifacts")

    async def _activate(self, state: PoolSession, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        timeout = resolve_agent_timeout(
            body.metadata, ceiling=min(self.config.timeout, state.request.agent_timeout_sec or self.config.timeout)
        )
        config = pool_config(
            base_url=self.model_endpoint(
                model_server=self.config.model_server, rollout_id=state.request.episode_id.capture_key
            ).base_url,
            model=self.config.model,
            context_window=self.config.context_window,
            timeout=timeout,
        )
        try:
            artifacts = await state.session.execute(
                stage_activation=lambda: state.stage(prompt=prompt_from_request(body), config=json.dumps(config)),
                collect=state.collect,
                timeout=timeout,
                close_timeout=self.config.session_close_timeout_seconds,
            )
            if artifacts.trajectory_error:
                raise RuntimeError("Pool did not produce a readable root trajectory")
            parsed = parse_trajectory(artifacts.trajectory)
            cleanup = state.session.cleanup
            if cleanup is None or not cleanup["cleanup_confirmed"]:
                raise RuntimeError("Pool ended without confirmed supervisor cleanup")
            if cleanup["error"]:
                raise RuntimeError(f"Pool supervision failed: {cleanup['error']}")
            if parsed.errors:
                raise RuntimeError("Pool session failed: " + "; ".join(parsed.errors))
            if parsed.exit_reason == "model_next_tool_call_http_error":
                raise RuntimeError("Pool model request failed: " + parsed.exit_reason)
            timed_out = cleanup["timed_out"]
            if parsed.exit_reason not in (None, "exit_tool_called"):
                raise RuntimeError(f"Pool failed: reason={parsed.exit_reason}")
            if timed_out and not parsed.output and not artifacts.completed_model_calls:
                raise RuntimeError("Pool timed out before any model response completed")
            if not timed_out:
                if cleanup["return_code"] not in (0, 4) or artifacts.exit_code not in (0, 4):
                    raise RuntimeError(
                        f"Pool runner failed: supervisor exit={cleanup['return_code']}, Pool exit={artifacts.exit_code}"
                    )
                if parsed.exit_reason != "exit_tool_called":
                    raise RuntimeError(f"Pool failed: reason={parsed.exit_reason}")
                if parsed.gaps:
                    raise RuntimeError("Pool completed without a complete trajectory: " + ", ".join(parsed.gaps))
            return NeMoGymResponse(
                id=f"resp_{uuid4().hex}",
                created_at=int(time()),
                object="response",
                model=self.config.model,
                output=parsed.output,
                usage=None,
                status="incomplete" if timed_out else "completed",
                tools=body.tools,
                tool_choice=body.tool_choice,
                parallel_tool_calls=body.parallel_tool_calls,
                metadata={
                    "harness_execution": "sandbox",
                    "pool_version": self.config.pool_version,
                    **({"runner_hostname": artifacts.runner_hostname} if artifacts.runner_hostname else {}),
                    **({"runner_pid": str(artifacts.runner_pid)} if artifacts.runner_pid is not None else {}),
                },
            )
        except BaseException as error:
            artifacts = state.session.artifacts
            if artifacts is not None and artifacts.stderr:
                LOG.error("Pool activation failed; stderr tail: %s", artifacts.stderr[-4096:])
            state.failure = type(error).__name__
            await self._preserve_failure(state)
            raise

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if self._shutting_down:
            raise HTTPException(503, "Pool server is shutting down")
        prompt_from_request(body)
        if body.model is not None and body.model != self.config.model:
            raise HTTPException(422, "Pool request model must match the configured model")
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(422, "Pool requires an agent session managed by an Environment Server")
        state = self._require_agent_session(session_id)
        if not isinstance(state, PoolSession):
            raise TypeError("Expected PoolSession")
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Pool activation must use its seeded rollout route")
        if state.session.closing:
            raise HTTPException(409, "Pool session is closing")
        if state.task is None:
            state.activation_request = body.model_copy(deep=True)
            state.task = asyncio.create_task(self._activate(state, state.activation_request))
        elif state.activation_request != body:
            raise HTTPException(409, "Pool sessions support one activation; retries must use the same input")
        return (await asyncio.shield(state.task)).model_copy(deep=True)

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        """Require Environment Server orchestration instead of owning benchmark verification."""
        raise HTTPException(422, "Run Pool through an Environment Server such as single_agent_turn")


if __name__ == "__main__":
    PoolAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = PoolAgent.run_webserver()
