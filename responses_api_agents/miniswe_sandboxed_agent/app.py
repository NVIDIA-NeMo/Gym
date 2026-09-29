# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE rollout orchestration using resource-owned sandboxes."""

import asyncio
import json
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, time
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, field_validator

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, rollout_context
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, create_provider, resolve_provider_config
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from nemo_gym.tasks.harbor.materialize import AGENT_TIMEOUT_METADATA_KEY
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

# The harbor resources server leaves the task's in-sandbox tool declarations here at seed time.
TASK_CONTEXT_FILE = "/tmp/.nemo-gym/task.json"


@dataclass
class MiniSWESession:
    """Execution state shared by /run and responses for one borrowed sandbox.

    The legacy ``/run`` path fills ``seed`` from the resources server's reply. An episode session
    (``/v1/agent_sessions``) receives the sandbox from the environment server instead: ``seed`` then
    carries only the identity, ``original_params`` is bound by the first ``/v1/responses`` call, whose
    input is the instruction, and ``provider`` is this process's transport, released at close.
    """

    sandbox: AsyncSandbox
    seed: SeedSessionResponse
    original_params: NeMoGymResponseCreateParamsNonStreaming | None
    rollout_id: str
    capture_model_calls: bool
    artifact_directory: Path | None = None
    setup_deadline: float | None = None
    setup_started_at: str | None = None
    result: AgentExecutionResult | None = None
    worker: asyncio.Task | None = None
    workdir: str | None = None
    provider: Any | None = None
    episode: AgentSeedSessionRequest | None = None
    observations: AgentObservationBundle | None = None


@dataclass
class MiniSWEEpisodeSession(AgentSessionState):
    """The base agent's record of an episode session; the execution state lives in ``_sessions``."""

    key: tuple[str, str]
    state: MiniSWESession


class MiniSWESandboxedConfig(BaseResponsesAPIAgentConfig):
    num_workers: Literal[1] = 1
    # The resources server the legacy `/run` path seeds and verifies against. Episode sessions
    # (`/v1/agent_sessions`) receive the sandbox from the environment server and need none.
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    sandbox_model_base_url: str | None = None
    harness: MiniSWEConfig = Field(default_factory=MiniSWEConfig)
    artifacts_dir: Path = Path("results/miniswe_sandboxed_agent")
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    # Wall-clock budget of one episode-session turn; the legacy path takes it from the seed reply.
    agent_timeout_sec: float = Field(default=28800, gt=0)
    setup_timeout_sec: float = Field(default=360, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)

    @field_validator("sandbox_model_base_url")
    @classmethod
    def normalize_sandbox_model_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("sandbox_model_base_url must be an HTTP(S) URL without a query or fragment")
        # This is the model server's address as seen by the sandbox, not the
        # /v1 endpoint. Accept either form to match other sandboxed agents.
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


# Harness outcomes that are not the agent's doing; the episode path refuses them with a 5xx.
INFRASTRUCTURE_TERMINATIONS = frozenset({"infrastructure_error", "cancelled"})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _instruction(input_value: Any) -> str:
    """The task instruction: the text of the row's input messages."""
    if isinstance(input_value, str):
        return input_value
    messages: list[str] = []
    for item in input_value or []:
        value = item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        if not isinstance(value, dict):
            messages.append(str(value))
            continue
        content = value.get("content", "")
        if isinstance(content, str):
            messages.append(content)
        elif isinstance(content, list):
            messages.extend(
                str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("text")
            )
    return "\n\n".join(part for part in messages if part)


async def _read_task_context(sandbox: AsyncSandbox) -> dict[str, Any]:
    """``mcp_servers`` and ``skills_dir`` the resources server recorded for this task, if any."""
    result = await sandbox.exec(f"cat {TASK_CONTEXT_FILE} 2>/dev/null || true", timeout_s=30)
    text = (result.stdout or "").strip()
    if not text:
        return {}
    try:
        context = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.warning(f"Ignoring unreadable {TASK_CONTEXT_FILE}")
        return {}
    return context if isinstance(context, dict) else {}


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
    ray_enabled = False
    config: MiniSWESandboxedConfig

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._runs = {}
        self._sessions: dict[tuple[str, str], MiniSWESession] = {}
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

    # -- episode sessions: the environment server seeds the sandbox and calls /v1/responses ------------

    def _model_base_url(self) -> str:
        """The Model Server address as the sandbox sees it, before the per-rollout prefix."""
        return self.config.sandbox_model_base_url or get_server_url(self.config.model_server.name)

    def _agent_timeout_sec(self, state: MiniSWESession, body: NeMoGymResponseCreateParamsNonStreaming) -> float:
        """The run budget: the seed's budget and the configured ceiling, bounded by the task's own timeout."""
        budget = min(state.seed.agent_timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
        raw = (body.metadata or {}).get(AGENT_TIMEOUT_METADATA_KEY)
        if raw is not None:
            try:
                task_timeout = float(raw)
            except (TypeError, ValueError):
                LOGGER.warning(f"Ignoring non-numeric {AGENT_TIMEOUT_METADATA_KEY}={raw!r} in the request metadata")
            else:
                if task_timeout > 0:
                    budget = min(budget, task_timeout)
        return budget

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> MiniSWEEpisodeSession:
        """Borrow the resources server's sandbox for this episode."""
        if body.sandbox_access is None:
            raise HTTPException(422, "mini-SWE needs sandbox_access from the resources server")
        connection = body.sandbox_access.connection
        if not isinstance(connection, DirectSandboxConnection):
            raise HTTPException(422, "mini-SWE supports only direct sandbox connections")
        provider_config = resolve_provider_config(connection.provider_config_ref, get_global_config_dict())
        # mini-SWE runs inside the sandbox and calls the Model Server itself, so a loopback URL (the
        # default without use_absolute_ip) would make every model call fail and the episode score 0.
        try:
            assert_model_url_reachable_from_sandbox(
                self._model_base_url(), provider_name=next(iter(provider_config), None)
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        provider = create_provider(provider_config)
        try:
            sandbox = await AsyncSandbox.connect(connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        # The environment server addresses the turn as /ng-rollout/<capture_key>/v1/responses and the base
        # agent's session cookie finds the state, so the owner half of the key only has to be unique.
        key = (f"episode:{body.agent_session_id}", body.episode_id.capture_key)
        state = MiniSWESession(
            sandbox=sandbox,
            seed=SeedSessionResponse(
                session_id=body.agent_session_id,
                task_id=body.task_id.task_id,
                agent_timeout_sec=self.config.agent_timeout_sec,
            ),
            original_params=None,
            rollout_id=body.episode_id.capture_key,
            capture_model_calls=True,
            artifact_directory=self.config.artifacts_dir / body.agent_session_id,
            workdir=body.sandbox_access.workdir,
            provider=provider,
            episode=body,
        )
        self._sessions[key] = state
        return MiniSWEEpisodeSession(request=body, key=key, state=state)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, MiniSWEEpisodeSession):
            raise TypeError("Expected a mini-SWE episode session state")
        session = state.state
        if self._sessions.get(state.key) is session:
            self._sessions.pop(state.key, None)
        try:
            await session.sandbox.disconnect()
        finally:
            if session.provider is not None:
                # Resources retains sandbox/Compose ownership; release only our transport.
                await session.provider.aclose()
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id, agent_observations=session.observations
        )

    async def _bind_episode_turn(
        self, state: MiniSWESession, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: str | None
    ) -> None:
        """First /v1/responses on an episode session: the input is the instruction, task.json the tool grants."""
        state.original_params = body.model_copy(deep=True)
        state.seed.instruction = _instruction(body.input)
        task_context = await _read_task_context(state.sandbox)
        state.seed.mcp_servers = task_context.get("mcp_servers") or []
        state.seed.skills_dir = task_context.get("skills_dir")
        # Model calls carry the /ng-rollout/<id> the environment server used, so capture correlates them.
        state.capture_model_calls = rollout_id is not None
        if rollout_id is not None:
            state.rollout_id = rollout_id

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        rollout_id = current_rollout_id()
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is not None:
            # An episode session: the base agent's cookie names the seeded state.
            record = self._require_agent_session(agent_session_id)
            if not isinstance(record, MiniSWEEpisodeSession):
                raise TypeError("Expected a mini-SWE episode session state")
            state = record.state
        else:
            owner = request.session[SESSION_ID_KEY]
            matches = [key for key in self._sessions if key[0] == owner]
            if rollout_id is None:
                if len(matches) > 1:
                    raise HTTPException(
                        409, "Multiple mini-SWE rollouts for this session; use /ng-rollout/<id>/v1/responses"
                    )
                state = self._sessions.get(matches[0]) if matches else None
            else:
                state = self._sessions.get((owner, rollout_id))
            if state is None:
                raise HTTPException(409, "No seeded mini-SWE sandbox for this session")
        if state.original_params is None:
            await self._bind_episode_turn(state, body, rollout_id)
        if body != state.original_params:
            raise HTTPException(409, "Agent session is already bound to another request")

        async def execute_message() -> AgentExecutionResult:
            params = body.model_copy(deep=True)
            seed = state.seed
            rollout_id = state.rollout_id
            sandbox = state.sandbox
            response = empty_response(params, self.config.model_server.name)
            termination = seed.termination or HarnessOutcome(
                reason="infrastructure_error", detail="Setup did not complete"
            )
            extra, timings = {}, {}
            agent_started = False
            harness = None
            with rollout_context(rollout_id if state.capture_model_calls else None):
                try:
                    if seed.termination is None:
                        timings["agent_setup"] = {"started_at": state.setup_started_at or now()}
                        deadline = state.setup_deadline or monotonic() + self.config.setup_timeout_sec
                        async with asyncio.timeout_at(deadline):
                            workdir = state.workdir
                            if workdir is None:
                                cwd = await sandbox.exec("pwd", timeout_s=30, user=seed.user)
                                if cwd.return_code or not (cwd.stdout or "").strip():
                                    raise RuntimeError("Unable to determine the task working directory")
                                workdir = (cwd.stdout or "").strip().splitlines()[-1]
                            context = HarnessContext(
                                session_id=seed.session_id,
                                task_id=seed.task_id,
                                rollout_id=rollout_id,
                                instruction=seed.instruction,
                                user=seed.user,
                                workdir=workdir,
                                setup_timeout_sec=self.config.setup_timeout_sec,
                                mcp_servers=seed.mcp_servers,
                                skills_dir=seed.skills_dir,
                            )
                            params.input = [NeMoGymEasyInputMessage(role="user", content=context.instruction)]

                            global_config = getattr(self.server_client, "global_config_dict", None)
                            harness = MiniSWEHarness(
                                sandbox=sandbox,
                                context=context,
                                config=self.config.harness,
                                observability_enabled=isinstance(global_config, Mapping)
                                and bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False)),
                                params=params,
                                model_base_url=self.base_url_for_run(
                                    base_url=self._model_base_url(), body={"_ng_rollout_id": rollout_id}
                                )
                                + "/v1",
                                model_name=self.config.model_server.name,
                                directory=state.artifact_directory or self.config.artifacts_dir / seed.session_id,
                            )
                            await harness.setup()
                        timings["agent_setup"]["finished_at"] = now()
                        timings["agent_execution"] = {"started_at": now()}
                        deadline = monotonic() + self._agent_timeout_sec(state, body)
                        agent_started = True
                        response, termination, extra = await harness.execute(max(0, deadline - monotonic()))
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
                    if harness is not None and not agent_started:
                        try:
                            await harness.close()
                        except Exception:
                            LOGGER.exception("Failed to clean up mini-SWE setup")
                    for timing in timings.values():
                        timing.setdefault("finished_at", now())
            return AgentExecutionResult(
                responses_create_params=params,
                response=response,
                termination=termination,
                agent_started=agent_started,
                agent_timings=timings,
                harness_metadata=extra,
            )

        if state.worker is None:
            state.worker = asyncio.create_task(execute_message())
        # Repeated cancellation must not release the transport or verify while
        # the message worker is still cleaning up sandbox processes.
        while not state.worker.done():
            try:
                await asyncio.shield(state.worker)
            except asyncio.CancelledError:
                if not state.worker.cancelling():
                    state.worker.cancel()
        state.result = state.worker.result()
        if state.episode is not None:
            return self._episode_turn_response(state, state.result)
        return state.result.response

    @staticmethod
    def _episode_turn_response(state: MiniSWESession, result: AgentExecutionResult) -> NeMoGymResponse:
        """The environment server sees one response; carry the harness outcome in its metadata.

        "completed", "nonzero_exit" and "timeout" are the agent's own result and are verified. An
        "infrastructure_error" or a cancelled run is not the agent's doing and is refused with a 503 so
        the episode is recorded as a masked failure instead of an unmasked zero. A repeated request for
        the same turn replays the same answer.
        """
        observations = result.harness_metadata.get("ng_agent_observations") if result.harness_metadata else None
        if observations:
            state.observations = AgentObservationBundle.model_validate(observations)
        if result.termination.reason in INFRASTRUCTURE_TERMINATIONS:
            detail = f"mini-SWE {result.termination.reason}"
            if result.termination.detail:
                detail += f": {result.termination.detail[:500]}"
            raise HTTPException(503, detail)
        response = result.response
        metadata = dict(response.metadata or {})
        metadata["miniswe_termination"] = result.termination.reason
        if result.termination.detail:
            metadata["miniswe_termination_detail"] = result.termination.detail[:500]
        metadata["miniswe_agent_started"] = str(result.agent_started).lower()
        response.metadata = metadata
        return response

    # -- legacy /run: this agent seeds and verifies against a TB4-style resources server ---------------

    async def run(self, request: Request, body: MiniSWERunRequest) -> MiniSWEVerifyResponse:
        if self._closing:
            raise HTTPException(503, "Agent server is shutting down")
        if self.config.resources_server is None:
            raise HTTPException(422, "The legacy /run path needs a resources_server; use episode sessions instead")
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
        result = AgentExecutionResult(
            responses_create_params=params,
            response=empty_response(params, self.config.model_server.name),
            termination=seed.termination or HarnessOutcome(reason="cancelled"),
        )
        provider = None
        state = None
        key = (request.session[SESSION_ID_KEY], payload["rollout_id"])
        try:
            if not cancelled and seed.termination is None:
                setup_started_at = now()
                setup_deadline = monotonic() + self.config.setup_timeout_sec
                result.agent_timings["agent_setup"] = {"started_at": setup_started_at}
                async with asyncio.timeout_at(setup_deadline):
                    provider = create_provider(resolve_provider_config(seed.sandbox_provider))
                    sandbox = await AsyncSandbox.connect(seed.sandbox_descriptor, provider=provider)
                if key in self._sessions:
                    raise HTTPException(409, "Agent session already has an active mini-SWE sandbox")
                state = MiniSWESession(
                    sandbox=sandbox,
                    seed=seed,
                    original_params=params.model_copy(deep=True),
                    rollout_id=payload.get("_ng_rollout_id") or payload["rollout_id"],
                    capture_model_calls=capture_model_calls,
                    artifact_directory=Path(payload["artifact_directory"])
                    if payload.get("artifact_directory")
                    else None,
                    setup_deadline=setup_deadline,
                    setup_started_at=setup_started_at,
                )
                self._sessions[key] = state
                with rollout_context(payload["rollout_id"]):
                    await self.responses(request, params)
                result = state.result
        except asyncio.CancelledError:
            result.termination = HarnessOutcome(reason="cancelled")
        except Exception as exc:
            result.termination = HarnessOutcome(
                reason="timeout" if isinstance(exc, TimeoutError) else "infrastructure_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if state is not None and self._sessions.get(key) is state:
                self._sessions.pop(key, None)
            for timing in result.agent_timings.values():
                timing.setdefault("finished_at", now())
            if provider is not None:
                try:
                    # Resources retains sandbox/Compose ownership; release only our transport.
                    await provider.aclose()
                except Exception:
                    LOGGER.exception("Failed to close the mini-SWE sandbox transport")
        verify_body = SandboxedVerifyRequest(session_id=seed.session_id, **result.model_dump())
        finalizer = asyncio.create_task(self._verify(verify_body, cookies))
        self._finalizers.add(finalizer)
        finalizer.add_done_callback(self._finalizers.discard)
        finalizer.add_done_callback(self._observe_background_task)
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
