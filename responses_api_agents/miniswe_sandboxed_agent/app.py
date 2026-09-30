# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE rollout orchestration using resource-owned sandboxes."""

import asyncio
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, time
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, field_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id, rollout_context
from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, create_provider, resolve_provider_config
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    get_server_url,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
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


def task_instruction(params: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """Read the task text from the prepared input without replacing its messages."""
    parts = []
    for message in params.input:
        if getattr(message, "role", None) != "user":
            continue
        if isinstance(message.content, str):
            parts.append(message.content)
        else:
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "input_text":
                    parts.append(item["text"])
                elif getattr(item, "type", None) == "input_text":
                    parts.append(item.text)
    instruction = "\n\n".join(parts)
    if not instruction.strip():
        raise ValueError("Prepared rows must contain a user task instruction; run gym eval prepare again")
    return instruction


@dataclass
class MiniSWESession:
    """Execution state shared by /run and responses for one borrowed sandbox."""

    sandbox: AsyncSandbox | None
    seed: SeedSessionResponse
    original_params: NeMoGymResponseCreateParamsNonStreaming
    rollout_id: str
    capture_model_calls: bool
    artifact_directory: Path | None = None
    setup_deadline: float | None = None
    setup_started_at: str | None = None
    result: AgentExecutionResult | None = None
    worker: asyncio.Task | None = None
    harness: MiniSWEHarness | None = None
    provider: SandboxProvider | None = None
    ready: bool = False
    closing: bool = False
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MiniSWESandboxedConfig(BaseResponsesAPIAgentConfig):
    num_workers: Literal[1] = 1
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    sandbox_model_base_url: str | None = None
    harness: MiniSWEConfig = Field(default_factory=MiniSWEConfig)
    artifacts_dir: Path = Path("results/miniswe_sandboxed_agent")
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
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

    async def responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        owner = request.session[SESSION_ID_KEY]
        rollout_id = current_rollout_id()
        if rollout_id is None:
            matches = [key for key in self._sessions if key[0] == owner]
            if len(matches) > 1:
                raise HTTPException(
                    409, "Multiple mini-SWE rollouts for this session; use /ng-rollout/<id>/v1/responses"
                )
            state = self._sessions.get(matches[0]) if matches else None
        else:
            state = self._sessions.get((owner, rollout_id))
        if state is None:
            raise HTTPException(409, "No seeded mini-SWE sandbox for this session")
        if not state.ready or state.closing:
            raise HTTPException(409, "mini-SWE session is not ready or is closing")
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
                            workdir = seed.sandbox_access.workdir if seed.sandbox_access is not None else seed.workdir
                            if workdir is not None and not workdir.startswith("/"):
                                raise ValueError("The resource sandbox workdir must be absolute")
                            if workdir is None:
                                cwd = await sandbox.exec("pwd", timeout_s=30, user=seed.user)
                                if cwd.return_code or not cwd.stdout.strip().startswith("/"):
                                    raise RuntimeError("Unable to determine the task working directory")
                                workdir = cwd.stdout.strip()
                            context = HarnessContext(
                                session_id=seed.session_id,
                                task_id=seed.task_id,
                                rollout_id=rollout_id,
                                instruction=task_instruction(params),
                                user=seed.user,
                                workdir=workdir,
                                setup_timeout_sec=self.config.setup_timeout_sec,
                                mcp_servers=seed.mcp_servers,
                                skills_dir=seed.skills_dir,
                            )

                            global_config = getattr(self.server_client, "global_config_dict", None)
                            harness = MiniSWEHarness(
                                sandbox=sandbox,
                                context=context,
                                config=self.config.harness,
                                observability_enabled=isinstance(global_config, Mapping)
                                and bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False)),
                                params=params,
                                model_base_url=self.base_url_for_run(
                                    base_url=self.config.sandbox_model_base_url
                                    or get_server_url(self.config.model_server.name),
                                    body={"_ng_rollout_id": rollout_id},
                                )
                                + "/v1",
                                model_name=self.config.model_server.name,
                                directory=state.artifact_directory or self.config.artifacts_dir / seed.session_id,
                            )
                            state.harness = harness
                            await harness.setup()
                        timings["agent_setup"]["finished_at"] = now()
                        timings["agent_execution"] = {"started_at": now()}
                        budget = min(seed.agent_timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
                        deadline = monotonic() + budget
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
        return state.result.response

    async def _attach_session(
        self,
        key: tuple[str, str],
        seed: SeedSessionResponse,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        rollout_id: str,
        artifact_directory: Path | None,
        capture_model_calls: bool = False,
        setup_deadline: float | None = None,
        setup_started_at: str | None = None,
    ) -> MiniSWESession:
        """Borrow a sandbox, retaining partial connection state for close."""
        if key in self._sessions:
            raise HTTPException(409, "Agent session already has an active mini-SWE sandbox")
        state = MiniSWESession(
            sandbox=None,
            seed=seed,
            original_params=params.model_copy(deep=True),
            rollout_id=rollout_id,
            capture_model_calls=capture_model_calls,
            artifact_directory=artifact_directory,
            setup_deadline=setup_deadline or monotonic() + self.config.setup_timeout_sec,
            setup_started_at=setup_started_at or now(),
        )
        self._sessions[key] = state
        async with asyncio.timeout_at(state.setup_deadline):
            global_config = getattr(self.server_client, "global_config_dict", None)
            if seed.sandbox_access is not None:
                connection = seed.sandbox_access.connection
                provider_config = resolve_provider_config(connection.provider_config_ref, global_config)
                descriptor = connection.descriptor
            else:
                # Older Resources servers still return inline provider settings.
                provider_config = resolve_provider_config(seed.sandbox_provider)
                descriptor = seed.sandbox_descriptor
            state.provider = create_provider(provider_config)
            state.sandbox = await AsyncSandbox.connect(descriptor, provider=state.provider)
            state.ready = True
        return state

    async def _close_session(self, key: tuple[str, str]) -> None:
        """Stop execution and disconnect; failed close retains state for retry."""
        state = self._sessions.get(key)
        if state is None:
            return
        state.closing = True
        async with state.close_lock:
            if self._sessions.get(key) is not state:
                return
            if state.worker is not None:
                if not state.worker.done():
                    state.worker.cancel()
                try:
                    state.result = await asyncio.shield(state.worker)
                except asyncio.CancelledError:
                    if not state.worker.cancelled():
                        raise
                except Exception:
                    LOGGER.exception("mini-SWE execution failed before close")
            if state.harness is not None:
                await state.harness.close()
            if state.provider is not None:
                await state.provider.aclose()
            self._sessions.pop(key)

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
                headers={"Idempotency-Key": uuid5(NAMESPACE_URL, f"{payload['client_session_id']}:{payload['rollout_id']}").hex},
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
        state = None
        key = (request.session[SESSION_ID_KEY], payload["rollout_id"])
        # Reject before claiming ownership: a competing run must not close another session.
        if key in self._sessions:
            raise HTTPException(409, "Agent session already has an active mini-SWE sandbox")
        try:
            if not cancelled and seed.termination is None:
                setup_started_at = now()
                setup_deadline = monotonic() + self.config.setup_timeout_sec
                result.agent_timings["agent_setup"] = {"started_at": setup_started_at}
                state = await self._attach_session(
                    key,
                    seed,
                    params,
                    rollout_id=payload.get("_ng_rollout_id") or payload["rollout_id"],
                    capture_model_calls=capture_model_calls,
                    artifact_directory=Path(payload["artifact_directory"])
                    if payload.get("artifact_directory")
                    else None,
                    setup_deadline=setup_deadline,
                    setup_started_at=setup_started_at,
                )
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
            for timing in result.agent_timings.values():
                timing.setdefault("finished_at", now())
            try:
                closing = asyncio.create_task(self._close_session(key))
                while not closing.done():
                    try:
                        await asyncio.shield(closing)
                    except asyncio.CancelledError:
                        result.termination = HarnessOutcome(reason="cancelled")
                closing.result()
            except Exception as exc:
                # No /verify until cleanup is acknowledged. Resources owns expiry
                # and sandbox destruction if this process cannot recover its transport.
                raise HTTPException(503, f"mini-SWE close failed; verification blocked: {exc}") from exc
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
