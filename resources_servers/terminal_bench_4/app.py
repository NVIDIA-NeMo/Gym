# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One resource-owned runner for provisioning, mini-SWE execution, grading, and cleanup."""

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic, time
from typing import ClassVar, Literal
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponse
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.server_utils import (
    SESSION_ID_KEY,
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
    rollout_path_prefix,
)
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.environment import EnvironmentConfig
from resources_servers.terminal_bench_4.lifecycle import NATIVE_VERSION, Session
from resources_servers.terminal_bench_4.models import (
    AgentTermination,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    SessionRequest,
)
from resources_servers.terminal_bench_4.task import PackageLoader
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig, MiniSWEHarness


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "terminal_bench_4"


class TerminalBench4Config(BaseResourcesServerConfig):
    num_workers: Literal[1] = 1
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    manifest_path: Path = BENCHMARK / "manifest.json"
    artifacts_dir: Path = Path("results/terminal_bench_4/resources")
    environment: EnvironmentConfig
    model_server: ModelServerRef
    harness: MiniSWEConfig = Field(default_factory=MiniSWEConfig)
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    max_concurrent_sessions: int = Field(default=8, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)
    task_download_dir: Path | None = None


class TerminalBench4RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_ref: str
    dataset_ref: str
    rollout_id: str = Field(min_length=1, max_length=256)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    capture_model_calls: bool = False
    capture_token_ids: bool = False
    artifact_directory: str | None = Field(default=None, min_length=1)


def empty_response(params, model):
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


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class TerminalBench4ResourcesServer(SimpleResourcesServer):
    config: TerminalBench4Config

    def model_post_init(self, context):
        super().model_post_init(context)
        self._manifest = json.loads(self.config.manifest_path.read_text())
        self._tasks = {"terminal-bench/" + task["name"]: task for task in self._manifest["tasks"]}
        self._sessions: dict[str, Session] = {}
        self._by_identity: dict[str, str] = {}
        self._slots = asyncio.Semaphore(self.config.max_concurrent_sessions)
        self._loader = PackageLoader(self.config.task_download_dir)
        self._closing = False
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def setup_webserver(self):
        app = super().setup_webserver()
        app.post("/run")(self.run)
        app.post("/cancel_session")(self.cancel_session)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                self._closing = True
                await lifecycle.shutdown(list(self._sessions.values()), self.config.shutdown_timeout_sec)

        app.router.lifespan_context = lifespan
        return app

    def _owner(self, request):
        return hashlib.sha256(
            request.session.get("tb4_client_session_id", request.session[SESSION_ID_KEY]).encode()
        ).hexdigest()

    def _state_path(self, identity):
        return self.config.artifacts_dir / f"{identity}.json"

    def _persist(self, session):
        resources = []
        for env in (session.environment, session.verifier_environment, session.shared_logs):
            if env is not None:
                resources.append(
                    {
                        "session_id": env.session_id,
                        "closed": env.closed,
                        "resources": env.resources or env.resource_identities(),
                        "cleanup_errors": env.cleanup_errors,
                    }
                )
        atomic_json(
            self._state_path(session.identity),
            {
                "record_version": 2,
                "runtime": "gym-tb4-native",
                "runtime_version": NATIVE_VERSION,
                "session_id": session.session_id,
                "owner": session.owner,
                "request": session.request.model_dump() | {"_ng_rollout_id": session.request.capture_rollout_id},
                "phase": session.phase,
                "subphase": session.subphase,
                "result": session.result,
                "identity": session.identity,
                "deadlines": session.deadlines,
                "termination": session.termination.model_dump() if session.termination else None,
                "verify_body": session.verify_body.model_dump(mode="json") if session.verify_body else None,
                "verified_response": session.verified_response.model_dump(mode="json")
                if session.verified_response
                else None,
                "resources": resources or session.recorded_resources,
                "diagnostics": session.diagnostics,
            },
        )
        lookup = self.config.artifacts_dir / f"{session.session_id}.state"
        temporary = lookup.with_suffix(".tmp")
        temporary.write_text(session.identity)
        temporary.replace(lookup)
        if session.directory.exists():
            atomic_json(session.directory / "result.json", session.result)

    def _new_session(self, identity, owner, body, session_id, **kwargs):
        kwargs.setdefault("result", {"runtime": "gym-tb4-native", "runtime_version": NATIVE_VERSION})
        session = Session(identity, owner, body, session_id, self.config.artifacts_dir / session_id, **kwargs)
        session.slots = self._slots
        session.config = self.config
        session.persist = lambda: self._persist(session)
        return session

    async def run(self, request: Request, body: TerminalBench4RunRequest) -> SandboxedVerifyResponse:
        if self._closing:
            raise HTTPException(503, "Resources server is shutting down")
        task = self._tasks.get(body.task_name)
        if task is None or body.task_ref != task["ref"] or body.dataset_ref != self._manifest["ref"]:
            raise HTTPException(422, "Task identity does not match the configured dataset pin")
        if body.client_session_id:
            request.session["tb4_client_session_id"] = body.client_session_id
        owner = self._owner(request)
        identity = hashlib.sha256(f"{owner}:{body.rollout_id}".encode()).hexdigest()
        session_id = self._by_identity.get(identity)
        if session_id is None:
            if self._state_path(identity).exists():
                state = json.loads(self._state_path(identity).read_text())
                session = self._session(request, state["session_id"])
            else:
                session_id = "tb4-" + uuid4().hex
                session = self._new_session(identity, owner, body.model_copy(deep=True), session_id)
                self._sessions[session_id] = session
                self._by_identity[identity] = session_id
                session.persist()
                session.execution = asyncio.create_task(self._run_session(session, dict(request.cookies)))
        else:
            session = self._sessions[session_id]
        if session.request != body:
            raise HTTPException(409, "Rollout identity is already bound to another request")
        if session.execution is not None:
            # HTTP retries/disconnects share a single runner; they never start a second harness.
            await asyncio.shield(session.execution)
        if session.verified_response is None:
            raise HTTPException(409, "Episode has no recorded response")
        return session.verified_response

    async def _run_session(self, session, cookies):
        session.started.set()
        body = session.request.model_copy(deep=True)
        response = empty_response(body.responses_create_params, self.config.model_server.name)
        extra = {}
        grade = False
        session.termination = AgentTermination(reason="infrastructure_error", detail="Setup did not complete")
        with rollout_context(body.capture_rollout_id):
            try:
                await lifecycle.prepare_session(session, self._loader)
                session.phase = "agent_setup"
                session.result["agent_setup"] = {"started_at": lifecycle.now()}
                session.deadlines["setup_started_at"] = lifecycle.now()
                session.persist()
                # Workdir discovery and harness setup share one budget; queueing/provisioning do not.
                async with asyncio.timeout(lifecycle.SETUP_TIMEOUT_SEC):
                    context = HarnessContext(
                        session_id=session.session_id,
                        task_id=body.task_name,
                        rollout_id=body.capture_rollout_id or body.rollout_id,
                        instruction=session.task.instruction,
                        user=session.task.config.agent.user,
                        workdir=await session.environment.agent_workdir(),
                        setup_timeout_sec=lifecycle.SETUP_TIMEOUT_SEC,
                        mcp_servers=[s.model_dump() for s in session.task.config.environment.mcp_servers],
                        skills_dir=session.task.config.environment.skills_dir,
                    )
                    body.responses_create_params.input = [
                        NeMoGymEasyInputMessage(role="user", content=context.instruction)
                    ]

                    async def query(params):
                        prefix = rollout_path_prefix(
                            body.capture_rollout_id if body.capture_model_calls else None,
                            token_capture=body.capture_token_ids,
                        )
                        model_response = await self.server_client.post(
                            server_name=self.config.model_server.name,
                            url_path=prefix + "/v1/responses",
                            json=params,
                            # Exact invocation ownership, including transport
                            # failures/retries that have no model response ID.
                            headers={"x-session-id": session.session_id},
                            cookies=cookies,
                        )
                        await raise_for_status(model_response)
                        return NeMoGymResponse.model_validate(await get_response_json(model_response))

                    global_config = getattr(self.server_client, "global_config_dict", None)
                    harness = MiniSWEHarness(
                        sandbox=session.environment.main,
                        context=context,
                        config=self.config.harness,
                        observability_enabled=isinstance(global_config, Mapping)
                        and bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False)),
                        params=body.responses_create_params,
                        query=query,
                        model_name=self.config.model_server.name,
                        directory=Path(body.artifact_directory)
                        if body.artifact_directory
                        else session.directory / "harness",
                    )
                    await harness.setup()
                session.result["agent_setup"]["finished_at"] = lifecycle.now()
                session.phase = "agent_running"
                session.result["agent_execution"] = {"started_at": lifecycle.now()}
                session.deadlines["agent_started_at"] = lifecycle.now()
                budget = min(session.task.config.agent.timeout_sec, self.config.agent_max_timeout_sec or float("inf"))
                deadline = monotonic() + budget
                session.persist()
                grade = True
                response, outcome, extra = await harness.execute(max(0, deadline - monotonic()))
                session.termination = AgentTermination.model_validate(outcome.model_dump())
                if monotonic() >= deadline:
                    session.termination.reason = "timeout"
                if session.termination.reason == "infrastructure_error":
                    lifecycle.exception(session, session.termination.detail, "AgentInfrastructureError")
            except asyncio.CancelledError:
                session.termination = AgentTermination(reason="cancelled")
                lifecycle.exception(session, "Episode cancelled", "CancelledError")
            except Exception as exc:
                setup_timeout = isinstance(exc, TimeoutError) and session.phase == "agent_setup"
                session.termination = AgentTermination(
                    reason="timeout" if setup_timeout else "infrastructure_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                lifecycle.exception(session, exc, "AgentSetupTimeoutError" if setup_timeout else None)
            finally:
                if grade:
                    session.result["agent_execution"]["finished_at"] = lifecycle.now()
                session.verify_body = SandboxedVerifyRequest(
                    **body.model_dump(),
                    session_id=session.session_id,
                    response=response,
                    termination=session.termination,
                )
                try:
                    session.directory.mkdir(parents=True, exist_ok=True)
                    (session.directory / "gym-agent.json").write_text(session.verify_body.model_dump_json(indent=2))
                except OSError as exc:
                    lifecycle.exception(session, exc, "AgentRecordError")
                session.phase = "verifying" if grade else "cleaning"
                session.finalization = asyncio.create_task(lifecycle.finalize_session(session, grade=grade))
                await asyncio.shield(session.finalization)
                session.verified_response = self._verified_response(session, extra)
                session.persist()

    def _verified_response(self, session, extra):
        result = session.result or {}
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        completed = "reward" in rewards
        failure = None
        if not completed:
            failure = (result.get("exception_info") or {}).get("exception_type", "MissingOfficialReward")
        elif session.termination.reason == "infrastructure_error":
            failure = session.termination.detail or "Agent infrastructure failure"
        return SandboxedVerifyResponse(
            **(
                session.verify_body.model_dump(exclude={"termination"})
                | extra
                | {"task_id": session.request.task_name}
            ),
            reward=float(rewards.get("reward", 0)),
            evaluation_completed=completed,
            termination=session.termination,
            infrastructure_error=failure,
            failure_reason=failure,
            artifacts={"trial": str(session.directory)},
            timings={
                key: result.get(key) for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
            },
            **({"_ng_failure_class": "infrastructure_error"} if failure else {}),
        )

    def _session(self, request, session_id):
        session = self._sessions.get(session_id)
        if session is None:
            if not re.fullmatch(r"tb4-[a-f0-9]{32}", session_id):
                raise HTTPException(404, "Unknown session")
            lookup = self.config.artifacts_dir / f"{session_id}.state"
            if lookup.exists():
                identity = lookup.read_text()
                if not re.fullmatch(r"[a-f0-9]{64}", identity):
                    raise HTTPException(409, "Invalid recorded episode identity")
                state = json.loads(self._state_path(identity).read_text())
                if state["owner"] != self._owner(request):
                    raise HTTPException(404, "Unknown session")
                if state["phase"] != "closed":
                    raise HTTPException(
                        409, "Resources process restarted; episode cannot resume; provider TTL applies"
                    )
                if state.get("record_version", 0) != 2:
                    raise HTTPException(409, "Unsupported recorded episode version")
                session = self._new_session(
                    state["identity"],
                    state["owner"],
                    TerminalBench4RunRequest.model_validate(state["request"]),
                    session_id,
                    phase="closed",
                    result=state.get("result") or {},
                    deadlines=state.get("deadlines") or {},
                    diagnostics=state.get("diagnostics") or [],
                    recorded_resources=state.get("resources") or [],
                )
                if state.get("termination"):
                    session.termination = AgentTermination.model_validate(state["termination"])
                if state.get("verify_body"):
                    session.verify_body = SandboxedVerifyRequest.model_validate(state["verify_body"])
                if state.get("verified_response"):
                    session.verified_response = SandboxedVerifyResponse.model_validate(state["verified_response"])
                self._sessions[session_id] = session
        if session is None or session.owner != self._owner(request):
            raise HTTPException(404, "Unknown session")
        return session

    async def cancel_session(self, request: Request, body: SessionRequest) -> dict:
        session = self._session(request, body.session_id)
        if session.execution is not None and not session.execution.done():
            await session.started.wait()
            if session.finalization is None and not session.execution.cancelling():
                session.execution.cancel()
            await asyncio.shield(session.execution)
        return {"session_id": body.session_id, "phase": session.phase}

    async def verify(self, request: Request, body: SandboxedVerifyRequest) -> SandboxedVerifyResponse:
        session = self._session(request, body.session_id)
        if session.verified_response is None:
            raise HTTPException(409, "Episode has no recorded response; the runner owns verification")
        return session.verified_response


if __name__ == "__main__":
    TerminalBench4ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4ResourcesServer.run_webserver()
