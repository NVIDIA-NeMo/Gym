# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned TB4 HTTP handoff backed by resource-owned native operations."""

import asyncio
import hashlib
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ClassVar, Literal
from uuid import uuid4

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig, ReverifyMode, SimpleResourcesServer
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.environment import EnvironmentConfig
from resources_servers.terminal_bench_4.handoff import (
    AgentTermination,
    SandboxedSeedResponse,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    SessionRequest,
)
from resources_servers.terminal_bench_4.lifecycle import NATIVE_VERSION, Session
from resources_servers.terminal_bench_4.task import PackageLoader


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "terminal_bench_4"


class TerminalBench4Config(BaseResourcesServerConfig):
    num_workers: Literal[1] = 1
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    manifest_path: Path = BENCHMARK / "manifest.json"
    artifacts_dir: Path = Path("results/terminal_bench_4/resources")
    environment: EnvironmentConfig
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    max_concurrent_sessions: int = Field(default=8, gt=0)
    shutdown_timeout_sec: float = Field(default=30, ge=0)
    task_download_dir: Path | None = None


class TerminalBench4SeedRequest(BaseModel):
    task_name: str
    task_ref: str
    dataset_ref: str
    rollout_id: str = Field(min_length=1, max_length=256)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    execution_id: str | None = Field(default=None, min_length=1, max_length=256)


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
        app.post("/start_session")(self.start_session)
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
                "record_version": 1,
                "runtime": "gym-tb4-native",
                "runtime_version": NATIVE_VERSION,
                "session_id": session.session_id,
                "owner": session.owner,
                "request": session.request.model_dump(),
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

    async def seed_session(self, request: Request, body: TerminalBench4SeedRequest) -> SandboxedSeedResponse:
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
                raise HTTPException(
                    409, "Recorded episode cannot be resumed; use its session ID to retry verification"
                )
            session_id = "tb4-" + uuid4().hex
            session = self._new_session(identity, owner, body, session_id)
            self._by_identity[identity] = session_id
            self._sessions[session_id] = session
            session.persist()
            session.preparation = asyncio.create_task(lifecycle.prepare_session(session, self._loader))
        session = self._sessions[session_id]
        if session.request != body:
            raise HTTPException(409, "Rollout identity is already bound to another task or worker execution")
        await asyncio.shield(session.preparation)
        if session.phase == "closed" or session.finalization is not None or session.seed is None:
            raise HTTPException(409, {"message": "Episode closed", "result": session.result})
        return session.seed

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
                if state.get("record_version", 0) not in (0, 1):
                    raise HTTPException(409, "Unsupported recorded episode version")
                # Both native and legacy closed records retain the small result
                # subset used below. No Harbor result model is needed for replay.
                session = self._new_session(
                    state["identity"],
                    state["owner"],
                    TerminalBench4SeedRequest.model_validate(state["request"]),
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

    async def start_session(self, request: Request, body: SessionRequest) -> dict:
        return await lifecycle.start_session(self._session(request, body.session_id))

    async def cancel_session(self, request: Request, body: SessionRequest) -> dict:
        session = self._session(request, body.session_id)
        finish = await lifecycle.request_finish(session, AgentTermination(reason="cancelled"))
        if finish is not None:
            await asyncio.shield(finish)
        return {"session_id": body.session_id, "phase": "closed"}

    async def verify(self, request: Request, body: SandboxedVerifyRequest) -> SandboxedVerifyResponse:
        session = self._session(request, body.session_id)
        if session.verified_response is not None:
            return session.verified_response
        finish = await lifecycle.request_finish(session, body.termination, body)
        if finish is not None:
            await asyncio.shield(finish)
        async with session.lock:
            if session.verified_response is not None:
                return session.verified_response
            result = session.result or {}
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            completed = "reward" in rewards
            termination = session.termination or body.termination
            failure = None
            if not completed:
                failure = (result.get("exception_info") or {}).get("exception_type", "MissingOfficialReward")
            elif termination.reason == "infrastructure_error":
                failure = termination.detail or "Agent infrastructure failure"
            session.verified_response = SandboxedVerifyResponse(
                **session.verify_body.model_dump(exclude={"termination"}),
                reward=float(rewards.get("reward", 0)),
                evaluation_completed=completed,
                termination=termination,
                infrastructure_error=failure,
                failure_reason=failure,
                artifacts={"trial": str(session.directory)},
                timings={
                    key: result.get(key) for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
                },
                **({"_ng_failure_class": "infrastructure_error"} if failure else {}),
            )
            session.persist()
            return session.verified_response


if __name__ == "__main__":
    TerminalBench4ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4ResourcesServer.run_webserver()
