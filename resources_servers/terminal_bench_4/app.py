# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned TB4 task provisioning and verification, independent of the harness."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar, Literal
from uuid import uuid4

from fastapi import HTTPException, Request
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig, ReverifyMode, SimpleResourcesServer
from nemo_gym.sandbox.handoff import (
    AgentTermination,
    SandboxedSeedResponse,
    SandboxedVerifyRequest,
    SandboxedVerifyResponse,
    SessionRequest,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_4.runtime import ExternalEpisode


BENCHMARK = Path(__file__).resolve().parents[2] / "benchmarks" / "terminal_bench_4"


class TerminalBench4Config(BaseResourcesServerConfig):
    num_workers: Literal[1] = 1
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    manifest_path: Path = BENCHMARK / "manifest.json"
    artifacts_dir: Path = Path("results/terminal_bench_4/resources")
    environment: dict[str, Any]
    # An explicit smoke cap may shorten, but never extend the official budget.
    agent_max_timeout_sec: float | None = Field(default=None, gt=0)
    max_concurrent_sessions: int = Field(default=8, gt=0)
    task_download_dir: Path | None = None


class TerminalBench4SeedRequest(BaseModel):
    task_name: str
    task_ref: str
    dataset_ref: str
    rollout_id: str = Field(min_length=1, max_length=256)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    execution_id: str | None = Field(default=None, min_length=1, max_length=256)


@dataclass
class Session:
    identity: str
    owner: str
    request: TerminalBench4SeedRequest
    episode: ExternalEpisode
    task: asyncio.Task | None = None
    trial: Any = None
    seed: SandboxedSeedResponse | None = None
    result: dict | None = None
    verify_body: SandboxedVerifyRequest | None = None
    verified_response: SandboxedVerifyResponse | None = None


class TerminalBench4ResourcesServer(SimpleResourcesServer):
    config: TerminalBench4Config

    def model_post_init(self, context):
        super().model_post_init(context)
        self._manifest = json.loads(self.config.manifest_path.read_text())
        self._tasks = {"terminal-bench/" + task["name"]: task for task in self._manifest["tasks"]}
        self._sessions: dict[str, Session] = {}
        self._by_identity: dict[str, str] = {}
        self._slots = asyncio.Semaphore(self.config.max_concurrent_sessions)
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
                tasks = [s.task for s in self._sessions.values() if s.task and not s.task.done()]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        app.router.lifespan_context = lifespan
        return app

    def _owner(self, request):
        return hashlib.sha256(
            request.session.get("tb4_client_session_id", request.session[SESSION_ID_KEY]).encode()
        ).hexdigest()

    def _state_path(self, identity):
        return self.config.artifacts_dir / f"{identity}.json"

    def _persist(self, session_id, session):
        path = self._state_path(session.identity)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "owner": session.owner,
                    "request": session.request.model_dump(),
                    "phase": session.episode.phase,
                    "result": session.result,
                    "identity": session.identity,
                    "termination": session.episode.termination.model_dump() if session.episode.termination else None,
                    "verified_response": session.verified_response.model_dump(mode="json")
                    if session.verified_response
                    else None,
                },
                indent=2,
            )
        )
        temporary.replace(path)
        # UUID-derived lookup for verify retries after a process restart.
        (self.config.artifacts_dir / f"{session_id}.state").write_text(session.identity)

    async def _execute(self, session_id, session):
        session.episode.on_phase_change = lambda: self._persist(session_id, session)
        try:
            async with self._slots:
                config = TrialConfig.model_validate(
                    {
                        "trial_name": session_id,
                        "trials_dir": str(self.config.artifacts_dir),
                        "task": {
                            "name": session.request.task_name,
                            "ref": session.request.task_ref,
                            "download_dir": str(self.config.task_download_dir)
                            if self.config.task_download_dir
                            else None,
                        },
                        "agent": {
                            "import_path": "resources_servers.terminal_bench_4.runtime:ExternalAgent",
                            "max_timeout_sec": self.config.agent_max_timeout_sec,
                        },
                        "environment": self.config.environment,
                    }
                )
                trial = session.trial = await Trial.create(config)
                if trial.task.has_steps:
                    raise ValueError("TB4 external episodes require single-step task packages")
                trial.agent.episode = session.episode
                trial.agent.session_id = session_id
                result = await trial.run()
                session.result = result.model_dump(mode="json")
        except asyncio.CancelledError:
            session.episode.termination = session.episode.termination or AgentTermination(reason="cancelled")
            raise
        except Exception as exc:
            session.result = {"exception_info": {"exception_type": type(exc).__name__, "exception_message": str(exc)}}
        finally:
            session.episode.phase = "closed"
            self._persist(session_id, session)

    async def _wait_event(self, session, event):
        if session.task is None or session.episode.phase == "closed":
            raise HTTPException(409, {"message": "Episode closed", "result": session.result})
        waiter = asyncio.create_task(event.wait())
        try:
            await asyncio.wait([waiter, session.task], return_when=asyncio.FIRST_COMPLETED)
            if session.task.done():
                raise HTTPException(409, {"message": "Episode closed", "result": session.result})
        finally:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def seed_session(self, request: Request, body: TerminalBench4SeedRequest) -> SandboxedSeedResponse:
        task = self._tasks.get(body.task_name)
        if task is None or body.task_ref != task["ref"] or body.dataset_ref != self._manifest["ref"]:
            raise HTTPException(422, "Task identity does not match the configured dataset pin")
        if body.client_session_id:
            # Stable across retries before the first response has returned a
            # resources cookie. This token belongs to the agent, not dataset rows.
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
            session = Session(identity, owner, body, ExternalEpisode())
            self._by_identity[identity] = session_id
            self._sessions[session_id] = session
            self._persist(session_id, session)
            session.task = asyncio.create_task(self._execute(session_id, session))
        session = self._sessions[session_id]
        if session.request != body:
            raise HTTPException(409, "Rollout identity is already bound to another task or worker execution")
        await self._wait_event(session, session.episode.prepared)
        if session.seed is None:
            trial = session.trial
            env = session.episode.environment
            session.seed = SandboxedSeedResponse(
                session_id=session_id,
                sandbox=await env.main_connection(),
                instruction=trial.task.instruction,
                user=trial.task.config.agent.user,
                agent_timeout_sec=min(
                    trial.task.config.agent.timeout_sec,
                    self.config.agent_max_timeout_sec or float("inf"),
                ),
                setup_timeout_sec=360,
                mcp_servers=[server.model_dump() for server in trial.agent.mcp_servers],
                skills_dir=trial.agent.skills_dir,
            )
        return session.seed

    def _session(self, request, session_id):
        session = self._sessions.get(session_id)
        if session is None:
            import re

            if not re.fullmatch(r"tb4-[a-f0-9]{32}", session_id):
                raise HTTPException(404, "Unknown session")
            lookup = self.config.artifacts_dir / f"{session_id}.state"
            if lookup.exists():
                state = json.loads(self._state_path(lookup.read_text()).read_text())
                if state["owner"] != self._owner(request):
                    raise HTTPException(404, "Unknown session")
                if state["phase"] != "closed":
                    raise HTTPException(
                        409, "Resources process restarted; episode cannot resume; provider TTL applies"
                    )
                session = Session(
                    state["identity"],
                    state["owner"],
                    TerminalBench4SeedRequest.model_validate(state["request"]),
                    ExternalEpisode(phase="closed"),
                    result=state["result"],
                )
                if state.get("termination"):
                    session.episode.termination = AgentTermination.model_validate(state["termination"])
                if state.get("verified_response"):
                    session.verified_response = SandboxedVerifyResponse.model_validate(state["verified_response"])
                self._sessions[session_id] = session
        if session is None or session.owner != self._owner(request):
            raise HTTPException(404, "Unknown session")
        return session

    async def start_session(self, request: Request, body: SessionRequest) -> dict:
        session = self._session(request, body.session_id)
        session.episode.setup_complete.set()
        await self._wait_event(session, session.episode.running)
        return {
            "agent_timeout_sec": max(0, session.seed.agent_timeout_sec - (monotonic() - session.episode.started_at))
        }

    async def cancel_session(self, request: Request, body: SessionRequest) -> dict:
        session = self._session(request, body.session_id)
        if session.episode.phase == "closed":
            return {"session_id": body.session_id, "phase": "closed"}
        session.episode.termination = AgentTermination(reason="cancelled")
        if session.episode.running.is_set():
            session.episode.finished.set()
        else:
            session.task.cancel()
        await asyncio.gather(asyncio.shield(session.task), return_exceptions=True)
        return {"session_id": body.session_id, "phase": "closed"}

    async def verify(self, request: Request, body: SandboxedVerifyRequest) -> SandboxedVerifyResponse:
        session = self._session(request, body.session_id)
        if session.verified_response is not None:
            return session.verified_response
        if session.episode.phase in {"preparing", "ready"}:
            raise HTTPException(409, "Agent setup has not completed")
        if session.verify_body is None:
            session.verify_body = body
            session.episode.termination = session.episode.termination or body.termination
            session.episode.termination.artifacts = list(
                dict.fromkeys(session.episode.termination.artifacts + body.termination.artifacts)
            )
            # Save termination/trajectory references before allowing collection.
            path = self.config.artifacts_dir / body.session_id / "gym-agent.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body.model_dump_json(indent=2))
            session.episode.finished.set()
        if session.task is not None and not session.task.cancelled():
            await asyncio.shield(session.task)
        result = session.result or {}
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        completed = "reward" in rewards
        termination = session.episode.termination or body.termination
        failure = None
        if not completed:
            exception = result.get("exception_info") or {}
            failure = exception.get("exception_type", "MissingOfficialReward")
        elif termination.reason == "infrastructure_error":
            failure = termination.detail or "Agent infrastructure failure"
        response = SandboxedVerifyResponse(
            **session.verify_body.model_dump(exclude={"termination"}),
            reward=float(rewards.get("reward", 0)),
            evaluation_completed=completed,
            termination=termination,
            infrastructure_error=failure,
            failure_reason=failure,
            artifacts={"trial": str(self.config.artifacts_dir / body.session_id)},
            timings={
                key: result.get(key) for key in ("environment_setup", "agent_setup", "agent_execution", "verifier")
            },
            provenance={
                "dataset_ref": session.request.dataset_ref,
                "task_ref": session.request.task_ref,
                "task_name": session.request.task_name,
                "harbor_version": "0.23.0",
            },
            **({"_ng_failure_class": "infrastructure_error"} if failure else {}),
        )
        session.verified_response = response
        self._persist(body.session_id, session)
        return response


if __name__ == "__main__":
    TerminalBench4ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TerminalBench4ResourcesServer.run_webserver()
