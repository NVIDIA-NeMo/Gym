# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EnterpriseOps-Gym resources server.

Each task runs against live enterprise systems from ServiceNow's EnterpriseOps-Gym: for every gym server the task
names, ``/seed_session`` starts a sandbox from that gym's image, waits for its MCP server and seeds a database from
the task's SQL dump. The first sandbox is lent to the agent. The agent calls the task's tools through
``POST /{tool_name}``, which forwards each call to the gym that owns the tool with the task database's ID and the
task's context headers. ``/verify`` runs the task's SQL verifiers against the final database state: the reward is 1
only when every verifier passes, and ``verifier_pass_rate`` is the share that pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import urllib.request
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from nemo_gym import CACHE_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import VERIFIER_ERROR
from nemo_gym.global_config import CACHE_DIR_KEY_NAME, get_global_config_dict
from nemo_gym.reward_profile import compute_pass_majority_metrics, compute_subset_metrics, highest_k_metrics
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import SandboxConfig, resolve_provider_config
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.enterpriseops_gym.gym_client import GymRequestError, GymServerClient, GymUnavailableError
from resources_servers.enterpriseops_gym.task_data import (
    DATABASE_ID_HEADER,
    DatabaseStateVerifier,
    GymServerTask,
    TaskData,
)
from resources_servers.enterpriseops_gym.verifier import (
    FinalStateVerifier,
    FinalStateVerifyRequest,
    QueryRejectedError,
    VerifierResult,
    evaluate,
)


LOG = logging.getLogger(__name__)


class GymImageConfig(BaseModel):
    """How to run one gym server: its image, the command that starts its HTTP server, and the port it serves."""

    model_config = ConfigDict(extra="forbid")

    image: str
    command: List[str] = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class EnterpriseOpsGymResourcesServerConfig(BaseResourcesServerConfig):
    # Names a top-level provider block such as nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml.
    sandbox_provider: str
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig)
    # Keyed by the gym server names the tasks use (`gym_servers[].name`).
    gym_images: Dict[str, GymImageConfig]
    # Working directory of the sandbox lent to the agent.
    agent_workdir: str = "/tmp"
    # The upstream `gym_dbs.zip` of task database dumps, pinned by checksum.
    seed_archive_url: str
    seed_archive_sha256: str
    # Defaults to `<cache_dir>/enterpriseops_gym`.
    seed_archive_dir: Optional[str] = None
    gym_ready_timeout_s: float = Field(default=300, gt=0)
    gym_request_timeout_s: float = Field(default=120, gt=0)
    # Sandbox endpoints of a provider whose proxy certificate cannot be verified need this off.
    verify_endpoint_tls: bool = False


class EnterpriseOpsGymVerifyRequest(BaseVerifyRequest):
    task_id: str
    domain: str


class EnterpriseOpsGymVerifyResponse(EnterpriseOpsGymVerifyRequest, BaseVerifyResponse):
    # False when the database rejected a verifier's query: a defect of the task, not a hard task.
    evaluation_completed: bool
    verifier_pass_rate: float
    num_verifiers: int
    num_verifiers_passed: int
    verifier_results: List[VerifierResult]


@dataclass
class _GymSandbox:
    """A started gym server and the headers that address its task database."""

    sandbox: AsyncSandbox
    client: Optional[GymServerClient] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class _TaskSession:
    episode_id: EpisodeId
    task_id: TaskId
    task: TaskData
    # In the task's order: the first one is lent to the agent.
    gyms: Dict[str, _GymSandbox]
    # Set by the first /verify, which stops the sandboxes; a repeated /verify reports the same results.
    verifier_results: Optional[List[VerifierResult]] = None


class SeedArchive:
    """The upstream archive of task database dumps, read by each dump's path inside it."""

    def __init__(self, path: Path) -> None:
        self._zip = zipfile.ZipFile(path)
        self._names = frozenset(self._zip.namelist())

    def read(self, name: str) -> str:
        if name not in self._names:
            raise ValueError(f"The seed archive has no database dump {name!r}")
        return self._zip.read(name).decode("utf-8")


def ensure_seed_archive(url: str, sha256: str, directory: Path) -> Path:
    """Download the seed archive into ``directory`` once and check its checksum."""
    path = directory / f"gym_dbs-{sha256[:16]}.zip"
    if path.is_file() and _sha256(path) == sha256:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as partial:
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                while chunk := response.read(1 << 20):
                    partial.write(chunk)
            partial.close()
            actual = _sha256(Path(partial.name))
            if actual != sha256:
                raise ValueError(f"Seed archive {url} has sha256 {actual}, expected {sha256}")
            os.replace(partial.name, path)
        finally:
            Path(partial.name).unlink(missing_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class EnterpriseOpsGymResourcesServer(SimpleResourcesServer):
    config: EnterpriseOpsGymResourcesServerConfig
    ray_enabled: ClassVar[bool] = False

    _sessions: Dict[str, _TaskSession] = PrivateAttr(default_factory=dict)
    _fence: ResourcesSessionFence = PrivateAttr()
    _seed_archive: Optional[SeedArchive] = PrivateAttr(default=None)
    _seed_archive_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("EnterpriseOps-Gym sessions are process-local and require num_workers=1")
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        # Last, so the routes registered above win over a tool of the same name.
        app.post("/{tool_name}")(self.call_tool)
        return app

    async def _archive(self) -> SeedArchive:
        """The seed archive, downloaded by the first seed that needs it."""
        async with self._seed_archive_lock:
            if self._seed_archive is None:
                directory = self.config.seed_archive_dir or Path(
                    get_global_config_dict().get(CACHE_DIR_KEY_NAME) or CACHE_DIR, "enterpriseops_gym"
                )
                path = await asyncio.to_thread(
                    ensure_seed_archive, self.config.seed_archive_url, self.config.seed_archive_sha256, Path(directory)
                )
                self._seed_archive = SeedArchive(path)
        return self._seed_archive

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            self._fence.raise_if_closed(session_id)
            session = self._sessions.get(session_id)
            if session is not None:
                if (session.episode_id, session.task_id) != (body.episode_id, body.task_id):
                    raise ResourcesSessionConflictError(
                        "resources_session_id is already bound to another episode or task"
                    )
                response = await self._seed_response(session_id, session)
            else:
                # A task this server cannot run is a terminal 422; a failure to start its gyms is a retryable 5xx.
                try:
                    task = TaskData.model_validate(body.task_data)
                except ValidationError as error:
                    raise ResourcesSeedRejectedError(f"Invalid EnterpriseOps-Gym task: {error}") from error
                if body.task_id.task_id != task.task_id:
                    raise ResourcesSeedRejectedError("TaskId does not match the EnterpriseOps-Gym task_id")
                missing = [gym.name for gym in task.gym_servers if gym.name not in self.config.gym_images]
                if missing:
                    raise ResourcesSeedRejectedError(f"No gym_images entry for gym servers {missing}")
                session = _TaskSession(body.episode_id, body.task_id, task, gyms={})
                try:
                    await self._start_gyms(session)
                    response = await self._seed_response(session_id, session)
                except BaseException:
                    await self._release(session)
                    raise
                self._sessions[session_id] = session
            # Only on success: a refusal must not hand the caller a cookie for a session another episode holds.
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def _start_gyms(self, session: _TaskSession) -> None:
        """Start every gym of the task, wait for it and seed its database, all at once."""
        archive = await self._archive()
        global_config_dict = get_global_config_dict()
        provider_config = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        # Every sandbox is tracked before it starts, so a failure anywhere stops the ones that did. The task's order
        # is kept: the first gym is the one lent to the agent.
        for gym_task in session.task.gym_servers:
            session.gyms[gym_task.name] = _GymSandbox(sandbox=AsyncSandbox(provider_config))

        async def start(gym_task: GymServerTask) -> None:
            image = self.config.gym_images[gym_task.name]
            gym = session.gyms[gym_task.name]
            try:
                sql = await asyncio.to_thread(archive.read, gym_task.seed_database_file)
            except ValueError as error:
                raise ResourcesSeedRejectedError(str(error)) from error
            spec = self.config.sandbox_config.spec(
                self.config.sandbox_provider,
                image=image.image,
                workdir=self.config.agent_workdir,
                metadata={"task_id": session.task.task_id[:63], "gym_server": gym_task.name[:63]},
                named_configs=global_config_dict,
            )
            await gym.sandbox.start(replace(spec, entrypoint=list(image.command), ports=[image.port]))
            gym.client = GymServerClient(
                await gym.sandbox.endpoint(image.port),
                verify_tls=self.config.verify_endpoint_tls,
                timeout_s=self.config.gym_request_timeout_s,
            )
            database_id = f"db_{uuid4().hex}"
            try:
                await gym.client.wait_until_healthy(self.config.gym_ready_timeout_s)
                await gym.client.seed_database(database_id, sql)
            except GymUnavailableError as error:
                raise HTTPException(503, f"Gym server {gym_task.name} is unavailable: {error}") from error
            except GymRequestError as error:
                raise ResourcesSeedRejectedError(
                    f"Gym server {gym_task.name} refused the task database: {error}"
                ) from error
            gym.headers = {**gym_task.headers, DATABASE_ID_HEADER: database_id}

        try:
            async with asyncio.TaskGroup() as group:
                for gym_task in session.task.gym_servers:
                    group.create_task(start(gym_task))
        except BaseExceptionGroup as failures:
            # Answer with one failure, the HTTP error that classifies it when there is one.
            errors = failures.exceptions
            raise next((error for error in errors if isinstance(error, HTTPException)), errors[0]) from failures

    async def _seed_response(self, session_id: str, session: _TaskSession) -> ResourcesSeedSessionResponse:
        host = next(iter(session.gyms.values()))
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    descriptor=await host.sandbox.serialize(),
                ),
                workdir=self.config.agent_workdir,
            ),
        )

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._fence.locked(session_id):
            closed_episode_id = self._fence.closed_episode(session_id)
            if closed_episode_id is None:
                session = self._sessions.get(session_id)
                if session is not None:
                    if session.episode_id != body.episode_id:
                        raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
                    await self._release(session)
                    del self._sessions[session_id]
                self._fence.mark_closed(session_id, body.episode_id)
            elif closed_episode_id != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
        request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _release(self, session: _TaskSession) -> None:
        """Stop the session's sandboxes. One that fails to stop stays in ``session.gyms`` for a later retry."""
        gyms = list(session.gyms.items())
        results = await asyncio.gather(*(gym.sandbox.stop() for _, gym in gyms), return_exceptions=True)
        errors = []
        for (name, _), result in zip(gyms, results):
            if isinstance(result, BaseException):
                errors.append(result)
            else:
                session.gyms.pop(name)
        if errors:
            raise errors[0]

    async def shutdown(self) -> None:
        for session_id, session in list(self._sessions.items()):
            try:
                await self._release(session)
            except Exception:
                LOG.exception("Failed to stop the sandboxes of EnterpriseOps-Gym session %s", session_id)
        self._sessions.clear()

    def _session(self, request: Request) -> Optional[_TaskSession]:
        return self._sessions.get(request.session.get(SESSION_ID_KEY))

    async def call_tool(
        self, request: Request, tool_name: str, arguments: Dict[str, Any] = Body(default_factory=dict)
    ) -> str:
        """Run one of the task's tools on the gym server that owns it.

        A call the tool rejects is a 400 whose detail the model sees; an unreachable gym is a 502, which fails the
        episode as an infrastructure error.
        """
        session = self._session(request)
        if session is None:
            raise ResourcesSessionLostError("No seeded EnterpriseOps-Gym session for this tool call")
        gym_name = session.task.tool_servers.get(tool_name)
        if gym_name is None:
            raise HTTPException(400, f"Tool {tool_name!r} is not available in this task")
        gym = session.gyms.get(gym_name)
        if gym is None or session.verifier_results is not None:
            raise HTTPException(409, "The task's gym servers are no longer running")
        try:
            result = await gym.client.call_tool(tool_name, arguments, headers=gym.headers)
        except GymUnavailableError as error:
            raise HTTPException(502, f"Gym server {gym_name} is unavailable: {error}") from error
        if result.is_error:
            raise HTTPException(400, result.text)
        return result.text

    async def verify(self, request: Request, body: EnterpriseOpsGymVerifyRequest) -> EnterpriseOpsGymVerifyResponse:
        session = self._session(request)
        if session is None:
            # Closed, never seeded here, or gone with a restart: the task databases went with it, so there is nothing
            # to score, and a reward of 0 would read as a model that solved nothing.
            lost = lost_session_verification("No seeded EnterpriseOps-Gym session for this verify")
            return EnterpriseOpsGymVerifyResponse(
                **(body.model_dump() | lost),
                evaluation_completed=False,
                verifier_pass_rate=0.0,
                num_verifiers=0,
                num_verifiers_passed=0,
                verifier_results=[],
            )
        if body.task_id != session.task.task_id:
            raise HTTPException(409, "The verify request names another task than the session")
        if session.verifier_results is None:

            async def run_sql(verifier: DatabaseStateVerifier) -> Dict[str, Any]:
                gym = session.gyms[verifier.gym_name]
                try:
                    return await gym.client.run_sql(verifier.validation_config.query, headers=gym.headers)
                except GymRequestError as error:
                    raise QueryRejectedError(str(error)) from error
                except GymUnavailableError as error:
                    # No verdict without the database: fail the request so the episode is retried.
                    raise HTTPException(502, f"Could not run verifier {verifier.name!r}: {error}") from error

            session.verifier_results = await evaluate(session.task.verifiers, run_sql)
            try:
                await self._release(session)
            except Exception:
                # The results stand; close_session retries the stop.
                LOG.exception("Failed to stop the sandboxes of task %s after verification", session.task.task_id)
        results = session.verifier_results
        passed = sum(result.passed for result in results)
        # A rejected query fails its verifier, as upstream scores it; the response names it so the row can be told
        # apart from a task the model failed.
        rejected = [result for result in results if result.error is not None]
        return EnterpriseOpsGymVerifyResponse(
            **body.model_dump(),
            reward=float(passed == len(results)),
            evaluation_completed=not rejected,
            failure_kind=VERIFIER_ERROR if rejected else None,
            failure_reason="; ".join(f"Verifier {result.name!r}: {result.error}" for result in rejected) or None,
            verifier_pass_rate=passed / len(results),
            num_verifiers=len(results),
            num_verifiers_passed=passed,
            verifier_results=results,
        )

    @staticmethod
    def _scores(rollout: Dict[str, Any]) -> Dict[str, float]:
        return {"success": rollout["reward"], "verifier_pass_rate": rollout.get("verifier_pass_rate", 0.0)}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._scores)
        metrics.pop("per_sample_aggregate", None)
        metrics.update(compute_subset_metrics(tasks, subset_key="domain", score_fn=self._scores))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key = {
            name: agent_metrics[name] for name in ("mean/input_tokens", "mean/output_tokens") if name in agent_metrics
        }
        key.update(
            highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["success", "verifier_pass_rate"])
        )
        return key


# Exercises the server's scoring against final database states built in SQLite: the verifiers need no gym servers.
VERIFIER_FIXTURE = VerifierFixture(
    server_factory=FinalStateVerifier,
    request_model=FinalStateVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    EnterpriseOpsGymResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = EnterpriseOpsGymResourcesServer.run_webserver()  # noqa: F401
