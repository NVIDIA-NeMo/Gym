# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources server for Harbor-format tasks.

One server instance serves one or more tasksets. Each taskset maps to a folder of
task folders in Gym's shared datasets folder plus the digest every task had when its
rows were materialized (see ``nemo_gym.tasks.harbor``).

- ``/seed_session`` checks the row's digest against the mapping and the folder,
  starts the task's image as a sandbox, creates the working directory, and hands the
  agent a ``SandboxAccess``.
- ``/verify`` runs ``tests/test.sh`` inside that same sandbox (Harbor's shared
  verifier mode), then reads ``/logs/verifier/reward.json`` or ``reward.txt``.
- ``/close_session`` stops the sandbox.

Reward rule: when ``test.sh`` ran, the sample is measured. A missing or invalid reward
file scores 0 with a ``failure_kind``. Only a Gym-side failure (sandbox lost, transfer
failed) masks the sample.
"""

import asyncio
import json
import logging
import math
import shlex
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from traceback import format_exc
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import failure_kinds
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.single_agent_turn_types import SingleAgentTurnResourcesVerifyRequest
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.task import HarborTaskError
from resources_servers.harbor.sandbox_io import download_dir, upload_dir


LOGGER = logging.getLogger(__name__)

DEFAULT_WORKDIR = "/app"
TESTS_DIR = "/tests"
VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"

# Namespaced failure kinds for outcomes the shared vocabulary does not name.
VERIFIER_TIMEOUT_KIND = "harbor:verifier_timeout"
MISSING_REWARD_KIND = "harbor:missing_reward"
INVALID_REWARD_KIND = "harbor:invalid_reward"


class HarborTasksetConfig(BaseModel):
    """Where one taskset's folders live and which digest each task was materialized with."""

    model_config = ConfigDict(extra="forbid")

    folder: Path
    tasks: dict[str, str]


class HarborResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    num_workers: Literal[1] = 1
    tasksets: dict[str, HarborTasksetConfig] = Field(default_factory=dict)
    # Name of the top-level sandbox provider block in the merged config.
    sandbox_provider: str = "sandbox"
    sandbox_ready_timeout_s: float = Field(default=900, gt=0)
    # Sandbox lifetime = agent timeout + verifier timeout + this slack.
    sandbox_ttl_slack_s: float = Field(default=900, ge=0)
    sandbox_provider_options: dict[str, Any] = Field(default_factory=dict)
    sandbox_metadata: dict[str, str] = Field(default_factory=dict)
    # Verifier logs are downloaded here, one folder per resources session.
    artifacts_dir: Path = Path("results/harbor/verifier")


@dataclass
class HarborSession:
    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str


def parse_reward_file(directory: Path) -> tuple[dict[str, float] | None, str | None]:
    """Read Harbor's reward file.

    Returns ``(rewards, problem)``: ``rewards`` is the parsed mapping (``reward.txt``
    becomes ``{"reward": value}``) or ``None``; ``problem`` names what went wrong.
    """
    json_path, text_path = directory / "reward.json", directory / "reward.txt"
    path = json_path if json_path.is_file() else text_path
    if not path.is_file():
        return None, "no reward.json or reward.txt was written"
    text = path.read_text().strip()
    if not text:
        return None, f"{path.name} is empty"
    try:
        raw = json.loads(text) if path == json_path else {"reward": float(text)}
    except ValueError as exc:
        return None, f"{path.name} is not valid: {exc}"
    if not isinstance(raw, dict) or not raw:
        return None, f"{path.name} must hold a non-empty JSON object"
    rewards: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, f"{path.name}: {key!r} is not a finite number"
        rewards[str(key)] = float(value)
    return rewards, None


def select_reward(rewards: dict[str, float]) -> float | None:
    """Harbor's rule: the ``reward`` key, else the only key. Several keys and none named ``reward`` is ambiguous."""
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


def _sandbox_resources(task: HarborTask) -> dict[str, Any]:
    environment = task.config.environment
    resources: dict[str, Any] = {}
    if environment.cpus is not None:
        resources["cpu"] = environment.cpus
    if environment.memory_mb is not None:
        resources["memory_mib"] = environment.memory_mb
    if environment.storage_mb is not None:
        resources["disk_gib"] = max(1, math.ceil(environment.storage_mb / 1024))
    if environment.gpus:
        resources["gpu"] = environment.gpus
        if environment.gpu_types:
            resources["gpu_type"] = environment.gpu_types[0]
    return resources


class HarborResourcesServer(SimpleResourcesServer):
    config: HarborResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sessions: dict[str, HarborSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._closed: set[str] = set()
        self._shutting_down = False

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/close_session")(self.close_session)
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def shutdown(self) -> None:
        self._shutting_down = True
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            try:
                await session.sandbox.stop()
            except Exception:
                print("Failed to stop abandoned Harbor sandbox", format_exc(), file=sys.stderr)

    # -- task resolution -------------------------------------------------------------------

    def _resolve_task(self, task_id: TaskId, task_data: dict[str, Any]) -> HarborTask:
        taskset = self.config.tasksets.get(task_id.taskset)
        if taskset is None:
            raise HTTPException(404, f"Taskset {task_id.taskset!r} is not served by this resources server")
        expected = taskset.tasks.get(task_id.task_id)
        if expected is None:
            raise HTTPException(404, f"Task {task_id.task_id!r} is not in taskset {task_id.taskset!r}")
        digest = task_data.get(DIGEST_KEY)
        if digest != expected:
            raise HTTPException(
                422,
                f"Row digest for {task_id.task_id!r} does not match the taskset mapping; re-materialize the taskset",
            )
        folder = Path(taskset.folder) / task_id.task_id
        try:
            task = load_task(folder)
        except HarborTaskError as exc:
            raise HTTPException(422, str(exc)) from exc
        if task.digest != expected:
            raise HTTPException(409, f"Task folder {folder} changed since materialization; re-materialize the taskset")
        return task

    # -- sandbox ---------------------------------------------------------------------------

    def _sandbox_spec(self, task: HarborTask, workdir: str) -> SandboxSpec:
        global_config_dict = get_global_config_dict()
        metadata = (
            resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
            | self.config.sandbox_metadata
            | {"nemo_gym_resources_server": self.config.name, "harbor_task": task.task_id[:63]}
        )
        ttl = task.config.agent.timeout_sec + task.config.verifier.timeout_sec + self.config.sandbox_ttl_slack_s
        return SandboxSpec(
            image=task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=dict(task.env),
            metadata=metadata,
            resources=_sandbox_resources(task),
            provider_options=dict(self.config.sandbox_provider_options),
        )

    async def _create_sandbox(self, task: HarborTask, workdir: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

    async def _prepare_workdir(self, sandbox: AsyncSandbox, task: HarborTask, workdir: str) -> None:
        commands = [f"mkdir -p {shlex.quote(workdir)}"]
        if task.user:
            commands.append(f"chown {shlex.quote(task.user)} {shlex.quote(workdir)}")
        result = await sandbox.exec(" && ".join(commands), cwd="/", timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(f"Could not prepare {workdir}: {result.stderr or result.stdout}")

    async def _sandbox_access(self, session: HarborSession) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=self.config.sandbox_provider,
                descriptor=await session.sandbox.serialize(),
            ),
            workdir=session.workdir,
        )

    # -- protocol --------------------------------------------------------------------------

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        if self._shutting_down:
            raise HTTPException(503, "Resources server is shutting down")
        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if session_id in self._closed:
                raise HTTPException(409, f"Resources session is already closed: {session_id}")
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.identity != (body.episode_id, body.task_id):
                    raise HTTPException(409, "resources_session_id is already bound to another episode or task")
                return ResourcesSeedSessionResponse(
                    resources_session_id=session_id, sandbox_access=await self._sandbox_access(existing)
                )

            task = self._resolve_task(body.task_id, body.task_data)
            if not task.needs_sandbox:
                raise HTTPException(
                    422, f"Task {task.task_id!r} declares no image; sandbox-less tasks are not supported yet"
                )
            workdir = task.workdir or DEFAULT_WORKDIR
            try:
                sandbox = await self._create_sandbox(task, workdir)
            except Exception as exc:
                LOGGER.exception(f"Sandbox creation failed for {task.task_id}")
                raise HTTPException(503, f"Could not start sandbox for {task.task_id!r}: {exc}") from exc
            try:
                await self._prepare_workdir(sandbox, task, workdir)
                session = HarborSession(
                    task=task,
                    taskset=body.task_id.taskset,
                    identity=(body.episode_id, body.task_id),
                    sandbox=sandbox,
                    workdir=workdir,
                )
                access = await self._sandbox_access(session)
            except Exception as exc:
                await sandbox.stop()
                LOGGER.exception(f"Sandbox setup failed for {task.task_id}")
                raise HTTPException(503, f"Could not set up sandbox for {task.task_id!r}: {exc}") from exc
            self._sessions[session_id] = session
            return ResourcesSeedSessionResponse(resources_session_id=session_id, sandbox_access=access)

    def _session_for(self, request: Request, episode_id: EpisodeId, task_id: TaskId) -> HarborSession:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            raise HTTPException(404, "Unknown Harbor resources session; seed it first")
        if session.identity != (episode_id, task_id):
            raise HTTPException(409, "Verification identity does not match the seeded resources session")
        return session

    async def verify(self, request: Request, body: SingleAgentTurnResourcesVerifyRequest) -> ResourcesVerifyResponse:
        session = self._session_for(request, body.episode_id, body.task_id)
        session_id = request.session[SESSION_ID_KEY]
        outcome = await self._run_verifier(session, session_id)
        return ResourcesVerifyResponse(
            responses_create_params=body.verification_input.responses_create_params,
            response=body.verification_input.response,
            **outcome,
        )

    async def _run_verifier(self, session: HarborSession, session_id: str) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote."""
        task = session.task
        sandbox = session.sandbox
        settings = task.config.verifier
        logs_dir = self.config.artifacts_dir / session_id
        try:
            prepare = await sandbox.exec(
                " && ".join(
                    [
                        f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                        f"find {TESTS_DIR} {VERIFIER_LOGS_DIR} -mindepth 1 -delete",
                        f"chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                    ]
                ),
                cwd="/",
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            await upload_dir(sandbox, task.path / "tests", TESTS_DIR)
            result = await sandbox.exec(
                f"bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                cwd=session.workdir,
                env=dict(settings.env),
                timeout_s=settings.timeout_sec,
                user=settings.user,
            )
            if result.error_type == "timeout":
                return self._measured(
                    0.0, VERIFIER_TIMEOUT_KIND, f"test.sh exceeded [verifier].timeout_sec={settings.timeout_sec}"
                )
            if result.error_type is not None:
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")

        extras = {"verifier_return_code": result.return_code, "verifier_logs_dir": str(logs_dir)}
        rewards, problem = parse_reward_file(logs_dir)
        if rewards is None:
            kind = MISSING_REWARD_KIND if "written" in (problem or "") else INVALID_REWARD_KIND
            return self._measured(0.0, kind, f"{problem}; tail: {_stdout_tail(logs_dir)}") | extras
        reward = select_reward(rewards)
        if reward is None:
            raise HTTPException(
                422,
                f"reward.json for {task.task_id!r} has several keys and none named 'reward': {sorted(rewards)}",
            )
        return {"reward": reward, "verifier_rewards": rewards} | extras

    @staticmethod
    def _measured(reward: float, kind: str, reason: str) -> dict[str, Any]:
        return {"reward": reward, "mask_sample": False, "failure_kind": kind, "failure_reason": reason[:2000]}

    @staticmethod
    def _masked(kind: str, reason: str) -> dict[str, Any]:
        return {"reward": 0.0, "mask_sample": True, "failure_kind": kind, "failure_reason": reason[:2000]}

    async def close_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self._sessions.get(session_id)
            if session is not None and session.identity[0] != body.episode_id:
                raise HTTPException(409, "episode_id does not match the seeded resources session")
            if session is not None:
                await session.sandbox.stop()
                del self._sessions[session_id]
            self._closed.add(session_id)
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)


def _stdout_tail(logs_dir: Path, limit: int = 600) -> str:
    path = logs_dir / Path(VERIFIER_STDOUT).name
    if not path.is_file():
        return "(no test-stdout.txt)"
    return path.read_text(errors="replace")[-limit:]


if __name__ == "__main__":
    HarborResourcesServer.run_webserver()
