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
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.models import HarborArtifact, HarborEnvironment
from nemo_gym.tasks.harbor.task import HarborTaskError
from resources_servers.harbor.sandbox_io import download_dir, download_path, upload_dir, upload_path


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"
VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
ARTIFACTS_DIR = "/logs/artifacts"
VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"

# Namespaced failure kinds for outcomes the shared vocabulary does not name.
VERIFIER_TIMEOUT_KIND = "harbor:verifier_timeout"
MISSING_REWARD_KIND = "harbor:missing_reward"
INVALID_REWARD_KIND = "harbor:invalid_reward"


class HarborVerifyRequest(BaseVerifyRequest):
    """The flat verify body the environment server posts: the row's ``task_data`` keys beside the params
    and the response. The session cookie identifies the episode; the digest, when present, must match it.
    """

    model_config = ConfigDict(populate_by_name=True)

    ng_digest: str | None = Field(default=None, alias=DIGEST_KEY)


class HarborVerifyResponse(BaseVerifyResponse):
    """The verify response plus what the Harbor verifier produced.

    ``verifier_rewards`` is the parsed ``reward.json`` (``reward.txt`` becomes ``{"reward": value}``).
    The ``verifier_*`` fields are ``None`` when ``test.sh`` never ran.
    """

    verifier_rewards: dict[str, float] | None = None
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None
    # "shared": test.sh ran in the agent's sandbox; "separate": in its own sandbox from [verifier.environment].
    verifier_mode: Literal["shared", "separate"] | None = None


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
    # Replaces the task's declared cpus, memory_mb and storage_mb when set (keys: cpu, memory_mib, disk_gib).
    # Used to run a benchmark at fixed resources, for example to compare against another server.
    sandbox_resources_override: dict[str, Any] | None = None
    # Export CPU-count env vars (OMP_NUM_THREADS and friends) matching the sandbox CPU limit.
    derive_cpu_env: bool = True
    # Operator environment for every sandbox; a task's own `[environment.env]` wins.
    sandbox_env: dict[str, str] = Field(default_factory=dict)
    # Shell commands run as root in every new sandbox before the agent sees it, for repairs the
    # operator owns rather than the task (for example pointing an end-of-life distro at an archive).
    # A failing command fails the seed with a retryable 503.
    sandbox_setup_commands: list[str] = Field(default_factory=list)
    sandbox_setup_timeout_s: float = Field(default=600, gt=0)
    # Extra seconds granted to `test.sh` beyond `[verifier].timeout_sec` before the in-container `timeout` kills it.
    verifier_grace_s: float = Field(default=30, ge=0)
    # Verifier logs are downloaded here, one folder per resources session.
    artifacts_dir: Path = Path("results/harbor/verifier")


@dataclass
class HarborSession:
    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str
    # The first verify's outcome; a retried /verify replays it instead of re-running test.sh.
    verify_outcome: dict[str, Any] | None = None
    agent_sandbox_stopped: bool = False


def parse_reward_file(directory: Path) -> tuple[dict[str, float] | None, str | None]:
    """Read Harbor's reward file.

    Returns ``(rewards, problem)``: ``rewards`` is the parsed mapping (``reward.txt``
    becomes ``{"reward": value}``) or ``None``; ``problem`` names what went wrong.
    ``reward.json`` is preferred; when it is missing or unusable, ``reward.txt`` is tried.
    """
    problems: list[str] = []
    for path in (directory / "reward.json", directory / "reward.txt"):
        if not path.is_file():
            continue
        rewards, problem = _parse_one_reward_file(path)
        if rewards is not None:
            return rewards, None
        problems.append(problem)
    if not problems:
        return None, "no reward.json or reward.txt was written"
    return None, "; ".join(problems)


def _parse_one_reward_file(path: Path) -> tuple[dict[str, float] | None, str]:
    # The verifier wrote these bytes; never let a stray non-UTF-8 byte escape as an exception.
    text = path.read_text(errors="replace").strip()
    if not text:
        return None, f"{path.name} is empty"
    try:
        raw = json.loads(text) if path.suffix == ".json" else {"reward": float(text)}
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


def _is_root(user: str | None) -> bool:
    return user is None or str(user).split(":")[0] in ("root", "0")


async def _exec_as_root_user(
    sandbox: AsyncSandbox, command: str, *, configured_user: str | None, cwd: str = "/", timeout_s: float = 60
):
    """Run ``command`` as root.

    ``configured_user`` is the user the sandbox runs commands as by default (the image's
    ``USER`` or ``[agent].user``). Only a non-root default needs the ``user="root"``
    override; root images keep the plain exec path every provider supports.
    """
    user = None if _is_root(configured_user) else "root"
    return await sandbox.exec(command, cwd=cwd, timeout_s=timeout_s, user=user)


def _verifier_image(task: HarborTask) -> str | None:
    """The prebuilt image a separate verifier runs in: ``[verifier.environment].docker_image``, else the agent's."""
    if task.config.is_shared_verifier:
        return None
    environment = task.config.verifier.environment
    if environment is not None and environment.docker_image:
        return environment.docker_image
    if environment is None:
        return task.image
    return None


def _sandbox_resources(environment: HarborEnvironment) -> dict[str, Any]:
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
    ray_enabled = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sessions: dict[str, HarborSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Parsed tasks by (folder, materialized digest): hash a task folder once per server, not per episode.
        self._tasks: dict[tuple[Path, str], HarborTask] = {}
        self._closed: set[str] = set()
        self._shutting_down = False

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
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

    async def _resolve_task(self, task_id: TaskId, task_data: dict[str, Any]) -> HarborTask:
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
        task = self._tasks.get((folder, expected))
        if task is not None:
            return task
        try:
            # Parsing task.toml and hashing the folder are blocking file work.
            task = await asyncio.to_thread(load_task, folder)
        except HarborTaskError as exc:
            raise HTTPException(422, str(exc)) from exc
        if task.digest != expected:
            raise HTTPException(409, f"Task folder {folder} changed since materialization; re-materialize the taskset")
        self._tasks[(folder, expected)] = task
        return task

    # -- sandbox ---------------------------------------------------------------------------

    def _sandbox_spec(
        self,
        task: HarborTask,
        workdir: str | None,
        *,
        environment: HarborEnvironment | None = None,
        image: str | None = None,
        role: str = "agent",
        ttl: float | None = None,
    ) -> SandboxSpec:
        """The agent's sandbox by default; pass the verifier's environment for a separate verifier."""
        global_config_dict = get_global_config_dict()
        if environment is None:
            # The agent's sandbox: Dockerfile ENV recorded by the loader, overridden by `[environment.env]`.
            environment, task_env = task.config.environment, task.env
        else:
            task_env = environment.env
        metadata = (
            resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
            | self.config.sandbox_metadata
            | {"nemo_gym_resources_server": self.config.name, "harbor_task": task.task_id[:63], "harbor_role": role}
        )
        if ttl is None:
            ttl = task.config.agent.timeout_sec + task.config.verifier.timeout_sec + self.config.sandbox_ttl_slack_s
        # The override replaces only the keys it names, so a GPU task routed to the GPU provider keeps `gpu`.
        resources = _sandbox_resources(environment) | (self.config.sandbox_resources_override or {})
        env = dict(self.config.sandbox_env) | dict(task_env)
        if self.config.derive_cpu_env:
            env = cpu_cap_env(resources.get("cpu")) | env
        return SandboxSpec(
            image=image or task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=env,
            metadata=metadata,
            resources=resources,
            provider_options=dict(self.config.sandbox_provider_options),
        )

    async def _create_sandbox(self, task: HarborTask, workdir: str | None) -> AsyncSandbox:
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

    async def _create_verifier_sandbox(self, task: HarborTask) -> AsyncSandbox:
        """A separate verifier runs in its own sandbox, sized by ``[verifier.environment]``."""
        environment = task.config.verifier.environment or task.config.environment
        spec = self._sandbox_spec(
            task,
            None,
            environment=environment,
            image=_verifier_image(task),
            role="verifier",
            ttl=task.config.verifier.timeout_sec + self.config.sandbox_ttl_slack_s,
        )
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        await sandbox.start(spec)
        return sandbox

    async def _prepare_workdir(self, sandbox: AsyncSandbox, task: HarborTask, workdir: str | None) -> str:
        """Create the working directory, or resolve the image's own WORKDIR when the task sets none."""
        if workdir is None:
            result = await sandbox.exec("pwd", timeout_s=60)
            if result.return_code != 0 or not (result.stdout or "").strip():
                raise RuntimeError(f"Could not resolve the image working directory: {result.stderr or result.stdout}")
            workdir = (result.stdout or "").strip().splitlines()[-1]
        commands = [f"mkdir -p {shlex.quote(workdir)}"]
        if task.user:
            commands.append(f"chown {shlex.quote(task.user)} {shlex.quote(workdir)}")
        result = await _exec_as_root_user(sandbox, " && ".join(commands), configured_user=task.user)
        if result.return_code != 0:
            raise RuntimeError(f"Could not prepare {workdir}: {result.stderr or result.stdout}")
        return workdir

    async def _run_setup_commands(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        for command in self.config.sandbox_setup_commands:
            result = await sandbox.exec(command, cwd="/", timeout_s=self.config.sandbox_setup_timeout_s, user="root")
            if result.return_code != 0:
                raise RuntimeError(
                    f"Sandbox setup command failed for {task.task_id!r} (exit {result.return_code}): "
                    f"{(result.stderr or result.stdout or '')[-500:]}"
                )

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

            task = await self._resolve_task(body.task_id, body.task_data)
            if not task.needs_sandbox:
                raise HTTPException(
                    422, f"Task {task.task_id!r} declares no image; sandbox-less tasks are not supported yet"
                )
            if not task.config.is_shared_verifier and _verifier_image(task) is None:
                raise HTTPException(
                    422,
                    f"Task {task.task_id!r} needs a separate verifier built from tests/Dockerfile; "
                    "only prebuilt verifier images are supported yet",
                )
            try:
                sandbox = await self._create_sandbox(task, task.workdir)
            except Exception as exc:
                LOGGER.exception(f"Sandbox creation failed for {task.task_id}")
                raise HTTPException(503, f"Could not start sandbox for {task.task_id!r}: {exc}") from exc
            try:
                workdir = await self._prepare_workdir(sandbox, task, task.workdir)
                await self._run_setup_commands(sandbox, task)
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

    def _session_for(self, request: Request, digest: str | None = None) -> HarborSession:
        session_id = request.session.get(SESSION_ID_KEY)
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            raise HTTPException(404, "Unknown Harbor resources session; seed it first")
        if digest is not None and digest != session.task.digest:
            raise HTTPException(409, "Verification digest does not match the seeded task")
        return session

    async def verify(self, request: Request, body: HarborVerifyRequest) -> HarborVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            raise HTTPException(404, "Unknown Harbor resources session; seed it first")
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self._session_for(request, body.ng_digest)
            if session.verify_outcome is None:
                session.verify_outcome = await self._run_verifier(session, session_id)
            outcome = session.verify_outcome
        return HarborVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            **outcome,
        )

    async def _run_verifier(self, session: HarborSession, session_id: str) -> dict[str, Any]:
        """Run ``tests/test.sh`` and read the reward it wrote, in the agent's sandbox or a separate one."""
        if session.task.config.is_shared_verifier:
            return await self._run_shared_verifier(session, session_id) | {"verifier_mode": "shared"}
        return await self._run_separate_verifier(session, session_id) | {"verifier_mode": "separate"}

    async def _run_separate_verifier(self, session: HarborSession, session_id: str) -> dict[str, Any]:
        """Harbor's separate mode: collect the agent's artifacts, stop its sandbox, verify in a fresh one.

        The verifier sandbox comes from ``[verifier.environment]`` (or the task image when the
        mode is ``separate`` without one). ``/logs/artifacts`` and every ``artifacts`` entry are
        copied through the host into the same paths, then ``tests/`` is uploaded and ``test.sh``
        runs there. Sidecar (Compose) artifacts and hooks are not collected yet.
        """
        task = session.task
        settings = task.config.verifier
        logs_dir = self.config.artifacts_dir / session_id
        artifacts_dir = logs_dir / "artifacts"
        started = asyncio.get_running_loop().time()
        verifier: AsyncSandbox | None = None
        try:
            for hook in settings.collect:
                if hook.service not in (None, "main"):
                    LOGGER.warning(
                        f"{task.task_id}: collect hook for sidecar {hook.service!r} skipped (no Compose yet)"
                    )
                    continue
                result = await session.sandbox.exec(
                    hook.command, cwd=session.workdir, timeout_s=hook.timeout_sec + 30, user=hook.user
                )
                if result.return_code:
                    LOGGER.warning(f"{task.task_id}: collect hook exited {result.return_code}: {hook.command!r}")
            restored: list[tuple[Path, str]] = []
            for artifact in _collected_artifacts(task):
                if artifact.service not in (None, "main"):
                    LOGGER.warning(
                        f"{task.task_id}: artifact {artifact.source!r} on sidecar {artifact.service!r} skipped"
                    )
                    continue
                host = artifacts_dir / artifact.host_path
                if await download_path(session.sandbox, artifact.source, host) is not None:
                    restored.append((host, artifact.source))
            # Harbor stops the agent's container before the verifier starts; nothing may keep running.
            await session.sandbox.stop()
            session.agent_sandbox_stopped = True

            verifier = await self._create_verifier_sandbox(task)
            prepare = await verifier.exec(
                f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR} && chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR}",
                cwd="/",
                timeout_s=60,
                user="root",
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            await upload_dir(verifier, task.path / "tests", TESTS_DIR)
            for host, source in restored:
                await upload_path(verifier, host, source)
            budget = int(settings.timeout_sec)
            result = await verifier.exec(
                f"chmod +x {TESTS_DIR}/test.sh; timeout --signal=KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                env=dict(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            if result.error_type is not None and result.error_type != "timeout":
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            await download_dir(verifier, VERIFIER_LOGS_DIR, logs_dir)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Separate verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        finally:
            if verifier is not None:
                try:
                    await verifier.stop()
                except Exception:
                    LOGGER.exception(f"Could not stop the verifier sandbox for {task.task_id}")
        return self._score(task, settings, result, logs_dir, started)

    async def _run_shared_verifier(self, session: HarborSession, session_id: str) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote."""
        task = session.task
        sandbox = session.sandbox
        settings = task.config.verifier
        logs_dir = self.config.artifacts_dir / session_id
        started = asyncio.get_running_loop().time()
        try:
            # /logs/verifier is root-owned from seed; a non-root image user cannot reset it.
            prepare = await _exec_as_root_user(
                sandbox,
                " && ".join(
                    [
                        f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                        f"find {TESTS_DIR} {VERIFIER_LOGS_DIR} -mindepth 1 -delete",
                        f"chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                    ]
                ),
                configured_user=task.user,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            await upload_dir(sandbox, task.path / "tests", TESTS_DIR)
            # One plain exec for the whole run. The provider keeps it alive for as long as
            # `timeout_s` says (OpenSandbox polls a background command, Docker streams), and
            # `timeout` inside the container is the cap that stops the tests themselves.
            # Commands must not be left running after the exec returns: OpenSandbox reaps
            # them when the command completes, so a background launch never finishes.
            budget = int(settings.timeout_sec)
            result = await sandbox.exec(
                f"timeout --signal=KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                cwd=session.workdir,
                env=dict(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            if result.error_type is not None and result.error_type != "timeout":
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            # Keep whatever the verifier wrote, including on a timeout, so a slow test.sh can be diagnosed.
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        return self._score(task, settings, result, logs_dir, started)

    def _score(
        self, task: HarborTask, settings: Any, result: SandboxExecResult, logs_dir: Path, started: float
    ) -> dict[str, Any]:
        """Turn the verifier's exit and its ``/logs/verifier`` download into the reward fields."""
        if result.error_type == "timeout" or result.return_code == 137:
            return self._measured(
                0.0,
                VERIFIER_TIMEOUT_KIND,
                f"test.sh exceeded [verifier].timeout_sec={settings.timeout_sec}; tail: {_stdout_tail(logs_dir)}",
            ) | {
                "verifier_logs_dir": str(Path(logs_dir).resolve()),
                "verifier_seconds": round(asyncio.get_running_loop().time() - started, 1),
            }
        # Absolute, so the row names a folder that exists from wherever the run was launched; the server
        # process runs from its own folder, where a relative artifacts_dir would otherwise resolve.
        extras = {
            "verifier_return_code": result.return_code,
            "verifier_logs_dir": str(Path(logs_dir).resolve()),
            "verifier_seconds": round(asyncio.get_running_loop().time() - started, 1),
        }
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

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self._sessions.get(session_id)
            if session is not None and session.identity[0] != body.episode_id:
                raise HTTPException(409, "episode_id does not match the seeded resources session")
            if session is not None:
                if not session.agent_sandbox_stopped:
                    await session.sandbox.stop()
                del self._sessions[session_id]
            self._closed.add(session_id)
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)


def _collected_artifacts(task: HarborTask) -> list[HarborArtifact]:
    """``/logs/artifacts`` first, then the task's own ``artifacts`` entries."""
    entries = list(task.config.artifacts)
    if not any(a.source.rstrip("/") == ARTIFACTS_DIR and a.service in (None, "main") for a in entries):
        entries.insert(0, HarborArtifact(source=ARTIFACTS_DIR))
    return entries


def _stdout_tail(logs_dir: Path, limit: int = 600) -> str:
    path = logs_dir / Path(VERIFIER_STDOUT).name
    if not path.is_file():
        return "(no test-stdout.txt)"
    return path.read_text(errors="replace")[-limit:]


if __name__ == "__main__":
    HarborResourcesServer.run_webserver()
