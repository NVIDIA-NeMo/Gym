# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resources Server for Harbor-format tasks.

Owns the parts of a Harbor trial that belong to the task: resolving the task from a Harbor dataset, starting its
sandbox, grading it with the task's Harbor verifier, and stopping the sandbox. The agent harness runs elsewhere and
borrows the sandbox through the ``SandboxAccess`` returned by ``seed_session``.
"""

import asyncio
import logging
import re
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from harbor.models.job.config import DatasetConfig
from harbor.models.task.config import NetworkMode, TaskOS
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode
from harbor.models.trial.config import VerifierConfig
from harbor.models.trial.paths import TrialPaths
from harbor.tasks.client import TaskClient
from harbor.utils.env import resolve_env_vars
from harbor.verifier.factory import VerifierFactory
from pydantic import Field, PositiveFloat, field_validator

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseMultiRewardVerifyResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.reward_profile import compute_pass_majority_metrics
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider, rewrite_image
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.adapters.harbor import DEFAULT_EXEC_TIMEOUT_SECONDS, HarborSandboxEnvironment
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.harbor_tasks.task_data import TaskData


LOG = logging.getLogger(__name__)


class HarborTasksResourcesServerConfig(BaseResourcesServerConfig):
    # Grading reads the sandbox the agent changed, so a rollout cannot be re-verified from its trajectory alone.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED

    harbor_datasets: dict[str, DatasetConfig] = Field(
        description="Dataset alias -> Harbor DatasetConfig (a local `path`, or a registry `name` and `version`)."
    )
    sandbox_provider: str = Field(description="Name of a top-level sandbox provider config, e.g. `sandbox`.")
    sandbox_config: dict[str, Any] = Field(
        default_factory=dict,
        description="SandboxSpec overrides: ttl_s, ready_timeout_s, env, metadata, provider_options, resources.",
    )
    image_rewrites: list[dict[str, str]] = Field(
        default_factory=list, description="Ordered [{from, to}] prefix rewrites applied to each task's docker_image."
    )
    harbor_verifier: VerifierConfig = Field(
        default_factory=VerifierConfig,
        description="Harbor trial VerifierConfig: timeout override and cap, extra env, custom verifier import path.",
    )
    verifier_timeout_multiplier: PositiveFloat = 1.0
    reward_key: str = Field(default="reward", min_length=1, description="Verifier reward key used as the reward.")
    artifacts_dir: Path = Field(
        default=Path("results/harbor_tasks"),
        description="Host directory for each episode's downloaded verifier logs, keyed by rollout capture key.",
    )
    exec_shell: str | None = "bash -c"
    default_exec_timeout_seconds: PositiveFloat = DEFAULT_EXEC_TIMEOUT_SECONDS
    stop_timeout_seconds: PositiveFloat = 300
    allow_unenforced_network_policy: bool = Field(
        default=False,
        description="Run tasks that ask for a restricted network even though Gym sandboxes do not enforce it.",
    )

    @field_validator("harbor_datasets", mode="after")
    @classmethod
    def resolve_dataset_paths(cls, datasets: dict[str, DatasetConfig]) -> dict[str, DatasetConfig]:
        # Server processes run from their own directory; config paths are relative to the Gym repository.
        return {
            alias: dataset.model_copy(update={"path": _repo_path(dataset.path)}) if dataset.path else dataset
            for alias, dataset in datasets.items()
        }

    @field_validator("artifacts_dir", mode="after")
    @classmethod
    def resolve_artifacts_dir(cls, artifacts_dir: Path) -> Path:
        return _repo_path(artifacts_dir)


class HarborTasksVerifyRequest(BaseVerifyRequest):
    harbor_dataset: str
    task_name: str


class HarborTasksVerifyResponse(BaseMultiRewardVerifyResponse):
    harbor_dataset: str
    task_name: str
    evaluation_completed: bool
    verifier_error: str | None = None
    verification_time_taken: float
    trial_dir: str


@dataclass
class HarborTaskSession:
    """One seeded task sandbox, owned by this server until close."""

    episode_id: EpisodeId
    task_id: TaskId
    task_data: TaskData
    task: Task
    sandbox: AsyncSandbox
    trial_paths: TrialPaths
    environment: HarborSandboxEnvironment | None = None
    workdir: str | None = None
    descriptor: dict[str, Any] | None = None


class HarborTasksResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: HarborTasksResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("Harbor task sessions are process-local and require num_workers=1")
        self._sessions: dict[str, HarborTaskSession] = {}
        self._closed_sessions: dict[str, EpisodeId] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._task_index: dict[str, dict[str, Path]] = {}
        self._task_index_lock = asyncio.Lock()

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
        return app

    async def shutdown(self) -> None:
        """Stop sandboxes whose episodes never closed them."""
        for session_id in list(self._sessions):
            try:
                await self._stop_session(session_id)
            except Exception:
                LOG.exception("Failed to stop abandoned Harbor task sandbox %s", session_id)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        async with self._session_locks.setdefault(session_id, asyncio.Lock()):
            if session_id in self._closed_sessions:
                raise ValueError(f"Resources session is already closed: {session_id}")
            existing = self._sessions.get(session_id)
            if existing is not None:
                if (existing.episode_id, existing.task_id) != (body.episode_id, body.task_id):
                    raise ValueError("resources_session_id is already bound to another episode or task")
                return self._seed_response(session_id, existing)

            task_data = TaskData.model_validate(body.task_data)
            task = await self.load_task(task_data)
            sandbox = AsyncSandbox(create_provider(self._provider_config()))
            trial_dir = self.config.artifacts_dir / _path_component(body.episode_id.capture_key)
            shutil.rmtree(trial_dir, ignore_errors=True)
            trial_paths = TrialPaths(trial_dir.resolve())
            trial_paths.mkdir()
            session = HarborTaskSession(
                episode_id=body.episode_id,
                task_id=body.task_id,
                task_data=task_data,
                task=task,
                sandbox=sandbox,
                trial_paths=trial_paths,
            )
            # Own the sandbox before it starts, so a failed start is still stopped.
            self._sessions[session_id] = session
            try:
                await sandbox.start(self._sandbox_spec(task))
                session.environment = HarborSandboxEnvironment(
                    sandbox,
                    environment_dir=task.paths.environment_dir,
                    environment_name=task.short_name,
                    session_id=f"{_path_component(body.episode_id.capture_key)}__env",
                    trial_paths=trial_paths,
                    task_env_config=task.config.environment.model_copy(deep=True),
                    exec_shell=self.config.exec_shell,
                    default_exec_timeout_seconds=self.config.default_exec_timeout_seconds,
                    logger=LOG,
                )
                await session.environment.start(force_build=False)
                await session.environment.run_healthcheck()
                await self.prepare_sandbox(session)
                session.workdir = await self._workdir(session)
                session.descriptor = await sandbox.serialize()
                return self._seed_response(session_id, session)
            except BaseException:
                try:
                    await self._stop_session(session_id)
                except Exception:
                    LOG.exception("Failed to stop partially seeded Harbor task sandbox %s", session_id)
                raise

    async def prepare_sandbox(self, session: HarborTaskSession) -> None:
        """Stage task resources into a started sandbox before the agent receives it.

        Benchmarks whose tasks need more than Harbor's own ``environment/`` upload override this.
        """

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        session_id = body.resources_session_id
        async with self._session_locks.setdefault(session_id, asyncio.Lock()):
            closed_episode_id = self._closed_sessions.get(session_id)
            if closed_episode_id is not None:
                if closed_episode_id != body.episode_id:
                    raise ValueError("episode_id does not match the closed resources session")
            else:
                session = self._sessions.get(session_id)
                if session is not None and session.episode_id != body.episode_id:
                    raise ValueError("episode_id does not match the seeded resources session")
                await self._stop_session(session_id)
                self._closed_sessions[session_id] = body.episode_id
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def verify(self, request: Request, body: HarborTasksVerifyRequest) -> HarborTasksVerifyResponse:
        session = self._sessions.get(request.session.get(SESSION_ID_KEY, ""))
        if session is None or session.environment is None:
            raise ValueError("verify requires a seeded Harbor task session")
        if (body.harbor_dataset, body.task_name) != (session.task_data.harbor_dataset, session.task_data.task_name):
            raise ValueError("verify request names a different task than the seeded session")

        task = session.task
        verifier_config = self.config.harbor_verifier
        rewards: dict[str, float | int] = {}
        verifier_error = None
        start = monotonic()
        if verifier_config.disable:
            verifier_error = "Verification is disabled by harbor_verifier.disable"
        else:
            # A repeated verify must not read the previous verify's reward file.
            shutil.rmtree(session.trial_paths.verifier_dir, ignore_errors=True)
            session.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            timeout_seconds = self._verifier_timeout_seconds(task)
            with session.environment.with_default_user(task.config.verifier.user):
                verifier = VerifierFactory.create_verifier_from_config(
                    verifier_config,
                    task=task,
                    trial_paths=session.trial_paths,
                    environment=session.environment,
                    override_env=verifier_config.env or None,
                    logger=LOG,
                )
                try:
                    rewards = (await asyncio.wait_for(verifier.verify(), timeout=timeout_seconds)).rewards or {}
                except TimeoutError:
                    verifier_error = f"VerifierTimeoutError: verifier exceeded {timeout_seconds} seconds"
                except Exception as error:
                    LOG.exception("Harbor verifier failed for %s", body.task_name)
                    verifier_error = f"{type(error).__name__}: {error}"[:2000]

        return HarborTasksVerifyResponse(
            **body.model_dump(),
            reward=self._reward(rewards, task_name=body.task_name),
            reward_components={key: float(value) for key, value in rewards.items()},
            evaluation_completed=verifier_error is None,
            verifier_error=verifier_error,
            verification_time_taken=monotonic() - start,
            trial_dir=str(session.trial_paths.trial_dir),
        )

    @staticmethod
    def _score_fn(rollout: dict) -> dict[str, float]:
        reward = rollout.get("reward")
        # A row without a reward is a rollout nobody scored, not a zero.
        return {} if reward is None else {"accuracy": reward}

    def compute_metrics(self, tasks: list[list[dict]]) -> dict:
        """pass@k over the verifier reward.

        A 0/1 benchmark gets the combinatorial estimator and a continuous one max-of-k; Harbor reports a score, not
        an extracted answer, so majority@k is not computed.
        """
        metrics, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn)
        return metrics

    async def load_task(self, task_data: TaskData) -> Task:
        """Resolve a row to a validated, supported Harbor task."""
        index = await self._dataset_index(task_data.harbor_dataset)
        task_dir = index.get(task_data.task_name)
        if task_dir is None:
            raise ValueError(f"Task {task_data.task_name!r} is not in Harbor dataset {task_data.harbor_dataset!r}")
        task = await asyncio.to_thread(Task, task_dir, disable_verification=self.config.harbor_verifier.disable)
        self._require_supported(task)
        return task

    async def _dataset_index(self, alias: str) -> dict[str, Path]:
        async with self._task_index_lock:
            if alias in self._task_index:
                return self._task_index[alias]
            dataset = self.config.harbor_datasets.get(alias)
            if dataset is None:
                raise ValueError(
                    f"Unknown harbor_dataset {alias!r}; configured: {sorted(self.config.harbor_datasets)}"
                )
            paths = await download_dataset_tasks(dataset, disable_verification=self.config.harbor_verifier.disable)
            index: dict[str, Path] = {}
            for path in paths:
                task = await asyncio.to_thread(Task, path, disable_verification=True)
                # Rows may name a task by its declared name or its directory name.
                index.setdefault(task.name, path)
                index.setdefault(path.name, path)
            self._task_index[alias] = index
            return index

    def _require_supported(self, task: Task) -> None:
        unsupported = unsupported_features(
            task, allow_unenforced_network_policy=self.config.allow_unenforced_network_policy
        )
        if unsupported:
            raise ValueError(f"Harbor task {task.name!r} uses unsupported features: {', '.join(unsupported)}")

    def _provider_config(self) -> dict[str, Any]:
        return resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())

    def _sandbox_spec(self, task: Task) -> SandboxSpec:
        environment = task.config.environment
        resources: dict[str, Any] = {}
        if environment.cpus:
            resources["cpu"] = float(environment.cpus)
        if environment.memory_mb:
            resources["memory_mib"] = int(environment.memory_mb)
        if environment.storage_mb:
            resources["disk_gib"] = max(1, round(environment.storage_mb / 1024))
        if environment.gpus:
            resources["gpu"] = int(environment.gpus)
        resources |= self.config.sandbox_config.get("resources") or {}
        # Harbor applies the task env to every command; container env gives the agent and verifier the same view.
        env = resolve_env_vars(environment.env) if environment.env else {}
        return SandboxSpec(
            image=rewrite_image(environment.docker_image, self.config.image_rewrites),
            ttl_s=self.config.sandbox_config.get("ttl_s"),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s"),
            workdir=environment.workdir,
            env=env | dict(self.config.sandbox_config.get("env") or {}),
            metadata=resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict())
            | dict(self.config.sandbox_config.get("metadata") or {})
            | {"nemo_gym_resources_server": self.config.name, "harbor_task": task.short_name[:63]},
            resources=SandboxResources.from_mapping(resources),
            provider_options=dict(self.config.sandbox_config.get("provider_options") or {}),
        )

    async def _workdir(self, session: HarborTaskSession) -> str:
        if session.task.config.environment.workdir:
            return session.task.config.environment.workdir
        result = await session.sandbox.exec("pwd", timeout_s=60)
        workdir = (result.stdout or "").strip()
        if result.return_code != 0 or not workdir.startswith("/"):
            raise RuntimeError(f"Cannot determine the sandbox working directory: {result.stderr or result.stdout}")
        return workdir

    def _seed_response(self, session_id: str, session: HarborTaskSession) -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    descriptor=session.descriptor,
                ),
                workdir=session.workdir,
            ),
        )

    def _verifier_timeout_seconds(self, task: Task) -> float:
        base = self.config.harbor_verifier.override_timeout_sec or task.config.verifier.timeout_sec
        cap = self.config.harbor_verifier.max_timeout_sec or float("inf")
        return min(base, cap) * self.config.verifier_timeout_multiplier

    def _reward(self, rewards: dict[str, float | int], *, task_name: str) -> float:
        if not rewards:
            return 0.0
        if self.config.reward_key in rewards:
            return float(rewards[self.config.reward_key])
        LOG.warning(
            "Harbor verifier rewards for %s have no %r key; using the first reward", task_name, self.config.reward_key
        )
        return float(next(iter(rewards.values())))

    async def _stop_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        async with asyncio.timeout(self.config.stop_timeout_seconds):
            await session.sandbox.stop()
        self._sessions.pop(session_id, None)


async def download_dataset_tasks(dataset: DatasetConfig, *, disable_verification: bool = False) -> list[Path]:
    """Resolve a Harbor dataset to local task directories, downloading registry tasks into Harbor's cache."""
    task_configs = await dataset.get_task_configs(disable_verification=disable_verification)
    downloaded = await TaskClient().download_tasks(
        [config.get_task_id() for config in task_configs],
        overwrite=dataset.overwrite,
        output_dir=dataset.download_dir,
    )
    return downloaded.paths


def unsupported_features(task: Task, *, allow_unenforced_network_policy: bool = False) -> list[str]:
    """List the task features this server cannot reproduce faithfully; empty when the task is supported."""
    unsupported = []
    if task.has_steps:
        unsupported.append("multi-step tasks ([[steps]])")
    if resolve_task_verifier_mode(task.config) == VerifierEnvironmentMode.SEPARATE:
        unsupported.append("separate verifier environments")
    if task.config.environment.os != TaskOS.LINUX:
        unsupported.append(f"{task.config.environment.os.value} tasks")
    if not task.config.environment.docker_image:
        unsupported.append("tasks without a prebuilt [environment].docker_image")
    if any((task.paths.environment_dir / name).exists() for name in ("docker-compose.yaml", "docker-compose.yml")):
        unsupported.append("docker-compose environments")
    network_modes = {
        task.config.environment.network_mode,
        task.config.agent.network_mode,
        task.config.verifier.network_mode,
    }
    if network_modes - {None, NetworkMode.PUBLIC} and not allow_unenforced_network_policy:
        unsupported.append("restricted network policies (set allow_unenforced_network_policy to run unenforced)")
    return unsupported


def _repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PARENT_DIR / path


def _path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "episode"


if __name__ == "__main__":
    HarborTasksResourcesServer.run_webserver()
