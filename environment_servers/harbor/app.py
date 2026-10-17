# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment server for Harbor-format tasks.

One server instance serves one or more tasksets. Each taskset maps to a folder of
task folders in Gym's shared datasets folder plus the digest every task had when its
rows were materialized (see ``nemo_gym.tasks.harbor``).

``/run`` owns one whole episode, in this order:

- seed: check the row's digest against the mapping and the folder, start the task's
  image as a sandbox through the configured provider, create the working directory;
- agent: open an agent session on the configured agent server, handing it a
  ``SandboxAccess`` for that sandbox, post the task's ``responses_create_params`` to it,
  then close the session and keep what the agent observed;
- verify: run ``tests/test.sh`` inside the same sandbox (Harbor's shared verifier mode),
  then read ``/logs/verifier/reward.json`` or ``reward.txt``;
- cleanup: stop the sandbox, whatever happened before.

Reward rule: when ``test.sh`` ran, the sample is measured. A missing or invalid reward
file scores 0 with a ``failure_kind``. Only a Gym-side failure (sandbox lost, transfer
failed) masks the sample. A task that cannot be set up is a handled episode failure:
terminal when the task itself is at fault (unknown task, changed folder, unsupported
settings), retryable when the sandbox provider failed.
"""

import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
import tarfile
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Literal
from uuid import uuid4

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from environment_servers.harbor.sandbox_io import (
    SandboxTransferError,
    _exec_as_root,
    download_dir,
    download_path,
    upload_dir,
    upload_path,
)
from nemo_gym import RESULTS_DIR, failure_kinds
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    HandledEpisodeError,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT, AgentServerRef, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_types import BaseEpisodeResponse, EpisodeId, TaskId
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK, get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnRequest, SingleAgentTurnResult
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.models import HarborArtifact, HarborEnvironment
from nemo_gym.tasks.harbor.task import HarborTaskError


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
# Added to `sandbox_ready_timeout_s` for the seed's own deadline on the create: room for the provider to
# report and clean up a create that hit its (equal) budget before the seed gives up on it.
SEED_CREATE_GRACE_S = 30.0


class HarborSeedTimeoutError(RuntimeError):
    """The sandbox was not ready within `sandbox_ready_timeout_s`: a provider-side seed failure, retried."""


class HarborEpisodeResult(SingleAgentTurnResult):
    """The episode's verify response plus what the Harbor verifier produced.

    ``verifier_rewards`` is the parsed ``reward.json`` (``reward.txt`` becomes ``{"reward": value}``).
    The ``verifier_*`` fields are ``None`` when ``test.sh`` never ran.
    """

    verifier_rewards: dict[str, float] | None = None
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None
    # "shared": test.sh ran in the agent's sandbox; "separate": in its own sandbox from [verifier.environment].
    verifier_mode: Literal["shared", "separate"] | None = None


class HarborEpisodeResponse(BaseEpisodeResponse[HarborEpisodeResult]):
    """Response for one Harbor episode: a result, or a staged failure with the agent's evidence."""

    failure: SingleAgentTurnFailure | None = None


class HarborTasksetConfig(BaseModel):
    """Where one taskset's folders live and which digest each task was materialized with."""

    model_config = ConfigDict(extra="forbid")

    folder: Path
    tasks: dict[str, str]


class HarborEnvironmentServerConfig(BaseEnvironmentServerConfig):
    model_config = ConfigDict(extra="forbid")

    # The harness that works in the sandbox; it receives a ``SandboxAccess`` when its session is seeded.
    agent_server: AgentServerRef
    tasksets: dict[str, HarborTasksetConfig] = Field(default_factory=dict)
    # Name of the top-level sandbox provider block in the merged config.
    sandbox_provider: str = "sandbox"
    sandbox_ready_timeout_s: float = Field(default=900, gt=0)
    # The sandbox outlives the episode's full budget (setup, queued agent work, verification) and the
    # final cleanup, so both must be finite here.
    default_episode_timeout_seconds: PositiveFloat = 21600
    cleanup_timeout_seconds: PositiveFloat = 180
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
    # A failing command fails the seed as a retryable episode failure.
    sandbox_setup_commands: list[str] = Field(default_factory=list)
    sandbox_setup_timeout_s: float = Field(default=600, gt=0)
    # Extra seconds granted to `test.sh` beyond `[verifier].timeout_sec` before the in-container `timeout` kills it.
    verifier_grace_s: float = Field(default=30, ge=0)
    # Verifier logs are downloaded here, one fresh folder per verification.
    artifacts_dir: Path | None = None


@dataclass
class HarborSession:
    """One episode's sandbox, from seed until the final cleanup stops it."""

    task: HarborTask
    taskset: str
    identity: tuple[EpisodeId, TaskId]
    sandbox: AsyncSandbox
    workdir: str
    # The episode key the sandbox is owned under (see ``_sandboxes``).
    session_id: str
    # Separate verification stops the agent's sandbox before test.sh runs; cleanup must not stop it twice.
    agent_sandbox_stopped: bool = False


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str]) -> dict[str, str]:
    """Resolve Harbor's whole-value host templates without changing the stored task."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
        elif match.group(1) in os.environ:
            resolved[key] = os.environ[match.group(1)]
        elif match.group(2) is not None:
            resolved[key] = match.group(2)
        else:
            raise HTTPException(422, f"Environment variable {match.group(1)!r} is required by the task")
    return resolved


def parse_reward_file(directory: Path) -> tuple[dict[str, float] | None, str | None]:
    """Read reward.json when present; only a missing JSON file falls back to reward.txt."""
    for path in (directory / "reward.json", directory / "reward.txt"):
        if path.is_file():
            return _parse_one_reward_file(path)
    return None, "no reward.json or reward.txt was written"


def _parse_one_reward_file(path: Path) -> tuple[dict[str, float] | None, str | None]:
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
        if not isinstance(value, (int, float)):
            return None, f"{path.name}: {key!r} is not a finite number"
        try:
            value = float(value)
        except OverflowError:
            return None, f"{path.name}: {key!r} is not a finite number"
        if not math.isfinite(value):
            return None, f"{path.name}: {key!r} is not a finite number"
        rewards[str(key)] = value
    return rewards, None


def select_reward(rewards: dict[str, float]) -> float | None:
    """Harbor's rule: the ``reward`` key, else the only key. Several keys and none named ``reward`` is ambiguous."""
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


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


class HarborEnvironmentServer(
    BaseEnvironmentServer[SingleAgentTurnRequest, HarborEpisodeResponse], AggregateMetricsMixin
):
    config: HarborEnvironmentServerConfig
    request_model = SingleAgentTurnRequest
    response_model = HarborEpisodeResponse

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        # Episode keys with an attempt inside run(); a redelivered key is refused while its attempt is live.
        self._active: set[str] = set()
        # Seeded sessions, by episode key; listed here until the sandbox is stopped.
        self._sessions: dict[str, HarborSession] = {}
        # Ownership starts before sandbox.start(), and ends only after stop() succeeds.
        self._sandboxes: dict[str, AsyncSandbox] = {}
        self._tasks: dict[tuple[Path, str], asyncio.Task[HarborTask]] = {}
        self._snapshots = tempfile.TemporaryDirectory(prefix="nemo-gym-harbor-tasks-")
        self._shutting_down = False
        artifacts = self.config.artifacts_dir
        if artifacts is None or not artifacts.is_absolute():
            results_root = Path(get_global_config_dict().get("results_dir", RESULTS_DIR)).resolve()
            artifacts = results_root / (artifacts if artifacts is not None else "harbor/verifier")
        self._artifacts_dir = artifacts.resolve()

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

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Aggregate the episode results; the verifier's fields sit at their top level, as a verify response's do."""
        return compute_aggregate_metrics(
            body.verify_responses,
            compute_metrics_fn=self.compute_metrics,
            get_key_metrics_fn=self.get_key_metrics,
            compute_repeat_metrics_fn=self.compute_repeat_metrics,
        )

    # -- lifecycle -------------------------------------------------------------------------

    async def _stop_sandbox(self, episode_key: str) -> None:
        """Stop the episode's sandbox; a failed stop keeps it owned for the next attempt or shutdown.

        Separate verification stops the agent's sandbox itself (``HarborSession.stop_agent_side``) while the
        episode key stays reserved, so a redelivered episode cannot start a second sandbox under it; the
        reservation is released here without a second stop.
        """
        sandbox = self._sandboxes.get(episode_key)
        session = self._sessions.get(episode_key)
        if sandbox is not None:
            if session is None or not session.agent_sandbox_stopped:
                await sandbox.stop()
            del self._sandboxes[episode_key]
        self._sessions.pop(episode_key, None)

    async def shutdown(self) -> None:
        self._shutting_down = True
        for episode_key in list(self._sandboxes):
            try:
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop abandoned Harbor sandbox")
        pending = [task for task in self._tasks.values() if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not self._sandboxes:
            await asyncio.to_thread(self._snapshots.cleanup)
            self._tasks.clear()

    def _snapshot_task(self, folder: Path, expected: str) -> HarborTask:
        snapshot_root = Path(tempfile.mkdtemp(dir=self._snapshots.name))
        try:
            snapshot = snapshot_root / folder.name
            shutil.copytree(folder, snapshot, symlinks=True)
            task = load_task(snapshot)
            if task.digest != expected:
                raise HTTPException(
                    409, f"Task folder {folder} changed since materialization; re-materialize the taskset"
                )
            return task
        except (OSError, HarborTaskError) as exc:
            shutil.rmtree(snapshot_root)
            raise HTTPException(422, f"Could not snapshot task {folder}: {exc}") from exc
        except BaseException:
            shutil.rmtree(snapshot_root)
            raise

    # -- task resolution -------------------------------------------------------------------

    async def _resolve_task(self, task_id: TaskId, task_data: dict[str, Any]) -> HarborTask:
        taskset = self.config.tasksets.get(task_id.taskset)
        if taskset is None:
            raise HTTPException(404, f"Taskset {task_id.taskset!r} is not served by this environment server")
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
        key = (folder, expected)
        pending = self._tasks.get(key)
        if pending is None:
            # Concurrent first episodes share the same copy/hash, including if a caller is cancelled.
            pending = asyncio.create_task(asyncio.to_thread(self._snapshot_task, folder, expected))
            self._tasks[key] = pending

            def remove_failed_load(done: asyncio.Task[HarborTask]) -> None:
                if done.cancelled() or done.exception() is not None:
                    self._tasks.pop(key, None)

            pending.add_done_callback(remove_failed_load)
        return pending.result() if pending.done() else await asyncio.shield(pending)

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
            | {"nemo_gym_environment_server": self.config.name, "harbor_task": task.task_id[:63], "harbor_role": role}
        )
        if ttl is None:
            ttl = (
                self.config.default_episode_timeout_seconds
                + self.config.cleanup_timeout_seconds
                + self.config.sandbox_ttl_slack_s
            )
        # The override replaces only the keys it names, so a GPU task routed to the GPU provider keeps `gpu`.
        resources = _sandbox_resources(environment) | (self.config.sandbox_resources_override or {})
        env = dict(self.config.sandbox_env) | resolve_env_templates(task_env)
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

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, episode_key: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        self._sandboxes[episode_key] = sandbox
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

    async def _create_sandbox_within_deadline(self, task: HarborTask, episode_key: str) -> AsyncSandbox:
        """The agent's ``_create_sandbox`` under the ready deadline; the timeout is a seed failure, retried."""
        return await self._within_ready_deadline(self._create_sandbox(task, task.workdir, episode_key=episode_key))

    async def _within_ready_deadline[T](self, create: Awaitable[T]) -> T:
        """A sandbox create bounded by the configured ready timeout, whatever the provider's retry budget.

        The provider retries a create on its own schedule and its request may block as long as the cluster
        takes to allocate the pod; `sandbox_ready_timeout_s` is this server's promise, so it is enforced
        here too, on the agent's sandbox and on the separate verifier's alike. The timeout raises
        ``HarborSeedTimeoutError``; the cancelled provider attempt deletes the sandbox it may have started.
        """
        deadline_s = self.config.sandbox_ready_timeout_s + SEED_CREATE_GRACE_S
        deadline = asyncio.timeout(deadline_s)
        try:
            async with deadline:
                return await create
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise HarborSeedTimeoutError(
                f"the sandbox was not ready within {deadline_s:g}s "
                f"(sandbox_ready_timeout_s={self.config.sandbox_ready_timeout_s:g} plus {SEED_CREATE_GRACE_S:g}s grace)"
            ) from exc

    async def _create_verifier_sandbox(self, task: HarborTask, *, episode_key: str | None = None) -> AsyncSandbox:
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
        if episode_key is not None:
            # Owned from before start(), like the agent's sandbox, so a failed or interrupted stop is retried.
            self._sandboxes[_verifier_key(episode_key)] = sandbox
        await self._within_ready_deadline(sandbox.start(spec))
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
        result = await _exec_as_root(sandbox, " && ".join(commands), timeout_s=60)
        if result.return_code != 0:
            raise RuntimeError(f"Could not prepare {workdir}: {result.stderr or result.stdout}")
        environment_dir = task.path / "environment"
        if (
            task.config.environment.docker_image
            and environment_dir.is_dir()
            and not any(
                (environment_dir / name).exists()
                for name in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")
            )
        ):
            await upload_dir(sandbox, environment_dir, workdir)
        return workdir

    async def _run_setup_commands(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        for command in self.config.sandbox_setup_commands:
            # Root first, like every other setup step; sandboxes that cannot switch users fall back.
            result = await _exec_as_root(sandbox, command, timeout_s=self.config.sandbox_setup_timeout_s)
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

    # -- the episode -----------------------------------------------------------------------

    async def run(self, request: SingleAgentTurnRequest, cleanup: CleanupContext) -> HarborEpisodeResponse:
        episode_key = request.episode_id.capture_key
        task_input = request.task.task_input

        # A lost reply makes the client resend the same episode while the first attempt may still be running.
        # That attempt owns the key's sandbox; the repeat must not stop it, so it is refused before anything
        # is registered. The client retries once the first attempt has returned (or been cleaned up).
        if episode_key in self._active:
            raise self._failure(
                stage="seed", failure_reason="A previous attempt of this episode is still running", terminal=False
            )
        self._active.add(episode_key)

        async def release_key() -> None:
            self._active.discard(episode_key)

        # Register before anything is provisioned, so a cancelled or timed-out seed still stops what it started.
        # Final cleanup runs after run() returns, outside the episode deadline, and retries a failed stop at shutdown.
        async def close_sandbox() -> None:
            try:
                await self._stop_sandbox(episode_key)
            finally:
                # A separate verifier owned under the derived key (started, or interrupted while starting)
                # belongs to this episode too.
                if _verifier_key(episode_key) in self._sandboxes:
                    await self._stop_sandbox(_verifier_key(episode_key))

        cleanup.register_cleanup("harbor episode key", release_key)
        cleanup.register_cleanup("harbor sandbox", close_sandbox)
        try:
            session = await self._seed(request, episode_key)
            access = await self._sandbox_access(session)
        except HTTPException as error:
            # 4xx: the task or row is at fault and a retry would fail the same way. 5xx: the provider failed.
            raise self._failure(stage="seed", failure_reason=str(error.detail), terminal=error.status_code < 500)
        except Exception as error:
            # The sandbox exists but cannot be described to the harness (serialize failed): a provider-side
            # seed failure, retried elsewhere; the cleanup stops what was claimed.
            raise self._failure(stage="seed", failure_reason=f"{type(error).__name__}: {error}", terminal=False)

        agent_session_id = f"agent-session-{uuid4().hex}"
        agent_cookies: dict[str, str] = {}
        agent_observations: AgentObservationBundle | None = None

        async def close_agent() -> None:
            nonlocal agent_observations
            close_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                ).model_dump(mode="json"),
                cookies=agent_cookies,
            )
            await raise_for_status(close_http_response)
            agent_close_response = AgentCloseSessionResponse.model_validate(
                await get_response_json(close_http_response)
            )
            if agent_close_response.agent_session_id != agent_session_id:
                raise ValueError("Agent close returned a different agent_session_id")
            agent_observations = agent_close_response.agent_observations

        # Register cleanup before seed so cancellation can close a remotely created session even if its response is lost.
        agent_cleanup = cleanup.register_cleanup("agent session", close_agent)

        try:
            agent_create_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    sandbox_access=access,
                ).model_dump(mode="json"),
            )
            await raise_for_status(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
            if agent_session.agent_session_id != agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            agent_cookies = _cookies(agent_create_http_response)
        except Exception as error:
            raise self._failure(
                stage="agent",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
            ) from error

        agent_response = None
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=task_input.responses_create_params,
                cookies=agent_cookies,
            )
            await raise_for_status(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                agent_cookies = response_cookies
            agent_response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
        except Exception as error:
            # The agent's evidence of the failed turn arrives with its session close.
            try:
                await agent_cleanup.close()
            except Exception:
                pass  # Final unwind retries the close and logs its failure.
            raise self._failure(
                stage="agent",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
                agent_observations=agent_observations,
            ) from error

        # Verification needs this close first: the harness must have let go of the sandbox, and its
        # observations travel with the result.
        try:
            await agent_cleanup.close()
        except Exception as error:
            raise self._failure(
                stage="cleanup",
                failure_reason=str(error),
                terminal=not _is_retryable_dependency_error(error),
                partial_response=agent_response,
            ) from error

        try:
            outcome = await self._run_verifier(session)
        except HTTPException as error:
            raise self._failure(
                stage="verification",
                failure_reason=str(error.detail),
                terminal=error.status_code < 500,
                partial_response=agent_response,
                agent_observations=agent_observations,
            ) from error

        result = HarborEpisodeResult(
            responses_create_params=task_input.responses_create_params,
            response=agent_response,
            **outcome,
            ng_agent_observations=agent_observations,
        )
        return HarborEpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=result)

    async def _seed(self, request: SingleAgentTurnRequest, episode_key: str) -> HarborSession:
        """Resolve the task and start its sandbox; the sandbox stays owned by the server until it is stopped."""
        if self._shutting_down:
            raise HTTPException(503, "Environment server is shutting down")
        task = await self._resolve_task(request.task.task_id, request.task.task_input.task_data)
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
        if task.user is not None:
            raise HTTPException(422, "Agent user overrides ([agent].user or Dockerfile USER) are not supported yet")
        # Fail before provisioning; retain templates in the snapshot and resolve at use time.
        resolve_env_templates(task.env)
        resolve_env_templates(task.config.verifier.env)
        if task.config.verifier.environment is not None:
            resolve_env_templates(task.config.verifier.environment.env)
        if self._shutting_down:
            raise HTTPException(503, "Environment server is shutting down")
        for stale_key in (episode_key, _verifier_key(episode_key)):
            if stale_key not in self._sandboxes:
                continue
            # Left by a finished attempt whose stop failed (run() refuses the key while an attempt is live):
            # the agent's sandbox or its separate verifier. Retry the stop here; the repeat goes ahead only
            # once the old sandbox is gone.
            try:
                await self._stop_sandbox(stale_key)
            except Exception as exc:
                LOGGER.exception(f"Could not stop the sandbox of a previous attempt of {episode_key}")
                raise HTTPException(
                    503, f"A sandbox of a previous attempt of this episode could not be stopped yet: {exc}"
                ) from exc
        try:
            sandbox = await self._create_sandbox_within_deadline(task, episode_key)
            self._sandboxes[episode_key] = sandbox
            workdir = await self._prepare_workdir(sandbox, task, task.workdir)
            await self._run_setup_commands(sandbox, task)
        except BaseException as exc:
            try:
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop Harbor sandbox after failed seed")
            if not isinstance(exc, Exception):
                raise
            LOGGER.exception("Sandbox setup failed for %s", task.task_id)
            raise HTTPException(503, f"Could not set up sandbox for {task.task_id!r}: {exc}") from exc
        session = HarborSession(
            task=task,
            taskset=request.task.task_id.taskset,
            identity=(request.episode_id, request.task.task_id),
            sandbox=sandbox,
            workdir=workdir,
            session_id=episode_key,
        )
        self._sessions[episode_key] = session
        return session

    def _agent_responses_path(self, request: SingleAgentTurnRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        agent_config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.agent_server.name,
        )
        token_capture = bool(block.get("enabled", False)) and (
            bool(block.get("all_agents", False)) or bool(agent_config.get("token_id_capture", False))
        )
        capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{capture_segment}/v1/responses"

    @staticmethod
    def _failure(
        *,
        stage: str,
        failure_reason: str,
        terminal: bool,
        partial_response: Any = None,
        agent_observations: Any = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            SingleAgentTurnFailure(
                stage=stage,
                failure_reason=failure_reason[:2000],
                terminal=terminal,
                partial_response=partial_response,
                agent_observations=agent_observations,
            )
        )

    # -- verification ----------------------------------------------------------------------

    async def _run_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Run ``tests/test.sh`` and read the reward it wrote, in the agent's sandbox or a separate one."""
        if session.task.config.is_shared_verifier:
            return await self._run_shared_verifier(session) | {"verifier_mode": "shared"}
        return await self._run_separate_verifier(session) | {"verifier_mode": "separate"}

    def _reserve_logs_dir(self) -> Path:
        """A fresh host directory for one verification's logs.

        Episode IDs are opaque caller input, never host paths. Reserve a fresh directory so
        neither traversal nor stale output from a reused ID can supply verifier files.
        """
        if self._artifacts_dir.resolve() != self._artifacts_dir:
            raise RuntimeError("Verifier artifact root was redirected after server startup")
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = self._artifacts_dir / f"session-{uuid4().hex}"
        # Exclusive mkdir keeps the directory fresh while honoring the process umask and parent ACLs.
        logs_dir.mkdir()
        logs_dir = logs_dir.resolve()
        if not logs_dir.is_relative_to(self._artifacts_dir):
            raise RuntimeError("Verifier log directory escapes the configured artifact root")
        return logs_dir

    async def _collect_reward(
        self, sandbox: AsyncSandbox, task: HarborTask, logs_dir: Path, *, timed_out: bool
    ) -> tuple[dict[str, float] | None, str | None]:
        """Download the reward files, then whatever else the verifier wrote.

        Reward transfer is required; optional diagnostics cannot invalidate a measured score, and
        a timed-out run has no reward to transfer but its partial logs are still worth keeping.
        """
        rewards = problem = None
        if not timed_out:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt"))
            rewards, problem = parse_reward_file(logs_dir)
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        return rewards, problem

    async def _run_separate_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Harbor's separate mode: collect the agent's artifacts, stop its sandbox, verify in a fresh one.

        The verifier sandbox comes from ``[verifier.environment]`` (or the task image when the
        mode is ``separate`` without one). ``/logs/artifacts`` and every ``artifacts`` entry are
        copied through the host into the same paths, then ``tests/`` is uploaded and ``test.sh``
        runs there. Sidecar (Compose) artifacts and hooks are not collected yet.
        """
        task = session.task
        settings = task.config.verifier
        started = asyncio.get_running_loop().time()
        try:
            logs_dir = self._reserve_logs_dir()
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        artifacts_dir = logs_dir / "artifacts"
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
                # Best effort per artifact, as Harbor collects them: what the agent left behind (an
                # unreadable tree, a link that escapes the folder, a file where a folder was expected)
                # cannot mask the sample. Symbolic links travel as links, because the restore target is
                # another sandbox, not the host. Anything else (a lost sandbox, a full disk, a dropped
                # connection) is Gym's outage and still masks.
                try:
                    kind = await download_path(session.sandbox, artifact.source, host, links="keep")
                except (
                    SandboxTransferError,
                    tarfile.TarError,
                    FileExistsError,
                    IsADirectoryError,
                    NotADirectoryError,
                ) as exc:
                    LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} not collected: {exc}")
                    continue
                if kind is not None:
                    restored.append((host, artifact.source))
            # Harbor stops the agent's container before the verifier starts; nothing may keep running.
            # The episode key stays reserved until the episode's final cleanup, which releases it without
            # stopping the agent side again.
            await session.sandbox.stop()
            session.agent_sandbox_stopped = True

            verifier = await self._create_verifier_sandbox(task, episode_key=session.session_id)
            prepare = await _exec_as_root(
                verifier,
                f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR} && chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR}",
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            await upload_dir(verifier, task.path / "tests", TESTS_DIR)
            for host, source in restored:
                await upload_path(verifier, host, source)
            budget = int(settings.timeout_sec)
            result = await verifier.exec(
                f"chmod +x {TESTS_DIR}/test.sh; timeout --signal=KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            timed_out = result.error_type == "timeout" or result.return_code == 137
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            rewards, problem = await self._collect_reward(verifier, task, logs_dir, timed_out=timed_out)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Separate verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        finally:
            # Owned from before start(): a create that failed or timed out is released here too.
            if _verifier_key(session.session_id) in self._sandboxes:
                try:
                    await self._stop_sandbox(_verifier_key(session.session_id))
                except Exception:
                    # Still owned: shutdown retries the stop.
                    LOGGER.exception(f"Could not stop the verifier sandbox for {task.task_id}")
        return self._score(
            task, settings, result, logs_dir, started, timed_out=timed_out, rewards=rewards, problem=problem
        )

    async def _run_shared_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote."""
        task = session.task
        sandbox = session.sandbox
        settings = task.config.verifier
        started = asyncio.get_running_loop().time()
        try:
            logs_dir = self._reserve_logs_dir()
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        try:
            # Replace agent-planted directory links before cleanup: find does not follow its starting
            # symlink, while tar does. Unlink /logs first without touching its former target.
            roots = (str(Path(VERIFIER_LOGS_DIR).parent), TESTS_DIR, VERIFIER_LOGS_DIR, AGENT_LOGS_DIR)
            unlink_roots = [f"if test -L {shlex.quote(path)}; then rm -- {shlex.quote(path)}; fi" for path in roots]
            prepare = await _exec_as_root(
                sandbox,
                " && ".join(
                    unlink_roots
                    + [
                        f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                        f"find {TESTS_DIR} {VERIFIER_LOGS_DIR} -mindepth 1 -delete",
                        f"chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                    ]
                ),
                timeout_s=60,
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
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            # `timeout` exits 137 when it had to kill test.sh; a provider that gave up first reports "timeout".
            timed_out = result.error_type == "timeout" or result.return_code == 137
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            rewards, problem = await self._collect_reward(sandbox, task, logs_dir, timed_out=timed_out)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        return self._score(
            task, settings, result, logs_dir, started, timed_out=timed_out, rewards=rewards, problem=problem
        )

    def _score(
        self,
        task: HarborTask,
        settings: Any,
        result: SandboxExecResult,
        logs_dir: Path,
        started: float,
        *,
        timed_out: bool,
        rewards: dict[str, float] | None,
        problem: str | None,
    ) -> dict[str, Any]:
        """Turn the verifier's exit and the reward it wrote into the reward fields."""
        # Absolute, so the row names a folder that exists from wherever the run was launched; the server
        # process runs from its own folder, where a relative artifacts_dir would otherwise resolve.
        extras = {
            "verifier_return_code": result.return_code,
            "verifier_logs_dir": str(Path(logs_dir).resolve()),
            "verifier_seconds": round(asyncio.get_running_loop().time() - started, 1),
        }
        if timed_out:
            return (
                self._measured(
                    0.0,
                    VERIFIER_TIMEOUT_KIND,
                    f"test.sh exceeded [verifier].timeout_sec={settings.timeout_sec}; tail: {_stdout_tail(logs_dir)}",
                )
                | extras
            )
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


def _verifier_key(episode_key: str) -> str:
    """The ownership key of an episode's separate-verifier sandbox."""
    return f"{episode_key}/verifier"


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


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable_dependency_error(error: Exception) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    # A dropped connection mid-body raises ClientPayloadError, which is transient like a refused connection.
    return isinstance(error, (ClientConnectionError, ClientPayloadError, TimeoutError))


if __name__ == "__main__":
    HarborEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HarborEnvironmentServer.run_webserver()  # noqa: F401
