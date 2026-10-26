# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment server for Harbor-format tasks.

One server instance serves one or more tasksets. Each taskset maps to a folder of
task folders in Gym's shared datasets folder plus the digest every task had when its
rows were materialized (see ``nemo_gym.tasks.harbor``).

``/run`` owns one whole episode, in this order:

- seed: check the row's digest against the mapping and the folder, start the task's
  image as a sandbox through the configured provider, run the ``RUN`` lines of an
  overlay-mode Dockerfile with the ``ENV``, ``WORKDIR`` and ``USER`` the loader resolved for
  each (see ``nemo_gym.tasks.harbor.dockerfile``), create the working directory, start the
  image's recorded ``ENTRYPOINT`` beside the provider's keepalive the way Harbor's Compose files
  leave it running (with ``sh -c "sleep infinity"`` as its arguments), then wait for the image's
  healthcheck and the task's. A ``RUN`` line that fails, or an entrypoint that exits non-zero
  before the hand-over, is a terminal seed failure (the task's Dockerfile does not apply); a
  provider that cannot run one is retryable;
- agent: open an agent session on the configured agent server, handing it a
  ``SandboxAccess`` for that sandbox (an agent instance implemented by the Harbor oracle,
  ``harbor_oracle_agent``, first gets the snapshot's ``solution/`` at ``/solution`` and the
  task's ``[solution].env``), post the task's ``responses_create_params`` to it, then close
  the session and keep what the agent observed;
- verify: write the agent transcript to ``/logs/agent/trajectory.json`` (ATIF v1.7), run
  ``tests/test.sh`` inside the same sandbox (Harbor's shared verifier mode) or in a fresh one
  (separate mode), then read ``/logs/verifier/reward.json`` or ``reward.txt``;
- cleanup: stop the sandbox, whatever happened before.

Every episode runs under its task's own time budget unless ``enforce_task_timeouts`` is off, and
the sandbox lives for that budget. Operator secrets reach the ``test.sh`` process only.

Reward rule: when ``test.sh`` ran, the sample is measured. A missing or invalid reward
file scores 0 with a ``failure_kind``. Only a Gym-side failure (sandbox lost, transfer
failed) masks the sample. A task that cannot be set up is a handled episode failure:
terminal when the task itself is at fault (unknown task, changed folder, unsupported
settings), retryable when the sandbox provider failed.
"""

import asyncio
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from environment_servers.harbor.entrypoint import (
    EntrypointLaunch,
    HarborEntrypointError,
    HarborEntrypointMixin,
    _cancel_entrypoint,
)
from environment_servers.harbor.overlay import HarborOverlayError, HarborOverlayMixin
from environment_servers.harbor.sandboxes import COMPOSE_IMAGES_FILE, PULL_RETRY_CONCURRENCY, HarborSandboxMixin
from environment_servers.harbor.session import HarborSession, HarborSessionMixin, resolve_env_templates
from environment_servers.harbor.users import HarborUsersMixin
from environment_servers.harbor.verifier import (
    COLLECT_HOOK_GRACE_S,
    HarborVerificationMixin,
    _verifier_gets_overlay,
    _verifier_image,
    _verifier_key,
)
from nemo_gym import RESULTS_DIR
from nemo_gym.atif_export import AtifExportError
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    HandledEpisodeError,
    dependency_failure,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT, AgentServerRef, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_types import BaseEpisodeResponse, TaskId
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK, get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose
from nemo_gym.server_utils import (
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnRequest, SingleAgentTurnResult
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE
from nemo_gym.tasks.harbor.verifier_env import TESTS_DOCKERFILE


LOGGER = logging.getLogger(__name__)


# The agent transcript could not be exported to ATIF; the turn is masked so a converter defect is never a reward.
TRAJECTORY_EXPORT_KIND = "harbor:trajectory_export_failed"
# The episode ran past the time budget of its task (see `enforce_task_timeouts`).
TASK_DEADLINE_KIND = "harbor:task_deadline"


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
    # The network the verifier ran under (the task's `[verifier].network_mode`, "public" when unset).
    verifier_network: str | None = None


class HarborEpisodeResponse(BaseEpisodeResponse[HarborEpisodeResult]):
    """Response for one Harbor episode: a result, or a staged failure with the agent's evidence."""

    failure: SingleAgentTurnFailure | None = None


class HarborTasksetConfig(BaseModel):
    """Where one taskset's folders live and which digest each task was materialized with.

    ``dataset_digest`` and ``dataset_toml_digest`` are the dataset's identity as prepare recorded it
    (``manifest.toml``'s ``[dataset]`` digests), so the run config says which dataset ran.
    """

    model_config = ConfigDict(extra="forbid")

    folder: Path
    tasks: dict[str, str]
    # Tasks found below the first level of the folder: task_id -> path relative to `folder`.
    paths: dict[str, str] = Field(default_factory=dict)
    dataset_digest: str | None = None
    dataset_toml_digest: str | None = None


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
    # Provider block for tasks that declare GPUs (agent or verifier); None runs them on `sandbox_provider`.
    gpu_sandbox_provider: str | None = None
    # Pass a task's `gpu_types` to the provider. Off: deployments without that filter reject it.
    request_gpu_type: bool = False
    # Extra seconds the provider exec waits beyond `[verifier].timeout_sec` for the in-container `timeout` to
    # kill test.sh and return; it does not extend the tests' own deadline, which is the task's `timeout_sec`.
    verifier_grace_s: float = Field(default=30, ge=0)
    # Verifier logs are downloaded here, one fresh folder per verification.
    artifacts_dir: Path | None = None
    # Operator secrets (judge API keys and the like), by environment variable name. They reach the `test.sh`
    # process only, and only the ones the task names: they fill `${NAME}` templates in `[verifier].env`, as
    # Harbor fills them from its own environment. A secret no template names is passed nowhere. They never
    # enter a sandbox's creation environment, the agent's turn, a task folder or a row.
    secrets: dict[str, str] = Field(default_factory=dict)
    # Give every episode the time budget of its task instead of the run-wide `default_episode_timeout_seconds`
    # alone: once its sandbox is ready, `harness_setup_allowance_s`, `[agent].timeout_sec`, every
    # `[[verifier.collect]]` hook's `timeout_sec` with its grace, `[verifier].timeout_sec` and the verifier
    # grace (plus a separate verifier's start). Sandbox setup, image pull retries included, is not charged
    # against it; the sandbox lifetime covers both. Off ignores task timeouts.
    enforce_task_timeouts: bool = True
    # Seconds a harness may spend setting itself up in the sandbox before the task's `[agent].timeout_sec` starts.
    harness_setup_allowance_s: float = Field(default=300, ge=0)
    # How many times a sandbox start that failed on an image pull (the agent's sandbox, a Compose group or
    # a separate verifier's sandbox) is retried here before the failure is reported as retryable. Bounded,
    # jittered and gated per server.
    image_pull_retries: int = Field(default=2, ge=0)
    # Verify tasks that declare `artifacts` in a fresh sandbox even when `task.toml` keeps Harbor's shared
    # mode: the agent cannot tamper with the grader. Tasks without artifacts keep their declared mode.
    separate_verifier_for_artifacts: bool = False


class HarborEnvironmentServer(
    BaseEnvironmentServer[SingleAgentTurnRequest, HarborEpisodeResponse],
    AggregateMetricsMixin,
    HarborSandboxMixin,
    HarborSessionMixin,
    HarborVerificationMixin,
    HarborEntrypointMixin,
    HarborOverlayMixin,
    HarborUsersMixin,
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
        self._sandboxes: dict[str, AsyncSandbox | AsyncSandboxCompose] = {}
        self._tasks: dict[tuple[Path, str], asyncio.Task[HarborTask]] = {}
        # Resolved Compose documents and the compose.yaml each is written to, by task snapshot and digest.
        self._compose_documents: dict[tuple[Path, str], asyncio.Task[tuple[dict[str, Any], Path]]] = {}
        # The user names an overlay resolved inside a Compose group's start (`_create_compose`), by episode key,
        # until the seed takes them.
        self._overlay_names: dict[str, dict[str, str]] = {}
        self._snapshots = tempfile.TemporaryDirectory(prefix="nemo-gym-harbor-tasks-")
        # Built or declared verifier images found without /tests/test.sh, each warned about once.
        self._verifier_images_without_tests: set[str] = set()
        self._pull_retry_gate = asyncio.Semaphore(PULL_RETRY_CONCURRENCY)
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
                if session is not None:
                    await _cancel_entrypoint(session.entrypoint)
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

    def _snapshot_task(self, folder: Path, expected: str, root: Path | None = None) -> HarborTask:
        snapshot_root = Path(tempfile.mkdtemp(dir=self._snapshots.name))
        root = folder.parent if root is None else root
        try:
            snapshot = snapshot_root / folder.name
            shutil.copytree(folder, snapshot, symlinks=True)
            # Dataset-level files live at the dataset root (the task's parent, or higher for a task
            # discovered deeper) and the loader reads them from the snapshot's parent, so the snapshot
            # gets copies there: the dataset's own settings (dataset.toml), the recorded image
            # configurations and the built images.
            for name in (DATASET_FILE, COMPOSE_IMAGES_FILE, BUILT_IMAGES_FILE):
                sidecar = root / name
                if sidecar.is_file():
                    shutil.copy2(sidecar, snapshot_root / name)
            task = load_task(snapshot)
            if task.digest != expected:
                raise HTTPException(
                    409, f"Task folder {folder} changed since materialization; re-materialize the taskset"
                )
            return task
        except HarborTaskError as exc:
            # The task folder itself is malformed: a retry would fail the same way.
            shutil.rmtree(snapshot_root)
            raise HTTPException(422, f"Could not snapshot task {folder}: {exc}") from exc
        except OSError as exc:
            # The host copy failed (disk full, transient I/O): not a task defect, so the episode is retried.
            shutil.rmtree(snapshot_root)
            raise HTTPException(503, f"Could not copy task {folder}: {exc}") from exc
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
        root = Path(taskset.folder)
        relative = Path(taskset.paths.get(task_id.task_id, task_id.task_id))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise HTTPException(422, f"Task {task_id.task_id!r} has an invalid path in the taskset mapping")
        folder = root / relative
        key = (folder, expected)
        pending = self._tasks.get(key)
        if pending is None:
            # Concurrent first episodes share the same copy/hash, including if a caller is cancelled.
            pending = asyncio.create_task(asyncio.to_thread(self._snapshot_task, folder, expected, root))
            self._tasks[key] = pending

            def remove_failed_load(done: asyncio.Task[HarborTask]) -> None:
                if done.cancelled() or done.exception() is not None:
                    self._tasks.pop(key, None)

            pending.add_done_callback(remove_failed_load)
        return pending.result() if pending.done() else await asyncio.shield(pending)

    def _task_budget_s(self, task: HarborTask) -> float:
        """The task's own time budget for one episode, from its `task.toml` and this server's allowances.

        One sandbox start and the Dockerfile overlay when there is one (the seed's nominal allowance),
        then everything the episode deadline covers (:meth:`_post_seed_budget_s`). Sizes the sandbox
        lifetime; the deadline itself starts when the sandbox is ready.
        """
        budget = self.config.sandbox_ready_timeout_s + self._post_seed_budget_s(task)
        if task.overlay:
            budget += task.config.environment.build_timeout_sec
        return budget

    def _post_seed_budget_s(self, task: HarborTask) -> float:
        """What an episode may take once its sandbox is ready: the per-task deadline.

        The harness's setup, `[agent].timeout_sec`, the `[[verifier.collect]]` hooks (run one after the
        other in both verifier modes, each within its own `timeout_sec` plus the exec grace), then
        `[verifier].timeout_sec` with its grace, plus a second sandbox start (and overlay) for a separate
        verifier. The seed is not counted: its own bounds are the sandbox ready timeout per start and the
        image pull retries, so a start that recovered on a retry begins the task with its whole budget.
        """
        separate = self._separate_verifier(task)
        budget = (
            self.config.harness_setup_allowance_s
            + task.config.agent.timeout_sec
            + sum(hook.timeout_sec + COLLECT_HOOK_GRACE_S for hook in task.config.verifier.collect)
            + task.config.verifier.timeout_sec
            + self.config.verifier_grace_s
        )
        if separate:
            budget += self.config.sandbox_ready_timeout_s
            if _verifier_gets_overlay(task, separate):
                budget += task.config.environment.build_timeout_sec
        return budget

    def _episode_budget_s(self, task: HarborTask) -> float:
        """Wall time one episode of this task may take: its own budget when enforced, within the run-wide deadline."""
        default = self.config.default_episode_timeout_seconds
        if not self.config.enforce_task_timeouts:
            return default
        return min(default, self._task_budget_s(task))

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
            # The sandbox is seeded; the oracle alone also gets the task's solution before its session opens.
            await self._seed_solution(session)
            access = await self._sandbox_access(session)
        except HTTPException as error:
            # 4xx: the task or row is at fault and a retry would fail the same way. 5xx: the provider failed.
            raise self._failure(stage="seed", failure_reason=str(error.detail), terminal=error.status_code < 500)
        except Exception as error:
            # The sandbox exists but cannot be described to the harness (serialize failed): a provider-side
            # seed failure, retried elsewhere; the cleanup stops what was claimed.
            raise dependency_failure(
                error,
                failure_model=SingleAgentTurnFailure,
                stage="seed",
                terminal=False,
                name_error_type=True,
            )

        agent_session_id = f"agent-session-{uuid4().hex}"
        agent_cookies: dict[str, str] = {}
        agent_observations: AgentObservationBundle | None = None
        # The harness's own turns (a TrajectoryRecord) from its close, forwarded on the result.
        agent_trajectory = None

        async def close_agent() -> None:
            nonlocal agent_observations, agent_trajectory
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
            agent_trajectory = agent_close_response.trajectory

        # Register cleanup before seed so cancellation can close a remotely created session even if its response is lost.
        agent_cleanup = cleanup.register_cleanup("agent session", close_agent)

        # The task's own time budget bounds the episode from here, the sandbox being ready: the seed (image
        # pulls, their retries, the overlay) has its own bounds and is not charged against it. It fires only
        # when the harness did not honour the `[agent].timeout_sec` it was handed.
        budget = self._post_seed_budget_s(session.task) if self.config.enforce_task_timeouts else None
        task_deadline = asyncio.timeout(budget) if budget is not None else None
        stage = "agent"
        agent_response = None
        try:
            async with task_deadline if task_deadline is not None else nullcontext():
                try:
                    agent_create_http_response = await self.server_client.post(
                        server_name=self.config.agent_server.name,
                        url_path="/v1/agent_sessions",
                        json=AgentSeedSessionRequest(
                            agent_session_id=agent_session_id,
                            episode_id=request.episode_id,
                            task_id=request.task.task_id,
                            sandbox_access=access,
                            agent_timeout_sec=session.task.config.agent.timeout_sec,
                        ).model_dump(mode="json"),
                    )
                    await raise_for_status(agent_create_http_response)
                    agent_session = AgentSeedSessionResponse.model_validate(
                        await get_response_json(agent_create_http_response)
                    )
                    if agent_session.agent_session_id != agent_session_id:
                        raise ValueError("Agent seed returned a different agent_session_id")
                    agent_cookies = _cookies(agent_create_http_response)
                    environment = session.task.config.environment
                    if (
                        environment.mcp_servers or environment.skills_dir
                    ) and not agent_session.supports_task_tool_context:
                        # The task's tools would never reach this harness, and the run would score a silent zero.
                        raise HTTPException(
                            422,
                            f"Task {session.task.task_id!r} declares MCP servers or a skills directory, which the "
                            f"agent {self.config.agent_server.name!r} does not support; run it with a harness that "
                            "declares supports_task_tool_context in its seed response",
                        )
                except HTTPException as error:
                    # The configured harness cannot take this task; a retry would be refused the same way.
                    raise self._failure(
                        stage="agent", failure_reason=str(error.detail), terminal=error.status_code < 500
                    ) from error
                except Exception as error:
                    raise dependency_failure(
                        error,
                        failure_model=SingleAgentTurnFailure,
                        stage="agent",
                    ) from error

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
                    raise dependency_failure(
                        error,
                        failure_model=SingleAgentTurnFailure,
                        stage="agent",
                        agent_observations=agent_observations,
                        trajectory=agent_trajectory,
                    ) from error

                # Verification needs this close first: the harness must have let go of the sandbox, and its
                # observations travel with the result.
                try:
                    await agent_cleanup.close()
                except TimeoutError as error:
                    # The turn is finished; a harness that is slow to tear down must not cost its score. Its
                    # observations are lost with the reply, and the final unwind retries the close.
                    LOGGER.warning(f"{session.task.task_id}: agent session close timed out, verifying anyway: {error}")
                    agent_observations = None
                except Exception as error:
                    raise dependency_failure(
                        error,
                        failure_model=SingleAgentTurnFailure,
                        stage="cleanup",
                        partial_response=agent_response,
                    ) from error

                stage = "verification"
                try:
                    trajectory = self._export_trajectory(request, agent_response)
                except AtifExportError as error:
                    LOGGER.error(f"{session.task.task_id}: could not export the agent transcript to ATIF: {error}")
                    outcome = self._masked(
                        TRAJECTORY_EXPORT_KIND, f"Could not export the agent transcript to ATIF: {error}"
                    )
                else:
                    try:
                        outcome = await self._run_verifier(session, trajectory)
                    except HTTPException as error:
                        raise self._failure(
                            stage="verification",
                            failure_reason=str(error.detail),
                            terminal=error.status_code < 500,
                            partial_response=agent_response,
                            agent_observations=agent_observations,
                            trajectory=agent_trajectory,
                        ) from error
        except TimeoutError:
            if task_deadline is None or not task_deadline.expired():
                raise
            # The final unwind closes the agent session and stops the sandbox.
            raise self._failure(
                stage=stage,
                failure_reason=(
                    f"Episode exceeded the time budget of task {session.task.task_id!r} ({budget:.0f}s from the "
                    f"sandbox being ready: harness setup, [agent].timeout_sec={session.task.config.agent.timeout_sec:g}, "
                    f"{len(session.task.config.verifier.collect)} [[verifier.collect]] hook(s) "
                    f"and [verifier].timeout_sec={session.task.config.verifier.timeout_sec:g}); set "
                    "enforce_task_timeouts: false to run without it"
                ),
                terminal=False,
                failure_kind=TASK_DEADLINE_KIND,
                partial_response=agent_response,
                agent_observations=agent_observations,
                trajectory=agent_trajectory,
            )

        result = HarborEpisodeResult(
            responses_create_params=task_input.responses_create_params,
            response=agent_response,
            **outcome,
            ng_agent_observations=agent_observations,
            ng_trajectory=agent_trajectory,
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
        separate = self._separate_verifier(task)
        if separate and _verifier_image(task, separate) is None:
            # The loader refuses this shape (no [verifier.environment].docker_image and no verifier image built from
            # tests/Dockerfile recorded); kept as the seed's invariant for a task object that did not come from it.
            raise HTTPException(
                422,
                f"Task {task.task_id!r} has no image for its separate verifier: [verifier.environment] names no "
                f"docker_image and no verifier image built from {TESTS_DOCKERFILE} is recorded in "
                f"{task.path.parent / BUILT_IMAGES_FILE}; prepare the dataset again (it builds the file when builds "
                "are enabled, and leaves the task out saying why otherwise), or name a prebuilt image",
            )
        # Fail before provisioning; retain templates in the snapshot and resolve at use time. Only the task.toml
        # layers are templates: a Dockerfile ENV in task.env is the literal docker build left (`_task_sandbox_env`).
        resolve_env_templates(task.config.environment.env)
        resolve_env_templates(task.config.verifier.env, self.config.secrets)
        if task.config.verifier.environment is not None:
            resolve_env_templates(task.config.verifier.environment.env)
        # The separate verifier's user is as static as its image: a `name:group` the provider cannot run as fails
        # here, before the rollout it would otherwise follow (`_sandbox_spec` refuses the agent's own the same way).
        self._refuse_verifier_group(task)
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
        compose: AsyncSandboxCompose | None = None
        launch: EntrypointLaunch | None = None
        try:
            if task.needs_compose:
                compose = await self._create_compose(task, episode_key)
                self._sandboxes[episode_key] = compose
                sandbox = compose.services["main"]
            else:
                sandbox = await self._create_sandbox_within_deadline(task, episode_key)
                self._sandboxes[episode_key] = sandbox
            session = HarborSession(
                task=task,
                taskset=request.task.task_id.taskset,
                identity=(request.episode_id, request.task.task_id),
                sandbox=sandbox,
                workdir="",
                session_id=episode_key,
                provider_ref=self._provider_ref(task),
                compose=compose,
            )
            # A Compose group ran the overlay on main inside its start, before main's uploads and command.
            if compose is not None:
                names = self._overlay_names.pop(episode_key, {})
            else:
                names = await self._apply_overlay(sandbox, task)
            await self._set_default_user(sandbox, task, names)
            image_user = await self._image_user_name(sandbox, task, names)
            # The working directory and /logs exist before the entrypoint starts, as Harbor's bind mounts do. For a
            # Compose group the adapter already created the /logs paths main bound, before its command; this mkdir
            # is idempotent there and still makes the whole tree.
            session.workdir = await self._prepare_workdir(sandbox, task, task.workdir)
            # A Compose group's main was started by the adapter, its entrypoint run and its image healthcheck waited.
            managed = compose is not None
            launch = session.entrypoint = await self._start_entrypoint(
                sandbox, task, image_user, prepared_workdir=session.workdir, compose_managed=managed
            )
            await self._wait_healthy(sandbox, task, launch, user=image_user, compose_managed=managed)
            await self._write_task_context(sandbox, task)
            if launch is not None:
                # The last look before the hand-over: an entrypoint that died during the seed's remaining steps
                # (a setup script failing after its first lines) is a seed failure, not a rollout on an empty task.
                await self._check_entrypoint(sandbox, task, launch)
        except HTTPException as exc:
            # A seed step that judged the failure itself keeps its status: the task's `name:group` user refused on
            # a provider that switches user by name only (`HarborUserError`, a 422 the task and the deployment
            # disagree on, which no retry mends) or an environment variable the task requires and the deployment
            # lacks. Refused before the sandbox exists on the create path; one created another way is stopped here.
            try:
                await _cancel_entrypoint(launch)
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop Harbor sandbox after a refused seed")
            LOGGER.error("Seed refused for %s: %s", task.task_id, exc.detail)
            raise
        except (HarborOverlayError, HarborEntrypointError) as exc:
            # The task's own Dockerfile (or the image it built) failed to apply: a task error, not a sandbox one.
            what = "Dockerfile overlay" if isinstance(exc, HarborOverlayError) else "Image ENTRYPOINT"
            try:
                await _cancel_entrypoint(launch)
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop Harbor sandbox after a failed %s", what.lower())
            LOGGER.error("%s failed for %s: %s", what, task.task_id, exc)
            raise HTTPException(422, f"{what} failed for {task.task_id!r}: {exc}") from exc
        except BaseException as exc:
            if not isinstance(exc, Exception):
                # Cancellation (the episode deadline fired): the registered `close_sandbox` cleanup stops
                # the owned sandbox under `cleanup_timeout_seconds`. Stopping it here would run unbounded,
                # since the deadline cancels only once, and hold the episode key for as long as the stop.
                raise
            try:
                await _cancel_entrypoint(launch)
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop Harbor sandbox after failed seed")
            LOGGER.exception("Sandbox setup failed for %s", task.task_id)
            raise HTTPException(503, f"Could not set up sandbox for {task.task_id!r}: {exc}") from exc
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
        trajectory: Any = None,
        failure_kind: str | None = None,
    ) -> HandledEpisodeError:
        # The harness's evidence of a failed turn (its observations and the decisions it took before the failure)
        # travels with the failure row, as it travels with a result.
        return HandledEpisodeError(
            SingleAgentTurnFailure(
                stage=stage,
                failure_reason=failure_reason[:2000],
                terminal=terminal,
                failure_kind=failure_kind,
                partial_response=partial_response,
                agent_observations=agent_observations,
                trajectory=trajectory,
            )
        )


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


if __name__ == "__main__":
    HarborEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HarborEnvironmentServer.run_webserver()  # noqa: F401
