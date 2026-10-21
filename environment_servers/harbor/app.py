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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

from environment_servers.harbor.entrypoint import (
    EntrypointLaunch,
    HarborEntrypointError,
    HarborEntrypointMixin,
    _cancel_entrypoint,
)
from environment_servers.harbor.overlay import HarborOverlayError, HarborOverlayMixin
from environment_servers.harbor.sandboxes import (
    COMPOSE_IMAGES_FILE,
    PULL_RETRY_CONCURRENCY,
    SEED_CREATE_GRACE_S,
    HarborSandboxMixin,
)
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
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, TaskId
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK, get_first_server_config_dict, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.server_utils import (
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResult
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_configs import load_compose_document
from nemo_gym.tasks.harbor.materialize import STEP_METADATA_KEY, HarborTaskInput
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE
from nemo_gym.tasks.harbor.registry import ImageConfigError
from nemo_gym.tasks.harbor.task import HarborTurn
from nemo_gym.tasks.harbor.verifier_env import TESTS_DOCKERFILE
from nemo_gym.tool_access import MCPStreamableHTTPConnection, MCPToolAccess, ToolAccess


LOGGER = logging.getLogger(__name__)

# A multi-turn episode in which no turn produced a verifier result to roll up.
NO_TURN_RESULT_KIND = "harbor:no_turn_result"


class HarborTurnResult(BaseModel):
    """One turn of a multi-step task: what the agent returned and what the step's verifier wrote.

    ``verifier_rewards`` is ``None`` when the turn produced no verifier result (``test.sh`` timed out
    or wrote no valid reward file); such a turn ends the episode.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    reward: float
    verifier_rewards: dict[str, float] | None = None
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None
    verifier_network: str | None = None
    response: NeMoGymResponse
    agent_observations: AgentObservationBundle | None = None
    # The harness's own turns from the session close, as ``ng_trajectory`` carries them for a single-turn task.
    trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)


@dataclass
class _TurnProgress:
    """What the running turn has produced so far: the evidence the task deadline's failure carries when it fires."""

    stage: str = "agent"
    response: NeMoGymResponse | None = None
    observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = None


# The agent transcript could not be exported to ATIF; the turn is masked so a converter defect is never a reward.
TRAJECTORY_EXPORT_KIND = "harbor:trajectory_export_failed"
# The episode ran past the time budget of its task (see `enforce_task_timeouts`).
TASK_DEADLINE_KIND = "harbor:task_deadline"


class HarborEpisodeResult(SingleAgentTurnResult):
    """The episode's verify response plus what the Harbor verifier produced.

    ``verifier_rewards`` is the parsed ``reward.json`` (``reward.txt`` becomes ``{"reward": value}``).
    The ``verifier_*`` fields are ``None`` when ``test.sh`` never ran.

    A multi-step task fills ``turns`` in order. The top-level reward follows
    ``turn_reward_strategy`` over them (Harbor's ``multi_step_reward_strategy``), and ``response``,
    ``responses_create_params`` and ``ng_agent_observations`` are those of the last turn that ran.
    """

    verifier_rewards: dict[str, float] | None = None
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None
    # "shared": test.sh ran in the agent's sandbox; "separate": in its own sandbox from [verifier.environment].
    verifier_mode: Literal["shared", "separate"] | None = None
    # The network the verifier ran under (the task's `[verifier].network_mode`, "public" when unset).
    verifier_network: str | None = None
    turns: list[HarborTurnResult] | None = None
    turn_reward_strategy: Literal["mean", "final"] | None = None
    turns_without_verifier_result: int | None = None
    # The turn after which the episode stopped early: a mask, no verifier result, or a missed min_reward.
    stopped_after_turn: str | None = None


class HarborEpisodeRequest(BaseEpisodeRequest[HarborTaskInput]):
    """Request for one Harbor episode: one turn, or a multi-step task's steps as turns."""


class HarborEpisodeResponse(BaseEpisodeResponse[HarborEpisodeResult]):
    """Response for one Harbor episode: a result, or a staged failure with the agent's evidence."""

    failure: SingleAgentTurnFailure | None = None


class ComposeVariable(BaseModel):
    """The deployment's value for one ``${NAME}`` a task's Compose file leaves undefined: exactly one source.

    ``value`` is literal, ``env`` names a variable of this server's process, and ``model_server`` names a Gym
    model server whose sandbox-reachable URL (with the episode's rollout prefix) becomes the value.
    """

    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    env: str | None = None
    model_server: str | None = None

    @model_validator(mode="after")
    def one_source(self) -> "ComposeVariable":
        if sum(source is not None for source in (self.value, self.env, self.model_server)) != 1:
            raise ValueError("A compose variable takes exactly one of value, env or model_server")
        return self


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
    # Grant the task's `[[environment.mcp_servers]]` URL entries to the agent as MCP tool accesses. "off" keeps
    # them in /tmp/.nemo-gym/task.json only; "required" makes an agent that cannot connect fail its seed.
    task_mcp_grants: Literal["off", "optional", "required"] = "off"
    # Values for the `${NAME}` references a task's Compose file leaves to the deployment, such as a sidecar's
    # model endpoint and key. Resolved at seed; the task and Harbor's own variables take precedence.
    compose_variables: dict[str, ComposeVariable] = Field(default_factory=dict)


def min_reward_failure(rewards: dict[str, float] | None, min_reward: float | dict[str, float] | None) -> str | None:
    """Why a turn misses its step's ``min_reward``, or ``None`` when it passes or sets none.

    As in Harbor, a number gates the ``reward`` key, a mapping gates each key it names, and a key the
    rewards lack (or a turn without a verifier result) counts as negative infinity.
    """
    if min_reward is None:
        return None
    thresholds = {"reward": float(min_reward)} if isinstance(min_reward, (int, float)) else min_reward
    for key, threshold in thresholds.items():
        actual = (rewards or {}).get(key, float("-inf"))
        if actual < threshold:
            return f"{key}={actual} is below min_reward {threshold}"
    return None


def roll_up_turns(turns: list["HarborTurnResult"], strategy: str) -> dict[str, Any]:
    """The episode's reward fields from its turns, by Harbor's ``multi_step_reward_strategy``.

    A masked turn masks the episode. ``final`` is the last turn that ran, as Harbor takes the last step
    result. ``mean`` averages over the turns that produced a verifier result: the reward is the mean of
    their rewards, and ``verifier_rewards`` the per-key means with a key a turn lacks counting 0. With
    nothing to roll up the episode is measured 0, as a single turn without a reward file is.
    """
    masked = next((turn for turn in turns if turn.mask_sample), None)
    if masked is not None:
        reason = f"turn {masked.name!r}: {masked.failure_reason}"[:2000]
        return {"reward": 0.0, "mask_sample": True, "failure_kind": masked.failure_kind, "failure_reason": reason}
    if strategy == "final":
        last = turns[-1]
        if last.verifier_rewards is None:
            reason = f"turn {last.name!r}: {last.failure_reason}"[:2000]
            return {"reward": 0.0, "mask_sample": False, "failure_kind": last.failure_kind, "failure_reason": reason}
        return {"reward": last.reward, "verifier_rewards": last.verifier_rewards}
    measured = [turn for turn in turns if turn.verifier_rewards is not None]
    if not measured:
        last = turns[-1]
        reason = f"no turn produced a verifier result; turn {last.name!r}: {last.failure_reason}"
        return HarborEnvironmentServer._measured(0.0, last.failure_kind or NO_TURN_RESULT_KIND, reason)
    keys = sorted({key for turn in measured for key in turn.verifier_rewards})
    means = {key: sum(turn.verifier_rewards.get(key, 0.0) for turn in measured) / len(measured) for key in keys}
    return {"reward": sum(turn.reward for turn in measured) / len(measured), "verifier_rewards": means}


class HarborEnvironmentServer(
    BaseEnvironmentServer[HarborEpisodeRequest, HarborEpisodeResponse],
    AggregateMetricsMixin,
    HarborSandboxMixin,
    HarborSessionMixin,
    HarborVerificationMixin,
    HarborEntrypointMixin,
    HarborOverlayMixin,
    HarborUsersMixin,
):
    config: HarborEnvironmentServerConfig
    request_model = HarborEpisodeRequest
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
        # Task digests whose Compose sidecars without deploy.resources were warned about once (`_warn_unsized_sidecars`).
        self._unsized_sidecars_warned: set[str] = set()
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
        hooks = sum(hook.timeout_sec + COLLECT_HOOK_GRACE_S for hook in task.config.verifier.collect)
        if task.turns:
            # Every step is a turn of its own: a fresh harness session, the hooks and the step's verifier.
            budget = sum(
                self.config.harness_setup_allowance_s
                + task.config.step_agent_timeout_sec(turn.config)
                + hooks
                + task.config.step_verifier_timeout_sec(turn.config)
                + self.config.verifier_grace_s
                for turn in task.turns
            )
        else:
            budget = (
                self.config.harness_setup_allowance_s
                + task.config.agent.timeout_sec
                + hooks
                + task.config.verifier.timeout_sec
                + self.config.verifier_grace_s
            )
        if separate:
            budget += self.config.sandbox_ready_timeout_s
            if _verifier_gets_overlay(task, separate):
                budget += task.config.environment.build_timeout_sec
        return budget

    def _seed_bound_s(self, task: HarborTask) -> float:
        """The most the seed may take before the task's own deadline opens: every sandbox start it may make
        (``sandbox_ready_timeout_s`` plus the create grace per attempt, ``image_pull_retries + 1`` attempts: a
        single sandbox runs its attempts under one such deadline, a Compose group each under its own) and the
        Dockerfile overlay when there is one. The nominal one-start allowance of :meth:`_task_budget_s` sizes
        the sandbox lifetime; this is the bound an episode deadline that starts at the request has to cover."""
        attempts = self.config.image_pull_retries + 1
        bound = (self.config.sandbox_ready_timeout_s + SEED_CREATE_GRACE_S) * attempts
        if task.overlay:
            bound += task.config.environment.build_timeout_sec
        return bound

    def _episode_bound_s(self, task: HarborTask) -> float:
        """What one episode of a multi-step task may take from the request: the seed's bound
        (:meth:`_seed_bound_s`) and everything after the sandbox is ready (:meth:`_post_seed_budget_s`). The
        quantity the episode deadline is raised to and the sandbox lifetime follows above
        ``default_episode_timeout_seconds``, so a seed that needs a pull retry or a slow sidecar start cannot eat
        the final turns' budget and end the episode with the base server's generic timeout. It is a worst case:
        the seed's refusal without enforcement compares the nominal one-start :meth:`_task_budget_s` with the
        default and names this bound beside it."""
        return self._seed_bound_s(task) + self._post_seed_budget_s(task)

    def _episode_budget_s(self, task: HarborTask) -> float:
        """Wall time one episode of this task may take: its own budget when enforced, within the run-wide deadline.

        A multi-step task is the exception to the cap: its steps may add up to more than
        ``default_episode_timeout_seconds`` (six steps of 1800 s agent and verifier time already do), and an
        episode cut off at the default would be retried, never scored, so with ``enforce_task_timeouts`` its
        budget is the task's own even above the default (:meth:`_episode_bound_s`, which
        :meth:`episode_timeout_seconds` raises the episode deadline to). Without enforcement the seed refuses a
        task whose nominal one-start episode (:meth:`_task_budget_s`) does not fit the default (:meth:`_seed`).
        """
        default = self.config.default_episode_timeout_seconds
        if not self.config.enforce_task_timeouts:
            return default
        if task.turns and (bound := self._episode_bound_s(task)) > default:
            return bound
        return min(default, self._task_budget_s(task))

    async def episode_timeout_seconds(self, request: HarborEpisodeRequest) -> float:
        """The episode's wall-clock bound: ``default_episode_timeout_seconds``, raised to the task's own bound
        for a multi-step task whose episode may take more (see :meth:`_episode_bound_s`), so the deadline the
        base server opens before the seed covers the seed's own bounds and matches the sandbox lifetime. A task
        that does not resolve keeps the default; the seed reports why.

        The lookup is the seed's own (:meth:`_resolve_task`: one shielded folder snapshot per task, shared by its
        concurrent episodes), run here before the deadline opens, so it is bounded by the default itself: a
        snapshot still copying after that long leaves the episode the default, and the seed finds the snapshot
        (or waits for it under the deadline) rather than copying the folder again."""
        default = self.config.default_episode_timeout_seconds
        if not self.config.enforce_task_timeouts:
            return default
        try:
            task = await asyncio.wait_for(
                self._resolve_task(request.task.task_id, request.task.task_input.task_data), timeout=default
            )
        except Exception:
            return default
        if not task.turns:
            return default
        return max(default, self._episode_bound_s(task))

    # -- the episode -----------------------------------------------------------------------

    async def run(self, request: HarborEpisodeRequest, cleanup: CleanupContext) -> HarborEpisodeResponse:
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

        # The task's own time budget bounds the episode from here, the sandbox being ready: the seed (image
        # pulls, their retries, the overlay) has its own bounds and is not charged against it. It fires only
        # when the harness did not honour the `[agent].timeout_sec` it was handed.
        budget = self._post_seed_budget_s(session.task) if self.config.enforce_task_timeouts else None
        task_deadline = asyncio.timeout(budget) if budget is not None else None
        progress = _TurnProgress()
        try:
            async with task_deadline if task_deadline is not None else nullcontext():
                if task_input.turns:
                    return await self._run_turns(request, session, access, cleanup, progress)

                agent_response, agent_observations, agent_trajectory = await self._agent_turn(
                    request,
                    access,
                    task_input.responses_create_params,
                    cleanup,
                    session.task,
                    progress,
                    agent_timeout_sec=session.task.config.agent.timeout_sec,
                )
                progress.stage = "verification"
                try:
                    outcome = await self._verified_outcome(session, request, agent_response)
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
            raise self._deadline_failure(session.task, budget, progress)

        result = HarborEpisodeResult(
            responses_create_params=task_input.responses_create_params,
            response=agent_response,
            **outcome,
            ng_agent_observations=agent_observations,
            ng_trajectory=agent_trajectory,
        )
        return HarborEpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=result)

    def _deadline_failure(self, task: HarborTask, budget: float, progress: "_TurnProgress") -> HandledEpisodeError:
        """The staged failure of an episode that ran past its task's time budget, with the turn's evidence so far."""
        if task.turns:
            budget_parts = (
                f"over {len(task.turns)} turn(s): harness setup, each step's [agent].timeout_sec, "
                f"{len(task.config.verifier.collect)} [[verifier.collect]] hook(s) and its [verifier].timeout_sec"
            )
        else:
            budget_parts = (
                f"harness setup, [agent].timeout_sec={task.config.agent.timeout_sec:g}, "
                f"{len(task.config.verifier.collect)} [[verifier.collect]] hook(s) "
                f"and [verifier].timeout_sec={task.config.verifier.timeout_sec:g}"
            )
        return self._failure(
            stage=progress.stage,
            failure_reason=(
                f"Episode exceeded the time budget of task {task.task_id!r} ({budget:.0f}s from the "
                f"sandbox being ready: {budget_parts}); set enforce_task_timeouts: false to run without it"
            ),
            terminal=False,
            failure_kind=TASK_DEADLINE_KIND,
            partial_response=progress.response,
            agent_observations=progress.observations,
            trajectory=progress.trajectory,
        )

    async def _verified_outcome(
        self,
        session: HarborSession,
        request: HarborEpisodeRequest,
        agent_response: NeMoGymResponse,
        turn: tuple[HarborTurn, Any] | None = None,
    ) -> dict[str, Any]:
        """Export the turn's transcript for the verifier and run it: the task's verifier for a single-turn
        episode, or the shared verifier with a step's settings and ``tests/`` for ``turn`` (the step and the
        request it was sent). A transcript the converter cannot represent masks the turn instead of grading it."""
        task = session.task
        try:
            if turn is None:
                trajectory = self._export_trajectory(request, agent_response)
            else:
                trajectory = self._export_trajectory(request, agent_response, turn[1])
        except AtifExportError as error:
            LOGGER.error(f"{task.task_id}: could not export the agent transcript to ATIF: {error}")
            return self._masked(TRAJECTORY_EXPORT_KIND, f"Could not export the agent transcript to ATIF: {error}")
        if turn is None:
            return await self._run_verifier(session, trajectory)
        step = turn[0]
        return await self._run_shared_verifier(
            session,
            trajectory,
            settings=self._turn_verifier_settings(task, step),
            test_dirs=(task.path / "tests", task.path / step.tests_dir),
        )

    async def _agent_turn(
        self,
        request: HarborEpisodeRequest,
        access: SandboxAccess,
        responses_create_params: Any,
        cleanup: CleanupContext,
        task: HarborTask,
        progress: "_TurnProgress",
        *,
        agent_timeout_sec: float,
    ) -> tuple[NeMoGymResponse, AgentObservationBundle | None, TrajectoryRecord | None]:
        """One agent turn in its own session on the episode's sandbox: seed, one activation, close.

        The session is closed before this returns, so the harness has let go of the sandbox and its
        observations are in hand. A failed seed or activation raises the staged episode failure.
        ``progress`` is kept current with what the turn has produced, for the task deadline's failure.
        """
        progress.stage, progress.response, progress.observations, progress.trajectory = "agent", None, None, None
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
            progress.observations = agent_observations
            progress.trajectory = agent_trajectory

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
                    tool_accesses=self._task_tool_accesses(task),
                    sandbox_access=access,
                    agent_timeout_sec=agent_timeout_sec,
                ).model_dump(mode="json"),
            )
            await raise_for_status(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
            if agent_session.agent_session_id != agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            agent_cookies = _cookies(agent_create_http_response)
            environment = task.config.environment
            if (environment.mcp_servers or environment.skills_dir) and not agent_session.supports_task_tool_context:
                # The task's tools would never reach this harness, and the run would score a silent zero.
                raise HTTPException(
                    422,
                    f"Task {task.task_id!r} declares MCP servers or a skills directory, which the agent "
                    f"{self.config.agent_server.name!r} does not support; run it with a harness that declares "
                    "supports_task_tool_context in its seed response",
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

        agent_response = None
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=_harness_params(responses_create_params),
                cookies=agent_cookies,
            )
            await raise_for_status(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                agent_cookies = response_cookies
            agent_response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
            progress.response = agent_response
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
            LOGGER.warning(f"{task.task_id}: agent session close timed out, verifying anyway: {error}")
            agent_observations = None
            progress.observations = None
        except Exception as error:
            raise dependency_failure(
                error,
                failure_model=SingleAgentTurnFailure,
                stage="cleanup",
                partial_response=agent_response,
            ) from error
        return agent_response, agent_observations, agent_trajectory

    async def _run_turns(
        self,
        request: HarborEpisodeRequest,
        session: HarborSession,
        access: SandboxAccess,
        cleanup: CleanupContext,
        progress: "_TurnProgress",
    ) -> HarborEpisodeResponse:
        """Run a multi-step task's steps as turns on one sandbox, as Harbor's multi-step trial does.

        Each turn opens a fresh agent session (Harbor's default, ``resume_trajectory = false``), sends the
        step's instruction, closes the session and runs the step's verifier: the task's ``tests/`` with
        ``steps/<name>/tests/`` on top. Before a turn the sandbox takes the step's ``[steps.agent].user`` when
        it has one and the oracle gets the step's own solution (:meth:`_prepare_turn`). The episode stops
        after a turn that is masked, produces no verifier result, or misses its ``min_reward``; the remaining
        turns do not run.
        """
        task = session.task
        inputs = request.task.task_input.turns
        if [turn.name for turn in inputs] != [turn.name for turn in task.turns]:
            raise self._failure(
                stage="seed",
                failure_reason=(
                    f"Row turns {[turn.name for turn in inputs]} do not match the steps of task {task.task_id!r} "
                    f"{[turn.name for turn in task.turns]}; re-materialize the taskset"
                ),
                terminal=True,
            )
        results: list[HarborTurnResult] = []
        stopped_after: str | None = None
        for turn, turn_input in zip(task.turns, inputs):
            try:
                access = await self._prepare_turn(session, turn, access)
            except HTTPException as error:
                raise self._failure(
                    stage="agent",
                    failure_reason=f"turn {turn.name!r}: {error.detail}",
                    terminal=error.status_code < 500,
                ) from error
            except Exception as error:
                # The sandbox could not take the turn's user or solution: a provider-side failure, retried elsewhere.
                raise self._failure(
                    stage="agent",
                    failure_reason=f"turn {turn.name!r}: {type(error).__name__}: {error}",
                    terminal=False,
                ) from error
            response, observations, trajectory = await self._agent_turn(
                request,
                access,
                turn_input.responses_create_params,
                cleanup,
                task,
                progress,
                agent_timeout_sec=task.config.step_agent_timeout_sec(turn.config),
            )
            progress.stage = "verification"
            try:
                # The step's verifier runs as its own user, never the step's agent user: the sandbox's default is
                # the seed's again for the hooks and test.sh (Harbor's `with_default_user` around the shared
                # verifier), then the step's user is back in place for the next turn's switch to compare with.
                with self._as_seed_default_user(session):
                    outcome = await self._verified_outcome(
                        session, request, response, (turn, turn_input.responses_create_params)
                    )
            except HTTPException as error:
                raise self._failure(
                    stage="verification",
                    failure_reason=f"turn {turn.name!r}: {error.detail}",
                    terminal=error.status_code < 500,
                    partial_response=response,
                    agent_observations=observations,
                    trajectory=trajectory,
                ) from error
            result = HarborTurnResult(
                name=turn.name, response=response, agent_observations=observations, trajectory=trajectory, **outcome
            )
            results.append(result)
            if (
                result.mask_sample
                or result.verifier_rewards is None
                or min_reward_failure(result.verifier_rewards, turn.config.min_reward) is not None
            ):
                stopped_after = turn.name if len(results) < len(task.turns) else None
                break

        strategy = task.config.multi_step_reward_strategy
        last = results[-1]
        episode = HarborEpisodeResult(
            responses_create_params=inputs[len(results) - 1].responses_create_params,
            response=last.response,
            ng_agent_observations=last.agent_observations,
            ng_trajectory=last.trajectory,
            verifier_mode="shared",
            turns=results,
            turn_reward_strategy=strategy,
            turns_without_verifier_result=sum(1 for result in results if result.verifier_rewards is None),
            stopped_after_turn=stopped_after,
            **roll_up_turns(results, strategy),
        )
        return HarborEpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=episode)

    async def _prepare_turn(self, session: HarborSession, turn: HarborTurn, access: SandboxAccess) -> SandboxAccess:
        """What a turn needs in place before its agent session opens: the step's user and the step's solution.

        Harbor runs each step as its own ``[steps.agent].user`` (else the task's ``[agent].user``), so when the
        turn's user differs from the one the sandbox runs as, the sandbox's default user, the working directory's
        owner and the user the harness is handed switch to it (:meth:`_apply_turn_user`), and the access the
        harness gets is rebuilt for it; the verifier keeps its own user and fallback. The oracle alone then gets
        the step's ``solution/`` at ``/solution`` (:meth:`_place_turn_solution`).
        """
        task = session.task
        user = str(turn.config.agent.user) if turn.config.agent.user is not None else task.user
        # Compared as written (the seed's or the previous turn's), not with the form the sandbox installed.
        if user != session.requested_user:
            await self._apply_turn_user(session, user)
            access = await self._sandbox_access(session)
        await self._place_turn_solution(session, turn)
        return access

    async def _apply_turn_user(self, session: HarborSession, user: str | None) -> None:
        """Switch the episode's sandbox to ``user`` for the turns that follow: the default of every exec that
        names no user (a numeric uid resolved to its name, as at seed), the working directory's owner, and the
        user the harness runs as. A ``name:group`` pair was refused at seed on a provider that cannot apply it.

        The working directory's owner as the seed left it (the task's user, or what the image or overlay gave
        it) is read once, before the first switch, and a turn that switches back to the image's own default
        (``None``) gives the directory back to that owner, as a step running as the image's non-root user
        would find it under Harbor, which never changes it.
        """
        task = session.task
        if session.workdir_owner is None:
            session.workdir_owner = await self._workdir_owner(session)
        await self._set_default_user(session.sandbox, task, session.user_names, user=user)
        # The sandbox now runs as the resolved form of the step's user (a uid resolved to its name, as at seed, or
        # the integer uid when it names no user); the working directory and the harness's own access follow the
        # same value, as installed, so the oracle runs as it too (an integer stays OpenSandbox's native uid, where
        # its string would be looked up as an account name).
        resolved = session.sandbox.default_user
        if resolved is None:
            await self._restore_workdir_owner(session)
        else:
            await self._own_workdir(session.sandbox, session.workdir, str(resolved))
        session.agent_user = resolved
        session.requested_user = user

    def _task_tool_accesses(self, task: HarborTask) -> list[ToolAccess]:
        """The task's MCP servers as agent tool grants, when ``task_mcp_grants`` asks for them.

        Only URL entries reached over streamable HTTP can be granted; a Compose sidecar's URL names its
        service host, which resolves inside the task's sandbox group, where the harness runs.
        """
        if self.config.task_mcp_grants == "off":
            return []
        accesses: list[ToolAccess] = []
        for server in task.config.environment.mcp_servers:
            if server.transport != "streamable-http" or not server.url:
                LOGGER.warning(
                    f"{task.task_id}: MCP server {server.name!r} ({server.transport}) cannot be granted; "
                    "only streamable-http URL entries can"
                )
                continue
            accesses.append(
                MCPToolAccess(
                    name=server.name,
                    required=self.config.task_mcp_grants == "required",
                    connection=MCPStreamableHTTPConnection(url=server.url),
                )
            )
        return accesses

    async def _seed(self, request: HarborEpisodeRequest, episode_key: str) -> HarborSession:
        """Resolve the task and start its sandbox; the sandbox stays owned by the server until it is stopped."""
        if self._shutting_down:
            raise HTTPException(503, "Environment server is shutting down")
        task = await self._resolve_task(request.task.task_id, request.task.task_input.task_data)
        if not task.needs_sandbox:
            raise HTTPException(
                422, f"Task {task.task_id!r} declares no image; sandbox-less tasks are not supported yet"
            )
        # The row's shape against the task's: a multi-step task's row names its steps as turns, in order, and a
        # single-turn task's row has none. Checked here, before anything is provisioned, in both directions: a row
        # with the digest but no `turns` (hand-built, or materialized before the steps were) would otherwise take the
        # single-turn path, one agent session and the task-level verifier, with the steps never run.
        row_turns = [turn.name for turn in request.task.task_input.turns]
        task_turns = [turn.name for turn in task.turns]
        if row_turns != task_turns:
            raise HTTPException(
                422,
                f"Row turns {row_turns} do not match the steps of task {task.task_id!r} {task_turns}; "
                "re-materialize the taskset",
            )
        separate = self._separate_verifier(task)
        if separate and task.turns:
            # The loader admits multi-step tasks in Harbor's shared mode only; `separate_verifier_for_artifacts`
            # must not force them into a mode their turns do not run.
            raise HTTPException(
                422,
                f"Task {task.task_id!r} is a multi-step task; separate verification of its turns is not supported "
                "yet (unset separate_verifier_for_artifacts or drop the task's artifacts)",
            )
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
        for turn in task.turns:
            # The sources the turn's verifier resolves its env from (`_verifier_env`: the server's secrets, then the
            # process environment), so a judge key only `secrets` supplies passes here as it does at the task level.
            resolve_env_templates(turn.config.verifier.env, self.config.secrets)
            # A step's own agent user is installed for its turn (`_apply_turn_user`); a pair the provider cannot
            # apply is refused before the earlier turns are paid for, as the task's own `[agent].user` is.
            self._refuse_unsupported_group(task, turn.config.agent.user, where=f" step {turn.name!r}")
        if task.needs_compose:
            # A Compose file the deployment leaves incomplete (a variable no compose_variables entry sets) is a
            # configuration error: refused before anything is provisioned, and not retried.
            try:
                load_compose_document(task, operator_variables=self._compose_variable_values(episode_key))
            except ImageConfigError as exc:
                raise HTTPException(422, f"Compose file of task {task.task_id!r}: {exc}") from exc
        if task.turns:
            # Three figures of one episode: what the steps need once the sandbox is ready (`_post_seed_budget_s`),
            # the nominal episode of one sandbox start, with the overlay build, before that (`_task_budget_s`), and
            # the worst case a deadline opened before the seed has to cover, every image pull attempt at its
            # timeout (`_episode_bound_s`), the one the deadline is raised to and the sandbox lifetime follows.
            # The refusal compares the nominal figure: a normal seed makes one start, and a task that fits the
            # default that way is run, not refused for the worst case; the WARNINGs name both figures.
            bound = self._episode_bound_s(task)
            nominal = self._task_budget_s(task)
            after_seed = self._post_seed_budget_s(task)
            default = self.config.default_episode_timeout_seconds
            need = (
                f"its {len(task.turns)} steps need up to {after_seed:.0f}s once the sandbox is ready (harness setup, "
                f"agent and verifier budgets, hooks and grace per step), {nominal:.0f}s with one sandbox start and "
                f"up to {bound:.0f}s with every image pull retry"
            )
            if not self.config.enforce_task_timeouts:
                # Nothing derives a longer deadline without enforcement, so the default is the deadline. An
                # episode that cannot fit it after one sandbox start would be cut short and retried, never scored:
                # refused with the setting named. One that fits only without a retried seed runs, with a WARNING
                # once per episode: a retried seed is a possibility whose outcome is a retryable timeout.
                if nominal > default:
                    raise HTTPException(
                        422,
                        f"Task {task.task_id!r}: {need}, above default_episode_timeout_seconds={default:.0f} after one "
                        f"sandbox start already, and enforce_task_timeouts is off, so the episode would be cut short "
                        f"and retried; raise default_episode_timeout_seconds to at least {nominal:.0f} ({bound:.0f} to "
                        "cover a retried seed), or set enforce_task_timeouts: true to give the episode the task's own "
                        "budget",
                    )
                if bound > default:
                    LOGGER.warning(
                        f"{task.task_id}: {need}; default_episode_timeout_seconds={default:.0f} covers the episode "
                        "after one sandbox start but not after a retried seed, and enforce_task_timeouts is off, so "
                        "an episode whose seed retries its image pulls may time out and be retried; set "
                        "enforce_task_timeouts: true to give the episode the task's own bound"
                    )
            elif bound > default:
                # Once per episode of such a task, at WARNING so the run's log shows which tasks run above the
                # run-wide default (a notable operational fact, not an error): the deadline and the sandbox
                # lifetime follow the task's own bound, so the episode is not cut short and retried.
                LOGGER.warning(
                    f"{task.task_id}: {need}; default_episode_timeout_seconds={default:.0f} is below that bound, so "
                    f"the episode deadline and the sandbox lifetime follow the task's own bound ({bound:.0f}s)"
                )
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
                requested_user=task.user,
            )
            # A Compose group ran the overlay on main inside its start, before main's uploads and command.
            if compose is not None:
                names = self._overlay_names.pop(episode_key, {})
            else:
                names = await self._apply_overlay(sandbox, task)
            await self._set_default_user(sandbox, task, names)
            session.user_names = names
            # What a turn of a multi-step task restores for its verifier (`_as_seed_default_user`).
            session.seed_default_user = sandbox.default_user
            # The harness is handed the user as installed, as a turn's switch leaves it (`_apply_turn_user`): the
            # name a uid resolved to, or the integer uid when it names no user (OpenSandbox's native `uid=` exec
            # path, where the string the task wrote would be looked up as an account name and fail).
            session.agent_user = sandbox.default_user
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

    def _agent_responses_path(self, request: HarborEpisodeRequest) -> str:
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


def _harness_params(responses_create_params: Any) -> Any:
    """The turn's request as the harness receives it: the row's step name (``harbor_step`` in ``metadata``) stays
    in the row and the result, where it names the turn, and is not sent. Harbor gives its agent no step name, and
    the sandbox harnesses (OpenCode, Hermes, Claude Code, OpenClaw) refuse a request whose ``metadata`` carries any
    key but the agent's time budget, so sending it would fail every multi-step task on those harnesses."""
    metadata = getattr(responses_create_params, "metadata", None)
    if isinstance(responses_create_params, dict):
        metadata = responses_create_params.get("metadata")
    if not metadata or STEP_METADATA_KEY not in metadata:
        return responses_create_params
    trimmed = {key: value for key, value in metadata.items() if key != STEP_METADATA_KEY}
    if isinstance(responses_create_params, dict):
        return {**responses_create_params, "metadata": trimmed}
    return responses_create_params.model_copy(update={"metadata": trimmed})


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


if __name__ == "__main__":
    HarborEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = HarborEnvironmentServer.run_webserver()  # noqa: F401
