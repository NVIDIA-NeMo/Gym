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

Reward rule: when ``test.sh`` ran, the sample is measured. In ``reward.json`` a ``reward``
key is the score and the other keys are metrics; without one every key is a component of the
score, which is their sum, and the result names them as ``reward_components`` (keys that sum to
``reward``, the contract a trainer asserts). A missing or invalid reward file scores 0 with a
``failure_kind``. Only a Gym-side failure (sandbox lost, transfer failed) masks the sample. A
failed agent response whose harness classified the cause (``ng_agent_outcome``) goes through
``agent_outcome_policy`` before verification, as on ``single_agent_turn``: by default a
transport or unknown cause is retried ungraded instead of scoring 0. A task that cannot be set up is a handled episode failure:
terminal when the task itself is at fault (unknown task, changed folder, unsupported
settings), retryable when the sandbox provider failed.
"""

import asyncio
import logging
import math
import shutil
import tempfile
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from aiohttp import ClientResponseError
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

from environment_servers.harbor.entrypoint import (
    ENTRYPOINT_FAILED_KIND,
    ENTRYPOINT_LOG_NAME,
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
from environment_servers.harbor.steps import HarborStepsMixin, StepPrepareError
from environment_servers.harbor.users import HarborUsersMixin
from environment_servers.harbor.verifier import (
    COLLECT_HOOK_GRACE_S,
    HarborVerificationMixin,
    _verifier_gets_overlay,
    _verifier_image,
    _verifier_key,
    summing_components,
)
from nemo_gym import RESULTS_DIR, failure_kinds
from nemo_gym.agent_activation_types import AgentContinuationRequirements
from nemo_gym.agent_outcome import AgentOutcomePolicyConfig, OutcomeAction, agent_outcome_policy
from nemo_gym.atif_export import AtifExportError
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    HandledEpisodeError,
    dependency_failure,
)
from nemo_gym.base_responses_api_agent import (
    AGENT_ACTIVATION_HEADER,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    AgentSessionCapabilitiesResponse,
)
from nemo_gym.config import (
    TOKEN_CAPTURE_PATH_SEGMENT,
    TOKEN_ID_CAPTURE_BLOCK,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.reward_profile import AggregateMetricsMixin, compute_aggregate_metrics
from nemo_gym.rollout_observability import AgentObservationBundle, TrajectoryRecord
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose, SandboxAccess
from nemo_gym.server_utils import (
    get_response_json,
    is_nemo_gym_fastapi_entrypoint,
    raise_for_status,
)
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnResult
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.cli import ORACLE_AGENT
from nemo_gym.tasks.harbor.compose_overlay import ComposeOverlayError, HarborComposeOverlay, wrap_first_instruction
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.materialize import STEP_METADATA_KEY, HarborTaskInput
from nemo_gym.tasks.harbor.models import HarborMCPServer
from nemo_gym.tasks.harbor.python_verifier import VERIFIER_FILE
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE
from nemo_gym.tasks.harbor.task import HarborTurn
from nemo_gym.tasks.harbor.task_data import (
    TaskDataError,
    TaskDataSchema,
    check_row_matches_folder,
    load_task_data_schema,
)
from nemo_gym.tasks.harbor.verifier_env import TESTS_DOCKERFILE
from nemo_gym.tool_access import MCPStreamableHTTPConnection, MCPToolAccess, ToolAccess


LOGGER = logging.getLogger(__name__)

# A multi-turn episode in which no turn produced a verifier result to roll up.
NO_TURN_RESULT_KIND = "harbor:no_turn_result"


class HarborTurnResult(BaseModel):
    """One turn of a multi-step task: what the agent returned and what the step's verifier wrote.

    ``verifier_rewards`` is ``None`` when the turn produced no verifier result (``test.sh`` timed out
    or wrote no valid reward file); such a turn ends the episode. ``response`` is ``None`` for a turn whose
    step preparation failed (``steps/<name>/workdir/setup.sh`` exited non-zero, or ``[steps.healthcheck]`` never
    passed): no agent session opened and no verifier ran, ``failure_kind`` says why, and the episode stops there
    too, as Harbor stops after a step whose preparation raised.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    reward: float
    verifier_rewards: dict[str, float] | None = None
    # The keys of reward.json that sum to `reward`, when the file names no `reward` key (see verifier.py).
    reward_components: dict[str, float] | None = Field(default=None, exclude_if=lambda value: value is None)
    mask_sample: bool = False
    failure_kind: str | None = None
    failure_reason: str | None = None
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None
    verifier_network: str | None = None
    # "shared": the step's test.sh ran in the agent's sandbox; "separate": in a sandbox of its own for this turn.
    verifier_mode: Literal["shared", "separate"] | None = None
    response: NeMoGymResponse | None = None
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


@dataclass
class _AgentSessionHandle:
    """An open agent session on the episode's sandbox: its id, the cookies that route to it, and, once closed,
    the harness's observations and its own trajectory (cumulative for a continuation session)."""

    agent_session_id: str
    cookies: dict[str, str]
    cleanup: Any
    continuation: bool = False
    observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = None
    closed: bool = False


# The agent transcript could not be exported to ATIF; the turn is masked so a converter defect is never a reward.
TRAJECTORY_EXPORT_KIND = "harbor:trajectory_export_failed"
# The episode ran past the time budget of its task (see `enforce_task_timeouts`).
TASK_DEADLINE_KIND = "harbor:task_deadline"


class HarborEpisodeResult(SingleAgentTurnResult):
    """The episode's verify response plus what the Harbor verifier produced.

    ``verifier_rewards`` is the parsed ``reward.json`` (``reward.txt`` becomes ``{"reward": value}``).
    ``reward_components`` names the keys that sum to ``reward``, present only when the file named no
    ``reward`` key (every key is then a component); with a ``reward`` key the other keys are metrics and
    stay in ``verifier_rewards`` alone. The ``verifier_*`` fields are ``None`` when ``test.sh`` never ran.

    That mapping is Harbor's optional, task-specific reading of ``reward.json``, not the shared multi-reward
    contract of :class:`~nemo_gym.base_resources_server.BaseMultiRewardVerifyResponse` (a required mapping whose
    objective keys are the same on every task of an environment): a task names whatever keys its ``test.sh``
    wrote, two tasks of one dataset may name different ones, and a measured episode names none when the file
    had a ``reward`` key, was missing or invalid (a measured 0), or its components did not sum to the roll-up.
    Its consumers are per-row inspection and a trainer that reads the mapping row by row and takes the scalar
    ``reward`` alone where it is absent. Gym pads no missing component and renames no key: a dataset meant for
    a fixed-objective trainer (GDPO and the like) must have every task write the same keys without a ``reward``
    key, which is the dataset's contract to keep, not this server's.

    A multi-step task fills ``turns`` in order. The top-level reward follows
    ``turn_reward_strategy`` over them (Harbor's ``multi_step_reward_strategy``), and ``response`` and
    ``responses_create_params`` are those of the last turn that ran; ``ng_agent_observations`` and
    ``ng_trajectory`` are the whole session's under ``turn_sessions: continue`` and, under ``fresh``, that one
    session's when a single turn's agent ran and empty when more did (each step's own session is on its turn, one record
    per step as Harbor keeps one ``steps/<name>/agent/`` per step; a merged episode record is issue #5063).
    Under ``continue`` that pair is the last activation's input and its new output alone (a continuation
    response carries only new work), produced with the earlier steps in the harness's own conversation, so it is
    not a standalone prompt and completion: the record of a continued episode is the cumulative ``ng_trajectory``
    (every trajectory turn names its activation) with, when model-call capture is enabled at the model server
    (``observability_enabled``, opt-in outside ``gym eval submit``), the captured model calls, which hold what the
    model saw and produced; the pair is the last step's request and reply, for inspection.
    """

    verifier_rewards: dict[str, float] | None = None
    reward_components: dict[str, float] | None = Field(default=None, exclude_if=lambda value: value is None)
    verifier_return_code: int | None = None
    verifier_logs_dir: str | None = None
    verifier_seconds: float | None = None
    # "shared": test.sh ran in the agent's sandbox; "separate": in its own sandbox from [verifier.environment]. A
    # multi-step task reports the mode its turns shared, or "mixed" when its steps resolved to both (each turn says).
    verifier_mode: Literal["shared", "separate", "mixed"] | None = None
    # The network the verifier ran under (the task's `[verifier].network_mode`, "public" when unset).
    verifier_network: str | None = None
    turns: list[HarborTurnResult] | None = None
    turn_reward_strategy: Literal["mean", "final"] | None = None
    turns_without_verifier_result: int | None = None
    # The turn after which the episode stopped early: a mask, no verifier result, or a missed min_reward.
    stopped_after_turn: str | None = None
    # How the turns reached the harness: a session per turn, or one continuation session (see `turn_sessions`).
    turn_sessions: Literal["fresh", "continue"] | None = None


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
    # Whether a failed agent response the harness classified is verified, retried or failed ungraded.
    agent_outcome_policy: AgentOutcomePolicyConfig = Field(default_factory=AgentOutcomePolicyConfig)
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
    # grace (plus a separate verifier's start), per step for a multi-step task, with each step's `setup.sh` bound
    # (`step_setup_timeout_s`) and its `[steps.healthcheck]` worst case when it has them. Sandbox setup, image pull
    # retries included, is not charged
    # against it; the sandbox lifetime covers both. Off ignores task timeouts.
    enforce_task_timeouts: bool = True
    # Seconds a harness may spend setting itself up in the sandbox before the task's `[agent].timeout_sec` starts.
    harness_setup_allowance_s: float = Field(default=300, ge=0)
    # Seconds a step's `workdir/setup.sh` may run before the step fails (Harbor bounds it by nothing but the trial's
    # own timeout). Unset: the step's agent budget, `[steps.agent].timeout_sec` else `[agent].timeout_sec`, the
    # task's own bound on that step's time. Counted in the episode budget for every step that ships the script.
    step_setup_timeout_s: float | None = Field(default=None, gt=0)
    # How many times a sandbox start that failed on an image pull (the agent's sandbox, a Compose group or
    # a separate verifier's sandbox) is retried here before the failure is reported as retryable. Bounded,
    # jittered and gated per server.
    image_pull_retries: int = Field(default=2, ge=0)
    # Verify tasks that declare `artifacts` in a fresh sandbox even when `task.toml` keeps Harbor's shared
    # mode: the agent cannot tamper with the grader. Tasks without artifacts keep their declared mode.
    separate_verifier_for_artifacts: bool = False
    # Grant the task's `[[environment.mcp_servers]]` URL entries to the agent as MCP tool accesses. "off" keeps
    # them in /tmp/.nemo-gym/task.json only; "required" makes an agent that cannot connect fail its seed or its
    # first turn, so a harness without supports_task_tool_context can take a task whose MCP servers are all granted.
    task_mcp_grants: Literal["off", "optional", "required"] = "off"
    # How a multi-step task's turns reach the harness. "fresh": one agent session per step (Harbor's default,
    # `resume_trajectory = false`). "continue": one continuation session for the episode, each step an activation
    # of it, so the agent keeps its conversation across steps (Harbor's `--resume-trajectory`); refused before any
    # sandbox for a harness without multi-turn continuation, as Harbor refuses an agent without SUPPORTS_RESUME.
    turn_sessions: Literal["fresh", "continue"] = "fresh"
    # Values for the `${NAME}` references a task's Compose file leaves to the deployment, such as a sidecar's
    # model endpoint and key. Resolved at seed; the task and Harbor's own variables take precedence.
    compose_variables: dict[str, ComposeVariable] = Field(default_factory=dict)
    # The job-level Compose overlay, as Harbor's `extra_docker_compose`, `agents[].mcp_servers` and
    # `prompt_template_path` together: Compose `files` merged onto every task's Compose group (a plain image task
    # becomes the one-service group `main` first), `mcp_servers` merged by name with the task's own and granted
    # with them, and a `prompt_template` rendered around the instruction of the episode's first turn. Prepare
    # reads the same block, so the overlay's images are recorded and its merge checked with the tasks'.
    compose_overlay: HarborComposeOverlay | None = None


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


def roll_up_turns(turns: list["HarborTurnResult"], strategy: str, *, task_id: str = "") -> dict[str, Any]:
    """The episode's reward fields from its turns, by Harbor's ``multi_step_reward_strategy``.

    A masked turn masks the episode. ``final`` is the last turn that ran, as Harbor takes the last step
    result, with its ``reward_components``. ``mean`` averages over every turn that ran: the reward is the mean of
    their rewards, and ``verifier_rewards`` the per-key means with a key a turn lacks counting 0. A turn that
    produced no verifier result (``test.sh`` timed out or wrote no valid reward file; it ends the episode) counts 0
    under every key, and the episode carries its failure kind and reason, measured: that is how the single-turn
    path scores such a verifier run and how ``final`` scores it as a last turn. Harbor's ``_aggregate_step_rewards``
    leaves such a step out of its denominator instead, which would let a later verifier the agent broke (a hung
    ``test.sh``, a reward file never written) lift the mean to the earlier turns' score. ``reward_components``
    are averaged the same way when every measured turn
    named some (the per-key means then sum to the mean reward); a mix of turns with and without
    components names none, since no set of keys would sum to the episode's reward. Components that do not
    sum to the rolled-up reward are dropped with a warning naming the task (``task_id``), the keys and both
    numbers: a results row names ``reward_components`` only when they sum to its ``reward``. With nothing to
    roll up the episode is measured 0, as a single turn without a reward file is.

    A turn whose agent never ran (``response`` is ``None``: its step's own ``workdir/`` merge, ``setup.sh`` or
    healthcheck failed) ends the episode too. As the first turn it masks the episode with the step's failure kind
    under both strategies: no agent has acted, so there is nothing to score (the server raises a terminal agent-stage
    failure for it before rolling up). As a later turn it is that same measured 0: it counts in ``mean`` and makes
    ``final`` 0, and the episode carries the step's failure kind and reason. That is a deliberate divergence from
    Harbor, whose ``_aggregate_step_rewards`` leaves a step without a verifier result out of its denominator and
    whose ``final`` then has no reward at all: a step's ``workdir/``, ``setup.sh`` and healthcheck run in the sandbox
    the earlier turns' agent had, where a process it left running, a working directory it corrupted or a service it
    killed can fail them, and a masked sample (or a mean over the earlier turns alone) would give an RL policy no
    signal against wrecking the next step's setup. The denominator is still the turns that ran: the steps after the
    failed one never ran and are not in it (Harbor's ``_run`` leaves its step loop after a step whose
    ``_prepare_step`` set ``exception_info``, so they are not in its ``step_results`` either), so in a task of three
    or more steps a wrecked second step scores the first turn's reward over 2, not over the task's step count; the
    policy loses the step it broke, not the steps after it. A task whose own step preparation is broken fails the
    same way in every rollout, the oracle's included, under that failure kind.
    """
    masked = next((turn for turn in turns if turn.mask_sample), None)
    if masked is not None:
        reason = f"turn {masked.name!r}: {masked.failure_reason}"[:2000]
        return {"reward": 0.0, "mask_sample": True, "failure_kind": masked.failure_kind, "failure_reason": reason}
    # A first turn whose agent never ran (its failed preparation ended the episode before any agent acted) masks the
    # episode under either strategy. A later turn's failed preparation is a turn without a verifier result below:
    # measured 0, counted in `mean`, the last turn under `final`, with the step's failure kind on the episode.
    if turns[0].response is None:
        reason = f"turn {turns[0].name!r}: {turns[0].failure_reason}"[:2000]
        return HarborEnvironmentServer._masked(turns[0].failure_kind or NO_TURN_RESULT_KIND, reason)
    if strategy == "final":
        last = turns[-1]
        if last.verifier_rewards is None:
            reason = f"turn {last.name!r}: {last.failure_reason}"[:2000]
            return {"reward": 0.0, "mask_sample": False, "failure_kind": last.failure_kind, "failure_reason": reason}
        rolled = {"reward": last.reward, "verifier_rewards": last.verifier_rewards}
        components = _named_components(
            last.reward, last.reward_components, task_id=task_id, where=f"turn {last.name!r}"
        )
        if components is not None:
            rolled["reward_components"] = components
        return rolled
    measured = [turn for turn in turns if turn.verifier_rewards is not None]
    if not measured:
        last = turns[-1]
        reason = f"no turn produced a verifier result; turn {last.name!r}: {last.failure_reason}"
        return HarborEnvironmentServer._measured(0.0, last.failure_kind or NO_TURN_RESULT_KIND, reason)
    keys = sorted({key for turn in measured for key in turn.verifier_rewards})
    # Every turn is in the denominator: a turn without a verifier result counts 0 under every key.
    means = {key: sum(turn.verifier_rewards.get(key, 0.0) for turn in measured) / len(turns) for key in keys}
    reward = sum(turn.reward for turn in measured) / len(turns)
    rolled = {"reward": reward, "verifier_rewards": means}
    # At most one turn has no result, since it ends the episode: the episode names its cause.
    unmeasured = next((turn for turn in turns if turn.verifier_rewards is None), None)
    if unmeasured is not None:
        reason = f"turn {unmeasured.name!r}: {unmeasured.failure_reason}"
        rolled |= HarborEnvironmentServer._measured(reward, unmeasured.failure_kind or NO_TURN_RESULT_KIND, reason)
    if all(turn.reward_components is not None for turn in measured):
        component_keys = sorted({key for turn in measured for key in turn.reward_components})
        component_means = {
            key: sum(turn.reward_components.get(key, 0.0) for turn in measured) / len(turns) for key in component_keys
        }
        components = _named_components(reward, component_means, task_id=task_id, where="the mean over turns")
        if components is not None:
            rolled["reward_components"] = components
    return rolled


def _named_components(
    reward: float, components: dict[str, float] | None, *, task_id: str, where: str
) -> dict[str, float] | None:
    """``components`` when they sum to ``reward``, else ``None`` with a warning that says what was dropped and why."""
    summing = summing_components(reward, components)
    if components is not None and summing is None:
        LOGGER.warning(
            "task %r: reward_components %s from %s sum to %r, not to the episode reward %r; the episode names "
            "no reward_components (a results row names only keys that sum to its reward)",
            task_id,
            sorted(components),
            where,
            math.fsum(components.values()),
            reward,
        )
    return summing


class HarborEnvironmentServer(
    BaseEnvironmentServer[HarborEpisodeRequest, HarborEpisodeResponse],
    AggregateMetricsMixin,
    HarborSandboxMixin,
    HarborSessionMixin,
    HarborVerificationMixin,
    HarborEntrypointMixin,
    HarborOverlayMixin,
    HarborUsersMixin,
    HarborStepsMixin,
):
    config: HarborEnvironmentServerConfig
    request_model = HarborEpisodeRequest
    response_model = HarborEpisodeResponse

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        # Episode keys with an attempt inside run(); a redelivered key is refused while its attempt is live.
        self._active: set[str] = set()
        # The agent server's session capabilities, asked once per process before the first run that needs them.
        self._agent_capabilities: AgentSessionCapabilitiesResponse | None = None
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
        # Job-level `compose_overlay.mcp_servers` entries that cannot be granted (an `sse` entry), each warned about
        # once per run rather than once per turn of every episode.
        self._ungrantable_overlay_servers: set[str] = set()
        self._pull_retry_gate = asyncio.Semaphore(PULL_RETRY_CONCURRENCY)
        self._shutting_down = False
        # Each taskset's dataset-level TaskData (task_data.py beside its task folders), imported once here: None
        # for a dataset that ships none, or the error that keeps every episode of the taskset from seeding.
        self._task_data_schemas: dict[str, TaskDataSchema | TaskDataError | None] = {}
        for name, taskset in self.config.tasksets.items():
            try:
                self._task_data_schemas[name] = load_task_data_schema(Path(taskset.folder))
            except TaskDataError as exc:
                LOGGER.error(f"Taskset {name!r}: its task_data.py does not load; its episodes will be refused: {exc}")
                self._task_data_schemas[name] = exc
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
            for name in (DATASET_FILE, COMPOSE_IMAGES_FILE, BUILT_IMAGES_FILE, VERIFIER_FILE):
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

    def _check_task_data(self, taskset: str, task: HarborTask, task_data: dict[str, Any]) -> None:
        """Refuse a row whose task fields the dataset's ``TaskData`` rejects, or that are not the pinned folder's.

        The digest check pins the folder; this pins the row's ``task_data`` to that folder's ``task_data.json``
        and to the dataset's ``task_data.py`` when it ships one. Both are task or row defects (422): a retry
        would fail the same way.
        """
        schema = self._task_data_schemas.get(taskset)
        if isinstance(schema, TaskDataError):
            raise HTTPException(422, f"Taskset {taskset!r}: {schema}")
        try:
            if schema is not None:
                schema.validate(task_data, task_id=task.task_id)
            check_row_matches_folder(task_data, task)
        except TaskDataError as exc:
            raise HTTPException(422, str(exc)) from exc

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
        verifier. A multi-step task sums that per step, plus the bound of each step's `setup.sh` when it ships
        one (:meth:`_step_setup_budget_s`) and the worst case of its `[steps.healthcheck]` when it declares one
        (:meth:`_step_healthcheck_budget_s`). The seed is not counted: its own bounds are the sandbox ready timeout
        per start and the image pull retries, so a start that recovered on a retry begins the task with its whole
        budget.
        """
        hooks = sum(hook.timeout_sec + COLLECT_HOOK_GRACE_S for hook in task.config.verifier.collect)
        if task.turns:
            # Every step is a turn of its own: a fresh harness session, the step's setup.sh and healthcheck when it
            # has them, the task's hooks and the step's own, and the step's verifier, plus a verifier sandbox start
            # (and overlay) for every turn that verifies separately.
            budget = sum(
                self.config.harness_setup_allowance_s
                + self._step_setup_budget_s(task, turn)
                + self._step_healthcheck_budget_s(turn)
                + task.config.step_agent_timeout_sec(turn.config)
                + hooks
                + sum(hook.timeout_sec + COLLECT_HOOK_GRACE_S for hook in turn.config.verifier.collect)
                + task.config.step_verifier_timeout_sec(turn.config)
                + self.config.verifier_grace_s
                for turn in task.turns
            )
            for turn in task.turns:
                if self._turn_separate(task, turn):
                    budget += self.config.sandbox_ready_timeout_s
                    if _verifier_gets_overlay(self._turn_verifier_view(task, turn), True):
                        budget += task.config.environment.build_timeout_sec
            return budget
        budget = (
            self.config.harness_setup_allowance_s
            + task.config.agent.timeout_sec
            + hooks
            + task.config.verifier.timeout_sec
            + self.config.verifier_grace_s
        )
        separate = self._separate_verifier(task)
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
            # The job-level prompt template wraps the first turn's instruction before anything is provisioned: the
            # harness, the transcript export and the result all see the rendered instruction, as Harbor's agent
            # renders it before its first turn. A template that does not render is the configuration's fault.
            request = self._with_prompt_template(request)
        except ComposeOverlayError as error:
            raise self._failure(stage="seed", failure_reason=str(error), terminal=True)
        task_input = request.task.task_input
        if len(task_input.turns) > 1 and self.config.turn_sessions == "continue":
            # Before anything is provisioned, as Harbor refuses `resume_trajectory` for an agent without resume
            # support before any environment spend; a single step could never continue, so the setting is moot.
            await self._require_turn_continuation()
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
        self._trace_score(session.task.task_id, episode_key, result)
        return HarborEpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=result)

    @staticmethod
    def _trace_score(task_id: str, episode_key: str, result: HarborEpisodeResult) -> None:
        """The episode's final score, at DEBUG (a ``--verbose`` run): one line for both execution paths, the single
        turn's and the multi-step roll-up's (:meth:`_run_turns`), so a verbose trace ends every scored episode the
        same way."""
        LOGGER.debug(
            "%s: episode %s scored %s%s",
            task_id,
            episode_key,
            result.reward,
            f" (masked: {result.failure_kind})" if result.mask_sample else "",
        )

    def _with_prompt_template(self, request: HarborEpisodeRequest) -> HarborEpisodeRequest:
        """``request`` with the overlay's prompt template rendered around the instruction of its first turn.

        Harbor's agent renders ``prompt_template_path`` around the instruction it is given
        (``render_prompt_template``, ``{{ instruction }}`` with ``StrictUndefined``); here the environment
        server does it once, on the first turn's user message, so every harness sees the same rendered
        instruction and the first turn's ATIF transcript (the one its verifier reads) carries it. The template was
        read and compiled when the block was read (``HarborComposeOverlay.compile_template``); each episode renders
        that compiled template (``render_instruction``), with no file read or parse on the event loop. A single-turn
        result's ``responses_create_params`` is the rendered request; a multi-step result records the last turn's
        params, so its rendered first turn is in that turn's transcript only (recording each turn's params on the
        turn result is a follow-up with M34). Later turns of a multi-step task are sent as written (wrapping each
        turn is a parked follow-up). Without a template the request is returned as it is.
        """
        overlay = self.config.compose_overlay
        if overlay is None or overlay.prompt_template is None:
            return request
        render = overlay.render_instruction

        def wrapped(params: Any) -> Any:
            # The row's params are a model here (a mapping on the wire): rendered on its JSON form, read back as one.
            # Only the fields the row set are dumped, so the model read back has the same fields set and the request
            # to the harness (``ServerClient.post`` serializes with ``exclude_unset=True``) carries exactly the row's
            # keys with the rendered content, not every default (``tools: []``, ``tool_choice``, ...) made explicit.
            if isinstance(params, BaseModel):
                dumped = params.model_dump(mode="json", exclude_unset=True)
                return type(params).model_validate(wrap_first_instruction(dumped, render))
            return wrap_first_instruction(params, render)

        request = request.model_copy(deep=True)
        task_input = request.task.task_input
        if task_input.turns:
            first = task_input.turns[0]
            first.responses_create_params = wrapped(first.responses_create_params)
        else:
            task_input.responses_create_params = wrapped(task_input.responses_create_params)
        return request

    def _deadline_failure(self, task: HarborTask, budget: float, progress: "_TurnProgress") -> HandledEpisodeError:
        """The staged failure of an episode that ran past its task's time budget, with the turn's evidence so far."""
        if task.turns:
            budget_parts = (
                f"over {len(task.turns)} turn(s): each step's workdir/setup.sh and [steps.healthcheck] bounds, "
                f"harness setup, its [agent].timeout_sec, {len(task.config.verifier.collect)} [[verifier.collect]] "
                "hook(s) plus its own, and its [verifier].timeout_sec"
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
        episode, or, for ``turn`` (the step and the request it was sent), the turn's collection pass
        (:meth:`_collect_turn_artifacts`) and then the step's verifier with its settings and ``tests/``, in the
        agent's sandbox or, for a step that verifies separately (:meth:`_turn_separate`), in a verifier sandbox of
        its own. A transcript the converter cannot represent masks the turn instead of grading it; a collection the
        sandbox could not serve masks it as an outage."""
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
            return await self._run_verifier(session, trajectory, response=agent_response)
        step = turn[0]
        # Harbor's per-step order: the collection pass (hooks, /logs/artifacts, the task's and the step's artifacts)
        # after the agent and before the verifier, into the turn's own logs folder. Nothing is stopped, except on the
        # last turn of a task whose last step verifies separately (Harbor's `_run_step`: `main` before the sidecar
        # pass, the whole agent environment before the step's verifier), as the single-step separate path stops it.
        separate = self._turn_separate(task, step)
        last_separate = separate and step.name == task.turns[-1].name
        try:
            logs_dir = self._reserve_logs_dir()
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not create verifier logs under %s", self._artifacts_dir)
            return self._masked(
                failure_kinds.PERSISTENCE_FAILED,
                f"Could not create verifier logs under {self._artifacts_dir}: {type(exc).__name__}: {exc}",
            )
        started = asyncio.get_running_loop().time()
        try:
            if separate and session.entrypoint is not None:
                # The agent image's entrypoint is checked while the agent's sandbox is still up, as the single-step
                # paths check it before grading: the collection pass below stops `main` on the last separate turn,
                # and the step's own verifier sandbox never held the entrypoint. A death masks the turn as in shared
                # mode (`_run_shared_verifier`); the log is kept whenever it was read, and always on the last turn,
                # whose agent sandbox is stopped before test.sh runs.
                died = await self._entrypoint_died_during_rollout(session.sandbox, task, session.entrypoint, logs_dir)
                if died is not None or last_separate:
                    await self._collect_entrypoint_log(session.sandbox, task, logs_dir / ENTRYPOINT_LOG_NAME)
                if died is not None:
                    return self._masked(ENTRYPOINT_FAILED_KIND, died) | {
                        "verifier_logs_dir": str(logs_dir.resolve()),
                        "verifier_mode": "separate",
                    }
            restored = await self._collect_turn_artifacts(
                session, step, logs_dir / "artifacts", stop_main=last_separate
            )
            if last_separate:
                # Nothing of the agent's may keep running while the verifier sandbox grades its last turn.
                await session.stop_agent_side()
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Artifact collection failed for {task.task_id} turn {step.name!r}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}") | {
                "verifier_logs_dir": str(logs_dir.resolve())
            }
        settings = self._turn_verifier_settings(task, step)
        test_dirs = (task.path / "tests", task.path / step.tests_dir)
        if separate:
            # Harbor's separate mode for this step alone: a verifier sandbox of its own around this verification,
            # fed the collected archives and the tests, while the agent's sandbox keeps running for the next turn.
            outcome = await self._verify_separately(
                session,
                self._turn_verifier_view(task, step),
                settings,
                trajectory,
                logs_dir,
                restored,
                started=started,
                test_dirs=test_dirs,
            )
            return outcome | {"verifier_mode": "separate"}
        # A shared turn restores nothing: the staged trees stay as the diagnostic, the archives have served their
        # purpose (a separate turn's are removed once restored, by `_verify_separately`).
        for archive, *_ in restored:
            archive.unlink(missing_ok=True)
        outcome = await self._run_shared_verifier(
            session, trajectory, settings=settings, test_dirs=test_dirs, logs_dir=logs_dir, collect_hooks=False
        )
        return outcome | {"verifier_mode": "shared"}

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
        handle = await self._open_agent_session(
            request, access, cleanup, task, progress, agent_timeout_sec=agent_timeout_sec, continuation=False
        )
        agent_response = await self._activate_agent_session(request, handle, responses_create_params, progress)
        # Verification needs this close first: the harness must have let go of the sandbox, and its
        # observations travel with the result.
        await self._close_agent_session(handle, task, progress, partial_response=agent_response)
        self._route_failed_response(agent_response, handle)
        return agent_response, handle.observations, handle.trajectory

    async def _open_agent_session(
        self,
        request: HarborEpisodeRequest,
        access: SandboxAccess,
        cleanup: CleanupContext,
        task: HarborTask,
        progress: "_TurnProgress",
        *,
        agent_timeout_sec: float,
        continuation: bool,
    ) -> _AgentSessionHandle:
        """Seed an agent session on the episode's sandbox and check the task's tools can reach the harness.

        With ``continuation`` the seed asks for the harness's multi-turn continuation (its saved session resumed
        across activations), so a multi-step task's steps can follow one another in one conversation. A failed or
        refused seed raises the staged episode failure.
        """
        progress.stage, progress.response, progress.observations, progress.trajectory = "agent", None, None, None
        handle = _AgentSessionHandle(
            agent_session_id=f"agent-session-{uuid4().hex}", cookies={}, cleanup=None, continuation=continuation
        )

        async def close_agent() -> None:
            close_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(
                    agent_session_id=handle.agent_session_id,
                    episode_id=request.episode_id,
                ).model_dump(mode="json"),
                cookies=handle.cookies,
            )
            await raise_for_status(close_http_response)
            agent_close_response = AgentCloseSessionResponse.model_validate(
                await get_response_json(close_http_response)
            )
            if agent_close_response.agent_session_id != handle.agent_session_id:
                raise ValueError("Agent close returned a different agent_session_id")
            handle.observations = agent_close_response.agent_observations
            handle.trajectory = agent_close_response.trajectory
            handle.closed = True
            progress.observations = handle.observations
            progress.trajectory = handle.trajectory

        # Register cleanup before seed so cancellation can close a remotely created session even if its response is lost.
        handle.cleanup = cleanup.register_cleanup("agent session", close_agent)

        # One source for the task's MCP declarations: what is granted and what the refusal below checks.
        mcp_servers = self._task_mcp_servers(task)
        try:
            # Inside the try: a declaration the grant schema rejects is a staged terminal agent failure, not a 500.
            tool_accesses = self._task_tool_accesses(task, mcp_servers)
            agent_create_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path="/v1/agent_sessions",
                json=AgentSeedSessionRequest(
                    agent_session_id=handle.agent_session_id,
                    episode_id=request.episode_id,
                    task_id=request.task.task_id,
                    tool_accesses=tool_accesses,
                    sandbox_access=access,
                    agent_timeout_sec=agent_timeout_sec,
                    continuation=AgentContinuationRequirements() if continuation else None,
                ).model_dump(mode="json"),
            )
            await raise_for_status(agent_create_http_response)
            agent_session = AgentSeedSessionResponse.model_validate(
                await get_response_json(agent_create_http_response)
            )
            if agent_session.agent_session_id != handle.agent_session_id:
                raise ValueError("Agent seed returned a different agent_session_id")
            handle.cookies = _cookies(agent_create_http_response)
            unreached = self._unreached_task_tools(task, mcp_servers, tool_accesses)
            if unreached and not agent_session.supports_task_tool_context:
                # The task's tools would never reach this harness, and the run would score a silent zero.
                # Advice only when following it would clear the refusal: every unreached entry is a grantable
                # streamable-http server and grants are not already required.
                grantable = "skills_dir" not in unreached and all(
                    server.transport == "streamable-http" and bool(server.url)
                    for server in mcp_servers
                    if f"MCP server {server.name!r}" in unreached
                )
                hint = (
                    ", or grant its streamable-http MCP servers with task_mcp_grants: required"
                    if grantable and self.config.task_mcp_grants != "required"
                    else ""
                )
                raise HTTPException(
                    422,
                    f"Task {task.task_id!r} declares MCP servers or a skills directory, which the agent "
                    f"{self.config.agent_server.name!r} does not support ({', '.join(unreached)}); run it with a "
                    f"harness that declares supports_task_tool_context in its seed response{hint}",
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
        return handle

    async def _activate_agent_session(
        self,
        request: HarborEpisodeRequest,
        handle: _AgentSessionHandle,
        responses_create_params: Any,
        progress: "_TurnProgress",
        *,
        activation: int | None = None,
    ) -> NeMoGymResponse:
        """Send one input to the open session: the turn's only activation, or activation ``activation`` (1-based)
        of a continuation session, named in the header the agent-session contract requires. A failed activation
        closes the session for its evidence and raises the staged episode failure."""
        progress.stage, progress.response = "agent", None
        headers = {AGENT_ACTIVATION_HEADER: str(activation)} if activation is not None else None
        try:
            agent_http_response = await self.server_client.post(
                server_name=self.config.agent_server.name,
                url_path=self._agent_responses_path(request),
                json=_harness_params(responses_create_params),
                cookies=handle.cookies,
                **({"headers": headers} if headers else {}),
            )
            await raise_for_status(agent_http_response)
            response_cookies = _cookies(agent_http_response)
            if response_cookies:
                handle.cookies = response_cookies
            agent_response = NeMoGymResponse.model_validate(await get_response_json(agent_http_response))
            progress.response = agent_response
        except Exception as error:
            # The agent's evidence of the failed turn arrives with its session close.
            try:
                await handle.cleanup.close()
            except Exception:
                pass  # Final unwind retries the close and logs its failure.
            raise dependency_failure(
                error,
                failure_model=SingleAgentTurnFailure,
                stage="agent",
                agent_observations=handle.observations,
                trajectory=handle.trajectory,
            ) from error
        return agent_response

    async def _close_agent_session(
        self,
        handle: _AgentSessionHandle,
        task: HarborTask,
        progress: "_TurnProgress",
        *,
        partial_response: NeMoGymResponse | None,
    ) -> None:
        """Close the session so the harness lets go of the sandbox; its observations and trajectory land on the
        handle. A close that times out is tolerated (the final unwind retries it); any other failure is a staged
        cleanup failure."""
        try:
            await handle.cleanup.close()
        except TimeoutError as error:
            # The turn is finished; a harness that is slow to tear down must not cost its score. Its
            # observations are lost with the reply, and the final unwind retries the close.
            LOGGER.warning(f"{task.task_id}: agent session close timed out, verifying anyway: {error}")
            handle.observations = None
            progress.observations = None
            # The episode is done with the session: the final unwind, not a second close here, retries it.
            handle.closed = True
        except Exception as error:
            raise dependency_failure(
                error,
                failure_model=SingleAgentTurnFailure,
                stage="cleanup",
                partial_response=partial_response,
            ) from error

    async def _close_before_failure(
        self,
        handle: _AgentSessionHandle | None,
        task: HarborTask,
        progress: "_TurnProgress",
        *,
        partial_response: NeMoGymResponse | None,
    ) -> None:
        """Close a continued session that is still open when a failure ends the episode mid-loop, so the evidence
        its close returns (the harness's observations and cumulative trajectory, on the handle) travels with the
        failure row. Without this the final unwind closes the session and retrieves both, after the failure response
        was built without them. The failure being built is kept whatever the close does: a close that fails is
        logged, never raised over the original error (the final unwind retries it)."""
        if handle is None or handle.closed:
            return
        progress.stage = "cleanup"
        try:
            await self._close_agent_session(handle, task, progress, partial_response=partial_response)
        except HandledEpisodeError as error:
            LOGGER.warning(
                f"{task.task_id}: agent session close failed while the episode was ending in a failure, which is "
                f"kept over the close's: {error.failure.failure_reason}"
            )

    def _route_failed_response(self, agent_response: NeMoGymResponse, handle: _AgentSessionHandle) -> None:
        """A failed response whose harness classified the cause is routed by the deployment's policy before any
        grading: a transport or unknown cause is retried ungraded, as on ``single_agent_turn``, instead of
        handing test.sh a half-finished sandbox and recording a real 0. In a multi-step task this ends the
        episode at the turn that failed; a verify cause is graded as before."""
        action = agent_outcome_policy(agent_response, config=self.config.agent_outcome_policy)
        if action != OutcomeAction.VERIFY:
            raise self._failure(
                stage="agent",
                failure_reason=agent_response.error.message if agent_response.error else "Agent reported failure",
                terminal=action == OutcomeAction.FAIL,
                partial_response=agent_response,
                agent_observations=handle.observations,
                trajectory=handle.trajectory,
            )

    async def _require_turn_continuation(self) -> None:
        """Refuse a ``turn_sessions: continue`` run before any sandbox when the harness cannot continue a session.

        The agent server's capabilities are asked once per process. An agent server without the route (an older
        build) cannot continue either. Harbor's multi-step trial refuses ``resume_trajectory`` the same way for an
        agent without ``SUPPORTS_RESUME``, before any environment spend.
        """
        if self._agent_capabilities is None:
            try:
                http_response = await self.server_client.get(
                    server_name=self.config.agent_server.name, url_path="/v1/agent_sessions/capabilities"
                )
                await raise_for_status(http_response)
                capabilities = AgentSessionCapabilitiesResponse.model_validate(await get_response_json(http_response))
            except ClientResponseError as error:
                if error.status not in (404, 405):
                    raise dependency_failure(
                        error,
                        failure_model=SingleAgentTurnFailure,
                        stage="seed",
                        terminal=False,
                    ) from error
                capabilities = AgentSessionCapabilitiesResponse()
            except Exception as error:
                raise dependency_failure(
                    error,
                    failure_model=SingleAgentTurnFailure,
                    stage="seed",
                ) from error
            self._agent_capabilities = capabilities
        if self._agent_capabilities.continuation is None:
            raise self._failure(
                stage="seed",
                failure_reason=(
                    f"turn_sessions: continue needs a harness with multi-turn continuation, which the agent "
                    f"{self.config.agent_server.name!r} does not declare; run the multi-step task with "
                    "turn_sessions: fresh (a session per step) or with a harness that resumes its session"
                ),
                terminal=True,
            )

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
        ``steps/<name>/tests/`` on top. With ``turn_sessions: continue`` (Harbor's ``--resume-trajectory``) the
        first turn seeds one continuation session and every turn is an activation of it, so the agent keeps its
        conversation; the session is closed once, after the last turn's activation and before that step's
        verifier, and its observations and cumulative trajectory are the episode's. The session's conversation state
        on disk and its borrowed connection to the sandbox then stay in place while every step's verifier but the
        last's runs (the harness process has exited between activations, as Harbor's resumed agent has between
        steps); a task whose steps switch the agent user cannot continue, since
        the session runs as the user it was seeded with (refused at seed, before any sandbox). Before a turn the previous turn's grading files are cleared, the sandbox
        takes the step's ``[steps.agent].user`` when it has one, the step's ``workdir/``, ``setup.sh`` and
        ``[steps.healthcheck]`` are applied and the oracle gets the step's own solution (:meth:`_prepare_turn`). The episode stops after a turn whose
        preparation failed (recorded as a failed turn with no agent response and no verifier result, Harbor's
        step ``exception_info``, scored 0 by :func:`roll_up_turns` when a turn ran before it), or that is masked,
        produces no verifier result, or misses its ``min_reward``;
        the remaining turns do not run. An episode whose first turn's preparation failed has no agent response
        at all, so it is a terminal failure at the agent stage with the step's failure kind, as Harbor's trial
        then ends with no verifier result.
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
        continued = self.config.turn_sessions == "continue" and len(task.turns) > 1
        handle: _AgentSessionHandle | None = None
        results: list[HarborTurnResult] = []
        stopped_after: str | None = None
        for index, (turn, turn_input) in enumerate(zip(task.turns, inputs), start=1):
            try:
                # The preparation (the step's workdir, setup.sh and healthcheck) is this turn's, before its agent: a
                # task deadline that fires in it reports the agent stage with no response yet, not the previous turn's
                # verification with that turn's response, which `_agent_turn` would only reset once the session opens.
                progress.stage, progress.response, progress.observations, progress.trajectory = (
                    "agent",
                    None,
                    None,
                    None,
                )
                access = await self._prepare_turn(session, turn, access)
            except StepPrepareError as error:
                # The step's own setup failed: a failed turn, no agent session, no verifier, and the episode stops.
                LOGGER.warning(f"{task.task_id}: {error.reason}")
                results.append(
                    HarborTurnResult(
                        name=turn.name,
                        reward=0.0,
                        mask_sample=False,
                        failure_kind=error.kind,
                        failure_reason=error.reason[:2000],
                    )
                )
                stopped_after = turn.name if len(results) < len(task.turns) else None
                break
            except HTTPException as error:
                # A continued session open from the earlier turns is closed first, for its evidence.
                last_response = results[-1].response if results else None
                await self._close_before_failure(handle, task, progress, partial_response=last_response)
                raise self._failure(
                    stage="agent",
                    failure_reason=f"turn {turn.name!r}: {error.detail}",
                    terminal=error.status_code < 500,
                    agent_observations=handle.observations if handle is not None else None,
                    trajectory=handle.trajectory if handle is not None else None,
                ) from error
            except Exception as error:
                # The sandbox could not clear the previous turn's grading files or take the turn's user or solution:
                # a provider-side failure, retried elsewhere; a task-caused non-zero exit (turn 1 replaced /tests
                # with a file) takes the same retryable path. The next agent never opens with those files at their
                # paths.
                last_response = results[-1].response if results else None
                await self._close_before_failure(handle, task, progress, partial_response=last_response)
                raise self._failure(
                    stage="agent",
                    failure_reason=f"turn {turn.name!r}: {type(error).__name__}: {error}",
                    terminal=False,
                    agent_observations=handle.observations if handle is not None else None,
                    trajectory=handle.trajectory if handle is not None else None,
                ) from error
            if continued:
                if handle is None:
                    handle = await self._open_agent_session(
                        request,
                        access,
                        cleanup,
                        task,
                        progress,
                        agent_timeout_sec=task.config.step_agent_timeout_sec(turn.config),
                        continuation=True,
                    )
                response = await self._activate_agent_session(
                    request, handle, turn_input.responses_create_params, progress, activation=index
                )
                try:
                    self._route_failed_response(response, handle)
                except HandledEpisodeError:
                    # The evidence of a continued session arrives with its close.
                    await self._close_agent_session(handle, task, progress, partial_response=response)
                    self._route_failed_response(response, handle)
                if index == len(task.turns):
                    # No activation follows the last step: the harness lets go of the sandbox before that step's
                    # verifier, as a fresh session does before every step's, and before a last step that verifies
                    # separately stops the agent's sandbox (`_verified_outcome`), where a close could no longer run
                    # in it (OpenCode's close removes its session directory from a borrowed sandbox).
                    await self._close_agent_session(handle, task, progress, partial_response=response)
                # A continued session's observations and trajectory are the whole episode's, at its close.
                observations, trajectory = None, None
            else:
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
                # A continued session's evidence sits on the handle once its close has returned: after the last
                # step's activation it already has, and a session still open at an earlier step's verifier failure
                # (a separate verifier whose sandbox would not start, say) is closed now, before the failure row is
                # built, so the row carries what the close returned instead of leaving it to the final unwind.
                # After a close that timed out there is none to carry.
                await self._close_before_failure(handle, task, progress, partial_response=response)
                raise self._failure(
                    stage="verification",
                    failure_reason=f"turn {turn.name!r}: {error.detail}",
                    terminal=error.status_code < 500,
                    partial_response=response,
                    agent_observations=handle.observations if handle is not None else observations,
                    trajectory=handle.trajectory if handle is not None else trajectory,
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

        if handle is not None and not handle.closed:
            # The episode stopped before its last step's activation: the harness lets go of the sandbox now.
            progress.stage = "cleanup"
            await self._close_agent_session(handle, task, progress, partial_response=results[-1].response)

        strategy = task.config.multi_step_reward_strategy
        last = results[-1]
        # The last turn whose agent ran: a turn that failed in its preparation has no response of its own.
        ran = next((result for result in reversed(results) if result.response is not None), None)
        if ran is None:
            raise self._failure(
                stage="agent",
                failure_reason=f"turn {last.name!r}: {last.failure_reason}",
                terminal=True,
                failure_kind=last.failure_kind,
            )
        sessions = sum(1 for result in results if result.response is not None)
        episode = HarborEpisodeResult(
            responses_create_params=inputs[results.index(ran)].responses_create_params,
            response=ran.response,
            # Fresh sessions leave one record per step on the turns, as Harbor's steps/<name>/agent/ does; the
            # episode-level slots carry a session's whole record (continue mode, or the one fresh session when a
            # single turn's agent ran; a turn whose preparation failed opened none), never one step's standing
            # for the episode (merging them is issue #5063).
            ng_agent_observations=handle.observations
            if handle is not None
            else (ran.agent_observations if sessions == 1 else None),
            ng_trajectory=handle.trajectory if handle is not None else (ran.trajectory if sessions == 1 else None),
            verifier_mode=_turns_verifier_mode(results),
            turns=results,
            turn_sessions="continue" if continued else "fresh",
            turn_reward_strategy=strategy,
            turns_without_verifier_result=sum(1 for result in results if result.verifier_rewards is None),
            stopped_after_turn=stopped_after,
            **roll_up_turns(results, strategy, task_id=request.task.task_id),
        )
        self._trace_score(task.task_id, request.episode_id.capture_key, episode)
        return HarborEpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=episode)

    async def _prepare_turn(self, session: HarborSession, turn: HarborTurn, access: SandboxAccess) -> SandboxAccess:
        """What a turn needs in place before its agent session opens: no grading files from the previous turn, the
        step's user, the step's ``workdir/`` and ``setup.sh``, and the step's solution.

        A turn that follows a turn verified in this sandbox (a shared turn, :meth:`_turn_separate` false) opens
        where that verification left its tests, its reward and the previous agent's transcript, so they are removed
        first (:meth:`_reset_shared_verifier_dirs`); the first turn, and a turn after one that verified in a
        sandbox of its own, keep whatever the image put in ``/tests``, since nothing was graded here. Harbor 0.22.0
        empties ``/tests`` and ``/logs/verifier`` only right before each step's shared verifier, so its next step's
        agent still sees the previous step's tests, and its separate mode never touches the agent's ``/tests``;
        Harbor main moved that reset into its step preparation, gated on the previous step having run a shared
        verifier there, and this follows main. Harbor also empties the whole ``/logs/agent`` before a step
        that does not resume (or moves the bind-mounted contents away); here only the server's
        ``trajectory.json`` goes and the agent's own files under ``/logs/agent`` stay. The reset removes files at
        those paths: a process the previous agent left running, or a copy it took during its verification, is out
        of scope, as under Harbor's shared mode. Harbor runs each step as its own ``[steps.agent].user``
        (else the task's ``[agent].user``), so when the turn's user differs from the one the sandbox runs as, the
        sandbox's default user, the working directory's owner and the user the harness is handed switch to it
        (:meth:`_apply_turn_user`), and the access the harness gets is rebuilt for it; the verifier keeps its own
        user and fallback. Then, as that user, the step's ``steps/<name>/workdir/`` is merged into the working
        directory and its ``setup.sh`` run, then the step's ``[steps.healthcheck]`` polled to a pass
        (:meth:`_prepare_step_workdir`, :meth:`_run_step_healthcheck`: Harbor's ``_prepare_step`` order; a failure
        in either is :class:`StepPrepareError`, the step's failure, and the healthcheck does not run after a failed
        setup). Nothing stops or pauses what the previous turn left running between the switch and the script, so a
        process it left behind can touch the working directory, ``setup.sh`` included, before the script runs; a
        step that takes root after a step that had less is where that matters (see :meth:`_prepare_step_workdir`).
        The oracle alone then gets the step's ``solution/`` at ``/solution`` (:meth:`_place_turn_solution`).
        """
        task = session.task
        index = next(i for i, candidate in enumerate(task.turns) if candidate is turn)
        if index and not self._turn_separate(task, task.turns[index - 1]):
            await self._reset_shared_verifier_dirs(session.sandbox, keep=task.config.verifier.inputs)
        user = str(turn.config.agent.user) if turn.config.agent.user is not None else task.user
        # Compared as written (the seed's or the previous turn's), not with the form the sandbox installed.
        if user != session.requested_user:
            await self._apply_turn_user(session, user)
            access = await self._sandbox_access(session)
        await self._prepare_step_workdir(session, turn)
        await self._run_step_healthcheck(session, turn)
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

    @staticmethod
    def _unreached_task_tools(
        task: HarborTask, mcp_servers: list[HarborMCPServer], tool_accesses: list[ToolAccess]
    ) -> list[str]:
        """The task's tool declarations a harness reaches only through ``/tmp/.nemo-gym/task.json``.

        ``mcp_servers`` is the list the grants were built from, so both sides of the check see the same servers.
        An MCP server granted as required reaches any harness that accepted the grant: by the tool-access contract
        a harness that cannot establish a required server fails its seed or its first turn (Hermes and Pi connect
        at the first activation) rather than running without it. Optional grants may be skipped, and a skills
        directory has no grant at all, so those stay unreached.
        """
        granted = {access.name for access in tool_accesses if isinstance(access, MCPToolAccess) and access.required}
        unreached = [f"MCP server {server.name!r}" for server in mcp_servers if server.name not in granted]
        if task.config.environment.skills_dir:
            unreached.append("skills_dir")
        return unreached

    def _task_tool_accesses(self, task: HarborTask, mcp_servers: list[HarborMCPServer]) -> list[ToolAccess]:
        """``mcp_servers`` (the task's, the job-level overlay's merged in by name) as agent tool grants, when
        ``task_mcp_grants`` asks for them: one grant list, as Harbor hands its agent one ``mcp_servers`` list.

        Only URL entries reached over streamable HTTP can be granted; a Compose sidecar's URL names its
        service host, which resolves inside the task's sandbox group, where the harness runs. An entry that cannot
        be granted still reaches the harness in ``/tmp/.nemo-gym/task.json``; a task's own such entry is warned about
        at every seed (the task author's signal), the job-level overlay's once per run, by name (the operator's
        choice, the memory team's Redis stack being ``sse`` by design, which would otherwise print one line per turn
        of every episode of the run).
        """
        if self.config.task_mcp_grants == "off":
            return []
        overlay = self.config.compose_overlay
        job_entries = {server.name for server in overlay.mcp_servers} if overlay is not None else set()
        accesses: list[ToolAccess] = []
        for server in mcp_servers:
            if server.transport != "streamable-http" or not server.url:
                if server.name not in job_entries:
                    LOGGER.warning(
                        f"{task.task_id}: MCP server {server.name!r} ({server.transport}) cannot be granted; "
                        "only streamable-http URL entries can"
                    )
                elif server.name not in self._ungrantable_overlay_servers:
                    self._ungrantable_overlay_servers.add(server.name)
                    LOGGER.warning(
                        f"compose_overlay: MCP server {server.name!r} ({server.transport}) cannot be granted; only "
                        "streamable-http URL entries can; it reaches the harness in /tmp/.nemo-gym/task.json "
                        "(said once per run)"
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

    def _check_seed_trajectory(self, task: HarborTask) -> None:
        """Refuse a task that ships Harbor's seed ``trajectory.json`` for every agent but the oracle, before anything
        is provisioned.

        Harbor loads the file into the agent's session for every agent but the oracle and nop
        (``Trial._resolve_load_trajectory``); no harness here can be seeded with it yet, so the oracle runs the task
        as Harbor's does and any other agent is refused, not retried. An agent block this server cannot read
        (several implementations under the instance, a client the merged config does not show) is not known to be
        any harness in particular: the task runs, with a warning, as :meth:`_seed_solution` places no solution for
        the same shape with a warning, rather than refusing the oracle wired that way.
        """
        implementation = self._agent_implementation()
        where = task.seed_trajectory.as_posix()
        if implementation is None:
            LOGGER.warning(
                "%s: ships Harbor's seed trajectory %s, and agent %r has no single responses_api_agents "
                "implementation in the run config, so the server cannot tell whether it is the oracle (which runs "
                "such a task as Harbor's does); the task runs without its prior context",
                task.task_id,
                where,
                self.config.agent_server.name,
            )
            return
        if implementation != ORACLE_AGENT:
            raise HTTPException(
                422,
                f"Task {task.task_id!r} ships Harbor's seed trajectory {where} (the prior context Harbor loads into "
                "the agent's session before its first turn, as --load-trajectory does, for every agent but the oracle "
                "and nop); loading a trajectory into a harness is not supported yet, so the task is refused for agent "
                f"{self.config.agent_server.name!r} rather than run without it. The oracle validates it, as Harbor's "
                "does; remove the file to run the task without its prior context",
            )

    async def _seed(self, request: HarborEpisodeRequest, episode_key: str) -> HarborSession:
        """Resolve the task and start its sandbox; the sandbox stays owned by the server until it is stopped."""
        if self._shutting_down:
            raise HTTPException(503, "Environment server is shutting down")
        task = await self._resolve_task(request.task.task_id, request.task.task_input.task_data)
        self._check_task_data(request.task.task_id.taskset, task, request.task.task_input.task_data)
        if task.seed_trajectory is not None:
            self._check_seed_trajectory(task)
        if self.config.turn_sessions == "continue" and len(task.turns) > 1:
            # A continued session runs as the user it was seeded with (the first step's, which `_prepare_turn`
            # puts in place before the session opens), where Harbor's resumed agent is relaunched as each step's
            # user. Each step's effective user is its own `[steps.agent].user`, else the task's, as `_prepare_turn`
            # resolves it: steps that all resolve to one user never switch, whether the task names it or every step
            # does; refused here, before any sandbox is created (or its image built or pulled), as the harness's
            # missing continuation is in `run()`.
            users = {
                str(turn.config.agent.user) if turn.config.agent.user is not None else task.user for turn in task.turns
            }
            if len(users) > 1:
                named = ", ".join(sorted(user if user is not None else "the image's default user" for user in users))
                raise HTTPException(
                    422,
                    f"Task {task.task_id!r} switches the agent user between steps ({named}), "
                    "which a continued harness session cannot follow; run it with turn_sessions: fresh",
                )
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
        # A multi-step task's separate turns each verify in a view of the task (`_turn_verifier_view`): the checks
        # below that read the task's verifier image, environment and network run on every such view too.
        views = [
            (f" step {turn.name!r}", self._turn_verifier_view(task, turn))
            for turn in task.turns
            if self._turn_separate(task, turn)
        ]
        for where, view in views:
            if _verifier_image(view, True) is None:
                raise HTTPException(
                    422,
                    f"Task {task.task_id!r}{where} has no image for its separate verifier: no [steps.verifier.environment] "
                    f"or [verifier.environment] names a docker_image and no verifier image built from "
                    f"{TESTS_DOCKERFILE} is recorded in {task.path.parent / BUILT_IMAGES_FILE}; prepare the dataset "
                    "again, or name a prebuilt image",
                )
            if view.config.verifier.environment is not None:
                resolve_env_templates(view.config.verifier.environment.env)
        if separate and not task.turns and _verifier_image(task, separate) is None:
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
        # So is its network: `[verifier].network_mode` becomes the separate verifier sandbox's egress policy at its
        # creation, which only OpenSandbox enforces; a provider that cannot, or an entry the policy cannot express,
        # fails here rather than after the rollout. The loader admits the phase policy for separate-mode tasks only.
        try:
            verifier_network_policy = self._verifier_network_policy(task)
        except ValueError as exc:
            raise HTTPException(422, f"Task {task.task_id!r} [verifier] network policy: {exc}") from exc
        shared_turns = [turn.name for turn in task.turns if not self._turn_separate(task, turn)]
        if verifier_network_policy is not None and (not separate or shared_turns):
            raise HTTPException(
                422,
                f"Task {task.task_id!r} restricts the verifier's network but verifies in the agent's sandbox"
                f"{' for step(s) ' + ', '.join(shared_turns) if shared_turns else ''}, whose network policy cannot be "
                "changed after it started; declare separate verification",
            )
        for turn in task.turns:
            # The sources the turn's verifier resolves its env from (`_verifier_env`: the server's secrets, then the
            # process environment), so a judge key only `secrets` supplies passes here as it does at the task level.
            resolve_env_templates(turn.config.verifier.env, self.config.secrets)
            # A step's own agent user is installed for its turn (`_apply_turn_user`); a pair the provider cannot
            # apply is refused before the earlier turns are paid for, as the task's own `[agent].user` is.
            self._refuse_unsupported_group(task, turn.config.agent.user, where=f" step {turn.name!r}")
        if self._uses_compose(task):
            # A Compose file the deployment leaves incomplete (a variable no compose_variables entry sets), or a
            # job-level overlay file that does not merge onto it, is a configuration error: refused before anything
            # is provisioned, and not retried. So is a group whose images prepare did not record (a dataset prepared
            # without this overlay): refused here, terminally, not raised from the group's start and retried. And so
            # is a merged document this deployment cannot run (a field the group cannot run, a bind source no
            # `volume_sources` entry maps, a service reset away that a collect hook names, extra_hosts on a non-root
            # sidecar): every check prepare ran, on the document the group would start, here rather than inside
            # `_create_compose`, where the failure would be retried for every episode. The overlay is a run-config
            # value, so it changes between prepare and serve more easily than a task file. The three are file work
            # (the Compose and overlay files, their env files, the image records), so they run in a worker thread.
            await asyncio.to_thread(self._compose_preflight, task, episode_key)
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
                f"setup.sh and healthcheck bounds, agent and verifier budgets, hooks and grace per step), "
                f"{nominal:.0f}s with one "
                f"sandbox start and "
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
            if self._uses_compose(task):
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
            # The task's [environment.healthcheck] runs in the prepared workdir, as Harbor's exec runs it; a group's
            # main runs its command in the Dockerfile's WORKDIR, which may be another directory.
            await self._wait_healthy(
                sandbox, task, launch, user=image_user, workdir=session.workdir, compose_managed=managed
            )
            await self._write_task_context(sandbox, task)
            if launch is not None:
                # The last look before the hand-over: an entrypoint that died during the seed's remaining steps
                # (a setup script failing after its first lines) is a seed failure, not a rollout on an empty task.
                await self._check_entrypoint(sandbox, task, launch)
            LOGGER.debug(
                "%s: episode %s seeded (workdir %s, %s verifier, %s)",
                task.task_id,
                episode_key,
                session.workdir,
                "separate" if separate else "shared",
                "Compose group" if compose is not None else "one sandbox",
            )
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


def _turns_verifier_mode(turns: list[HarborTurnResult]) -> Literal["shared", "separate", "mixed"] | None:
    """The episode's ``verifier_mode`` from its turns': the one mode every graded turn used, ``mixed`` when the steps
    resolved to both (Harbor's ``resolve_step_verifier_mode`` is per step), ``None`` when no turn was graded."""
    modes = {turn.verifier_mode for turn in turns if turn.verifier_mode is not None}
    if not modes:
        return None
    return next(iter(modes)) if len(modes) == 1 else "mixed"


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
