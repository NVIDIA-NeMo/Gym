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
  ``SandboxAccess`` for that sandbox, post the task's ``responses_create_params`` to it,
  then close the session and keep what the agent observed;
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
import json
import logging
import math
import os
import posixpath
import random
import re
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import yaml
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from environment_servers.harbor.sandbox_io import (
    SandboxTransferError,
    _exec_as_root,
    download_dir,
    download_path,
    remote_kind,
    upload_dir,
    upload_path,
)
from nemo_gym import RESULTS_DIR, __version__, failure_kinds
from nemo_gym.atif_export import AtifExportError, encode_atif_trajectory, transcript_to_atif
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
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose, SandboxExecResult, SandboxSpec, clamp_resource_requests
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.compose_config import (
    adapt_non_root_sidecars,
    deployment_platform,
    opensandbox_shm_labels,
    resolve_compose,
    sandbox_platform,
    service_gpus,
)
from nemo_gym.sandbox.config import (
    resolve_provider_config,
    resolve_provider_metadata,
    resolve_provider_volume_sources,
)
from nemo_gym.sandbox.providers.base import ORPHAN_REAP_TIMEOUT_S, SandboxCreateError
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnRequest, SingleAgentTurnResult
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.dockerfile import OverlayRun
from nemo_gym.tasks.harbor.image_configs import (
    check_task_compose_fields,
    load_compose_document,
    plan_task_compose_volumes,
    task_declares_gpus,
    without_pull_policy,
    without_restated_main_build,
)
from nemo_gym.tasks.harbor.models import HarborArtifact, HarborEnvironment, HarborHealthcheck
from nemo_gym.tasks.harbor.task import BUILT_IMAGES_FILE, TESTS_DOCKERFILE, HarborTaskError


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"
VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
ARTIFACTS_DIR = "/logs/artifacts"
COMPOSE_IMAGES_FILE = "compose-images.json"
TASK_CONTEXT_DIR = "/tmp/.nemo-gym"
TASK_CONTEXT_FILE = f"{TASK_CONTEXT_DIR}/task.json"
VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"
# The agent transcript in ATIF v1.7, written under AGENT_LOGS_DIR before every verify; Harbor verifiers may read it.
TRAJECTORY_FILE_NAME = "trajectory.json"

# Namespaced failure kinds for outcomes the shared vocabulary does not name.
VERIFIER_TIMEOUT_KIND = "harbor:verifier_timeout"
MISSING_REWARD_KIND = "harbor:missing_reward"
INVALID_REWARD_KIND = "harbor:invalid_reward"
# Added to `sandbox_ready_timeout_s` for this server's own deadline on a create. The provider caps its create
# request at the ready timeout, so a pod that never schedules fails the request by then; the grace is room for
# the provider to find and delete what that create left behind (its orphan-reap allowance) and report what it
# found, before this server gives up on it. A provider attempt budget configured above the ready timeout
# (`create.timeout_s`) is not covered: the deadline here is this server's promise.
SEED_CREATE_GRACE_S = ORPHAN_REAP_TIMEOUT_S + 15.0


class HarborSeedTimeoutError(RuntimeError):
    """The sandbox was not ready within `sandbox_ready_timeout_s`: a provider-side seed failure, retried."""


# The separate verifier's sandbox could not take the task's Dockerfile overlay after the rollout ran (a shared-mode
# task the server verifies separately, see `separate_verifier_for_artifacts`).
OVERLAY_FAILED_KIND = "harbor:overlay_failed"
# The agent transcript could not be exported to ATIF; the turn is masked so a converter defect is never a reward.
TRAJECTORY_EXPORT_KIND = "harbor:trajectory_export_failed"
# The episode ran past the time budget of its task (see `enforce_task_timeouts`).
TASK_DEADLINE_KIND = "harbor:task_deadline"

# Image pull failures the server retries itself before it reports a retryable seed failure.
IMAGE_PULL_FAILURE_MARKERS = (
    "errimagepull",
    "imagepullbackoff",
    "failed to pull",
    "error pulling",
    "image pull",
    "pulling image",
)
# Retried starts queue behind this gate so a registry hiccup cannot turn into a pull storm.
PULL_RETRY_CONCURRENCY = 4
PULL_RETRY_BASE_DELAY_S = 5.0
PULL_RETRY_MAX_DELAY_S = 60.0

# How much of a failed overlay RUN line's output the seed error carries.
OVERLAY_OUTPUT_TAIL = 1500


class HarborOverlayError(RuntimeError):
    """A ``RUN`` line of the task's overlay-mode Dockerfile failed; the task cannot be set up."""


class HarborOverlayUnavailable(RuntimeError):
    """The sandbox provider could not run an overlay step at all (not the step failing): retry elsewhere."""


# Where the launcher keeps the image entrypoint's markers and log inside the sandbox, the way the Compose adapter
# keeps a service's; world-writable so an entrypoint that runs as the image's user can write them.
ENTRYPOINT_DIR = "/tmp/.nemo-gym-entrypoint"
ENTRYPOINT_LOG = f"{ENTRYPOINT_DIR}/log"
# The entrypoint's arguments: what Harbor's Compose files give it in place of CMD (``command: sh -c "sleep infinity"``).
ENTRYPOINT_ARGS = ("sh", "-c", "sleep infinity")
# How long the launcher waits for its started marker before the provider is judged unable to run it.
ENTRYPOINT_START_TIMEOUT_S = 60.0
ENTRYPOINT_POLL_INTERVAL_S = 0.5
# The entrypoint's output, collected best effort into the verification's logs folder, and the exit status
# it had recorded by the time the agent's sandbox was verified.
ENTRYPOINT_LOG_NAME = "entrypoint-stdout.txt"
VERIFIER_ENTRYPOINT_LOG_NAME = "verifier-entrypoint-stdout.txt"
ENTRYPOINT_EXIT_NAME = "entrypoint-exit.txt"
# The image entrypoint exited non-zero after the seed: during the rollout (when the exec holding it returned),
# or in the separate verifier's fresh sandbox. Under Harbor the container would have died and the trial errored.
ENTRYPOINT_FAILED_KIND = "harbor:entrypoint_failed"


class HarborEntrypointError(RuntimeError):
    """The image's ``ENTRYPOINT`` exited non-zero at start; the task cannot be set up."""


class HarborEntrypointUnavailable(RuntimeError):
    """The sandbox provider could not run the entrypoint launcher at all (not the entrypoint failing): retry."""


@dataclass
class EntrypointLaunch:
    """The image entrypoint running in a sandbox: its argv and the exec that holds it open."""

    argv: list[str]
    process: asyncio.Task
    # The entrypoint returned 0 without running its keepalive argument (warned once, then left alone).
    returned: bool = False
    # The provider ended the holding exec while no exit status was recorded (logged once; see _check_entrypoint).
    noticed: bool = False


_SHELLS = frozenset({"sh", "bash", "dash"})
_KEEPALIVE_PROGRAMS = frozenset({"sleep", "tail"})
# A shell script that is one sleep or tail command (``sleep infinity``, ``exec tail -f /dev/null``) or loops
# over a sleep. A compound script that starts with one (``sleep 5; service redis-server start``) does work and runs.
_KEEPALIVE_SCRIPT = re.compile(r"^\s*(?:exec\s+)?(?:sleep|tail)\s+[^;&|\n]*$")
_KEEPALIVE_LOOP = re.compile(r"^\s*while\s+(?:true|:)\s*;\s*do\s+sleep\s+\S+\s*;?\s*done\s*;?\s*$")


def is_keepalive_entrypoint(argv: list[str] | tuple[str, ...]) -> bool:
    """Whether ``argv`` only keeps a container alive, which the sandbox's own keepalive already does."""
    if not argv:
        return False
    if posixpath.basename(argv[0]) in _KEEPALIVE_PROGRAMS:
        return True
    if len(argv) == 3 and posixpath.basename(argv[0]) in _SHELLS and argv[1] == "-c":
        script = argv[2]
        return bool(_KEEPALIVE_SCRIPT.match(script) or _KEEPALIVE_LOOP.match(script))
    return False


def entrypoint_to_run(task: HarborTask) -> list[str] | None:
    """The image ``ENTRYPOINT`` the seed starts beside the keepalive, or ``None`` when nothing is to run.

    Nothing runs for a Compose task (the adapter runs ``main``'s entrypoint), an image whose recorded
    configuration has no entrypoint or none is recorded (a prebuilt ``docker_image``: today's behaviour),
    a Dockerfile that cleared it with ``ENTRYPOINT []``, and an entrypoint that is itself a keepalive.
    """
    if task.needs_compose or not task.entrypoint:
        return None
    argv = list(task.entrypoint)
    return None if is_keepalive_entrypoint(argv) else argv


def entrypoint_script(argv: list[str], runtime: str = ENTRYPOINT_DIR) -> str:
    """The ``run.sh`` that marks the start, runs ``argv`` with Harbor's arguments and records the exit status."""
    command = shlex.join([*argv, *ENTRYPOINT_ARGS])
    return f"touch {runtime}/started\n{command}\nstatus=$?\nprintf '%s' \"$status\" > {runtime}/exit\n"


def image_healthcheck_settings(record: Mapping[str, Any] | None) -> HarborHealthcheck | None:
    """The recorded Dockerfile ``HEALTHCHECK`` as the settings the seed polls, or ``None`` when it has none.

    Docker records ``Test`` as ``["CMD", ...]`` or ``["CMD-SHELL", script]`` (``["NONE"]`` or empty
    disables it) and the durations in nanoseconds, absent when the Dockerfile left Docker's defaults.
    """
    if not record:
        return None
    test = record.get("Test") or []
    if not test or test[0] == "NONE":
        return None
    if test[0] == "CMD-SHELL" and len(test) == 2:
        command = str(test[1])
    elif test[0] == "CMD" and len(test) > 1:
        command = shlex.join(str(part) for part in test[1:])
    else:
        return None

    def seconds(key: str, default: float) -> float:
        value = record.get(key)
        return value / 1_000_000_000 if value else default

    return HarborHealthcheck(
        command=command,
        interval_sec=seconds("Interval", 30.0),
        timeout_sec=seconds("Timeout", 30.0),
        start_period_sec=seconds("StartPeriod", 0.0),
        start_interval_sec=seconds("StartInterval", 5.0),
        retries=int(record.get("Retries") or 3),
    )


async def _cancel_entrypoint(launch: EntrypointLaunch | None) -> None:
    """End the exec that holds the entrypoint open; the sandbox is stopped right after."""
    if launch is None:
        return
    launch.process.cancel()
    await asyncio.gather(launch.process, return_exceptions=True)


def _launch_outcome(launch: EntrypointLaunch) -> str | None:
    """How the exec holding the entrypoint ended, or ``None`` while it is still held open."""
    process = launch.process
    if not process.done():
        return None
    if process.cancelled():
        return "cancelled"
    error = process.exception()
    if error is not None:
        return f"{type(error).__name__}: {error}"
    result = process.result()
    return result.error_type or f"exit {result.return_code}"


def _launch_returned(launch: EntrypointLaunch) -> bool:
    """Whether the holding exec returned on its own: ``run.sh`` ran to its end, so the entrypoint had exited.

    A provider that capped the exec (docker's ``exec.default_timeout_s``, OpenSandbox's request timeout
    without ``background_exec``) reports ``error_type`` instead, and the entrypoint may well be running still.
    """
    process = launch.process
    if not process.done() or process.cancelled() or process.exception() is not None:
        return False
    return process.result().error_type is None


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
    # Extra seconds granted to `test.sh` beyond `[verifier].timeout_sec` before the in-container `timeout` kills it.
    verifier_grace_s: float = Field(default=30, ge=0)
    # Verifier logs are downloaded here, one fresh folder per verification.
    artifacts_dir: Path | None = None
    # Operator secrets (judge API keys and the like), by environment variable name. They reach the `test.sh`
    # process only, and only the ones the task names: they fill `${NAME}` templates in `[verifier].env`, as
    # Harbor fills them from its own environment. A secret no template names is passed nowhere. They never
    # enter a sandbox's creation environment, the agent's turn, a task folder or a row.
    secrets: dict[str, str] = Field(default_factory=dict)
    # Give every episode the time budget of its task instead of the run-wide `default_episode_timeout_seconds`
    # alone: once its sandbox is ready, `harness_setup_allowance_s`, `[agent].timeout_sec`,
    # `[verifier].timeout_sec` and the verifier grace (plus a separate verifier's start). Sandbox setup,
    # image pull retries included, is not charged against it; the sandbox lifetime covers both. Off
    # ignores task timeouts.
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
    provider_ref: str
    compose: AsyncSandboxCompose | None = None
    # The image entrypoint the seed started beside the keepalive, cancelled before the sandbox is stopped.
    entrypoint: EntrypointLaunch | None = None
    # Separate verification stops the agent's sandbox before test.sh runs; cleanup must not stop it twice.
    agent_sandbox_stopped: bool = False

    def service(self, name: str | None) -> AsyncSandbox:
        """The sandbox of a Compose service; ``None`` or ``"main"`` is the agent's own."""
        if name in (None, "main"):
            return self.sandbox
        if self.compose is None or name not in self.compose.services:
            raise KeyError(f"No Compose service {name!r} in this task")
        return self.compose.services[name]

    async def stop_agent_side(self) -> None:
        """Stop the agent's sandbox, and its sidecars when the task is a Compose group."""
        if self.agent_sandbox_stopped:
            return
        await _cancel_entrypoint(self.entrypoint)
        if self.compose is not None:
            await self.compose.stop()
        else:
            await self.sandbox.stop()
        # Only a stop that succeeded releases the episode's final cleanup from stopping again.
        self.agent_sandbox_stopped = True


_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def resolve_env_templates(env: dict[str, str], secrets: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve Harbor's whole-value host templates without changing the stored task.

    ``secrets`` (the server's operator secrets) are consulted before the process environment; they are
    passed only when resolving the verifier process's environment.
    """
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = _ENV_TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
        elif secrets and match.group(1) in secrets:
            resolved[key] = secrets[match.group(1)]
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


def _is_root(user: str | None) -> bool:
    return user is None or str(user).split(":")[0] in ("root", "0")


async def _exec_as_root_user(
    sandbox: AsyncSandbox,
    command: str,
    *,
    configured_user: str | None,
    spec_user: str | None = None,
    cwd: str | None = "/",
    timeout_s: float = 60,
):
    """Run ``command`` as root (``cwd=None`` is the sandbox's default directory).

    ``configured_user`` is the OCI ``User`` of the image the sandbox started from
    (``HarborTask.sandbox_user``); ``spec_user`` is the default the sandbox was created with
    (``SandboxSpec.user``, the Dockerfile's final ``USER`` or ``[agent].user``), which the sandbox
    facade applies to every exec that names no user. Only when both are root does the plain exec
    path, which every provider supports, already run as root; otherwise ``user="root"`` is asked for.
    """
    user = None if _is_root(configured_user) and _is_root(spec_user) else "root"
    return await sandbox.exec(command, cwd=cwd, timeout_s=timeout_s, user=user)


def overlay_shell_command(step: OverlayRun, sandbox_env: Mapping[str, str] | None = None) -> str:
    """The shell line that runs one Dockerfile ``RUN`` with its ``ENV`` and ``WORKDIR`` in effect.

    The step's environment is already resolved to literals by the loader, so every value is
    exported quoted (nothing expands again here). ``sandbox_env`` is what the sandbox was
    created with (the Dockerfile's final ``ENV`` plus ``task.toml``); a key of it the step does
    not have yet is unset first, so the line sees the environment of its point in the Dockerfile,
    as a build step would, not the finished image's. The working directory exists by the time
    the line runs (see :func:`overlay_workdir_command`), so the line only enters it. The
    command's own exit status is the line's.
    """
    parts = [f"unset {shlex.quote(key)}" for key in sorted(set(sandbox_env or ()) - set(step.env))]
    parts += [f"export {key}={shlex.quote(value)}" for key, value in step.env.items()]
    if step.workdir:
        parts.append(f"cd {shlex.quote(step.workdir)} || exit 1")
    parts.append(step.command)
    return "; ".join(parts)


def overlay_workdir_command(workdir: str, user: str | None) -> str:
    """Create a ``WORKDIR`` as root the way BuildKit does: missing directories are made and owned by the user."""
    quoted = shlex.quote(workdir)
    if _is_root(user):
        return f"mkdir -p {quoted}"
    return f"test -d {quoted} || (mkdir -p {quoted} && chown {shlex.quote(user)} {quoted})"


def _is_numeric_user(user: str | None) -> bool:
    return user is not None and str(user).split(":")[0].isdigit() and not _is_root(user)


def _output_tail(result: SandboxExecResult, limit: int = OVERLAY_OUTPUT_TAIL) -> str:
    text = "\n".join(part.rstrip() for part in (result.stdout, result.stderr) if part and part.strip())
    return text[-limit:] if text else "(no output)"


def _check_overlay_result(result: SandboxExecResult, label: str, what: str | None, budget: float) -> None:
    """Classify one overlay exec: the task's fault (:class:`HarborOverlayError`) or the provider's."""
    subject = f"{label}: {what}" if what else label
    if result.error_type == "timeout":
        raise HarborOverlayError(f"{subject} exceeded [environment].build_timeout_sec={budget}")
    if result.error_type is not None:
        raise HarborOverlayUnavailable(f"{subject} could not run in the sandbox: {result.error_type}")
    if result.return_code != 0:
        raise HarborOverlayError(f"{subject} exited {result.return_code}; output tail: {_output_tail(result)}")


def _separate_mode(task: HarborTask, separate: bool | None) -> bool:
    """The verifier mode in effect: the task's own unless the server forced separate verification."""
    return (not task.config.is_shared_verifier) if separate is None else separate


def _verifier_runs_task_image(task: HarborTask, separate: bool | None = None) -> bool:
    """A separate verifier with no ``[verifier.environment]`` and no image built from ``tests/Dockerfile`` grades in
    the task's own image: Harbor's effective verifier environment is then a copy of ``[environment]``, and the loader
    only lets that shape through when the copy carries ``[environment].docker_image`` (Harbor's prebuilt verifier), so
    the task has no overlay (a declared image resolves none) and the verifier needs none. A shared-mode task the server
    verifies separately (``separate``, see ``separate_verifier_for_artifacts``) grades in the task image too, overlay
    and all (:func:`_verifier_gets_overlay`)."""
    return _separate_mode(task, separate) and task.config.verifier.environment is None and task.verifier_image is None


def _verifier_gets_overlay(task: HarborTask, separate: bool | None = None) -> bool:
    """A separate verifier that runs in the task's own image needs the same Dockerfile overlay as the agent. Only a
    shared-mode task forced into separate verification can have one: the loader admits a declared separate verifier in
    the task image with a prebuilt ``docker_image`` only."""
    return bool(task.overlay) and _verifier_runs_task_image(task, separate)


def _verifier_image(task: HarborTask, separate: bool | None = None) -> str | None:
    """The image a separate verifier runs in: ``[verifier.environment].docker_image``, else the one prepare built
    from ``tests/Dockerfile`` (``HarborTask.verifier_image`` covers both), else the agent's when the task declares no
    ``[verifier.environment]``; ``None`` in shared mode (the task's own, unless ``separate`` says what the server
    decided). A loaded task in separate mode always has one (the loader refuses the task otherwise); ``None`` with a
    ``[verifier.environment]`` is an invariant the seed still checks."""
    if not _separate_mode(task, separate):
        return None
    if task.verifier_image:
        return task.verifier_image
    if task.config.verifier.environment is None:
        return task.image
    return None


def _verifier_start(task: HarborTask, separate: bool | None = None) -> HarborTask | None:
    """The separate verifier's image as the entrypoint launcher reads one, or ``None`` when nothing is to start.

    Harbor ``up --wait``s the verifier's image, so its ``ENTRYPOINT`` runs and its ``HEALTHCHECK`` gates
    ``test.sh``, as the task image's do at seed. The verifier grades in the task's own image (the task
    itself, with its recorded start) or in the image prepare built from ``tests/Dockerfile``: the task with
    that image's recorded configuration in place of the agent image's (``HarborTask.verifier_image_config``:
    its entrypoint, the user and working directory it starts with, its healthcheck; no overlay, and no spec
    user, as :meth:`HarborEnvironmentServer._sandbox_spec` creates the sandbox). A declared prebuilt verifier
    image has no record, like a prebuilt ``docker_image``, and nothing is started in it.
    """
    if not _separate_mode(task, separate):
        return None
    if _verifier_runs_task_image(task, separate):
        return task
    config = task.verifier_image_config
    if config is None or task.verifier_image is None:
        return None
    return replace(
        task,
        image=task.verifier_image,
        built_image=task.verifier_image,
        overlay=(),
        user=None,
        sandbox_user=config.user,
        entrypoint=config.entrypoint,
        entrypoint_user=config.user,
        entrypoint_workdir=config.workdir,
        image_healthcheck=config.healthcheck,
    )


def _sandbox_resources(environment: HarborEnvironment, *, request_gpu_type: bool = False) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    if environment.cpus is not None:
        resources["cpu"] = environment.cpus
    if environment.memory_mb is not None:
        resources["memory_mib"] = environment.memory_mb
    if environment.storage_mb is not None:
        resources["disk_gib"] = max(1, math.ceil(environment.storage_mb / 1024))
    if environment.gpus:
        resources["gpu"] = environment.gpus
        if environment.gpu_types and request_gpu_type:
            resources["gpu_type"] = environment.gpu_types[0]
    return resources


def _is_image_pull_failure(error: BaseException) -> bool:
    """Whether a failed sandbox start names an image pull as its cause.

    Walks the cause/context chain and the members of exception groups: a Compose group whose cleanup
    after a failed pull failed too surfaces as an ``ExceptionGroup`` of the cleanup errors, raised while
    the pull failure was being handled.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current).lower()
        if any(marker in message for marker in IMAGE_PULL_FAILURE_MARKERS):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        chained = current.__cause__ or current.__context__
        if chained is not None:
            pending.append(chained)
    return False


def _pull_retry_delay(attempt: int) -> float:
    """Exponential backoff with jitter before retrying a failed image pull; ``attempt`` is the one that failed."""
    base = min(PULL_RETRY_MAX_DELAY_S, PULL_RETRY_BASE_DELAY_S * 2 ** (attempt - 1))
    return base * random.uniform(0.5, 1.5)


def _provider_name(provider_config: Any) -> str | None:
    """The registry name of a resolved single-key provider config; ``None`` for a provider instance (tests)."""
    return next(iter(provider_config)) if isinstance(provider_config, Mapping) else None


def _needs_gpu(task: HarborTask) -> bool:
    # The same predicate prepare routes the Compose capability check by (image_configs.check_task_compose_fields).
    return task_declares_gpus(task)


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
        self._sandboxes: dict[str, AsyncSandbox | AsyncSandboxCompose] = {}
        self._tasks: dict[tuple[Path, str], asyncio.Task[HarborTask]] = {}
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
            resolve_provider_metadata(self._provider_ref(task), global_config_dict)
            | self.config.sandbox_metadata
            | {"nemo_gym_environment_server": self.config.name, "harbor_task": task.task_id[:63], "harbor_role": role}
        )
        if ttl is None:
            ttl = self._episode_budget_s(task) + self.config.cleanup_timeout_seconds + self.config.sandbox_ttl_slack_s
        # The override replaces only the keys it names, so a GPU task routed to the GPU provider keeps `gpu`.
        resources = _sandbox_resources(environment, request_gpu_type=self.config.request_gpu_type) | (
            self.config.sandbox_resources_override or {}
        )
        # The Dockerfile's final USER (or [agent].user) is the default for every exec in the task's own
        # image. A separate verifier in another image keeps that image's own default. An overlay task
        # starts without it, because its RUN lines may be what creates the user; `_set_default_user`
        # switches the sandbox over once the overlay has run.
        user = task.user if (image is None or image == task.image) and not task.overlay else None
        return SandboxSpec(
            image=image or task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=resolve_env_templates(task_env),
            metadata=metadata,
            resources=resources,
            provider_options=self._provider_options_within(resources, task, role),
            user=user,
        )

    def _provider_options_within(self, resources: Mapping[str, Any], task: HarborTask, role: str) -> dict[str, Any]:
        """The configured provider options, with scheduling requests capped at the task's declared limits.

        A deployment's ``resource_requests`` (say ``disk_gib: 10`` for every sandbox) can exceed what a
        task declares (``storage = "4G"``); Kubernetes never schedules that pod and the provider would only
        see it pending. The provider clamps too; the warning here names the task.
        """
        options = dict(self.config.sandbox_provider_options)
        requests = options.get("resource_requests")
        if not isinstance(requests, Mapping):
            return options
        clamped, changed = clamp_resource_requests(requests, resources)
        for key, (request, limit) in changed.items():
            LOGGER.warning(
                "sandbox_provider_options.resource_requests.%s=%s exceeds the %s limit %s task %s declares; "
                "clamping the request to the limit so the sandbox can be scheduled",
                key,
                request,
                key,
                limit,
                f"{task.task_id} ({role})",
            )
        if changed:
            options["resource_requests"] = clamped
        return options

    def _separate_verifier(self, task: HarborTask) -> bool:
        """Whether this episode verifies in a fresh sandbox: the task's mode, or forced for tasks with artifacts."""
        return not task.config.is_shared_verifier or (
            self.config.separate_verifier_for_artifacts and bool(task.config.artifacts)
        )

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

        The harness's setup, `[agent].timeout_sec`, then `[verifier].timeout_sec` with its grace, plus
        a second sandbox start (and overlay) for a separate verifier. The seed is not counted: its own
        bounds are the sandbox ready timeout per start and the image pull retries, so a start that
        recovered on a retry begins the task with its whole budget.
        """
        separate = self._separate_verifier(task)
        budget = (
            self.config.harness_setup_allowance_s
            + task.config.agent.timeout_sec
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

    def _provider_ref(self, task: HarborTask) -> str:
        """Which top-level sandbox block runs this task: the GPU one when the task declares GPUs."""
        if self.config.gpu_sandbox_provider and _needs_gpu(task):
            return self.config.gpu_sandbox_provider
        return self.config.sandbox_provider

    async def _start_with_pull_retries(
        self,
        task: HarborTask,
        *,
        key: str,
        what: str,
        make: Callable[[], Any],
        start: Callable[[Any], Awaitable[None]],
    ) -> Any:
        """Start a sandbox or Compose group, retrying a start that failed on an image pull `image_pull_retries` times.

        ``make`` builds a fresh object (neither accepts a second ``start``), owned under ``key`` from before
        ``start`` so that a failed or half-started one is stopped whole. Only pull failures are retried here
        (a provider that rejects the request, or a sandbox that never becomes ready, is reported at once);
        the retries back off with jitter and queue behind a per-server gate, so one registry hiccup cannot
        multiply into a pull storm.
        """
        attempts = self.config.image_pull_retries + 1
        for attempt in range(1, attempts + 1):
            sandbox = make()
            self._sandboxes[key] = sandbox
            try:
                if attempt == 1:
                    await start(sandbox)
                else:
                    async with self._pull_retry_gate:
                        await start(sandbox)
                return sandbox
            except (SandboxCreateError, RuntimeError, OSError, ExceptionGroup) as exc:
                # A Compose group whose cleanup failed too raises an ExceptionGroup over the pull failure.
                if attempt >= attempts or not _is_image_pull_failure(exc):
                    raise
                LOGGER.warning(
                    f"{task.task_id}: image pull failed starting {what} (attempt {attempt}/{attempts}): {exc}"
                )
                try:
                    await self._stop_sandbox(key)
                except Exception:
                    # Still owned: shutdown retries the stop. Do not start a second sandbox under the same key.
                    LOGGER.exception(f"{task.task_id}: could not stop the {what} of a failed image pull")
                    raise exc from None
                await asyncio.sleep(_pull_retry_delay(attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, episode_key: str) -> AsyncSandbox:
        """Start the agent's sandbox, with the in-server image pull retries."""
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        spec = self._sandbox_spec(task, workdir)
        return await self._start_with_pull_retries(
            task,
            key=episode_key,
            what=repr(task.image),
            make=lambda: AsyncSandbox(provider_config),
            start=lambda sandbox: sandbox.start(spec),
        )

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

    def _compose_image_configs(self, task: HarborTask) -> dict[str, Any]:
        """The image configurations `gym dataset fetch` recorded next to the task folders."""
        path = task.path.parent / COMPOSE_IMAGES_FILE
        if not path.is_file():
            raise RuntimeError(
                f"Compose task {task.task_id!r} needs recorded image configurations at {path}; "
                "prepare the dataset again to record them"
            )
        return json.loads(path.read_text())

    def _volume_sources(self, task: HarborTask) -> dict[str, str]:
        """The provider block's ``volume_sources``: external volumes and foreign bind sources on shared storage."""
        return resolve_provider_volume_sources(self._provider_ref(task), get_global_config_dict())

    def _compose_document(self, task: HarborTask) -> dict[str, Any]:
        """The published overlay resolved against recorded image metadata, ready for the Compose adapter."""
        image_configs = self._compose_image_configs(task)
        global_config_dict = get_global_config_dict()
        provider = resolve_provider_config(self._provider_ref(task), global_config_dict)
        # The same reader prepare used: env_file inlined, ${VAR} resolved from the task, not the process, and the
        # images prepare built standing in for the sidecars it built (and main when the task runs from its built
        # image), their build: and pull_policy dropped. The record was read once by load_task: no hashing here.
        document = load_compose_document(task)
        # The sandbox has no pull step to configure, so ``pull_policy`` is dropped from every service (the loader
        # dropped it, with ``build:``, from the services prepare built). Prepare ran the same field check on this
        # same document (``nemo_gym.tasks.harbor.image_configs``).
        document = without_pull_policy(document)
        if task.built_image is None:
            # A pull or overlay task's main build: that only restates Harbor's own build of the Dockerfile is
            # dropped; a key that would change the image is the task's error, which prepare already reported.
            document, reason = without_restated_main_build(task, document)
            if reason is not None:
                raise HarborTaskError(f"{task.task_id}: {reason}")
        # The same field check prepare ran, with the same inputs: deploy.resources, networks, extra_hosts, post_start
        # and the rest take their normalized forms, each service is checked against the provider it runs on (a GPU
        # sidecar against the GPU block's) and `platform` against this deployment's, so a field the deployment cannot
        # apply fails here, before a sandbox. A provider instance in place of the config (tests) has no registry
        # name; the adapter checks it live.
        gpu_provider = (
            resolve_provider_config(self.config.gpu_sandbox_provider, global_config_dict)
            if self.config.gpu_sandbox_provider
            else None
        )
        check_task_compose_fields(
            task,
            document,
            provider=_provider_name(provider),
            gpu_provider=_provider_name(gpu_provider),
            platform=self._compose_platform(),
        )
        # The same planner prepare used: task-file bind mounts become uploads, named volumes stay for shared
        # storage, external ones resolve through the provider block. Sources point into the server's task copy.
        plan_task_compose_volumes(task, document, volume_sources=self._volume_sources(task))
        # Every service of the group, GPU-routed or not, starts with the one configured set of provider options,
        # so the platform they request (OpenSandbox ``platform: {os, arch}``; linux/amd64 when unset) is the one
        # each recorded image has to be for (and the one `_compose_platform` spelled for the field check above).
        platform = sandbox_platform(self.config.sandbox_provider_options)
        document = resolve_compose(document, task.image, image_configs, platform=platform)
        # A non-root sidecar runs without host injection and so takes no extra_hosts: prepare refused that already.
        adapt_non_root_sidecars(document, image_configs)
        if "opensandbox" in provider:
            opensandbox_shm_labels(document)
        sidecars = {a.service for a in task.config.artifacts} | {h.service for h in task.config.verifier.collect}
        missing = sidecars - {None, "main"} - set(document["services"])
        if missing:
            raise RuntimeError(
                f"Artifacts or collect hooks name Compose services that do not exist: {sorted(missing)}"
            )
        return document

    def _sidecar_provider_ref(self, task: HarborTask, service: dict[str, Any]) -> str:
        """Which sandbox block runs a Compose sidecar: the GPU one when the service reserves GPUs through
        ``deploy.resources`` and a GPU block is set, as ``_provider_ref`` routes the main service."""
        if self.config.gpu_sandbox_provider and service_gpus(service):
            return self.config.gpu_sandbox_provider
        return self._provider_ref(task)

    def _sidecar_spec(self, task: HarborTask, main_spec: SandboxSpec, service: dict[str, Any]) -> SandboxSpec:
        """A sidecar's spec: the main spec's deployment settings, no task resources, env or user (the sidecar keeps
        its own image's defaults; its ``deploy.resources`` are applied by the adapter). A GPU sidecar carries
        the task's ``gpu_types[0]`` when ``request_gpu_type`` is on, as the main service does, and the GPU
        block's ``default_metadata``. A sidecar keeps its own image's working directory (the adapter applies the
        recorded WorkingDir); main's workdir is a path that exists only in main's image."""
        resources: dict[str, Any] = {}
        metadata = dict(main_spec.metadata)
        if service_gpus(service):
            gpu_types = task.config.environment.gpu_types
            if gpu_types and self.config.request_gpu_type:
                resources["gpu_type"] = gpu_types[0]
            metadata = (
                resolve_provider_metadata(self._sidecar_provider_ref(task, service), get_global_config_dict())
                | self.config.sandbox_metadata
                | {key: metadata[key] for key in ("nemo_gym_environment_server", "harbor_task", "harbor_role")}
            )
        return replace(
            main_spec,
            resources=resources,
            env={},
            user=None,
            workdir=None,
            metadata=metadata,
            provider_options=dict(self.config.sandbox_provider_options),
        )

    def _compose_platform(self) -> str:
        """The platform this deployment runs, for a service's ``platform`` to match: the provider options'
        ``platform`` (OpenSandbox's ``{os, arch}``) when set, else linux/amd64, the one prepare records images for.
        Prepare reads the same ``sandbox_provider_options`` (``compose_platform_from_overrides``), so both agree."""
        return deployment_platform(self.config.sandbox_provider_options)

    async def _create_compose(self, task: HarborTask, episode_key: str) -> AsyncSandboxCompose:
        """Start the task's Compose group; the ``main`` service is the agent's sandbox."""
        # File reads (the Compose file, env_file, compose-images.json) off the event loop: seeds run concurrently.
        document = await asyncio.to_thread(self._compose_document, task)
        # Episode IDs are opaque caller input, never host paths: the document gets a fresh folder.
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="compose-", dir=self._artifacts_dir)) / "compose.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        main_spec = self._sandbox_spec(task, task.workdir)
        # Harbor's order for main: image ENV (baked in, or the Dockerfile ENV the loader folded into task.env for a
        # pull or overlay task) < the task's Compose file < Harbor's env override file, which holds only the
        # host-resolved [environment].env (harbor 0.22.0 environments/docker/docker.py 363-370, environments/base.py
        # 278-285). _sandbox_spec's env is task.env, the image layer with [environment].env already over it; the
        # file goes between the two here, and the adapter applies the whole spec env over the file (``{**env,
        # **spec.env}``), which leaves exactly those three layers in that order.
        compose_env = {
            key: str(value)
            for key, value in ((document["services"].get("main") or {}).get("environment") or {}).items()
            if value is not None
        }
        main_spec = replace(
            main_spec,
            env={**main_spec.env, **compose_env, **resolve_env_templates(task.config.environment.env or {})},
        )
        global_config_dict = get_global_config_dict()
        task_ref = self._provider_ref(task)
        # Sidecars keep their own image's default user; the task's USER belongs to the main image only. A sidecar
        # that reserves GPUs runs on the GPU block, per service, the way the main service does for the task.
        service_specs: dict[str, SandboxSpec] = {}
        service_providers: dict[str, Any] = {}
        for name, service in document["services"].items():
            if name == "main":
                service_specs[name] = main_spec
                continue
            service_specs[name] = self._sidecar_spec(task, main_spec, service)
            ref = self._sidecar_provider_ref(task, service)
            if ref != task_ref:
                service_providers[name] = resolve_provider_config(ref, global_config_dict)
        provider_config = resolve_provider_config(task_ref, global_config_dict)
        platform = self._compose_platform()
        # Ownership starts before start(): a group that fails half-way is still stopped whole, and a pull
        # failure of any service is retried with a fresh group, like a single sandbox's.
        return await self._start_with_pull_retries(
            task,
            key=episode_key,
            what="Compose group",
            make=lambda: AsyncSandboxCompose(
                provider_config,
                path,
                service_specs=service_specs,
                timeout_s=self.config.sandbox_ready_timeout_s,
                service_providers=service_providers,
                platform=platform,
                volume_sources=self._volume_sources(task),
                # Upload sources may only come from the server's own copy of the task, the one the planner saw.
                upload_root=task.path,
            ),
            start=lambda compose: compose.start(),
        )

    async def _set_default_user(self, sandbox: AsyncSandbox, task: HarborTask, names: dict[str, str]) -> None:
        """Make the task's user the sandbox's default for every exec, once it exists in the sandbox.

        A digit-only ``USER`` is resolved to its name with ``getent`` (``names`` is what the overlay
        already looked up), because OpenSandbox and PTY sessions switch user by name. When the uid
        names no user the numeric value stays, with a warning: Docker accepts it as it is.
        """
        user = task.user
        if user is None:
            return
        if _is_numeric_user(user):
            uid, _, group = str(user).partition(":")
            name = await self._user_name(sandbox, task, uid, names)
            if name is not None:
                user = name + (f":{group}" if group else "")
        sandbox.set_default_user(user)

    async def _user_name(self, sandbox: AsyncSandbox, task: HarborTask, uid: str, names: dict[str, str]) -> str | None:
        """The name of ``uid`` in the image, looked up once with ``getent``; ``None`` (warned once) when it has none."""
        if uid not in names:
            lookup = await _exec_as_root_user(
                sandbox, f"getent passwd {shlex.quote(uid)} | cut -d: -f1", configured_user=task.sandbox_user
            )
            lines = (lookup.stdout or "").strip().splitlines()
            names[uid] = lines[-1] if lookup.return_code == 0 and lines and lines[-1] else ""
            if not names[uid]:
                LOGGER.warning(
                    f"{task.task_id}: USER {uid} names no user in {task.image}; providers that switch user by name "
                    "cannot run commands as it"
                )
        return names[uid] or None

    async def _image_user_name(self, sandbox: AsyncSandbox, task: HarborTask, names: dict[str, str]) -> str | None:
        """The user the image starts as (its recorded ``User``; ``None`` is root), by name when it was numeric.

        It is what the entrypoint and the image ``HEALTHCHECK`` run as. Docker runs a numeric ``User``
        without a passwd entry; a provider that switches user by name cannot, so the number stays
        (with :meth:`_user_name`'s warning) and the launcher's own failure then names the cause.
        """
        user = task.entrypoint_user
        if not _is_numeric_user(user):
            return user
        uid, _, group = str(user).partition(":")
        name = await self._user_name(sandbox, task, uid, names)
        return user if name is None else name + (f":{group}" if group else "")

    async def _apply_overlay(self, sandbox: AsyncSandbox, task: HarborTask) -> dict[str, str]:
        """Run the Dockerfile ``RUN`` lines the loader recorded, in order, before anything else uses the sandbox.

        Each line runs with the ``ENV``, ``WORKDIR`` and ``USER`` the loader resolved for it, the
        way ``docker build`` would: the directory is created as root first (owned by the line's
        user when that is not root), a numeric ``USER`` is mapped to its name in the image, then
        the line runs. The whole overlay shares ``[environment].build_timeout_sec``, as a build
        would. A line that exits non-zero, runs out of budget or names a user the image does not
        have raises :class:`HarborOverlayError` with the command and the tail of its output: the
        task's Dockerfile does not apply, so retrying would not help. A provider that cannot run a
        step at all raises :class:`HarborOverlayUnavailable` instead, so the seed is retried.
        """
        user_names: dict[str, str] = {}
        if not task.overlay:
            return user_names
        budget = task.config.environment.build_timeout_sec
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        total = len(task.overlay)
        prepared_workdirs: set[str] = {"/"}  # the root always exists; every other WORKDIR is made on first use

        async def as_root(command: str, label: str, what: str) -> SandboxExecResult:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HarborOverlayError(f"{label} not run: [environment].build_timeout_sec={budget} is used up")
            result = await _exec_as_root_user(
                sandbox, command, configured_user=task.sandbox_user, spec_user=task.user, timeout_s=remaining
            )
            _check_overlay_result(result, label, what, budget)
            return result

        for index, step in enumerate(task.overlay, 1):
            label = f"RUN line {index}/{total} {step.command!r}"
            user = step.user
            if _is_numeric_user(user):
                uid, _, group = str(user).partition(":")
                if uid not in user_names:
                    lookup = await as_root(
                        f"getent passwd {shlex.quote(uid)} | cut -d: -f1", label, f"USER {uid} lookup"
                    )
                    name = (lookup.stdout or "").strip().splitlines()
                    if not name or not name[-1]:
                        raise HarborOverlayError(f"{label}: USER {uid} names no user in {task.image}")
                    user_names[uid] = name[-1]
                user = user_names[uid] + (f":{group}" if group else "")
            if step.workdir not in prepared_workdirs:
                await as_root(overlay_workdir_command(step.workdir, user), label, f"WORKDIR {step.workdir}")
                prepared_workdirs.add(step.workdir)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HarborOverlayError(f"{label} not run: [environment].build_timeout_sec={budget} is used up")
            command = overlay_shell_command(step, task.env)
            if _is_root(user):
                result = await _exec_as_root_user(
                    sandbox, command, configured_user=task.sandbox_user, spec_user=task.user, timeout_s=remaining
                )
            else:
                result = await sandbox.exec(command, cwd="/", timeout_s=remaining, user=user)
            _check_overlay_result(result, label, None, budget)
        return user_names

    async def _start_entrypoint(
        self, sandbox: AsyncSandbox, task: HarborTask, user: str | None, *, prepared_workdir: str | None = None
    ) -> EntrypointLaunch | None:
        """Start the image's ``ENTRYPOINT`` beside the keepalive, as Harbor's Compose start leaves it running.

        Harbor starts the image with ``command: sh -c "sleep infinity"``, which replaces CMD only, so the
        entrypoint is PID 1 with that as its arguments, as the image's user, in its working directory.
        Gym's providers start their own keepalive instead, so the same process is started here as a
        long-lived exec through a ``run.sh`` that records a started marker, the exit status and the
        output under :data:`ENTRYPOINT_DIR`, the way the Compose adapter starts a service. ``user`` is
        the image's user by name (:meth:`_image_user_name`). For a Dockerfile the loader resolved (not a
        built image, which holds its ``WorkingDir``) the entrypoint's working directory is created as
        ``docker build`` would when it is not ``prepared_workdir``, the one the seed made. The call
        returns once the marker exists and the first exit probe has passed (:meth:`_check_entrypoint`).
        A launcher that ends before the marker exists is classified as an overlay step is: the provider
        could not run it (:class:`HarborEntrypointUnavailable`, retried), or its own shell failed (a
        ``su`` or ``chdir`` for the image's user or directory: :class:`HarborEntrypointError`, terminal).
        """
        argv = entrypoint_to_run(task)
        if argv is None:
            return None
        made = await _exec_as_root(sandbox, f"mkdir -p {ENTRYPOINT_DIR} && chmod 777 {ENTRYPOINT_DIR}", timeout_s=60)
        if made.return_code:
            raise HarborEntrypointUnavailable(f"could not create {ENTRYPOINT_DIR}: {made.stderr or made.stdout}")
        # The recorded User and WorkingDir, named explicitly: `[agent].user` and `[environment].workdir` shape
        # the sandbox's defaults for the commands run beside the entrypoint, not the entrypoint itself.
        cwd = task.entrypoint_workdir or "/"
        if task.built_image is None and task.entrypoint_workdir and task.entrypoint_workdir != prepared_workdir:
            # The Dockerfile's final WORKDIR, which `docker build` creates in the image; only a RUN line after it
            # made the overlay create it, and the seed makes the task's workdir, which may be another.
            workdir = await _exec_as_root_user(
                sandbox, overlay_workdir_command(cwd, user), configured_user=task.sandbox_user, spec_user=task.user
            )
            if workdir.error_type is not None:
                raise HarborEntrypointUnavailable(f"WORKDIR {cwd} could not be created: {workdir.error_type}")
            if workdir.return_code:
                raise HarborEntrypointError(
                    f"WORKDIR {cwd} for ENTRYPOINT {shlex.join(argv)} could not be created; "
                    f"output tail: {_output_tail(workdir)}"
                )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.sh"
            path.write_text(entrypoint_script(argv))
            await sandbox.upload(path, f"{ENTRYPOINT_DIR}/run.sh")
        command = f"sh {ENTRYPOINT_DIR}/run.sh >{ENTRYPOINT_LOG} 2>&1 </dev/null"
        if _is_root(user):
            run = _exec_as_root_user(
                sandbox, command, configured_user=task.sandbox_user, spec_user=task.user, cwd=cwd, timeout_s=None
            )
        else:
            run = sandbox.exec(command, cwd=cwd, timeout_s=None, user=user)
        # The exec is held open for as long as the provider allows: some providers reap background children as
        # soon as their launching shell exits. Docker caps it at `exec.default_timeout_s` and OpenSandbox without
        # `operations.background_exec` at `connection.request_timeout_s`; past that the process is untracked
        # (noticed and logged by `_check_entrypoint`). It is cancelled before the sandbox is stopped.
        launch = EntrypointLaunch(argv=argv, process=asyncio.create_task(run))
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + ENTRYPOINT_START_TIMEOUT_S
            while True:
                finished = launch.process.done()
                started = await sandbox.exec(f"test -f {ENTRYPOINT_DIR}/started", timeout_s=60)
                if started.return_code == 0:
                    break
                if finished:
                    error = launch.process.exception()
                    if error is not None:
                        raise HarborEntrypointUnavailable(f"the entrypoint launcher could not run: {error}") from error
                    outcome = launch.process.result()
                    if outcome.error_type is None and outcome.return_code != 0:
                        # `touch started` is run.sh's first line, in a world-writable folder: the launcher's own
                        # shell (`su`, `chdir`, `sh`) never reached the script. The image's fault, as a RUN
                        # line's non-zero exit is; another sandbox would repeat it.
                        raise HarborEntrypointError(
                            f"the launcher for ENTRYPOINT {shlex.join(argv)} (as {user or 'root'} in {cwd}) exited "
                            f"{outcome.return_code} before starting it; output tail: {_output_tail(outcome)}"
                        )
                    raise HarborEntrypointUnavailable(
                        f"the entrypoint launcher exited before starting: "
                        f"{outcome.error_type or outcome.stderr or outcome.stdout or 'no output'}"
                    )
                if loop.time() >= deadline:
                    raise HarborEntrypointUnavailable(
                        f"the entrypoint launcher did not start within {ENTRYPOINT_START_TIMEOUT_S:g}s"
                    )
                await asyncio.sleep(ENTRYPOINT_POLL_INTERVAL_S)
            await self._check_entrypoint(sandbox, task, launch)
        except BaseException:
            await _cancel_entrypoint(launch)
            raise
        return launch

    async def _entrypoint_exit_status(self, sandbox: AsyncSandbox) -> str | None:
        """What ``run.sh`` recorded as the entrypoint's exit status, or ``None`` while it has not exited."""
        probe = await sandbox.exec(
            f"if test -f {ENTRYPOINT_DIR}/exit; then cat {ENTRYPOINT_DIR}/exit; fi", timeout_s=60
        )
        if probe.return_code:
            raise HarborEntrypointUnavailable(
                f"cannot read the entrypoint's exit status: {probe.stderr or probe.stdout}"
            )
        return (probe.stdout or "").strip() or None

    async def _entrypoint_output_tail(self, sandbox: AsyncSandbox) -> str:
        tail = await sandbox.exec(f"tail -c {OVERLAY_OUTPUT_TAIL} {ENTRYPOINT_LOG}", timeout_s=60)
        return (tail.stdout or "").strip() or "(no output)"

    @staticmethod
    def _notice_launcher_end(task: HarborTask, launch: EntrypointLaunch) -> None:
        """Log once when the provider ended the exec holding the entrypoint while it had not exited.

        Docker's exec returns ``timeout`` at ``exec.default_timeout_s`` with the process still running,
        OpenSandbox's foreground exec at ``connection.request_timeout_s``: from then on the entrypoint
        is untracked (its exit file is still probed), and the cancel at stop has nothing to end.
        """
        if launch.noticed:
            return
        outcome = _launch_outcome(launch)
        if outcome is None:
            return
        launch.noticed = True
        LOGGER.warning(
            "%s: the exec holding ENTRYPOINT %s ended (%s) before the entrypoint exited: the provider capped it, "
            "so the process is no longer tracked (OpenSandbox deployments need operations.background_exec for "
            "long-lived launchers)",
            task.task_id,
            shlex.join(launch.argv),
            outcome,
        )

    async def _check_entrypoint(self, sandbox: AsyncSandbox, task: HarborTask, launch: EntrypointLaunch) -> None:
        """Fail the seed when the entrypoint has exited non-zero; warn once when it returned 0.

        Under Harbor an entrypoint that returns ends the container and ``up --wait`` fails the trial
        either way. A non-zero status is the task's fault (:class:`HarborEntrypointError`, with the
        output tail). A zero status means it returned without running its keepalive argument: the
        sandbox's own keepalive is still up, so the episode goes on, the one deliberate leniency.
        """
        if launch.returned:
            return
        status = await self._entrypoint_exit_status(sandbox)
        if status is None:
            self._notice_launcher_end(task, launch)
            return
        output = await self._entrypoint_output_tail(sandbox)
        if status != "0":
            raise HarborEntrypointError(f"ENTRYPOINT {shlex.join(launch.argv)} exited {status}; output tail: {output}")
        launch.returned = True
        LOGGER.warning(
            "%s: ENTRYPOINT %s exited 0 without running its keepalive argument (under Harbor the container would have "
            "exited); continuing with the sandbox's own keepalive. Output tail: %s",
            task.task_id,
            shlex.join(launch.argv),
            output,
        )

    async def _entrypoint_died_during_rollout(
        self, sandbox: AsyncSandbox, task: HarborTask, launch: EntrypointLaunch, logs_dir: Path
    ) -> str | None:
        """Read the entrypoint's exit status back after the rollout; the reason to mask the sample when it died.

        Under Harbor the entrypoint is PID 1: when it exits the container dies and the trial errors
        instead of scoring. Here the keepalive outlives it, so the status is read from the runtime folder
        and kept as :data:`ENTRYPOINT_EXIT_NAME` beside the output. That folder is world-writable (a
        non-root entrypoint writes the markers) and the agent had the sandbox, so the file alone does not
        void a rollout: the sample is masked only when the exec holding the entrypoint returned on its own
        (``run.sh`` finished, so the entrypoint had exited) and the status is non-zero. A status the exec
        does not corroborate is logged as an error and the sample is scored. A zero status gets the seed's
        leniency. Nothing here fails the verification: a probe that cannot run is logged.
        """
        if launch.returned:
            return None
        try:
            status = await self._entrypoint_exit_status(sandbox)
        except Exception:
            LOGGER.warning("Could not read the entrypoint's exit status for %s", task.task_id, exc_info=True)
            return None
        if status is None:
            self._notice_launcher_end(task, launch)
            return None
        try:
            (logs_dir / ENTRYPOINT_EXIT_NAME).write_text(status + "\n")
        except OSError:
            LOGGER.warning("Could not record the entrypoint's exit status for %s", task.task_id, exc_info=True)
        argv = shlex.join(launch.argv)
        if status == "0":
            launch.returned = True
            LOGGER.warning(
                "%s: ENTRYPOINT %s exited 0 during the rollout without running its keepalive argument (under Harbor "
                "the container would have exited); the sample is scored",
                task.task_id,
                argv,
            )
            return None
        try:
            output = await self._entrypoint_output_tail(sandbox)
        except Exception:
            output = "(unavailable)"
        if _launch_returned(launch):
            LOGGER.error(
                "Image ENTRYPOINT %s exited %s during the rollout for %s; the sample is masked (under Harbor the "
                "container would have died). Output tail: %s",
                argv,
                status,
                task.task_id,
                output,
            )
            return f"Image ENTRYPOINT {argv} exited {status} during the rollout; output tail: {output}"
        LOGGER.error(
            "%s: ENTRYPOINT %s has exit status %s recorded, but the exec holding it did not return (%s): scored, "
            "not masked; the status is kept as %s. Output tail: %s",
            task.task_id,
            argv,
            status,
            _launch_outcome(launch) or "still held",
            ENTRYPOINT_EXIT_NAME,
            output,
        )
        return None

    async def _wait_healthy(
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        entrypoint: EntrypointLaunch | None = None,
        *,
        user: str | None = None,
        task_check: bool = True,
    ) -> None:
        """Poll the image's ``HEALTHCHECK``, then the task's ``[environment.healthcheck]``, until each passes.

        That is Harbor's order: ``up --wait`` gates on the image HEALTHCHECK, then ``run_healthcheck()``
        runs the task's, before the agent user is set. Both run as the container's user (``user``, the
        image's ``User`` by name; ``None`` is root): the daemon runs the image check in the image
        ``WorkingDir``, Harbor's exec runs the task check in the task workdir. ``task_check=False`` leaves
        the task's check out (the separate verifier's sandbox, where Harbor never runs it). An entrypoint
        started by the seed is probed before every attempt, so one that dies fails fast with its output
        instead of waiting out the retries. A check that never passes is a retryable seed failure.
        """
        checks: list[tuple[str, HarborHealthcheck, str | None]] = []
        # A Compose task's main service already waited for its recorded healthcheck in the adapter.
        image_check = None if task.needs_compose else image_healthcheck_settings(task.image_healthcheck)
        if image_check is not None:
            checks.append((" (the image HEALTHCHECK)", image_check, task.entrypoint_workdir or "/"))
        if task_check and task.config.environment.healthcheck is not None:
            checks.append(("", task.config.environment.healthcheck, None))
        loop = asyncio.get_running_loop()
        for label, check, cwd in checks:
            grace_until = loop.time() + check.start_period_sec
            failures = 0
            while True:
                if entrypoint is not None:
                    await self._check_entrypoint(sandbox, task, entrypoint)
                if _is_root(user):
                    result = await _exec_as_root_user(
                        sandbox,
                        check.command,
                        configured_user=task.sandbox_user,
                        spec_user=task.user,
                        cwd=cwd,
                        timeout_s=check.timeout_sec + 5,
                    )
                else:
                    result = await sandbox.exec(check.command, cwd=cwd, timeout_s=check.timeout_sec + 5, user=user)
                if result.return_code == 0:
                    break
                in_grace = loop.time() < grace_until
                if not in_grace:
                    failures += 1
                    if failures >= check.retries:
                        ended = ""
                        if entrypoint is not None and entrypoint.noticed:
                            ended = f" after the exec holding the entrypoint ended ({_launch_outcome(entrypoint)})"
                        raise RuntimeError(
                            f"Healthcheck failed {check.retries} times: {check.command!r}{label}{ended}"
                        )
                await asyncio.sleep(check.start_interval_sec if in_grace else check.interval_sec)

    async def _write_task_context(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        """Leave the task's in-sandbox tool declarations where a harness can read them.

        ``/tmp/.nemo-gym/task.json`` carries ``mcp_servers`` (Harbor's ``[[environment.mcp_servers]]``)
        and ``skills_dir``. Harnesses that know how to talk to task MCP servers from inside the sandbox
        read it at session start; nothing in the episode protocol needs to change for that.
        """
        environment = task.config.environment
        if not environment.mcp_servers and not environment.skills_dir:
            return
        context = {
            "mcp_servers": [server.model_dump(mode="json") for server in environment.mcp_servers],
            "skills_dir": environment.skills_dir,
        }
        payload = shlex.quote(json.dumps(context))
        # Root first, falling back to the default user on sandboxes that cannot switch identities; /tmp
        # is writable either way, and the modes only need the agent's own user to read the file.
        result = await _exec_as_root(
            sandbox,
            f"mkdir -p {TASK_CONTEXT_DIR} && printf %s {payload} > {TASK_CONTEXT_FILE} "
            f"&& (chmod 755 {TASK_CONTEXT_DIR} && chmod 644 {TASK_CONTEXT_FILE} || true)",
            timeout_s=60,
        )
        if result.return_code:
            raise RuntimeError(f"Could not write {TASK_CONTEXT_FILE}: {result.stderr or result.stdout}")

    async def _create_verifier_sandbox(self, task: HarborTask, *, episode_key: str | None = None) -> AsyncSandbox:
        """A separate verifier runs in its own sandbox, sized by ``[verifier.environment]``.

        Without a ``[verifier.environment]`` and a built verifier image the verifier runs in the task's own prebuilt
        image, so it is created like the agent's sandbox: the Dockerfile's resolved ``ENV`` under ``task.toml``'s.
        """
        ttl = task.config.verifier.timeout_sec + self.config.sandbox_ttl_slack_s
        if _verifier_gets_overlay(task, True):
            ttl += task.config.environment.build_timeout_sec
        environment = task.config.verifier.environment
        if _verifier_runs_task_image(task, True):
            # The task image keeps the task's working directory, as the agent had it, and the agent's resolved
            # environment (Dockerfile ENV under task.toml's).
            workdir = task.workdir
        elif environment is None:
            # Built from tests/Dockerfile under Harbor's copy of [environment]: that block's settings and
            # `workdir` (else the image's own WORKDIR), not the agent image's Dockerfile ENV and WORKDIR.
            environment = task.config.environment
            workdir = environment.workdir
        else:
            # An explicit verifier image starts in its `[verifier.environment].workdir`, else its own WORKDIR.
            workdir = environment.workdir
        spec = self._sandbox_spec(
            task,
            workdir,
            environment=environment,
            image=_verifier_image(task, True),
            role="verifier",
            ttl=ttl,
        )
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        if episode_key is None:
            sandbox = AsyncSandbox(provider_config)
            await self._within_ready_deadline(sandbox.start(spec))
            return sandbox
        # Owned from before start(), like the agent's sandbox, so a failed or interrupted stop is retried;
        # a start that failed on an image pull is retried like the agent's, under the same ready deadline.
        return await self._within_ready_deadline(
            self._start_with_pull_retries(
                task,
                key=_verifier_key(episode_key),
                what=f"verifier {spec.image!r}",
                make=lambda: AsyncSandbox(provider_config),
                start=lambda sandbox: sandbox.start(spec),
            )
        )

    async def _prepare_workdir(self, sandbox: AsyncSandbox, task: HarborTask, workdir: str | None) -> str:
        """Create the working directory, or resolve the image's own WORKDIR when the task sets none."""
        if workdir is None:
            result = await sandbox.exec("pwd", timeout_s=60)
            if result.return_code != 0 or not (result.stdout or "").strip():
                raise RuntimeError(f"Could not resolve the image working directory: {result.stderr or result.stdout}")
            workdir = (result.stdout or "").strip().splitlines()[-1]
        commands = [f"mkdir -p {shlex.quote(workdir)}"]
        if not _is_root(task.user) and workdir != "/":
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
        # Harbor mounts /logs into the agent's container; tasks may write to /logs/artifacts during the
        # episode. Best effort: an image that runs as a non-root user without the capability to switch
        # to root cannot create top-level folders, and the agent may still not need them.
        logs = await _exec_as_root(
            sandbox,
            f"mkdir -p {AGENT_LOGS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR} "
            f"&& chmod 777 {AGENT_LOGS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR}",
            timeout_s=60,
        )
        if logs.return_code != 0:
            LOGGER.warning(
                f"{task.task_id}: could not create /logs folders in the agent sandbox: {logs.stderr or logs.stdout}"
            )
        return workdir

    async def _sandbox_access(self, session: HarborSession) -> SandboxAccess:
        return SandboxAccess(
            connection=DirectSandboxConnection(
                provider_config_ref=session.provider_ref,
                descriptor=await session.sandbox.serialize(),
            ),
            workdir=session.workdir,
            user=session.task.user,
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
                except Exception as error:
                    raise self._failure(
                        stage="agent",
                        failure_reason=str(error),
                        terminal=not _is_retryable_dependency_error(error),
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
                except TimeoutError as error:
                    # The turn is finished; a harness that is slow to tear down must not cost its score. Its
                    # observations are lost with the reply, and the final unwind retries the close.
                    LOGGER.warning(f"{session.task.task_id}: agent session close timed out, verifying anyway: {error}")
                    agent_observations = None
                except Exception as error:
                    raise self._failure(
                        stage="cleanup",
                        failure_reason=str(error),
                        terminal=not _is_retryable_dependency_error(error),
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
                        ) from error
        except TimeoutError:
            if task_deadline is None or not task_deadline.expired():
                raise
            # The final unwind closes the agent session and stops the sandbox.
            raise self._failure(
                stage=stage,
                failure_reason=(
                    f"Episode exceeded the time budget of task {session.task.task_id!r} ({budget:.0f}s from the "
                    f"sandbox being ready: harness setup, [agent].timeout_sec={session.task.config.agent.timeout_sec:g} "
                    f"and [verifier].timeout_sec={session.task.config.verifier.timeout_sec:g}); set "
                    "enforce_task_timeouts: false to run without it"
                ),
                terminal=False,
                failure_kind=TASK_DEADLINE_KIND,
                partial_response=agent_response,
                agent_observations=agent_observations,
            )

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
        # Fail before provisioning; retain templates in the snapshot and resolve at use time.
        resolve_env_templates(task.env)
        resolve_env_templates(task.config.verifier.env, self.config.secrets)
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
            names = await self._apply_overlay(sandbox, task)
            await self._set_default_user(sandbox, task, names)
            image_user = await self._image_user_name(sandbox, task, names)
            # The working directory and /logs exist before the entrypoint starts, as Harbor's bind mounts do.
            session.workdir = await self._prepare_workdir(sandbox, task, task.workdir)
            launch = session.entrypoint = await self._start_entrypoint(
                sandbox, task, image_user, prepared_workdir=session.workdir
            )
            await self._wait_healthy(sandbox, task, launch, user=image_user)
            await self._write_task_context(sandbox, task)
            if launch is not None:
                # The last look before the hand-over: an entrypoint that died during the seed's remaining steps
                # (a setup script failing after its first lines) is a seed failure, not a rollout on an empty task.
                await self._check_entrypoint(sandbox, task, launch)
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
            try:
                await _cancel_entrypoint(launch)
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop Harbor sandbox after failed seed")
            if not isinstance(exc, Exception):
                raise
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
        failure_kind: str | None = None,
    ) -> HandledEpisodeError:
        return HandledEpisodeError(
            SingleAgentTurnFailure(
                stage=stage,
                failure_reason=failure_reason[:2000],
                terminal=terminal,
                failure_kind=failure_kind,
                partial_response=partial_response,
                agent_observations=agent_observations,
            )
        )

    def _export_trajectory(self, request: SingleAgentTurnRequest, agent_response: NeMoGymResponse) -> bytes:
        """The agent's transcript as ATIF v1.7 JSON, for ``/logs/agent/trajectory.json``.

        Exported from the Gym transcript (the task's input and the harness's output); raises
        :class:`AtifExportError` for anything the converter cannot represent, which masks the turn.
        """
        params: Any = request.task.task_input.responses_create_params
        input_items = params.get("input") if isinstance(params, Mapping) else getattr(params, "input", None)
        usage = agent_response.usage.model_dump(mode="json") if agent_response.usage is not None else None
        trajectory = transcript_to_atif(
            input_items,
            [item.model_dump(mode="json") for item in agent_response.output],
            session_id=request.episode_id.capture_key,
            agent_name=self.config.agent_server.name,
            agent_version=__version__,
            model_name=agent_response.model,
            usage=usage,
        )
        return encode_atif_trajectory(trajectory)

    # -- verification ----------------------------------------------------------------------

    async def _run_verifier(self, session: HarborSession, trajectory: bytes) -> dict[str, Any]:
        """Run ``tests/test.sh`` and read the reward it wrote, in the agent's sandbox or a separate one.

        ``trajectory`` is the exported ATIF transcript, written to ``/logs/agent/trajectory.json`` in the
        sandbox that runs ``test.sh`` before it starts.
        """
        if self._separate_verifier(session.task):
            return await self._run_separate_verifier(session, trajectory) | {"verifier_mode": "separate"}
        return await self._run_shared_verifier(session, trajectory) | {"verifier_mode": "shared"}

    async def _upload_trajectory(self, sandbox: AsyncSandbox, trajectory: bytes, logs_dir: Path) -> None:
        """Keep a host copy of the transcript beside the verifier logs and place it where the verifier reads it."""
        host = logs_dir / TRAJECTORY_FILE_NAME
        host.write_bytes(trajectory)
        await upload_path(sandbox, host, f"{AGENT_LOGS_DIR}/{TRAJECTORY_FILE_NAME}")

    def _verifier_env(self, settings: Any) -> dict[str, str]:
        """The `test.sh` process environment: the task's resolved `[verifier].env`, and nothing else.

        Operator secrets reach it only through the `${NAME}` templates the task writes (Harbor's
        verifier resolves its env the same way), so a task that names none runs test.sh without any.
        """
        return resolve_env_templates(settings.env, self.config.secrets)

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
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        logs_dir: Path,
        *,
        timed_out: bool,
        entrypoint_log: str | None = None,
    ) -> tuple[dict[str, float] | None, str | None]:
        """Download the reward files, then whatever else the verifier wrote.

        Reward transfer is required; optional diagnostics cannot invalidate a measured score, and
        a timed-out run has no reward to transfer but its partial logs are still worth keeping.
        With ``entrypoint_log``, the sandbox's entrypoint output lands under that name too.
        """
        rewards = problem = None
        if not timed_out:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt"))
            rewards, problem = parse_reward_file(logs_dir)
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        if entrypoint_log is not None:
            await self._collect_entrypoint_log(sandbox, task, logs_dir / entrypoint_log)
        return rewards, problem

    async def _collect_entrypoint_log(self, sandbox: AsyncSandbox, task: HarborTask, target: Path) -> None:
        """Keep the entrypoint's output with the verification's logs; best effort, never a masked sample."""
        try:
            kind = await download_path(sandbox, ENTRYPOINT_LOG, target)
        except Exception:
            LOGGER.warning("Could not collect the entrypoint log for %s", task.task_id, exc_info=True)
            return
        if kind is None:
            LOGGER.warning("%s: no entrypoint log at %s to collect", task.task_id, ENTRYPOINT_LOG)

    async def _run_separate_verifier(self, session: HarborSession, trajectory: bytes) -> dict[str, Any]:
        """Harbor's separate mode: collect the agent's artifacts, stop its sandbox, verify in a fresh one.

        The verifier sandbox comes from ``[verifier.environment]``, the image prepare built from
        ``tests/Dockerfile``, or the task's prebuilt image (``_verifier_image``; no overlay runs there, the loader
        admits the task-image shape only with a declared image). ``/logs/artifacts`` and every ``artifacts``
        entry are copied through the host into the same paths, the verifier image's recorded entrypoint
        and ``HEALTHCHECK`` run as they do at seed (:func:`_verifier_start`), then ``tests/`` is uploaded
        unless the image owns ``/tests/test.sh``, and ``test.sh`` runs there. Sidecar (Compose) artifacts
        and hooks are not collected yet.
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
        verifier_entrypoint: EntrypointLaunch | None = None
        try:
            for hook in settings.collect:
                try:
                    sandbox = session.service(hook.service)
                except KeyError as exc:
                    LOGGER.warning(f"{task.task_id}: collect hook skipped: {exc}")
                    continue
                main = hook.service in (None, "main")
                result = await sandbox.exec(
                    hook.command if main else f"sh -c {shlex.quote(hook.command)}",
                    cwd=session.workdir if main else None,
                    timeout_s=hook.timeout_sec + 30,
                    user=hook.user,
                )
                if result.return_code:
                    LOGGER.warning(f"{task.task_id}: collect hook exited {result.return_code}: {hook.command!r}")
            restored: list[tuple[Path, str]] = []
            for artifact in _collected_artifacts(task):
                try:
                    sandbox = session.service(artifact.service)
                except KeyError as exc:
                    LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} skipped: {exc}")
                    continue
                host = artifacts_dir / artifact.host_path
                # Best effort per artifact, as Harbor collects them: what the agent left behind (an
                # unreadable tree, a link that escapes the folder, a file where a folder was expected)
                # cannot mask the sample. Symbolic links travel as links, because the restore target is
                # another sandbox, not the host. Anything else (a lost sandbox, a full disk, a dropped
                # connection) is Gym's outage and still masks.
                try:
                    kind = await download_path(sandbox, artifact.source, host, links="keep")
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
            died = None
            if session.entrypoint is not None:
                await self._collect_entrypoint_log(session.sandbox, task, logs_dir / ENTRYPOINT_LOG_NAME)
                died = await self._entrypoint_died_during_rollout(session.sandbox, task, session.entrypoint, logs_dir)
            # Harbor stops the agent's containers before the verifier starts; nothing may keep running.
            # The episode key stays reserved until the episode's final cleanup, which releases it without
            # stopping the agent side again.
            await session.stop_agent_side()
            if died is not None:
                return self._masked(ENTRYPOINT_FAILED_KIND, died) | {"verifier_logs_dir": str(logs_dir.resolve())}

            verifier = await self._create_verifier_sandbox(task, episode_key=session.session_id)
            names: dict[str, str] = {}
            if _verifier_gets_overlay(task, True):
                # A shared-mode task verified separately runs in the task's own image, so it needs the same overlay
                # as the agent did (a declared separate verifier in the task image is a prebuilt image: no overlay).
                names = await self._apply_overlay(verifier, task)
                await self._set_default_user(verifier, task, names)
            # The verifier's /logs folders exist before its entrypoint starts, as in the seed and as Harbor's
            # /logs/verifier bind mount does; /tests and the artifacts follow, as Harbor uploads them after start.
            prepare = await _exec_as_root(
                verifier,
                f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR} {ARTIFACTS_DIR} "
                f"&& chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            start = _verifier_start(task, True)
            if start is not None:
                # Harbor only `up --wait`s the verifier's image (built from tests/Dockerfile, or the task's own when
                # its effective verifier environment is a copy of [environment] with a docker_image): that image's
                # entrypoint and HEALTHCHECK run here as its user; the task's [environment.healthcheck] is not run,
                # as Harbor does not run it in the verifier.
                image_user = await self._image_user_name(verifier, start, names)
                verifier_entrypoint = await self._start_entrypoint(verifier, start, image_user)
                await self._wait_healthy(verifier, start, verifier_entrypoint, user=image_user, task_check=False)
            # A prebuilt verifier image usually ships /tests already, with the modes its tests expect, and one
            # built from tests/Dockerfile owns /tests the way its file copied it (Harbor uploads nothing into it,
            # `skip_tests_upload`). Only upload the package's tests when the image has none, as the old server did.
            if await remote_kind(verifier, f"{TESTS_DIR}/test.sh") is None:
                if task.verifier_image and task.verifier_image not in self._verifier_images_without_tests:
                    self._verifier_images_without_tests.add(task.verifier_image)
                    LOGGER.warning(
                        "%s: verifier image %s has no %s/test.sh; uploading the task's tests/ into it (Harbor uploads "
                        "nothing into a separate verifier's image, so its tests/Dockerfile usually copies the folder)",
                        task.task_id,
                        task.verifier_image,
                        TESTS_DIR,
                    )
                await upload_dir(verifier, task.path / "tests", TESTS_DIR)
            for host, source in restored:
                await upload_path(verifier, host, source)
            await self._upload_trajectory(verifier, trajectory, logs_dir)
            if verifier_entrypoint is not None:
                # The last look before test.sh, in a fresh sandbox the agent never had: its exit status is trusted.
                await self._check_entrypoint(verifier, start, verifier_entrypoint)
            budget = int(settings.timeout_sec)
            result = await verifier.exec(
                f"chmod +x {TESTS_DIR}/test.sh; timeout --signal=KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                env=self._verifier_env(settings),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            timed_out = result.error_type == "timeout" or result.return_code == 137
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            rewards, problem = await self._collect_reward(
                verifier,
                task,
                logs_dir,
                timed_out=timed_out,
                entrypoint_log=VERIFIER_ENTRYPOINT_LOG_NAME if verifier_entrypoint is not None else None,
            )
        except HTTPException:
            raise
        except HarborOverlayError as exc:
            # The rollout is complete; losing the verifier's sandbox to its overlay must not discard it.
            LOGGER.error(f"Dockerfile overlay failed in the verifier sandbox for {task.task_id}: {exc}")
            return self._masked(OVERLAY_FAILED_KIND, f"Dockerfile overlay failed in the verifier sandbox: {exc}")
        except HarborEntrypointError as exc:
            # The rollout is complete; the verifier image's entrypoint failing in its fresh sandbox must not discard it.
            LOGGER.error(f"Image ENTRYPOINT failed in the verifier sandbox for {task.task_id}: {exc}")
            return self._masked(ENTRYPOINT_FAILED_KIND, f"Image ENTRYPOINT failed in the verifier sandbox: {exc}")
        except Exception as exc:
            LOGGER.exception(f"Separate verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        finally:
            await _cancel_entrypoint(verifier_entrypoint)
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

    async def _run_shared_verifier(self, session: HarborSession, trajectory: bytes) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote.

        ``/logs/agent`` is left as the agent wrote it, except for ``trajectory.json``, which the server
        writes there so verifiers that read the transcript find it.
        """
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
            if session.entrypoint is not None:
                died = await self._entrypoint_died_during_rollout(sandbox, task, session.entrypoint, logs_dir)
                if died is not None:
                    await self._collect_entrypoint_log(sandbox, task, logs_dir / ENTRYPOINT_LOG_NAME)
                    return self._masked(ENTRYPOINT_FAILED_KIND, died) | {"verifier_logs_dir": str(logs_dir.resolve())}
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
            await self._upload_trajectory(sandbox, trajectory, logs_dir)
            # One plain exec for the whole run. The provider keeps it alive for as long as
            # `timeout_s` says (OpenSandbox polls a background command, Docker streams), and
            # `timeout` inside the container is the cap that stops the tests themselves.
            # Commands must not be left running after the exec returns: OpenSandbox reaps
            # them when the command completes, so a background launch never finishes.
            budget = int(settings.timeout_sec)
            result = await sandbox.exec(
                f"timeout --signal=KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1",
                cwd=session.workdir,
                env=self._verifier_env(settings),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            # `timeout` exits 137 when it had to kill test.sh; a provider that gave up first reports "timeout".
            timed_out = result.error_type == "timeout" or result.return_code == 137
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            rewards, problem = await self._collect_reward(
                sandbox,
                task,
                logs_dir,
                timed_out=timed_out,
                entrypoint_log=ENTRYPOINT_LOG_NAME if session.entrypoint is not None else None,
            )
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
            "verifier_network": settings.network_mode or "public",
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
