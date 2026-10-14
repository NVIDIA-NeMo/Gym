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
  each (see ``nemo_gym.tasks.harbor.dockerfile``), create the working directory. A ``RUN``
  line that fails is a terminal seed failure (the task's Dockerfile does not apply); a
  provider that cannot run one is retryable;
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
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Literal
from uuid import uuid4

import yaml
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from environment_servers.harbor.sandbox_io import (
    SandboxProbeError,
    SandboxTransferError,
    _exec_as_root,
    download_dir,
    download_path,
    remote_kind,
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
from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose, SandboxExecResult, SandboxSpec, clamp_resource_requests
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.compose_config import adapt_non_root_sidecars, opensandbox_shm_labels, resolve_compose
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.providers.base import ORPHAN_REAP_TIMEOUT_S
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure, SingleAgentTurnRequest, SingleAgentTurnResult
from nemo_gym.tasks.harbor import DIGEST_KEY, HarborTask, load_task
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.dockerfile import DEFAULT_SHELL, OverlayRun
from nemo_gym.tasks.harbor.models import HarborArtifact, HarborEnvironment
from nemo_gym.tasks.harbor.task import HarborTaskError


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"
VERIFIER_LOGS_DIR = "/logs/verifier"
AGENT_LOGS_DIR = "/logs/agent"
ARTIFACTS_DIR = "/logs/artifacts"
COMPOSE_IMAGES_FILE = "compose-images.json"
TASK_CONTEXT_DIR = "/tmp/.nemo-gym"
TASK_CONTEXT_FILE = f"{TASK_CONTEXT_DIR}/task.json"
VERIFIER_STDOUT = f"{VERIFIER_LOGS_DIR}/test-stdout.txt"
# Written by the verifier wrapper once `timeout` returns: the exit code and the start and end times
# (epoch seconds) of the run, so the server can tell the timer firing from a test.sh that exits 137.
EXIT_RECORD_NAME = ".ng-exit"
# Runs before any redirection: an image without `timeout` (or `bash`) exits 127 here, leaving neither a
# stdout file nor an exit record, which is how the caller tells it from a script that exits 127 itself.
TIMEOUT_WRAPPER_GUARD = "command -v timeout >/dev/null 2>&1 && command -v bash >/dev/null 2>&1 || exit 127"

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


# The separate verifier's sandbox could not take the task's Dockerfile overlay after the rollout ran.
OVERLAY_FAILED_KIND = "harbor:overlay_failed"

# How much of a failed overlay RUN line's output the seed error carries.
OVERLAY_OUTPUT_TAIL = 1500


class HarborOverlayError(RuntimeError):
    """A ``RUN`` line of the task's overlay-mode Dockerfile failed; the task cannot be set up."""


class HarborOverlayUnavailable(RuntimeError):
    """The sandbox provider could not run an overlay step at all (not the step failing): retry elsewhere."""


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
    # Provider block for tasks that declare GPUs (agent or verifier); None runs them on `sandbox_provider`.
    gpu_sandbox_provider: str | None = None
    # Pass a task's `gpu_types` to the provider. Off: deployments without that filter reject it.
    request_gpu_type: bool = False
    # Extra seconds the provider exec waits beyond `[verifier].timeout_sec` for the in-container `timeout` to
    # kill test.sh and return; it does not extend the tests' own deadline, which is the task's `timeout_sec`.
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
    provider_ref: str
    compose: AsyncSandboxCompose | None = None
    # Separate verification stops the agent's sandbox before test.sh runs; cleanup must not stop it twice.
    agent_sandbox_stopped: bool = False
    # In a Compose group the agent's own container is stopped before the sidecars' hooks run (see ``stop_main``).
    main_stopped: bool = False

    def service(self, name: str | None) -> AsyncSandbox:
        """The sandbox of a Compose service; ``None`` or ``"main"`` is the agent's own."""
        if name in (None, "main"):
            return self.sandbox
        if self.compose is None or name not in self.compose.services:
            raise KeyError(f"No Compose service {name!r} in this task")
        return self.compose.services[name]

    async def stop_main(self) -> None:
        """Stop the agent's own container, leaving a Compose group's sidecars running.

        Harbor collects in phases: the main container's hooks and artifacts, then its stop, then the
        sidecars' hooks and artifacts, so nothing the agent left running can change a sidecar's
        state while it is read. Without a group this is the whole agent side.
        """
        if self.compose is None:
            await self.stop_agent_side()
            return
        if not self.main_stopped:
            await self.sandbox.stop()
            self.main_stopped = True

    async def stop_agent_side(self) -> None:
        """Stop the agent's sandbox, and its sidecars when the task is a Compose group."""
        if self.agent_sandbox_stopped:
            return
        if self.compose is not None:
            # A main container stopped by ``stop_main`` is skipped: a sandbox stop is idempotent.
            await self.compose.stop()
        else:
            await self.sandbox.stop()
        # Only a stop that succeeded releases the episode's final cleanup from stopping again.
        self.agent_sandbox_stopped = True


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


def verifier_command(budget: int) -> str:
    """One in-container run of ``tests/test.sh`` under ``timeout``, with the exit record written after it."""
    return (
        f"{TIMEOUT_WRAPPER_GUARD}; start=$(date +%s); "
        f"timeout -s KILL {budget} bash {TESTS_DIR}/test.sh > {VERIFIER_STDOUT} 2>&1; rc=$?; "
        f'echo "$rc $start $(date +%s)" > {VERIFIER_LOGS_DIR}/{EXIT_RECORD_NAME}; exit $rc'
    )


def parse_exit_record(directory: Path) -> tuple[int, int | None] | None:
    """``(exit code, elapsed seconds)`` from the wrapper's record, or None when the wrapper never finished.

    The elapsed time is None when the record carries no usable timestamps (an image without ``date``).
    """
    path = directory / EXIT_RECORD_NAME
    if not path.is_file():
        return None
    parts = path.read_text(errors="replace").split()
    try:
        return_code = int(parts[0])
    except (IndexError, ValueError):
        return None
    try:
        return return_code, int(parts[2]) - int(parts[1])
    except (IndexError, ValueError):
        return return_code, None


def verifier_timed_out(result: SandboxExecResult, exit_record: tuple[int, int | None] | None, budget: int) -> bool:
    """Whether the timer stopped test.sh: the provider gave up first, or ``timeout`` killed it at the budget.

    GNU ``timeout -s KILL`` exits 137 when the timer fired, but so does a script whose own helper was
    killed; the timer is the cause only once the run has lasted the whole budget. An image without
    ``date`` leaves no elapsed time, and 137 then means the timer.
    """
    if result.error_type == "timeout":
        return True
    if exit_record is None or exit_record[0] != 137:
        return False
    elapsed = exit_record[1]
    return elapsed is None or elapsed >= budget


def select_reward(rewards: dict[str, float]) -> float | None:
    """Harbor's rule: the ``reward`` key, else the only key. Several keys and none named ``reward`` is ambiguous."""
    if "reward" in rewards:
        return rewards["reward"]
    if len(rewards) == 1:
        return next(iter(rewards.values()))
    return None


def _is_root(user: str | None) -> bool:
    """Whether ``user`` runs with uid 0, whatever group it names."""
    return user is None or str(user).split(":")[0] in ("root", "0")


def _is_plain_root(user: str | None) -> bool:
    """Whether ``user`` is root with no group of its own (``root:1234`` is not: the group must be kept)."""
    return user is None or str(user) in ("root", "0")


async def _exec_as_root_user(
    sandbox: AsyncSandbox, command: str, *, configured_user: str | None, cwd: str = "/", timeout_s: float = 60
):
    """Run ``command`` as root.

    ``configured_user`` is the user the sandbox runs commands as by default: the OCI ``User``
    of the image the sandbox started from (``HarborTask.sandbox_user``), which is not the
    Dockerfile's final ``USER`` or ``[agent].user``. Only a non-root default needs the
    ``user="root"`` override; root images keep the plain exec path every provider supports.
    """
    user = None if _is_root(configured_user) else "root"
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

    The line itself runs as the build would run it: a shell-form line through the image's
    ``Shell`` when that is not Docker's default (``bash -o pipefail -c`` makes ``false | true``
    fail, as it did in the build), an exec-form line as its argument vector with no shell at all.
    With the default shell the line is left as written for the provider's shell to run.
    """
    parts = [f"unset {shlex.quote(key)}" for key in sorted(set(sandbox_env or ()) - set(step.env))]
    parts += [f"export {key}={shlex.quote(value)}" for key, value in step.env.items()]
    if step.workdir:
        parts.append(f"cd {shlex.quote(step.workdir)} || exit 1")
    if step.shell is None:
        parts.append(f"exec {step.command}")
    elif tuple(step.shell) != DEFAULT_SHELL:
        parts.append(f"exec {shlex.join(step.shell)} {shlex.quote(step.command)}")
    else:
        parts.append(step.command)
    return "; ".join(parts)


def overlay_workdir_command(workdir: str, user: str | None) -> str:
    """Create a ``WORKDIR`` as root the way BuildKit does: missing directories are made and owned by the user.

    An existing directory (one the image ships) is left as it is. ``root:1234`` is not plain root: the
    directory is handed to that user and group, as a build would.
    """
    quoted = shlex.quote(workdir)
    if _is_plain_root(user):
        return f"mkdir -p {quoted}"
    return f"test -d {quoted} || (mkdir -p {quoted} && chown {shlex.quote(user)} {quoted})"


def _is_numeric_user(user: str | None) -> bool:
    return user is not None and str(user).split(":")[0].isdigit() and not _is_root(user)


def _exec_user(user: str) -> str | int:
    """The ``user`` of a ``RUN`` line's exec: a bare uid as the number (a uid the image has no name for), the rest as written.

    A numeric string reaches the ``su``-backed providers as an account name, which no uid is;
    the number takes their uid path. A ``name:group`` or ``uid:gid`` pair stays a string: only
    Docker's ``--user`` takes the pair, and the other providers fail the line with their own message.
    """
    return int(user) if user.isdigit() else user


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


def _verifier_gets_overlay(task: HarborTask) -> bool:
    """A separate verifier that runs in the task's own image needs the same Dockerfile overlay as the agent."""
    return bool(task.overlay) and not task.config.is_shared_verifier and task.config.verifier.environment is None


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


def _needs_gpu(task: HarborTask) -> bool:
    verifier_environment = task.config.verifier.environment
    return bool(task.config.environment.gpus or (verifier_environment is not None and verifier_environment.gpus))


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
        # Resolved Compose documents and the compose.yaml each is written to, by task snapshot and digest.
        self._compose_documents: dict[tuple[Path, str], asyncio.Task[tuple[dict[str, Any], Path]]] = {}
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
            # Dataset-level files live beside the task folders and the loader reads them from the
            # task's parent, so the snapshot gets copies at the same relative place: the dataset's
            # own settings (dataset.toml) and the recorded image configurations.
            for name in (DATASET_FILE, COMPOSE_IMAGES_FILE):
                sidecar = folder.parent / name
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
            resolve_provider_metadata(self._provider_ref(task), global_config_dict)
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
        resources = _sandbox_resources(environment, request_gpu_type=self.config.request_gpu_type) | (
            self.config.sandbox_resources_override or {}
        )
        return SandboxSpec(
            image=image or task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=resolve_env_templates(task_env),
            metadata=metadata,
            resources=resources,
            provider_options=self._provider_options_within(resources, task, role),
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

    def _provider_ref(self, task: HarborTask) -> str:
        """Which top-level sandbox block runs this task: the GPU one when the task declares GPUs."""
        if self.config.gpu_sandbox_provider and _needs_gpu(task):
            return self.config.gpu_sandbox_provider
        return self.config.sandbox_provider

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, episode_key: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        self._sandboxes[episode_key] = sandbox
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

    async def _create_sandbox_within_deadline(self, task: HarborTask, episode_key: str) -> AsyncSandbox:
        """The agent's ``_create_sandbox`` under the ready deadline; the timeout is a seed failure, retried.

        An overlay task starts in the base image's own working directory: a provider creates the spec's
        directory as root before anything runs, and the overlay's ``WORKDIR`` steps (then ``_prepare_workdir``)
        would find it existing and leave it root-owned. Created by the overlay instead, it belongs to the
        user in effect at that ``WORKDIR``, as in the build. The ``main`` service of a Compose group starts
        the same way (:meth:`_create_compose`).
        """
        workdir = None if task.overlay else task.workdir
        return await self._within_ready_deadline(self._create_sandbox(task, workdir, episode_key=episode_key))

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

    def _compose_document(self, task: HarborTask) -> dict[str, Any]:
        """The published overlay resolved against recorded image metadata, ready for the Compose adapter."""
        compose_file = next(
            task.path / "environment" / name
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
            if (task.path / "environment" / name).is_file()
        )
        image_configs = self._compose_image_configs(task)
        document = resolve_compose(yaml.safe_load(compose_file.read_text()), task.image, image_configs)
        adapt_non_root_sidecars(document, image_configs)
        provider = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
        if "opensandbox" in provider:
            opensandbox_shm_labels(document)
        sidecars = {a.service for a in task.config.artifacts} | {h.service for h in task.config.verifier.collect}
        missing = sidecars - {None, "main"} - set(document["services"])
        if missing:
            raise RuntimeError(
                f"Artifacts or collect hooks name Compose services that do not exist: {sorted(missing)}"
            )
        return document

    def _write_compose_document(self, task: HarborTask) -> tuple[dict[str, Any], Path]:
        """Resolve the task's Compose document and write it where the adapter reads it (blocking file work)."""
        document = self._compose_document(task)
        # Episode IDs are opaque caller input, never host paths: the document gets a fresh folder.
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="compose-", dir=self._artifacts_dir)) / "compose.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        return document, path

    async def _compose_document_for(self, task: HarborTask) -> tuple[dict[str, Any], Path]:
        """The resolved document and its compose.yaml, parsed and written once per task digest.

        The document is the same for every episode of a task (the published overlay, the recorded image
        metadata and the provider block do not change under a running server), so the YAML and JSON
        reads and the write happen once, in a thread; concurrent first episodes share the one resolution,
        and a failed one is forgotten so the next episode tries again.
        """
        key = (task.path, task.digest)
        pending = self._compose_documents.get(key)
        if pending is None:
            pending = asyncio.create_task(asyncio.to_thread(self._write_compose_document, task))
            self._compose_documents[key] = pending

            def forget_failed(done: asyncio.Task[tuple[dict[str, Any], Path]]) -> None:
                if done.cancelled() or done.exception() is not None:
                    self._compose_documents.pop(key, None)

            pending.add_done_callback(forget_failed)
        return pending.result() if pending.done() else await asyncio.shield(pending)

    async def _create_compose(self, task: HarborTask, episode_key: str) -> AsyncSandboxCompose:
        """Start the task's Compose group; the ``main`` service is the agent's sandbox."""
        document, path = await self._compose_document_for(task)
        # An overlay task's main service starts without the task workdir, as its single sandbox would (see
        # `_create_sandbox_within_deadline`); the sidecars derive from it and start in their images' own.
        main_spec = self._sandbox_spec(task, None if task.overlay else task.workdir)
        sidecar_spec = replace(
            main_spec, resources={}, env={}, provider_options=dict(self.config.sandbox_provider_options)
        )
        compose = AsyncSandboxCompose(
            resolve_provider_config(self._provider_ref(task), get_global_config_dict()),
            path,
            service_specs={name: main_spec if name == "main" else sidecar_spec for name in document["services"]},
            timeout_s=self.config.sandbox_ready_timeout_s,
        )
        # Ownership starts before start(): a group that fails half-way is still stopped whole.
        self._sandboxes[episode_key] = compose
        await compose.start()
        return compose

    async def _apply_overlay(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        """Run the Dockerfile ``RUN`` lines the loader recorded, in order, before anything else uses the sandbox.

        Each line runs with the ``ENV``, ``WORKDIR`` and ``USER`` the loader resolved for it, the
        way ``docker build`` would: every ``WORKDIR`` instruction creates its directory as root, in
        Dockerfile order (owned by the user and group in effect at it when that is not plain root, left
        alone when the image ships it), a numeric ``USER`` is mapped to its name in the image (a uid
        without a passwd entry is passed as the number: Docker's ``--user`` and OpenSandbox's ``uid`` take
        it, the ``su``-backed providers refuse it with their own message), then the line runs. A ``USER``
        with uid 0 and a group of its own (``root:1234``, ``0:0``) keeps that group on its ``WORKDIR`` but
        runs its lines as plain root: the exec API has no user:group form, and the ``su``-backed providers
        (OpenSandbox, enroot, apptainer) reject ``root:1234`` as an account name. The whole overlay
        shares ``[environment].build_timeout_sec``, as a build would. A line that exits non-zero or
        runs out of budget raises :class:`HarborOverlayError` with the command and the tail of its
        output: the task's Dockerfile does not apply, so retrying would not help. A provider that
        cannot run a step at all raises :class:`HarborOverlayUnavailable` instead, so the seed is retried.
        """
        if not task.overlay:
            return
        budget = task.config.environment.build_timeout_sec
        loop = asyncio.get_running_loop()
        deadline = loop.time() + budget
        total = len(task.overlay)
        prepared_workdirs: set[str] = {"/"}  # the root always exists; every other WORKDIR is made once
        user_names: dict[str, str] = {}

        async def as_root(command: str, label: str, what: str) -> SandboxExecResult:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HarborOverlayError(f"{label} not run: [environment].build_timeout_sec={budget} is used up")
            result = await _exec_as_root_user(sandbox, command, configured_user=task.sandbox_user, timeout_s=remaining)
            _check_overlay_result(result, label, what, budget)
            return result

        async def effective_user(user: str | None, label: str) -> str | None:
            """A numeric ``USER`` by the name the image gives it; a uid without a passwd entry stays numeric."""
            if not _is_numeric_user(user):
                return user
            uid, _, group = str(user).partition(":")
            if uid not in user_names:
                lookup = await as_root(f"getent passwd {shlex.quote(uid)} | cut -d: -f1", label, f"USER {uid} lookup")
                name = (lookup.stdout or "").strip().splitlines()
                user_names[uid] = name[-1] if name and name[-1] else uid
            return user_names[uid] + (f":{group}" if group else "")

        async def make_workdir(path: str, user: str | None, label: str) -> None:
            if path in prepared_workdirs:
                return
            await as_root(overlay_workdir_command(path, await effective_user(user, label)), label, f"WORKDIR {path}")
            prepared_workdirs.add(path)

        workdirs = list(task.overlay_workdirs)
        for index, step in enumerate(task.overlay, 1):
            label = f"RUN line {index}/{total} {step.command!r}"
            # The WORKDIR instructions before this line, in their order, each owned by the user of its point.
            while workdirs and workdirs[0].before < index:
                instruction = workdirs.pop(0)
                await make_workdir(instruction.path, instruction.user, label)
            user = await effective_user(step.user, label)
            await make_workdir(step.workdir, user, label)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HarborOverlayError(f"{label} not run: [environment].build_timeout_sec={budget} is used up")
            command = overlay_shell_command(step, task.env)
            if _is_root(user):
                # Root with a group of its own included: the group stays on the WORKDIR, the line runs as root.
                result = await _exec_as_root_user(
                    sandbox, command, configured_user=task.sandbox_user, timeout_s=remaining
                )
            else:
                result = await sandbox.exec(command, cwd="/", timeout_s=remaining, user=_exec_user(user))
            _check_overlay_result(result, label, None, budget)
        # WORKDIR instructions after the last RUN line exist in the built image too.
        for instruction in workdirs:
            await make_workdir(instruction.path, instruction.user, f"after RUN line {total}/{total}")

    async def _wait_healthy(self, sandbox: AsyncSandbox, task: HarborTask) -> None:
        """Poll the task's ``[environment.healthcheck]`` until it passes or its retries run out."""
        check = task.config.environment.healthcheck
        if check is None:
            return
        loop = asyncio.get_running_loop()
        grace_until = loop.time() + check.start_period_sec
        failures = 0
        while True:
            # Decided before the attempt, as Harbor does: a check that starts inside the start period and
            # fails after it has ended is still a start-period attempt, not a counted failure.
            in_grace = loop.time() < grace_until
            result = await sandbox.exec(check.command, timeout_s=check.timeout_sec + 5)
            if result.return_code == 0:
                return
            if not in_grace:
                failures += 1
                if failures >= check.retries:
                    raise RuntimeError(f"Healthcheck failed {check.retries} times: {check.command!r}")
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

        Without a ``[verifier.environment]`` the verifier runs in the task's own image, so it is
        created like the agent's sandbox: the Dockerfile's resolved ``ENV`` under ``task.toml``'s.

        Its lifetime is the episode-sized default the agent's sandbox gets, not ``timeout_sec`` plus
        slack: the sandbox also has to come up, run the overlay when it gets one, receive ``/tests``
        and the artifacts, run ``test.sh`` with its grace and hand back the reward, and a provider's
        renew interval must fit under it. The default covers the overlay for the agent, so it does
        for the verifier too.
        """
        environment = task.config.verifier.environment
        spec = self._sandbox_spec(
            task,
            # The task image keeps the task's working directory, as the agent had it; an explicit
            # verifier image starts in its `[verifier.environment].workdir`, else its own WORKDIR.
            task.workdir if environment is None else environment.workdir,
            environment=environment,
            image=_verifier_image(task),
            role="verifier",
        )
        provider_config = resolve_provider_config(self._provider_ref(task), get_global_config_dict())
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
        if workdir != "/":
            # Only a directory the task or image named below the root is created and, with a user
            # override, handed over; an image whose WORKDIR is `/` must keep root's ownership of it.
            commands = [f"mkdir -p {shlex.quote(workdir)}"]
            if not _is_root(task.user):
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
        compose: AsyncSandboxCompose | None = None
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
            await self._apply_overlay(sandbox, task)
            session.workdir = await self._prepare_workdir(sandbox, task, task.workdir)
            await self._wait_healthy(sandbox, task)
            await self._write_task_context(sandbox, task)
        except HarborOverlayError as exc:
            # The task's own Dockerfile failed to apply: a task error, not a sandbox one.
            try:
                await self._stop_sandbox(episode_key)
            except Exception:
                LOGGER.exception("Failed to stop Harbor sandbox after a failed overlay")
            LOGGER.error("Dockerfile overlay failed for %s: %s", task.task_id, exc)
            raise HTTPException(422, f"Dockerfile overlay failed for {task.task_id!r}: {exc}") from exc
        except BaseException as exc:
            if not isinstance(exc, Exception):
                # Cancellation (the episode deadline fired): the registered `close_sandbox` cleanup stops
                # the owned sandbox under `cleanup_timeout_seconds`. Stopping it here would run unbounded,
                # since the deadline cancels only once, and hold the episode key for as long as the stop.
                raise
            try:
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
        self, sandbox: AsyncSandbox, task: HarborTask, logs_dir: Path
    ) -> tuple[tuple[int, int | None] | None, dict[str, float] | None, str | None]:
        """Download the reward files and the wrapper's exit record, then whatever else the verifier wrote.

        The reward and the exit record are the required transfer, whatever the exit code: a grader may
        have written its reward before a late exit, and a timed-out run's partial logs are still worth
        keeping. Optional diagnostics cannot invalidate a measured score. Returns the exit record (None
        when the wrapper never reached its end), the rewards and the reward problem.
        """
        await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt", EXIT_RECORD_NAME))
        exit_record = parse_exit_record(logs_dir)
        rewards, problem = parse_reward_file(logs_dir)
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        return exit_record, rewards, problem

    async def _run_separate_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Harbor's separate mode: collect the agent's artifacts, stop its sandbox, verify in a fresh one.

        The verifier sandbox comes from ``[verifier.environment]`` (or the task image when the
        mode is ``separate`` without one). ``/logs/artifacts`` and every ``artifacts`` entry are
        copied through the host into the same paths, then ``tests/`` is uploaded and ``test.sh``
        runs there. Collection is phased as Harbor's is: the main container's hooks and artifacts,
        its stop, then the sidecars' hooks and artifacts, then the rest of the group is stopped.
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
            main_hooks = [hook for hook in settings.collect if _is_main_service(hook.service)]
            sidecar_hooks = [hook for hook in settings.collect if not _is_main_service(hook.service)]
            # Each artifact stages under its own numbered root (see ``_download_artifacts``), across both phases.
            artifacts = list(enumerate(_collected_artifacts(task)))
            main_artifacts = [(index, artifact) for index, artifact in artifacts if _is_main_service(artifact.service)]
            sidecar_artifacts = [(i, artifact) for i, artifact in artifacts if not _is_main_service(artifact.service)]
            await self._run_collect_hooks(session, main_hooks)
            restored = await self._download_artifacts(session, main_artifacts, artifacts_dir)
            # Harbor's phased collection: the agent's own container is stopped before the sidecars' hooks
            # run and their artifacts are read, so no agent process can change a sidecar's state meanwhile.
            await session.stop_main()
            await self._run_collect_hooks(session, sidecar_hooks)
            restored += await self._download_artifacts(session, sidecar_artifacts, artifacts_dir)
            # Harbor stops the agent's containers before the verifier starts; nothing may keep running.
            # The episode key stays reserved until the episode's final cleanup, which releases it without
            # stopping the agent side again.
            await session.stop_agent_side()

            verifier = await self._create_verifier_sandbox(task, episode_key=session.session_id)
            if _verifier_gets_overlay(task):
                # The verifier runs in the task's own image, so it needs the same overlay as the agent did.
                await self._apply_overlay(verifier, task)
            prepare = await _exec_as_root(
                verifier,
                f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {ARTIFACTS_DIR} && chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR}",
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            # A prebuilt verifier image usually ships /tests already, with the modes its tests expect.
            # Only upload the package's tests when the image has none, as the old server did.
            if await remote_kind(verifier, f"{TESTS_DIR}/test.sh") is None:
                await upload_dir(verifier, task.path / "tests", TESTS_DIR)
            for host, source in restored:
                await upload_path(verifier, host, source)
            # A sub-second task budget still gets one whole second: `timeout 0` would mean no limit.
            budget = max(1, math.ceil(settings.timeout_sec))
            result = await verifier.exec(
                f"chmod +x {TESTS_DIR}/test.sh; {verifier_command(budget)}",
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            exit_record, rewards, problem = await self._collect_reward(verifier, task, logs_dir)
            if exit_record is None and result.error_type != "timeout":
                # The wrapper never reached its end: the guard found no `timeout` or `bash` (exit 127
                # before any redirection), or the shell itself could not run. test.sh never started.
                return self._masked(
                    failure_kinds.VERIFIER_ERROR,
                    f"test.sh could not be started: timeout wrapper or bash missing (exit {result.return_code})",
                )
            timed_out = verifier_timed_out(result, exit_record, budget)
        except HTTPException:
            raise
        except HarborOverlayError as exc:
            # The rollout is complete; losing the verifier's sandbox to its overlay must not discard it.
            LOGGER.error(f"Dockerfile overlay failed in the verifier sandbox for {task.task_id}: {exc}")
            return self._masked(OVERLAY_FAILED_KIND, f"Dockerfile overlay failed in the verifier sandbox: {exc}")
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

    async def _run_collect_hooks(self, session: HarborSession, hooks: list[Any]) -> None:
        """Run ``[[verifier.collect]]`` hooks in their services, before anything is collected from them.

        Harbor runs them in both verifier modes. A hook that exits non-zero is the task's business and
        only warns; a hook the sandbox could not run at all is an outage and raises.
        """
        task = session.task
        for hook in hooks:
            try:
                sandbox = session.service(hook.service)
            except KeyError as exc:
                LOGGER.warning(f"{task.task_id}: collect hook skipped: {exc}")
                continue
            main = _is_main_service(hook.service)
            result = await sandbox.exec(
                hook.command if main else f"sh -c {shlex.quote(hook.command)}",
                cwd=session.workdir if main else None,
                timeout_s=hook.timeout_sec + 30,
                user=hook.user,
            )
            if result.error_type:
                raise SandboxProbeError(f"collect hook could not run: {result.error_type}: {hook.command!r}")
            if result.return_code:
                LOGGER.warning(f"{task.task_id}: collect hook exited {result.return_code}: {hook.command!r}")

    async def _download_artifacts(
        self, session: HarborSession, artifacts: list[tuple[int, HarborArtifact]], artifacts_dir: Path
    ) -> list[tuple[Path, str]]:
        """Pull the numbered ``artifacts`` out of their services; returns ``(host copy, sandbox path)`` for the ones found.

        Each artifact stages under its own numbered root. Links are kept (below), so one artifact's tree may
        hold a link to anywhere; with a shared root a later artifact's path could run through it.
        """
        task = session.task
        restored: list[tuple[Path, str]] = []
        for index, artifact in artifacts:
            try:
                sandbox = session.service(artifact.service)
            except KeyError as exc:
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} skipped: {exc}")
                continue
            host = artifacts_dir / f"{index:02d}" / artifact.host_path
            # Best effort per artifact, as Harbor collects them: what the agent left behind (an
            # unreadable tree, a link that escapes the folder, a file where a folder was expected)
            # cannot mask the sample. Symbolic links travel as links, because the restore target is
            # another sandbox, not the host. Anything else (a lost sandbox, a full disk, a dropped
            # connection) is Gym's outage and still masks.
            try:
                kind = await download_path(
                    sandbox, artifact.source, host, links="keep", exclude=tuple(artifact.exclude)
                )
            except (
                SandboxTransferError,
                tarfile.TarError,
                FileExistsError,
                IsADirectoryError,
                NotADirectoryError,
            ) as exc:
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} not collected: {exc}")
                continue
            if kind is not None and host.is_symlink():
                # The artifact itself is a link: nothing to restore without following it on this host.
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} is a symbolic link, not restored")
                continue
            if kind is not None:
                restored.append((host, artifact.source))
        return restored

    async def _run_shared_verifier(self, session: HarborSession) -> dict[str, Any]:
        """Run ``tests/test.sh`` in the agent's sandbox and read the reward it wrote.

        ``[[verifier.collect]]`` hooks run first, as in Harbor: what they produce under ``/logs/artifacts``
        is there for ``test.sh``, and for the run's logs.
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
            await self._run_collect_hooks(session, settings.collect)
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
            # A sub-second task budget still gets one whole second: `timeout 0` would mean no limit.
            budget = max(1, math.ceil(settings.timeout_sec))
            result = await sandbox.exec(
                verifier_command(budget),
                cwd=session.workdir,
                env=resolve_env_templates(settings.env),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user,
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            exit_record, rewards, problem = await self._collect_reward(sandbox, task, logs_dir)
            if exit_record is None and result.error_type != "timeout":
                # The wrapper never reached its end: the guard found no `timeout` or `bash` (exit 127
                # before any redirection), or the shell itself could not run. test.sh never started.
                return self._masked(
                    failure_kinds.VERIFIER_ERROR,
                    f"test.sh could not be started: timeout wrapper or bash missing (exit {result.return_code})",
                )
            timed_out = verifier_timed_out(result, exit_record, budget)
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
        if rewards is None and timed_out:
            # A reward written before the kill still counts; only a run that left none is the timeout.
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


def _is_main_service(name: str | None) -> bool:
    """``None`` or ``"main"`` name the agent's own container; anything else is a Compose sidecar."""
    return name in (None, "main")


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
