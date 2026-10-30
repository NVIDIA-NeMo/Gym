# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grade an episode: the test.sh wrapper protocol, the reward files and the verifier sandbox.

The verifier runs Harbor's tests/test.sh under a timeout wrapper that writes an exit record,
reads reward.json or reward.txt, applies the reward rule (``reward`` key, else the keys summed as
``reward_components``) and turns the exit into a measured or masked sample. Module level are the
in-sandbox paths the protocol uses, the failure kinds, the wrapper command, the reward and exit
record parsers, the timeout rule, and the choice of image and user the verifier runs as in shared
and separate mode. The mixin holds the shared and separate verification choreography unchanged,
the reward and log collection, the collect hooks and artifact download, and the scoring methods.
Creating the verifier sandbox stays with the other provisioning code in sandboxes.py.
"""

import asyncio
import errno
import json
import logging
import math
import posixpath
import shlex
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from environment_servers.harbor.entrypoint import (
    ENTRYPOINT_EXIT_NAME,
    ENTRYPOINT_FAILED_KIND,
    ENTRYPOINT_LOG_NAME,
    VERIFIER_ENTRYPOINT_LOG_NAME,
    EntrypointLaunch,
    HarborEntrypointError,
    _cancel_entrypoint,
)
from environment_servers.harbor.overlay import OVERLAY_FAILED_KIND, HarborOverlayError
from environment_servers.harbor.session import (
    AGENT_LOGS_DIR,
    ARTIFACTS_DIR,
    VERIFIER_LOGS_DIR,
    HarborSession,
    resolve_env_templates,
)
from nemo_gym import __version__
from nemo_gym.core import NeMoGymResponse, failure_kinds
from nemo_gym.data import (
    HarborArtifact,
    HarborTask,
    HarborTaskStep,
    HarborTurn,
    HarborVerifier,
    VerifierShim,
    is_ignored,
    write_shim,
)
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxExecResult,
    SandboxProbeError,
    SandboxTransferError,
    download_dir,
    download_path,
    exec_as_root,
    remote_kind,
    upload_archive,
    upload_dir,
    upload_path,
)
from nemo_gym.server_runtime import BaseEpisodeRequest
from nemo_gym.trajectory import encode_atif_trajectory, transcript_to_atif


LOGGER = logging.getLogger(__name__)

TESTS_DIR = "/tests"

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


def _uploads_task_tests(task: HarborTask) -> bool:
    """Whether the task's own ``tests/`` must reach ``/tests``: a ``test.sh`` task's, and a Python verifier's when the
    task's ``tests/`` carries the module (the shim's ``test.sh`` names ``/tests/verifier.py``, which only that upload
    provides). A dataset-level verifier travels in the shim alone: a task it grades has no ``tests/`` of its own.

    Decided from where the loaded task's verifier is, not from whether the folder is on disk now: a snapshot whose
    ``tests/`` disappeared under the episode then fails its upload and masks the sample, as a ``test.sh`` task's does,
    instead of running the shim against a missing module and scoring the missing reward as a zero.
    """
    return task.python_verifier is None or task.python_verifier.parent == task.path / "tests"


async def _make_test_script_executable(sandbox: AsyncSandbox) -> None:
    """``chmod +x`` on ``test.sh`` as root before it runs, as Harbor's verifier does.

    Harbor uploads ``tests/`` as ``docker cp`` leaves it (root-owned, the package's modes), chmods the test
    script alone because it is a ``.sh`` (``needs_chmod``), and runs it as the verifier user; it chowns
    nothing and widens nothing, so a package built under ``umask 077`` is readable by a root verifier and by
    no other user, there as here. The chmod's exit status is not read, as Harbor does not read it.
    """
    await exec_as_root(sandbox, f"chmod +x {TESTS_DIR}/test.sh", timeout_s=60)


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
    """The score a parsed reward file gives: the ``reward`` key when present, else the sum of every key.

    Gym's reading of ``reward.json``: a ``reward`` key is the score and the other keys are metrics; without
    one every key is a component of the reward (:func:`reward_components`), which is their sum. One key
    without ``reward`` is Harbor's own rule too. ``None`` when the components do not sum to a finite number.
    """
    if "reward" in rewards:
        return rewards["reward"]
    try:
        total = math.fsum(rewards.values())
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def reward_components(rewards: dict[str, float]) -> dict[str, float] | None:
    """The keys that sum to the reward, or ``None`` when the file names the reward itself.

    With a ``reward`` key the other keys are metrics, not parts of the score, so there are no components to
    name; without one every key is a component and :func:`select_reward` is their sum.
    """
    if "reward" in rewards:
        return None
    return dict(rewards)


def summing_components(reward: float, components: dict[str, float] | None) -> dict[str, float] | None:
    """``components`` when their values sum to ``reward`` (within floating-point tolerance), else ``None``.

    A results row may name ``reward_components`` only when they sum to its ``reward``, the contract a
    trainer asserts; this is the check at the point a verifier result becomes a result.
    """
    if components is None:
        return None
    if math.isclose(math.fsum(components.values()), reward, rel_tol=1e-9, abs_tol=1e-9):
        return components
    return None


def _verifier_user(task: HarborTask) -> str | int | None:
    """The user ``test.sh`` runs as, named explicitly because the sandbox's default may be ``[agent].user``:
    ``[verifier].user`` when set, else the sandbox's own default for a separate verifier in another image
    (that image's ``USER``, which :meth:`HarborEnvironmentServer._sandbox_spec` leaves in place, never the task
    image's), else the task image's own ``USER`` (what ``docker compose exec`` gives Harbor: a numeric one as a
    uid, and a ``name:group`` or ``uid:gid`` with its group, as Harbor's ``-u`` passes the pair; root's own group
    is plain root), else root when an agent override is installed and the image's ``USER`` is not recorded
    (Harbor's environment default), else the sandbox's own default. Only Docker's ``--user`` applies a group;
    :meth:`HarborVerificationMixin._refuse_verifier_group` refuses the pair at seed on the other providers."""
    settings = task.config.verifier
    if settings.user is not None:
        return settings.user
    if not task.config.is_shared_verifier and not _verifier_runs_task_image(task):
        # Another image: its own USER is already the sandbox's default (see `_sandbox_spec`), never the task image's.
        return None
    user = task.entrypoint_user
    if user is None:
        return "root" if task.user is not None else None
    name, _, group = str(user).partition(":")
    if group and not (name in ("root", "0") and group in ("root", "0")):
        # The pair as written, group included (`users._user_group`'s rule: root's own group is plain root): test.sh
        # keeps the files readable only by that group, as the sandbox's default and Harbor's `compose exec -u` do.
        return str(user)
    return int(name) if name.isdigit() else name


def _shared_verifier_task(task: HarborTask) -> HarborTask:
    """``task`` as a verification in the agent's own sandbox sees it: its ``[verifier]`` table in shared mode, with no
    separate environment or image. :func:`_verifier_user` then resolves the fallback user of a ``test.sh`` that runs
    in the agent's sandbox by the shared rules (the task image's own ``USER``, else root under an agent override, else
    the sandbox's default) even when the task's own mode is separate in another image and a step opted out into shared
    mode: that image's default (``None``, which the separate rule answers) is nobody's in the agent's sandbox, where
    ``None`` would run ``test.sh`` as the agent. An explicit ``[verifier].user`` keeps its precedence."""
    if task.config.is_shared_verifier:
        return task
    verifier = task.config.verifier.model_copy(update={"environment_mode": "shared", "environment": None})
    return replace(
        task,
        config=task.config.model_copy(update={"verifier": verifier}),
        verifier_image=None,
        verifier_image_config=None,
    )


def _verifier_runs_task_image(task: HarborTask, separate: bool | None = None) -> bool:
    """A separate verifier with no ``[verifier.environment]`` and no image built from ``tests/Dockerfile`` grades in
    the task's own image: Harbor's effective verifier environment is then a copy of ``[environment]``, and the loader
    only lets that shape through when the copy carries ``[environment].docker_image`` (Harbor's prebuilt verifier), so
    the task has no overlay (a declared image resolves none) and the verifier needs none. A shared-mode task the server
    verifies separately (``separate``, see ``separate_verifier_for_artifacts``) grades in the task image too, overlay
    and all (:func:`_verifier_gets_overlay`)."""
    return _separate_mode(task, separate) and task.config.verifier.environment is None and task.verifier_image is None


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
    its entrypoint, the user and working directory it starts with, its healthcheck and the shell that runs
    it; no overlay, and no spec user, as :meth:`HarborEnvironmentServer._sandbox_spec` creates the sandbox).
    A declared prebuilt verifier image starts the same way when ``compose-images.json`` records it, which it
    does when the image is the base of a task's Dockerfile, this task's own included: Harbor ``up --wait``s
    the declared image as it is, so its own entrypoint runs as its own ``User`` in its own ``WorkingDir`` and
    its ``HEALTHCHECK`` gates ``test.sh``, with none of the Dockerfile's ``RUN`` lines, ``USER`` or ``WORKDIR``
    (the task's resolved start describes the agent's overlaid image, not this one). Only the start is the
    record's; the sandbox is still created from ``[verifier.environment]``, see
    :meth:`HarborEnvironmentServer._create_verifier_sandbox`, and its user default and overlay follow
    :func:`_verifier_user` and :func:`_verifier_gets_overlay`. A declared image nothing is recorded for starts
    nothing, like a prebuilt ``docker_image``. The result keeps the task's path, so ``needs_compose`` still
    describes the agent's environment: the verifier's sandbox is a plain one whatever that says, and its
    callers start it as one.
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
        image_shell=config.shell,
    )


def _verifier_key(episode_key: str) -> str:
    """The ownership key of an episode's separate-verifier sandbox."""
    return f"{episode_key}/verifier"


def _is_main_service(name: str | None) -> bool:
    """``None`` or ``"main"`` name the agent's own container; anything else is a Compose sidecar."""
    return name in (None, "main")


# The record of one collection pass, written beside the staged artifacts as Harbor's artifact handler writes its own.
ARTIFACT_MANIFEST_NAME = "manifest.json"


def _collected_artifacts(task: HarborTask, step: HarborTaskStep | None = None) -> list[tuple[HarborArtifact, str]]:
    """The artifacts one collection pass reads, each with where it was declared (``convention``, ``task`` or
    ``step``): ``/logs/artifacts`` first, then the task's own ``artifacts`` entries, then the step's for a turn of a
    multi-step task. Harbor's ``with_convention_entry([*task artifacts, *step artifacts])``: the implicit
    ``/logs/artifacts`` entry is prepended unless the task or the step declares that path for ``main`` itself."""
    entries = [(artifact, "task") for artifact in task.config.artifacts]
    if step is not None:
        entries += [(artifact, "step") for artifact in step.artifacts]
    if not any(a.source.rstrip("/") == ARTIFACTS_DIR and a.service in (None, "main") for a, _ in entries):
        entries.insert(0, (HarborArtifact(source=ARTIFACTS_DIR), "convention"))
    return entries


def _numbered_artifacts(task: HarborTask, step: HarborTaskStep | None = None) -> list[tuple[int, HarborArtifact, str]]:
    """:func:`_collected_artifacts` with each entry's staging number (its ``NN/`` root under ``artifacts/``)."""
    return [(index, artifact, origin) for index, (artifact, origin) in enumerate(_collected_artifacts(task, step))]


def _stdout_tail(logs_dir: Path, limit: int = 600) -> str:
    path = logs_dir / Path(VERIFIER_STDOUT).name
    if not path.is_file():
        return "(no test-stdout.txt)"
    return path.read_text(errors="replace")[-limit:]


def _verifier_gets_overlay(task: HarborTask, separate: bool | None = None) -> bool:
    """A separate verifier that runs in the task's own image needs the same Dockerfile overlay as the agent. Only a
    shared-mode task forced into separate verification can have one: the loader admits a declared separate verifier in
    the task image with a prebuilt ``docker_image`` only."""
    return bool(task.overlay) and _verifier_runs_task_image(task, separate)


def _separate_mode(task: HarborTask, separate: bool | None) -> bool:
    """The verifier mode in effect: the task's own unless the server forced separate verification."""
    return (not task.config.is_shared_verifier) if separate is None else separate


# The agent transcript in ATIF v1.7, written under AGENT_LOGS_DIR before every verify; Harbor verifiers may read it.
TRAJECTORY_FILE_NAME = "trajectory.json"

# Top-level names in a verification's host logs folder that this server writes itself, beside what it downloads from
# /logs/verifier: the host copy of the transcript, the artifacts staging tree with its manifest, and the entrypoint
# records. The optional download of /logs/verifier (a folder the agent could write in shared mode) leaves these names
# out (`download_dir(protect=...)`), so an agent-planted trajectory.json or artifacts/manifest.json cannot replace
# Gym's record of the turn.
SERVER_OWNED_LOG_NAMES = (
    TRAJECTORY_FILE_NAME,
    "artifacts",
    ENTRYPOINT_LOG_NAME,
    ENTRYPOINT_EXIT_NAME,
    VERIFIER_ENTRYPOINT_LOG_NAME,
)


def _restores_under(source: str, path: str) -> bool:
    """Whether restoring an artifact collected from ``source`` writes ``path``: it is that path or an ancestor."""
    # One leading slash on both: PurePosixPath keeps "//x" apart from "/x", as POSIX allows.
    return PurePosixPath("/" + path.lstrip("/")).is_relative_to(PurePosixPath("/" + source.lstrip("/")))


# Extra seconds the provider exec of a `[[verifier.collect]]` hook waits beyond the hook's own `timeout_sec`;
# counted in the episode budget, so the task deadline cannot cut off a hook within its declared limit.
COLLECT_HOOK_GRACE_S = 30


@dataclass(frozen=True)
class VerifierSettings:
    """The verifier settings one ``test.sh`` run uses: the task's, or a step's over the task's.

    ``collect`` is the task's ``[[verifier.collect]]`` hooks, with a step's own ``[[steps.verifier.collect]]`` hooks
    after them for a turn, as Harbor runs them for every step's collection pass.
    """

    timeout_sec: float
    env: dict[str, str]
    user: str | int | None
    collect: list[Any] = field(default_factory=list)
    # The task's `[verifier].network_mode` (a step's over it), reported as `verifier_network`.
    network_mode: str | None = None


class HarborVerificationMixin:
    """Grade an episode: the test.sh wrapper protocol, the reward files and the verifier sandbox.

    The verifier runs Harbor's tests/test.sh under a timeout wrapper that writes an exit record,
    reads reward.json or reward.txt, applies the reward rule (``reward`` key, else the keys summed as
    ``reward_components``) and turns the exit into a measured or masked sample. Module level are the
    in-sandbox paths the protocol uses, the failure kinds, the wrapper command, the reward and exit
    record parsers, the timeout rule, and the choice of image and user the verifier runs as in shared
    and separate mode. The mixin holds the shared and separate verification choreography unchanged,
    the reward and log collection, the collect hooks and artifact download, and the scoring methods.
    Creating the verifier sandbox stays with the other provisioning code in sandboxes.py.

    """

    async def _run_verifier(
        self, session: HarborSession, trajectory: bytes, *, response: NeMoGymResponse | None = None
    ) -> dict[str, Any]:
        """Run ``tests/test.sh`` and read the reward it wrote, in the agent's sandbox or a separate one.

        ``trajectory`` is the exported ATIF transcript, written to ``/logs/agent/trajectory.json`` in the
        sandbox that runs ``test.sh`` before it starts. ``response`` is the agent's final response, handed to a
        Python verifier that declares it (:meth:`_verifier_shim`).
        """
        if self._separate_verifier(session.task):
            return await self._run_separate_verifier(session, trajectory, response=response) | {
                "verifier_mode": "separate"
            }
        return await self._run_shared_verifier(session, trajectory, response=response) | {"verifier_mode": "shared"}

    async def _upload_verifier_shim(
        self, sandbox: AsyncSandbox, session: HarborSession, response: NeMoGymResponse | None
    ) -> bool:
        """Upload the shim over ``/tests``, after the task's own ``tests/``, when a Python verifier grades the task.

        It holds the ``test.sh`` that runs ``verify()`` inside the sandbox with the workspace path, the runner,
        a vendored ``nemo_gym.verifiers``, the task's fields and the agent's final response; no verifier code
        runs on this host (:mod:`nemo_gym.data.tasks.harbor.python_verifier`). The shim is per episode (it embeds the
        response), so it is written to a folder that lives only for the upload. ``False`` for a ``test.sh`` task.
        """
        task = session.task
        if task.python_verifier is None:
            return False
        shim = VerifierShim(
            verifier=task.python_verifier,
            in_task_tests=task.python_verifier.parent == task.path / "tests",
            workspace=session.workdir,
            task_data=task.task_data,
            response=response.model_dump(mode="json") if response is not None else None,
        )
        with tempfile.TemporaryDirectory(prefix="verifier-shim-", dir=self._snapshots.name) as folder:
            await upload_dir(sandbox, write_shim(Path(folder), shim), TESTS_DIR, ignore=is_ignored)
        return True

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

    @staticmethod
    def _within(path: str, directory: str) -> bool:
        """Whether the sandbox path ``path`` is ``directory`` or lies inside it."""
        # One leading slash first: POSIX lets "//x" name something other than "/x", and normpath keeps the pair.
        path, directory = (posixpath.normpath("/" + p.lstrip("/")) for p in (path, directory))
        return path == directory or path.startswith(directory.rstrip("/") + "/")

    @classmethod
    def _protected_restore(cls, path: str) -> str | None:
        """The grader's directory a restore written at ``path`` would replace, or ``None``.

        ``/tests`` and everything in it is the trusted test script and its helpers, and a restore over it
        (``/``) replaces it as surely as one inside it. ``/logs/verifier`` and everything in it is where
        ``test.sh`` writes its reward. A restore over ``/logs/verifier`` (``/logs``) is not refused here: it
        brings the agent's own folders back, and :meth:`_restore_artifact` empties ``/logs/verifier`` again
        after it.
        """
        if cls._within(path, TESTS_DIR) or cls._within(TESTS_DIR, path):
            return TESTS_DIR
        if cls._within(path, VERIFIER_LOGS_DIR):
            return VERIFIER_LOGS_DIR
        return None

    async def _restore_artifact(
        self, verifier: AsyncSandbox, task: HarborTask, archive: Path, kind: str, source: str
    ) -> None:
        """Restore one collected artifact at its own path in the verifier sandbox, clear of the grader's files.

        Harbor empties ``/logs/verifier``, restores every artifact at its own path and only then runs ``test.sh``
        from the verifier image's own ``/tests``, so a valid ``artifacts = ["/logs"]`` brings back whatever the
        agent wrote under ``/logs/verifier``, a ``reward.json`` included, which the ``reward.txt`` that ``test.sh``
        then writes does not displace. Here an artifact that would be written at ``/tests``, inside or over it,
        or at or inside ``/logs/verifier`` (:meth:`_protected_restore`, the loader already refuses those declared
        paths) is collected on the host as a diagnostic but never restored, and one a link in the fresh sandbox
        would divert onto either directory (an earlier
        artifact's tree may hold such a link) is refused by :func:`upload_archive` once its target is resolved
        there; a refusal only warns, as a collection the agent's tree defeats does, and the sample is graded on
        the trusted files. A restore that covers ``/logs/verifier`` is followed by that directory being emptied
        again, so ``test.sh`` starts from a clean one. The transcript's place under ``/logs/agent`` is
        :meth:`_upload_trajectory`'s.
        """
        protected = self._protected_restore(source)
        if protected is not None:
            LOGGER.warning(
                "%s: artifact %r not restored: it would replace the grader's %s", task.task_id, source, protected
            )
            return
        if not await upload_archive(verifier, archive, kind, source, keep_out=(TESTS_DIR, VERIFIER_LOGS_DIR)):
            LOGGER.warning(
                "%s: artifact %r not restored: a link in the verifier sandbox diverts it onto the grader's files",
                task.task_id,
                source,
            )
            return
        if self._within(VERIFIER_LOGS_DIR, source):
            reset = await exec_as_root(
                verifier,
                f"rm -rf -- {VERIFIER_LOGS_DIR} && mkdir -p {VERIFIER_LOGS_DIR} && chmod 777 {VERIFIER_LOGS_DIR}",
                timeout_s=60,
            )
            if reset.return_code != 0:
                raise RuntimeError(
                    f"Could not empty {VERIFIER_LOGS_DIR} after restoring {source}: {reset.stderr or reset.stdout}"
                )

    async def _collect_reward(
        self,
        sandbox: AsyncSandbox,
        task: HarborTask,
        logs_dir: Path,
        *,
        entrypoint_log: str | None = None,
    ) -> tuple[tuple[int, int | None] | None, dict[str, float] | None, str | None]:
        """Download the reward files and the wrapper's exit record, then whatever else the verifier wrote.

        The reward and the exit record are the required transfer, whatever the exit code: a grader may
        have written its reward before a late exit, and a timed-out run's partial logs are still worth
        keeping. Optional diagnostics cannot invalidate a measured score. With ``entrypoint_log``, the
        sandbox's entrypoint output lands under that name too. Returns the exit record (None when the
        wrapper never reached its end), the rewards and the reward problem.
        """
        await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, names=("reward.json", "reward.txt", EXIT_RECORD_NAME))
        exit_record = parse_exit_record(logs_dir)
        rewards, problem = parse_reward_file(logs_dir)
        try:
            await download_dir(sandbox, VERIFIER_LOGS_DIR, logs_dir, skip_unsafe=True, protect=SERVER_OWNED_LOG_NAMES)
        except Exception:
            LOGGER.warning("Could not collect optional verifier logs for %s", task.task_id, exc_info=True)
        if entrypoint_log is not None:
            await self._collect_entrypoint_log(sandbox, task, logs_dir / entrypoint_log)
        return exit_record, rewards, problem

    async def _run_separate_verifier(
        self, session: HarborSession, trajectory: bytes, *, response: NeMoGymResponse | None = None
    ) -> dict[str, Any]:
        """Harbor's separate mode: collect the agent's artifacts, stop its sandbox, verify in a fresh one.

        The verifier sandbox comes from ``[verifier.environment]``, the image prepare built from
        ``tests/Dockerfile``, or the task's prebuilt image (``_verifier_image``; no overlay runs there, the loader
        admits the task-image shape only with a declared image). ``/logs/artifacts`` and every ``artifacts``
        entry are collected as the archives the agent's sandbox packs, staged through the host (the filtered
        host copy is the safety gate and a diagnostic) and restored from those archives into the same paths,
        with the modes the agent left (see :func:`upload_archive`), the verifier image's recorded entrypoint
        and ``HEALTHCHECK`` run as they do at seed (:func:`_verifier_start`), then ``tests/`` is uploaded
        unless the image owns ``/tests/test.sh``, and ``test.sh`` runs there. Collection is phased as
        Harbor's is: the main container's hooks and artifacts, its stop, then the sidecars' hooks and
        artifacts, then the rest of the group is stopped.
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
        try:
            main_hooks = [hook for hook in settings.collect if _is_main_service(hook.service)]
            sidecar_hooks = [hook for hook in settings.collect if not _is_main_service(hook.service)]
            # Each artifact stages under its own numbered root (see ``_download_artifacts``), across both phases.
            artifacts = _numbered_artifacts(task)
            main_artifacts = [entry for entry in artifacts if _is_main_service(entry[1].service)]
            sidecar_artifacts = [entry for entry in artifacts if not _is_main_service(entry[1].service)]
            manifest: list[dict[str, Any]] = []
            await self._run_collect_hooks(session, main_hooks)
            restored = await self._download_artifacts(session, main_artifacts, artifacts_dir, manifest)
            died = None
            if session.entrypoint is not None:
                await self._collect_entrypoint_log(session.sandbox, task, logs_dir / ENTRYPOINT_LOG_NAME)
                died = await self._entrypoint_died_during_rollout(session.sandbox, task, session.entrypoint, logs_dir)
            # Harbor's phased collection: the agent's own container is stopped before the sidecars' hooks
            # run and their artifacts are read, so no agent process can change a sidecar's state meanwhile.
            await session.stop_main()
            await self._run_collect_hooks(session, sidecar_hooks)
            restored += await self._download_artifacts(session, sidecar_artifacts, artifacts_dir, manifest)
            self._write_artifact_manifest(artifacts_dir, manifest)
            # Harbor stops the agent's containers before the verifier starts; nothing may keep running.
            # The episode key stays reserved until the episode's final cleanup, which releases it without
            # stopping the agent side again.
            await session.stop_agent_side()
            if died is not None:
                return self._masked(ENTRYPOINT_FAILED_KIND, died) | {"verifier_logs_dir": str(logs_dir.resolve())}
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception(f"Separate verification infrastructure failed for {task.task_id}")
            return self._masked(failure_kinds.PROVIDER_UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        return await self._verify_separately(
            session, task, settings, trajectory, logs_dir, restored, started=started, response=response
        )

    async def _verify_separately(
        self,
        session: HarborSession,
        task: HarborTask,
        settings: Any,
        trajectory: bytes,
        logs_dir: Path,
        restored: list[tuple[Path, str, str]],
        *,
        started: float,
        test_dirs: tuple[Path, ...] | None = None,
        response: NeMoGymResponse | None = None,
    ) -> dict[str, Any]:
        """Grade in a fresh sandbox what was collected from the agent's: the second half of Harbor's separate mode.

        ``task`` is the task whose verifier settings and image apply: the session's own for a single-step task, or
        a turn's view of it (:meth:`_turn_verifier_view`) for a step of a multi-step task that verifies separately;
        ``settings`` is its ``[verifier]`` table or the turn's :class:`VerifierSettings`. The verifier sandbox is
        created from that, receives the ``restored`` archives at their own paths, the tests (``test_dirs`` in order,
        a step's over the task's, when given; the task's ``tests/`` otherwise, unless the image ships its own) and
        the transcript, runs ``test.sh`` and is stopped, whatever happened. Nothing else reaches it from the agent's
        sandbox, which this method never touches: a multi-step task's agent sandbox keeps running across turns, so
        each turn's verifier sandbox lives only around that turn's verification, under the episode's verifier key.
        """
        verifier: AsyncSandbox | None = None
        verifier_entrypoint: EntrypointLaunch | None = None
        key = _verifier_key(session.session_id)
        if key in self._sandboxes:
            # The previous turn's verifier is still owned: its stop failed (the `finally` below). Starting this
            # turn's under the same key would drop that sandbox from the table and leak it until its `ttl_s`, so
            # the stop is retried once here; a verifier that still cannot be stopped keeps the key for the episode
            # cleanup (`close_sandbox`, then shutdown) and this turn is masked, as a verifier sandbox that could not
            # be created is: the rollout is complete and the provider's fault must not become a reward.
            try:
                await self._stop_sandbox(key)
            except Exception as exc:
                LOGGER.exception(f"Could not stop the previous turn's verifier sandbox for {task.task_id}")
                return self._masked(
                    failure_kinds.PROVIDER_UNAVAILABLE,
                    f"the previous turn's verifier sandbox could not be stopped: {type(exc).__name__}: {exc}",
                )
        try:
            verifier = await self._create_verifier_sandbox(task, episode_key=session.session_id)
            names: dict[str, str] = {}
            overlaid = _verifier_gets_overlay(task, True)
            if overlaid:
                # A shared-mode task verified separately runs in the task's own image, so it needs the same overlay
                # as the agent did (a declared separate verifier in the task image is a prebuilt image: no overlay).
                names = await self._apply_overlay(verifier, task)
            if _verifier_runs_task_image(task):
                # Its default is that image's own USER, never [agent].user, resolved by name once it exists.
                await self._set_default_user(verifier, task, names, role="verifier")
            if overlaid and task.workdir is not None:
                # The overlay made the Dockerfile's WORKDIR directories only. The task workdir ([environment].workdir
                # when task.toml sets one, else the Dockerfile's final WORKDIR) is where test.sh is sent below, so it
                # is made and handed to the task's user here as the seed's `_prepare_workdir` leaves it in the agent's
                # sandbox, which is how the shared verifier finds it; a Docker exec into a missing directory fails
                # before test.sh starts.
                await self._own_workdir(verifier, task.workdir, task.user, create=True)
            # The verifier's /logs folders exist before its entrypoint starts, as in the seed and as Harbor's
            # /logs/verifier bind mount does; /tests and the artifacts follow, as Harbor uploads them after start.
            prepare = await exec_as_root(
                verifier,
                f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR} {ARTIFACTS_DIR} "
                f"&& chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                timeout_s=60,
            )
            if prepare.return_code != 0:
                raise RuntimeError(f"Could not prepare verifier directories: {prepare.stderr or prepare.stdout}")
            start = _verifier_start(task, True)
            if start is not None:
                # Harbor only `up --wait`s the verifier's image (built from tests/Dockerfile, declared by
                # [verifier.environment] and recorded as a Dockerfile's base, or the task's own when its effective
                # verifier environment is a copy of [environment] with a docker_image): that image's
                # entrypoint and HEALTHCHECK run here as its user; the task's [environment.healthcheck] is not run,
                # as Harbor does not run it in the verifier. The verifier sandbox is a plain one whatever the agent's
                # environment was (a Compose group included), so nothing is left to an adapter here.
                image_user = await self._image_user_name(verifier, start, names)
                verifier_entrypoint = await self._start_entrypoint(verifier, start, image_user)
                await self._wait_healthy(verifier, start, verifier_entrypoint, user=image_user, task_check=False)
            # A prebuilt verifier image usually ships /tests already, with the modes its tests expect, and one
            # built from tests/Dockerfile owns /tests the way its file copied it (Harbor uploads nothing into it,
            # `skip_tests_upload`). Only upload the package's tests when the image has none, as the old server did.
            # A task graded by a Python verifier has no test.sh of its own: a test.sh the image ships is not its
            # grader, so its tests/ (the module and its helpers) is uploaded regardless and the shim's test.sh,
            # uploaded last, is the one that runs.
            image_has_tests = await remote_kind(verifier, f"{TESTS_DIR}/test.sh") is not None
            if not image_has_tests or task.python_verifier is not None:
                unseen = task.verifier_image and task.verifier_image not in self._verifier_images_without_tests
                if not image_has_tests and unseen:
                    self._verifier_images_without_tests.add(task.verifier_image)
                    LOGGER.warning(
                        "%s: verifier image %s has no %s/test.sh; uploading the task's tests/ into it (Harbor uploads "
                        "nothing into a separate verifier's image, so its tests/Dockerfile usually copies the folder)",
                        task.task_id,
                        task.verifier_image,
                        TESTS_DIR,
                    )
                if test_dirs is not None:
                    # A turn's layers: the task's tests/ then the step's, each over the last, absent ones skipped.
                    for tests in test_dirs:
                        if tests.is_dir():
                            await upload_dir(verifier, tests, TESTS_DIR, ignore=is_ignored)
                elif _uploads_task_tests(task):
                    await upload_dir(verifier, task.path / "tests", TESTS_DIR, ignore=is_ignored)
            await self._upload_verifier_shim(verifier, session, response)
            for archive, kind, source in restored:
                await self._restore_artifact(verifier, task, archive, kind, source)
                # The staged tree stays as the diagnostic; the archive served its purpose.
                archive.unlink(missing_ok=True)
            # An artifact that covers /logs/agent brought the agent's tree, planted entries included, into
            # this fresh sandbox: the transcript's destination is cleared again before it is written.
            await self._upload_trajectory(verifier, trajectory, logs_dir, restored=[source for *_, source in restored])
            if verifier_entrypoint is not None:
                # The last look before test.sh, in a fresh sandbox the agent never had: its exit status is trusted.
                await self._check_entrypoint(verifier, start, verifier_entrypoint)
            # A sub-second task budget still gets one whole second: `timeout 0` would mean no limit.
            budget = max(1, math.ceil(settings.timeout_sec))
            await _make_test_script_executable(verifier)
            # An overlaid verifier sandbox was created without the task workdir, as the agent's is (see
            # `_create_verifier_sandbox`), so its default directory is the base image's own: test.sh is sent to the
            # task workdir made above, as the shared verifier's exec is sent to the session's. Harbor's exec passes
            # `[environment].workdir` whenever task.toml sets one and lands in the built image's WORKDIR otherwise,
            # which is what `task.workdir` resolves. Every other verifier sandbox starts in the directory its spec
            # named, or its image's own.
            result = await verifier.exec(
                verifier_command(budget),
                cwd=task.workdir if overlaid else None,
                env=self._verifier_env(settings),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=_verifier_user(task),
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            exit_record, rewards, problem = await self._collect_reward(
                verifier,
                task,
                logs_dir,
                entrypoint_log=VERIFIER_ENTRYPOINT_LOG_NAME if verifier_entrypoint is not None else None,
            )
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
            if key in self._sandboxes:
                try:
                    await self._stop_sandbox(key)
                except Exception:
                    # Still owned: the next separate turn retries the stop before it reuses the key (above), and
                    # the episode cleanup and shutdown retry it after.
                    LOGGER.exception(f"Could not stop the verifier sandbox for {task.task_id}")
        return self._score(
            task, settings, result, logs_dir, started, timed_out=timed_out, rewards=rewards, problem=problem
        )

    async def _reset_shared_verifier_dirs(self, sandbox: AsyncSandbox, *, keep: Sequence[str] = ()) -> None:
        """Empty ``/tests`` and ``/logs/verifier`` and remove the server's ``/logs/agent/trajectory.json`` in the
        agent's own sandbox, as root. Raises :class:`RuntimeError` when the sandbox could not.

        The shared verifier does this before it uploads its tests, and a multi-step episode before every turn after
        the first, as Harbor main's ``_prepare_step`` does (0.22.0 does not, so its next step's agent sees the
        previous step's tests): the previous turn's verifier left its tests (hidden ones included), its reward and
        the agent's transcript where the next turn's fresh agent would read them, and before ``test.sh`` the agent
        may have left anything in a folder it could write. Emptying ``/logs/verifier`` is a deliberate difference
        from Harbor's single-step shared mode, which merges tests and reads whatever the folder holds: a planted
        ``reward.json`` would outrank the ``reward.txt`` the trusted ``test.sh`` writes, and a planted test report
        (an all-pass ``results.xml`` left behind by a solution that exits during pytest's import) would be scored
        as a pass by a ``test.sh`` that reads one. ``keep`` names the files that survive (``[verifier].inputs``, the
        task's declared sidecar inputs: kumo's ``verifier`` service appends ``/logs/verifier/actions.jsonl``, shared
        with ``main`` by the planner, the input ``main``'s ``tests/score.py`` reads, which a whole-folder reset
        would delete); each is kept only as a regular file at that top-level name, so a planted directory or link
        under a kept name goes with everything else. They survive every reset of the episode, a deliberate
        difference from 0.22.0's ``_reset_shared_step_verifier_dirs``, which empties the folder before each step's
        verifier. Agent-planted directory links at the roots go first: find does not follow its starting symlink,
        while tar does. /logs is unlinked without touching its former target. The reserved trajectory.json
        destination goes too (`rm -rf` removes a link itself, or a planted directory), so the agent cannot plant an
        entry the upload would follow or fail on; the rest of /logs/agent is the agent's and is kept.
        """
        roots = (str(Path(VERIFIER_LOGS_DIR).parent), TESTS_DIR, VERIFIER_LOGS_DIR, AGENT_LOGS_DIR)
        unlink_roots = [f"if test -L {shlex.quote(path)}; then rm -- {shlex.quote(path)}; fi" for path in roots]
        if keep:
            kept = " -o ".join(f"-path {shlex.quote(f'{VERIFIER_LOGS_DIR}/{name}')}" for name in keep)
            clear = [
                f"find {TESTS_DIR} -mindepth 1 -delete",
                f"find {VERIFIER_LOGS_DIR} -mindepth 1 ! \\( \\( {kept} \\) -type f \\) -delete",
            ]
        else:
            clear = [f"find {TESTS_DIR} {VERIFIER_LOGS_DIR} -mindepth 1 -delete"]
        result = await exec_as_root(
            sandbox,
            " && ".join(
                unlink_roots
                + [
                    f"mkdir -p {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                    f"rm -rf -- {AGENT_LOGS_DIR}/{TRAJECTORY_FILE_NAME}",
                    *clear,
                    f"chmod 777 {TESTS_DIR} {VERIFIER_LOGS_DIR} {AGENT_LOGS_DIR}",
                ]
            ),
            timeout_s=60,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Could not clear /tests, /logs/verifier and the transcript in the agent's sandbox: "
                f"{result.stderr or result.stdout}"
            )

    async def _run_collect_hooks(self, session: HarborSession, hooks: list[Any]) -> None:
        """Run ``[[verifier.collect]]`` hooks in their services, before anything is collected from them.

        Harbor runs them in both verifier modes. A hook that exits non-zero, or runs past its
        ``timeout_sec``, is the task's business and only warns, as Harbor's best-effort hooks do; a hook
        the sandbox could not run at all is an outage and raises.
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
                timeout_s=hook.timeout_sec + COLLECT_HOOK_GRACE_S,
                user=hook.user,
            )
            if result.error_type == "timeout":
                LOGGER.warning(f"{task.task_id}: collect hook timed out after {hook.timeout_sec:g}s: {hook.command!r}")
                continue
            if result.error_type:
                raise SandboxProbeError(f"collect hook could not run: {result.error_type}: {hook.command!r}")
            if result.return_code:
                LOGGER.warning(f"{task.task_id}: collect hook exited {result.return_code}: {hook.command!r}")

    @staticmethod
    def _write_artifact_manifest(artifacts_dir: Path, entries: list[dict[str, Any]]) -> None:
        """Record one collection pass as ``artifacts/manifest.json``, one entry per declared artifact in staging
        order, as Harbor's artifact handler writes its manifest beside what it collected: the ``source`` and
        ``service`` as declared, where the entry was declared (``origin``: ``convention``, ``task`` or ``step``), its
        staging folder (``destination``, relative to the verifier logs folder), what the sandbox held there (``type``:
        ``directory``, ``file`` or ``null``) and the outcome (``status``: ``ok`` when staged and kept for a restore,
        ``absent`` when the sandbox had nothing at the path, ``link`` for a symbolic link that is not restored,
        ``failed`` when the agent's tree could not be collected, ``skipped`` when the entry names a service the task
        has not got). Best effort, like the collection itself: a manifest that cannot be written is logged.
        """
        if not entries:
            return
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(entries, indent=2) + "\n")
        except OSError:
            LOGGER.warning("Could not write %s under %s", ARTIFACT_MANIFEST_NAME, artifacts_dir, exc_info=True)

    async def _download_artifacts(
        self,
        session: HarborSession,
        artifacts: list[tuple[int, HarborArtifact, str]],
        artifacts_dir: Path,
        manifest: list[dict[str, Any]] | None = None,
    ) -> list[tuple[Path, str, str]]:
        """Pull the numbered ``artifacts`` out of their services; returns ``(archive, kind, sandbox path)`` for the
        ones found, and appends one record per entry to ``manifest`` (:meth:`_write_artifact_manifest`).

        Each artifact stages under its own numbered root, and the archive its sandbox packed is kept beside
        that root (``NN.tar.gz``) for the restore: the staged copy went through the host's filter and carries
        widened modes, the archive carries the modes the agent left. Links are kept (below), so one
        artifact's tree may hold a link to anywhere; with a shared root a later artifact's path could run
        through it.
        """
        task = session.task
        restored: list[tuple[Path, str, str]] = []
        records = manifest if manifest is not None else []
        for index, artifact, origin in artifacts:
            host = artifacts_dir / f"{index:02d}" / artifact.host_path
            archive = artifacts_dir / f"{index:02d}.tar.gz"
            record: dict[str, Any] = {
                "index": index,
                "source": artifact.source,
                "service": artifact.service,
                "origin": origin,
                "destination": f"artifacts/{index:02d}/{artifact.host_path.as_posix()}",
                "type": None,
                "status": "failed",
            }
            records.append(record)
            try:
                sandbox = session.service(artifact.service)
            except KeyError as exc:
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} skipped: {exc}")
                record["status"] = "skipped"
                continue
            # Best effort per artifact, as Harbor collects them: what the agent left behind (an
            # unreadable tree, a link that escapes the folder, a device node, a file where a folder was
            # expected, a path too long for this host) cannot mask the sample. Symbolic links travel as
            # links, because the restore target is another sandbox, not the host. Anything else (a lost
            # sandbox, a full disk, a dropped connection) is Gym's outage and still masks.
            try:
                kind = await download_path(
                    sandbox,
                    artifact.source,
                    host,
                    links="keep",
                    exclude=tuple(artifact.exclude),
                    archive_copy=archive,
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
            except OSError as exc:
                # Only the name-length error is the agent's tree (a path this host's filesystem cannot
                # hold); ENOSPC, EIO and the rest are this host's failure and must still mask, so no
                # blanket OSError clause.
                if exc.errno != errno.ENAMETOOLONG:
                    raise
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} not collected: {exc}")
                continue
            record["type"] = {"dir": "directory", "file": "file"}.get(kind)
            if kind is None:
                record["status"] = "absent"
                continue
            if host.is_symlink():
                # The artifact itself is a link: nothing to restore without following it on this host.
                LOGGER.warning(f"{task.task_id}: artifact {artifact.source!r} is a symbolic link, not restored")
                archive.unlink(missing_ok=True)
                record["status"] = "link"
                continue
            record["status"] = "ok"
            restored.append((archive, kind, artifact.source))
        return restored

    async def _collect_turn_artifacts(
        self, session: HarborSession, turn: HarborTurn, artifacts_dir: Path, *, stop_main: bool = False
    ) -> list[tuple[Path, str, str]]:
        """Harbor's per-step collection pass (``_collect_step_artifacts`` of its multi-step trial, 0.22.0), after the
        turn's agent and before its verifier, in shared mode too: the task's ``[[verifier.collect]]`` hooks and then
        the step's own ``[[steps.verifier.collect]]`` hooks for ``main`` (Harbor's ``_collect_hooks_for``: the task's,
        then the step's) and the ``main`` entries of ``/logs/artifacts``, the task's ``artifacts`` and the step's own
        (:func:`_collected_artifacts`), then the sidecars' hooks and entries, each staged under its numbered root in
        ``artifacts_dir`` with the pass recorded in its manifest (:meth:`_download_artifacts`,
        :meth:`_write_artifact_manifest`). Nothing is stopped unless ``stop_main`` says so: the agent's sandbox and its
        sidecars keep running for the turns that follow. With ``stop_main`` (the last turn of a task whose last step
        verifies separately, Harbor's ``stop_main_before_sidecars``) the agent's own container is stopped between the
        main pass and the sidecars' hooks, so nothing it left running can change a sidecar's state while it is read,
        as the single-step separate path does. Returns the archives a separate verifier of this turn restores.
        """
        task = session.task
        hooks = [*task.config.verifier.collect, *turn.config.verifier.collect]
        artifacts = _numbered_artifacts(task, turn.config)
        manifest: list[dict[str, Any]] = []
        await self._run_collect_hooks(session, [hook for hook in hooks if _is_main_service(hook.service)])
        main = [entry for entry in artifacts if _is_main_service(entry[1].service)]
        restored = await self._download_artifacts(session, main, artifacts_dir, manifest)
        if stop_main:
            await session.stop_main()
        await self._run_collect_hooks(session, [hook for hook in hooks if not _is_main_service(hook.service)])
        sidecars = [entry for entry in artifacts if not _is_main_service(entry[1].service)]
        restored += await self._download_artifacts(session, sidecars, artifacts_dir, manifest)
        self._write_artifact_manifest(artifacts_dir, manifest)
        return restored

    async def _run_shared_verifier(
        self,
        session: HarborSession,
        trajectory: bytes,
        *,
        settings: VerifierSettings | None = None,
        test_dirs: tuple[Path, ...] | None = None,
        response: NeMoGymResponse | None = None,
        logs_dir: Path | None = None,
        collect_hooks: bool = True,
    ) -> dict[str, Any]:
        """Run ``test.sh`` in the agent's sandbox and read the reward it wrote.

        ``[[verifier.collect]]`` hooks run first, as in Harbor: what they produce under ``/logs/artifacts``
        is there for ``test.sh``, and for the run's logs. ``/logs/agent`` is left as the agent wrote it,
        except for ``trajectory.json``: whatever the agent left at that reserved path (a link, a directory)
        is removed and the server writes the transcript there, so verifiers that read it find it.

        ``test_dirs`` are uploaded to ``/tests`` in order, each over the last (a step's ``tests/`` over the
        task's); missing folders are skipped. By default the task's ``tests/`` and ``[verifier]`` apply.
        ``logs_dir`` is the verification's host folder when the caller reserved it already (a turn's collection
        pass staged its artifacts there first), and ``collect_hooks=False`` says that pass ran the hooks.
        """
        task = session.task
        sandbox = session.sandbox
        if settings is None:
            verifier = task.config.verifier
            settings = VerifierSettings(
                timeout_sec=verifier.timeout_sec,
                env=verifier.env,
                user=verifier.user,
                collect=verifier.collect,
                network_mode=verifier.network_mode,
            )
        # A step's tests/ (and a multi-step task's top-level tests/) may be absent; a single-turn task's tests/ is
        # uploaded whatever the host says, so a snapshot removed under the episode fails the upload and masks.
        optional_tests = test_dirs is not None
        if test_dirs is None:
            # A task graded by the dataset's verifier.py has no tests/ of its own: the shim below is its /tests.
            test_dirs = (task.path / "tests",) if _uploads_task_tests(task) else ()
        started = asyncio.get_running_loop().time()
        if logs_dir is None:
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
            if collect_hooks:
                await self._run_collect_hooks(session, settings.collect)
            await self._reset_shared_verifier_dirs(sandbox, keep=task.config.verifier.inputs)
            for tests in test_dirs:
                if optional_tests and not tests.is_dir():
                    continue
                await upload_dir(sandbox, tests, TESTS_DIR, ignore=is_ignored)
            await self._upload_verifier_shim(sandbox, session, response)
            await _make_test_script_executable(sandbox)
            await self._upload_trajectory(sandbox, trajectory, logs_dir)
            # One plain exec for the whole run. The provider keeps it alive for as long as
            # `timeout_s` says (OpenSandbox polls a background command, Docker streams), and
            # `timeout` inside the container is the cap that stops the tests themselves.
            # Commands must not be left running after the exec returns: OpenSandbox reaps
            # them when the command completes, so a background launch never finishes.
            # A sub-second task budget still gets one whole second: `timeout 0` would mean no limit.
            budget = max(1, math.ceil(settings.timeout_sec))
            # The verifier's own user, not the agent's: the sandbox's default may be [agent].user. The fallback is the
            # shared rule's whatever the task's own mode (a step of a separate-mode task may opt out into this path).
            result = await sandbox.exec(
                verifier_command(budget),
                cwd=session.workdir,
                env=self._verifier_env(settings),
                timeout_s=settings.timeout_sec + self.config.verifier_grace_s,
                user=settings.user if settings.user is not None else _verifier_user(_shared_verifier_task(task)),
            )
            if result.error_type not in (None, "timeout"):
                return self._masked(failure_kinds.VERIFIER_ERROR, f"test.sh could not run: {result.error_type}")
            exit_record, rewards, problem = await self._collect_reward(
                sandbox,
                task,
                logs_dir,
                entrypoint_log=ENTRYPOINT_LOG_NAME if session.entrypoint is not None else None,
            )
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
            "verifier_network": settings.network_mode or "public",
        }
        LOGGER.debug(
            "%s: test.sh exited %s after %ss%s; rewards %s; logs in %s",
            task.task_id,
            result.return_code,
            extras["verifier_seconds"],
            " (timed out)" if timed_out else "",
            rewards,
            extras["verifier_logs_dir"],
        )
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
            # Finite components whose sum overflows: a reward file no trainer can use, measured like any other
            # invalid reward file.
            reason = (
                f"reward.json keys {sorted(rewards)} do not sum to a finite reward; tail: {_stdout_tail(logs_dir)}"
            )
            return self._measured(0.0, INVALID_REWARD_KIND, reason) | extras
        outcome = {"reward": reward, "verifier_rewards": rewards}
        components = summing_components(reward, reward_components(rewards))
        if components is not None:
            outcome["reward_components"] = components
        return outcome | extras

    def _refuse_verifier_group(self, task: HarborTask) -> None:
        """Refuse at seed the ``name:group`` ``test.sh`` would run as, in either verifier mode.

        Two users are as static as the task and the provider, which are both known before any sandbox exists:
        the one :func:`_verifier_user` dispatches ``test.sh`` as (``[verifier].user``, or the task image's own
        ``USER`` with its group), and the image's own ``USER`` a separate verifier graded in the task's image
        installs as its default (:meth:`HarborUsersMixin._set_default_user` with ``role="verifier"``). Only
        Docker applies a group. Refused here, the episode fails as a seed failure instead of paying the whole
        rollout and failing at verification; a verifier in another image keeps that image's default untouched
        and names no user. A multi-step task's turn grades as its step's own ``[steps.verifier].user`` when the
        step sets one (:meth:`_turn_verifier_settings` dispatches the pair as written), which is as static as the
        task's: every step's pair is refused here the same way, before the first turn runs, with the step named
        (``<task_id> step 'build': user ...``), as the loader names the step in its own per-step refusals. A shared
        turn without a user of its own dispatches as the shared fallback (:func:`_shared_verifier_task`), the task
        image's own ``USER`` with its group, even in a separate-mode task: that pair is refused for the step too.
        """
        if _verifier_runs_task_image(task):
            self._refuse_unsupported_group(task, task.entrypoint_user)
        self._refuse_unsupported_group(task, _verifier_user(task))
        for turn in task.turns:
            where = f" step {turn.name!r}"
            self._refuse_unsupported_group(task, turn.config.verifier.user, where=where)
            if self._turn_separate(task, turn):
                if _verifier_runs_task_image(self._turn_verifier_view(task, turn)):
                    # The turn's own verifier sandbox in the task image installs the image's USER as its default.
                    self._refuse_unsupported_group(task, task.entrypoint_user, where=where)
            elif turn.config.verifier.user is None:
                self._refuse_unsupported_group(task, _verifier_user(_shared_verifier_task(task)), where=where)

    @staticmethod
    def _measured(reward: float, kind: str, reason: str) -> dict[str, Any]:
        return {"reward": reward, "mask_sample": False, "failure_kind": kind, "failure_reason": reason[:2000]}

    @staticmethod
    def _masked(kind: str, reason: str) -> dict[str, Any]:
        return {"reward": 0.0, "mask_sample": True, "failure_kind": kind, "failure_reason": reason[:2000]}

    def _separate_verifier(self, task: HarborTask) -> bool:
        """Whether this episode verifies in a fresh sandbox: the task's mode, or forced for tasks with artifacts.

        For a multi-step task the question is per turn (:meth:`_turn_separate`); this is whether any turn does.
        """
        if task.turns:
            return any(self._turn_separate(task, turn) for turn in task.turns)
        return not task.config.is_shared_verifier or (
            self.config.separate_verifier_for_artifacts and bool(task.config.artifacts)
        )

    def _turn_separate(self, task: HarborTask, turn: HarborTurn) -> bool:
        """Whether a turn verifies in a fresh sandbox: the step's own mode as Harbor resolves it
        (:attr:`HarborTurn.verifier_mode`: the step's explicit mode, else ``separate`` with a
        ``[steps.verifier.environment]``, else the task's), or forced by ``separate_verifier_for_artifacts`` when the
        task or the step declares artifacts, as a single-step task with artifacts is forced."""
        if turn.verifier_mode == "separate":
            return True
        return self.config.separate_verifier_for_artifacts and bool(task.config.artifacts or turn.config.artifacts)

    @staticmethod
    def _turn_verifier_view(task: HarborTask, turn: HarborTurn) -> HarborTask:
        """``task`` as a turn's separate verifier sees it: the step's ``[steps.verifier]`` over the task's
        ``[verifier]`` (user, env and environment, Harbor's step-over-task resolution; the timeout is the step's own
        or Harbor's 600 s step default, never the task's), in separate mode,
        with the turn's verifier image and its recorded configuration in place of the task's. Everything the
        separate-mode machinery reads off a task (:func:`_verifier_image`, :func:`_verifier_runs_task_image`,
        :func:`_verifier_user`, the sandbox spec, the network policy) then answers for the turn, and a step without
        its own ``[steps.verifier.environment]`` inherits the task's or runs the task image, as in Harbor.

        A turn whose own mode is shared and which the server forces into a fresh sandbox
        (``separate_verifier_for_artifacts``, :meth:`_turn_separate`) has no verifier image or environment of its own
        (the loader resolves those for separate turns alone), so its view is the one a forced single-step shared task
        gets: no verifier environment, the task's own image with its overlay (:func:`_verifier_runs_task_image`),
        never the task's separate ``[verifier.environment]`` with its image missing, which the seed would refuse."""
        step = turn.config.verifier
        base = task.config.verifier
        forced = turn.verifier_mode != "separate"
        verifier = HarborVerifier(
            timeout_sec=task.config.step_verifier_timeout_sec(turn.config),
            user=step.user if step.user is not None else base.user,
            env=base.env | step.env,
            environment_mode="separate",
            environment=None if forced else (step.environment if step.environment is not None else base.environment),
            collect=[*base.collect, *step.collect],
            network_mode=step.network_mode if step.network_mode is not None else base.network_mode,
            allowed_hosts=step.allowed_hosts if step.allowed_hosts is not None else base.allowed_hosts,
        )
        return replace(
            task,
            config=task.config.model_copy(update={"verifier": verifier}),
            verifier_image=None if forced else turn.verifier_image,
            verifier_image_config=None if forced else turn.verifier_image_config,
        )

    def _verifier_env(self, settings: Any) -> dict[str, str]:
        """The `test.sh` process environment: the task's resolved `[verifier].env`, and nothing else.

        Operator secrets reach it only through the `${NAME}` templates the task writes (Harbor's
        verifier resolves its env the same way), so a task that names none runs test.sh without any.
        """
        return resolve_env_templates(settings.env, self.config.secrets)

    async def _upload_trajectory(
        self, sandbox: AsyncSandbox, trajectory: bytes, logs_dir: Path, *, restored: Sequence[str] = ()
    ) -> None:
        """Keep a host copy of the transcript beside the verifier logs and place it where the verifier reads it.

        ``restored`` names the artifact sources re-materialized in ``sandbox`` before this upload (the separate
        verifier restores every collected artifact at its own path). When one covers the reserved destination,
        the agent's tree came back with it, so whatever the agent planted there (a link the upload would follow,
        a directory it would fail on) is removed first, as the shared-mode prepare does. Otherwise the sandbox is
        fresh at that path, or it was already cleared, and the upload needs no extra round trip.
        """
        host = logs_dir / TRAJECTORY_FILE_NAME
        host.write_bytes(trajectory)
        target = f"{AGENT_LOGS_DIR}/{TRAJECTORY_FILE_NAME}"
        if any(_restores_under(source, target) for source in restored):
            clear = await exec_as_root(
                sandbox,
                f"if test -L {AGENT_LOGS_DIR}; then rm -- {AGENT_LOGS_DIR}; fi "
                f"&& mkdir -p {AGENT_LOGS_DIR} && rm -rf -- {target}",
                timeout_s=60,
            )
            if clear.return_code != 0:
                raise RuntimeError(f"Could not clear the restored {target}: {clear.stderr or clear.stdout}")
        await upload_path(sandbox, host, target)
        # The host copy is created under the server's umask and the archive carries that mode, which root `tar`
        # keeps: under umask 077 the transcript would land root-owned 0600, unreadable by a non-root `[verifier].user`
        # (Harbor's agents write the file as themselves, so it is theirs to read). The reserved file alone is opened;
        # the rest of /logs/agent is the agent's, with the modes it left.
        readable = await exec_as_root(sandbox, f"chmod a+r {target}", timeout_s=60)
        if readable.return_code != 0:
            raise RuntimeError(f"Could not make {target} readable: {readable.stderr or readable.stdout}")

    def _export_trajectory(
        self, request: BaseEpisodeRequest[Any], agent_response: NeMoGymResponse, responses_create_params: Any = None
    ) -> bytes:
        """The agent's transcript as ATIF v1.7 JSON, for ``/logs/agent/trajectory.json``.

        Exported from the Gym transcript (the task's input and the harness's output); raises
        :class:`AtifExportError` for anything the converter cannot represent, which masks the turn.
        ``responses_create_params`` is the request the turn was sent when it is not the row's own (a
        multi-step task's turn); by default the row's ``responses_create_params`` is the input.
        """
        params: Any = request.task.task_input.responses_create_params
        if responses_create_params is not None:
            params = responses_create_params
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

    @staticmethod
    def _turn_verifier_settings(task: HarborTask, turn: HarborTurn) -> VerifierSettings:
        """A step's verifier settings over the task's, as Harbor resolves them (its hooks after the task's; the
        timeout is the step's own or Harbor's 600 s step default, :meth:`HarborTaskConfig.step_verifier_timeout_sec`)."""
        step = turn.config.verifier
        return VerifierSettings(
            timeout_sec=task.config.step_verifier_timeout_sec(turn.config),
            env=task.config.verifier.env | step.env,
            user=step.user if step.user is not None else task.config.verifier.user,
            collect=[*task.config.verifier.collect, *step.collect],
            network_mode=step.network_mode if step.network_mode is not None else task.config.verifier.network_mode,
        )
