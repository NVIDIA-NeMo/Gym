# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import hashlib
import json
import logging
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.dataset_config import (
    GymDatasetConfig,
    GymTaskOverride,
    apply_dataset_config,
    read_dataset_config,
)
from nemo_gym.tasks.harbor.digest import content_hash, context_digest
from nemo_gym.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    OverlayRun,
    OverlayWorkdir,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.models import HARBOR_STEP_VERIFIER_TIMEOUT_SEC, HarborTaskConfig, HarborTaskStep
from nemo_gym.tasks.harbor.python_verifier import VERIFIER_FILE, python_verifier_for
from nemo_gym.tasks.harbor.records import (
    BUILT_IMAGES_FILE,
    IMAGE_CONFIGS_FILE,
    MAIN_SERVICE,
    VERIFIER_SERVICE,
    read_built_images,
    read_image_configs,
    recorded_forced_tasks,
    recorded_service_images,
)
from nemo_gym.tasks.harbor.task_data import read_task_fields
from nemo_gym.tasks.harbor.verifier_env import (
    TESTS_COMPOSE_FILE,
    TESTS_DOCKERFILE,
    _verifier_build_required,
    _verifier_definition_required,
    effective_verifier_environment,
    step_verifier_build_context,
    step_verifier_environment,
    step_verifier_mode,
    verifier_build_context,
    verifier_compose_main_overrides,
    verifier_compose_refusal,
    verifier_compose_sidecars,
)


logger = logging.getLogger(__name__)

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
# Harbor's seed trajectory (``TaskPaths.TRAJECTORY_FILENAME``): an ATIF document a task ships beside its instruction
# (``steps/<first step>/`` for a multi-step task, where a top-level one is ignored) that Harbor 0.22.0 loads into the
# agent's session before its first turn, for every agent but the oracle and nop (``Trial._resolve_load_trajectory``).
# Gym cannot seed a harness with it yet: the loader records the file (:attr:`HarborTask.seed_trajectory`) and the
# environment server runs the task for the oracle alone, refusing it for any other agent rather than running it
# without its prior context.
TRAJECTORY_FILE = "trajectory.json"
STEPS_DIR = "steps"
# A step's files under ``steps/<name>/``: the folder merged into the working directory before the turn and the
# script run from there (Harbor's ``_upload_step_workdir`` and ``_run_step_setup``: the script lives inside the folder).
STEP_WORKDIR = "workdir"
STEP_SETUP_SCRIPT = "setup.sh"
# The one value of the pre-1.3 `multi_step_reward_mode` key that older multi-step generators wrote: the per-step
# average, which is the default `mean` roll-up the task gets once the key is dropped, so it is dropped silently.
LEGACY_AGGREGATE_REWARD_MODE = "aggregate"
DOCKERFILE = Path("environment") / "Dockerfile"
# The names the loader accepts for a task's Compose overlay under ``environment/``, first match wins
# (``nemo_gym.tasks.harbor.image_configs.compose_file``); Harbor's Docker environment reads the first one only.
COMPOSE_FILE_NAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")


# How deep below a dataset root `discover_tasks` looks for task folders. Datasets group tasks by
# category at most a few levels deep; hidden folders are never entered.
MAX_DISCOVERY_DEPTH = 6


@dataclass(frozen=True)
class HarborTurn:
    """One task step loaded with its ``steps/<name>/`` folder, which the harbor server runs as one agent turn.

    ``config`` is the step as ``task.toml`` writes it (:class:`HarborTaskStep`) and ``instruction`` is
    ``steps/<name>/instruction.md``, so ``HarborTaskStep`` is to ``HarborTurn`` what ``HarborTaskConfig`` is to
    ``HarborTask``. One turn per task step is this server's choice, as in Harbor's multi-step trial, not something
    a task step implies: Harbor's simulated-user trials run many turns of the agent under test inside one step.

    ``verifier_mode`` is the step's own, as Harbor's ``resolve_step_verifier_mode`` resolves it (the step's explicit
    mode, else ``separate`` when it declares a ``[steps.verifier.environment]``, else the task's). ``verifier_image``
    and ``verifier_image_config`` are, for a separate turn, what :attr:`HarborTask.verifier_image` and
    :attr:`HarborTask.verifier_image_config` are for a single-step task: the image the turn's verifier runs in when
    it is not the task's own (``None`` runs the task image), by the same rules, and that image's recorded
    configuration when there is one.
    """

    instruction: str
    config: HarborTaskStep
    verifier_mode: str = "shared"
    verifier_image: str | None = None
    verifier_image_config: BaseImageConfig | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tests_dir(self) -> Path:
        """Relative to the task folder; may be absent, in which case the task's ``tests/`` is used alone."""
        return Path(STEPS_DIR) / self.name / "tests"

    @property
    def solution_dir(self) -> Path:
        """The step's own reference solution, relative to the task folder (Harbor's ``steps/<name>/solution/``);
        may be absent, in which case the task's top-level ``solution/`` stands in for it (:meth:`HarborTask.turn_solution_dir`)."""
        return Path(STEPS_DIR) / self.name / "solution"

    @property
    def workdir_dir(self) -> Path:
        """``steps/<name>/workdir/``, relative to the task folder: merged into the sandbox's working directory
        before the turn's agent session, as Harbor's ``_upload_step_workdir`` uploads it; may be absent."""
        return Path(STEPS_DIR) / self.name / STEP_WORKDIR

    @property
    def setup_script(self) -> Path:
        """``steps/<name>/workdir/setup.sh``, relative to the task folder: run as ``bash <workdir>/setup.sh`` once the
        folder is in place, as Harbor's ``_run_step_setup`` runs it; may be absent."""
        return self.workdir_dir / STEP_SETUP_SCRIPT


@dataclass(frozen=True)
class HarborTask:
    """A task folder read into memory.

    ``task_id`` is the folder name: ``[task].name`` is an ``org/name`` label that is
    not unique across a dataset. ``folder_digest`` is the folder's content hash alone
    (:func:`~nemo_gym.tasks.harbor.digest.content_hash`, what ``manifest.toml`` records per task and
    a fetch verifies); ``digest`` is the one rows carry and the server checks, that hash plus the
    ``dataset.toml`` settings applied to the task (:func:`task_digest`), equal to ``folder_digest``
    when the dataset has no ``[gym]`` table. ``image`` is the prebuilt image when the task
    declares one, else the base image of a pull-mode or overlay-mode Dockerfile pinned to
    the digest its recorded configuration was read at (the multi-arch index behind the
    written tag, see :mod:`nemo_gym.tasks.harbor.image_configs`; the written reference
    when nothing is recorded), else ``None`` (a build is required). ``workdir``, ``env``
    and ``user`` merge ``task.toml`` with what the Dockerfile recorded; ``task.toml`` wins.

    Dockerfile ``ENV``, ``WORKDIR`` and ``USER`` are resolved to literals as ``docker build``
    resolves them, starting from the base image's configuration recorded in
    :data:`IMAGE_CONFIGS_FILE` next to the task folders (see
    :mod:`nemo_gym.tasks.harbor.dockerfile`).

    ``overlay`` is Gym-derived data, not a Harbor field: the Dockerfile's ``RUN`` lines
    (with the ``WORKDIR``, ``ENV`` and ``USER`` in effect for each) that the server runs
    in the pulled base image at seed. Empty for a prebuilt image or a pull-mode Dockerfile.

    ``overlay_workdirs`` are the Dockerfile's ``WORKDIR`` instructions for an overlay-mode task,
    in order with the user in effect at each, so the server creates every one of them as a build
    would, not only the ones a ``RUN`` line happens to use.

    ``sandbox_user`` is the user the sandbox runs commands as when no user is asked for: the
    default user of the image the sandbox starts from. For a Dockerfile Gym runs itself that is
    the base image's recorded ``User`` (``None`` is root), not the Dockerfile's final ``USER`` or
    ``[agent].user``; for an image prepare built (forced over a declared ``docker_image`` too) it is
    the built image's recorded ``User``, whatever ``[agent].user`` says; for a declared ``docker_image``
    built from a pull-mode Dockerfile it is that Dockerfile's final ``USER``. When neither is known the
    task's ``user`` stands in for it.

    ``built_image`` is the image prepare built from the task's ``environment/`` folder when the
    Dockerfile needed a build (``built-images.json``); ``image`` is then that image and the task
    runs in pull mode from it. ``None`` in the other two modes and for a prebuilt image.

    ``verifier_image`` is the image a separate verifier runs in when it is not the task's own: the
    ``[verifier.environment].docker_image`` the task declares, else the image prepare built from ``tests/Dockerfile``
    (service ``verifier`` under Harbor's hash of ``tests/`` in ``built-images.json``, see
    :func:`verifier_build_context`). ``None`` in shared mode, and in separate mode when the verifier runs in the
    task's own image because the effective verifier environment is a copy of ``[environment]`` that carries its
    ``docker_image`` (Harbor's prebuilt verifier). Every other separate-mode shape loads with a verifier image or not
    at all: a task whose effective verifier environment names no image and ships no ``tests/Dockerfile`` is refused as
    Harbor refuses it, and one whose ``tests/Dockerfile`` has no recorded build is refused like an unrecorded
    ``environment/`` build (:func:`load_task`).

    ``context_id`` is Harbor's environment id of ``environment/`` (:func:`~nemo_gym.tasks.harbor.digest.context_digest`,
    the key of ``built-images.json`` and the body of the ``hb__<hash>`` image names), computed once here for every
    task with an ``environment/`` folder (the build-mode branches read it, and the Compose loader needs it at every
    seed, for a Compose file or for the one-service group a job-level overlay extends) and ``None`` for a task
    without one; ``built_images`` is
    every image that record holds for the context, by Compose service name (``main`` and the sidecars with a
    ``build:`` section, :func:`recorded_service_images`), ``{}`` when none is recorded. The Compose loader
    substitutes them at prepare and at seed without hashing the folder again; ``main``'s entry is applied only
    when ``built_image`` is set, since a record of ``main`` may be another task's build of the same context or
    one that predates a ``docker_image`` override.

    ``entrypoint`` is the image's ``Entrypoint`` as Docker would start it: the built image's recorded
    one, or the base image's for a Dockerfile the loader resolves unless that Dockerfile wrote
    ``ENTRYPOINT []``. ``None`` when the image records none or no configuration is recorded (a prebuilt
    ``docker_image``). ``image_shell`` is the image's ``Shell`` (Docker's ``/bin/sh -c`` when it sets none),
    what the daemon runs a ``CMD-SHELL`` healthcheck through, ``None`` likewise when nothing is recorded.
    ``entrypoint_user`` and ``entrypoint_workdir`` are the ``User`` and ``WorkingDir``
    the image starts with (``None`` is root and ``/``): what Docker gives its ``ENTRYPOINT`` and its
    ``HEALTHCHECK``, independent of ``[agent].user`` and ``[environment].workdir``, which only shape the
    commands run beside them. They are known whenever the image's configuration is, with or without an
    entrypoint. ``image_healthcheck`` is the recorded Dockerfile ``HEALTHCHECK`` block (``Test``,
    ``Interval``, ... as Docker records them), or ``None``.

    ``verifier_image_config`` is the recorded configuration of the separate verifier's image, ``verifier_image``
    (what Docker starts it with: its ``Entrypoint``, ``User``, ``WorkingDir`` and ``Healthcheck``, the way the fields
    above describe the task image's): the image prepare built from ``tests/Dockerfile``, or a declared
    ``[verifier.environment].docker_image`` whose reference ``compose-images.json`` records, as it does the base
    image of this task's Dockerfile (pull or overlay mode) and of the other tasks'. The record is the image's own,
    never the Dockerfile's final ``USER`` and ``WORKDIR`` or its ``RUN`` lines: Harbor ``up --wait``s the declared
    image as it is. ``None`` for a declared image nothing is recorded for (as for a prebuilt ``docker_image``),
    when the verifier grades in the task's own image (the task's own fields describe that start), and in shared
    mode.

    ``turns`` holds the task's ``[[steps]]`` in order; empty for a single-turn task. A multi-turn task
    never sends ``instruction`` to the agent: each turn sends its own, and the top-level file may hold
    material the agent must not see (SWE-Interact puts the full hidden specification there).

    ``task_data`` is the task's own field values, read from ``task_data.json`` in its folder (a
    parameterized task's rendered row), ``{}`` for a task without one. Materialize writes them into the
    row's ``task_data`` beside the reserved ``_ng_*`` keys, and the dataset's ``task_data.py`` validates
    them when it ships one (:mod:`nemo_gym.tasks.harbor.task_data`).

    ``seed_trajectory`` is Harbor's seed trajectory when the task ships one, relative to the task folder:
    ``steps/<first step>/trajectory.json`` for a multi-step task, else ``trajectory.json`` (one or the other, as
    Harbor's ``_task_trajectory_path`` picks it); ``None`` otherwise. The environment server runs such a task for
    the oracle alone and refuses it for any other agent, naming the file.
    """

    path: Path
    task_id: str
    config: HarborTaskConfig
    instruction: str
    digest: str
    folder_digest: str
    image: str | None
    workdir: str | None
    env: dict[str, str]
    user: str | None
    overlay: tuple[OverlayRun, ...] = ()
    sandbox_user: str | None = None
    built_image: str | None = None
    context_id: str | None = None
    built_images: Mapping[str, str] = field(default_factory=dict)
    verifier_image: str | None = None
    entrypoint: tuple[str, ...] | None = None
    entrypoint_user: str | None = None
    image_shell: tuple[str, ...] | None = None
    entrypoint_workdir: str | None = None
    image_healthcheck: dict[str, Any] | None = None
    verifier_image_config: BaseImageConfig | None = None
    overlay_workdirs: tuple[OverlayWorkdir, ...] = ()
    turns: tuple[HarborTurn, ...] = ()
    # The task's own field values (``task_data.json`` in its folder), written into the row's ``task_data``
    # beside the reserved keys and validated against the dataset's ``task_data.py`` when it ships one.
    task_data: Mapping[str, Any] = field(default_factory=dict)
    # The Python verifier that grades the task (``tests/verifier.py``, or the dataset root's ``verifier.py`` for a
    # task without ``tests/``), run inside the sandbox through the test.sh shim; ``None`` when ``tests/test.sh`` does.
    python_verifier: Path | None = None
    # Harbor's seed trajectory, relative to the task folder, when the task ships one (see the class docstring).
    seed_trajectory: Path | None = None

    @property
    def needs_sandbox(self) -> bool:
        """A task runs in a sandbox exactly when it declares an image or a Dockerfile."""
        return self.image is not None or (self.path / DOCKERFILE).is_file()

    @property
    def needs_compose(self) -> bool:
        """The task's environment is a Compose group: ``main`` plus the sidecars its overlay declares."""
        return has_compose_file(self.path)

    @property
    def has_solution(self) -> bool:
        """Whether the oracle can run this task: ``solution/solve.sh``, or ``solution/answer.md`` it submits as its
        response, for a single-turn task; for a multi-step task a ``solve.sh`` for every step, its own under
        ``steps/<name>/solution/`` or the task's top-level one (:meth:`turn_solution_dir`), as Harbor 0.22.0's
        oracle resolves each step's solution (``agents/oracle.py``)."""
        if self.turns:
            return all((self.turn_solution_dir(turn) / "solve.sh").is_file() for turn in self.turns)
        return (self.path / "solution" / "solve.sh").is_file() or self.has_answer

    @property
    def has_answer(self) -> bool:
        return (self.path / "solution" / "answer.md").is_file()

    def turn_solution_dir(self, turn: HarborTurn) -> Path:
        """The folder the oracle gets at ``/solution`` for ``turn``: the step's own ``steps/<name>/solution/`` when
        that folder exists, else the task's top-level ``solution/`` (which may be absent too), the fallback Harbor's
        oracle applies per step. A step folder that exists is the step's answer even without a ``solve.sh``, as in
        Harbor, where such a step fails rather than running the task's script."""
        own = self.path / turn.solution_dir
        return own if own.is_dir() else self.path / "solution"


def has_compose_file(task_folder: Path) -> bool:
    """Whether the task ships a Compose overlay under ``environment/`` (:data:`COMPOSE_FILE_NAMES`)."""
    return any((Path(task_folder) / "environment" / name).is_file() for name in COMPOSE_FILE_NAMES)


def is_task_folder(path: Path) -> bool:
    return (Path(path) / TASK_FILE).is_file()


def built_service_images(task_folder: Path, root: Path | None = None) -> dict[str, str]:
    """The images prepare built for the task's ``environment/`` context, by Compose service name (``{}`` when none
    are recorded): ``main`` for the task's own Dockerfile, each built sidecar under its service name.

    ``root`` is the dataset folder the record sits in (the task's parent by default).
    """
    task_folder = Path(task_folder)
    context = task_folder / DOCKERFILE.parent
    if not context.is_dir():
        return {}
    folder = Path(root) if root is not None else task_folder.parent
    return recorded_service_images(read_built_images(folder).get(context_digest(context)))


def built_image(task_folder: Path, root: Path | None = None) -> str | None:
    """The image prepare built for the task's Dockerfile, when ``built-images.json`` records one.

    ``root`` is the dataset folder the record sits in (the task's parent by default).
    """
    return built_service_images(task_folder, root).get(MAIN_SERVICE)


def _recorded_as_forced(root: Path, task_name: str) -> bool:
    """Whether any ``built-images.json`` entry of ``root`` records ``task_name`` as built under ``builds.force``
    over its declared image: one file read, so a declared-image task whose Dockerfile the loader can run pays
    Harbor's hash of ``environment/`` (the key its own record sits under) only when such a record may exist."""
    return any(task_name in recorded_forced_tasks(entry) for entry in read_built_images(root).values())


def _build_required(dockerfile: Path, reason: Exception, root: Path) -> HarborTaskError:
    """``root`` is the dataset folder whose ``built-images.json`` the loader read: the task's parent, or the dataset
    root of a task discovered deeper (the record is never next to a group folder)."""
    return HarborTaskError(
        f"{dockerfile} needs a build ({reason}) and no built image is recorded in "
        f"{root / BUILT_IMAGES_FILE}. `gym eval run` and `gym dataset validate` build it "
        "when buildah, podman or docker is installed (configured by the `builds:` block, see "
        "environment_servers/harbor/README.md), or declare [environment].docker_image with a prebuilt image."
    )


def dockerfile_base_image(task_folder: Path, declared_image: str | None) -> str | None:
    """The image whose recorded configuration the task's Dockerfile resolves against, if it has one.

    A prebuilt ``docker_image`` already holds whatever its Dockerfile's ``RUN`` lines did, so only a
    pull-mode Dockerfile contributes settings next to a declared image. Dockerfiles that need a build
    contribute nothing (``load_task`` reports them).
    """
    dockerfile = Path(task_folder) / DOCKERFILE
    if not dockerfile.is_file():
        return None
    try:
        parsed = parse_dockerfile(dockerfile.read_text())
    except (DockerfileNeedsBuild, UnicodeDecodeError):
        return None
    if declared_image is not None and parsed.has_runs:
        return None
    return parsed.image


_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def read_instruction(path: Path) -> str:
    """``instruction.md`` as the agent sees it: leading canary marker lines and blank lines dropped.

    Harbor tasks open with an HTML comment or heading carrying a canary GUID for contamination
    tracking. It is not part of the task, so it is removed the way Gym's Terminal Bench servers
    always did; everything after it is kept byte for byte.
    """
    lines = path.read_text().split("\n")
    removed_canary = False
    while lines and _CANARY_LINE.match(lines[0].strip()):
        lines.pop(0)
        removed_canary = True
    # Only the blank lines that separated the canary from the task go with it. A file with no canary is
    # the author's prompt byte for byte, leading newline included, as the legacy servers passed it.
    while removed_canary and lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def load_task(path: Path, *, root: Path | None = None, image_configs: Mapping[str, Any] | None = None) -> HarborTask:
    """Read ``path`` as one task. Raises :class:`HarborTaskError` when it is not one.

    ``root`` is the dataset folder whose files shape the task (``dataset.toml`` and the recorded
    image configurations); it is the task's parent folder unless the task was discovered deeper.
    ``image_configs`` is that folder's recorded image configurations when the caller has read them
    already (``discover_tasks`` reads the shared file once); otherwise they are read here.
    """
    path = Path(path)
    if path.is_symlink():
        raise HarborTaskError(f"Task folder must not be a symlink: {path}")
    path = path.resolve()
    root = path.parent if root is None else Path(root).resolve()
    if not is_task_folder(path):
        raise HarborTaskError(f"{path} has no {TASK_FILE}")
    try:
        data = tomllib.loads((path / TASK_FILE).read_text())
        for key in HarborTaskConfig.unknown_keys(data):
            logger.warning("%s: ignoring unknown key `%s`", path / TASK_FILE, key)
        legacy_mode = data.get("multi_step_reward_mode")
        if legacy_mode is not None and legacy_mode != LEGACY_AGGREGATE_REWARD_MODE:
            # Harbor 0.22.0 drops the key whatever it says and rolls the steps up with `mean` (or the task's own
            # `multi_step_reward_strategy`); so does the mirror, and the task author hears which value was dropped.
            logger.warning(
                "%s: ignoring legacy key `multi_step_reward_mode = %r` (Harbor reads `multi_step_reward_strategy`, "
                "`mean` when unset)",
                path / TASK_FILE,
                legacy_mode,
            )
        config = HarborTaskConfig.model_validate(data)
    except (OSError, ValueError) as exc:
        raise HarborTaskError(f"{path / TASK_FILE}: {exc}") from exc
    # The dataset's own settings sit beside the task folders and shape the effective task.toml.
    try:
        dataset = read_dataset_config(root)
        config = apply_dataset_config(config, path.name, dataset)
    except ValueError as exc:
        raise HarborTaskError(str(exc)) from exc

    phases = {"environment": config.environment, "agent": config.agent, "verifier": config.verifier}
    if config.verifier.environment is not None:
        phases["verifier.environment"] = config.verifier.environment
    for step in config.steps:
        phases[f"steps.{step.name}.agent"] = step.agent
        phases[f"steps.{step.name}.verifier"] = step.verifier
        if step.verifier.environment is not None:
            # A step's own verifier environment carries the same restriction fields as the task's; the server creates
            # the step's verifier sandbox from it with no egress policy, so a restriction there is refused like the
            # task-level one rather than dropped.
            phases[f"steps.{step.name}.verifier.environment"] = step.verifier.environment
    for name, phase in phases.items():
        if phase.network_mode in (None, "public") and not phase.allowed_hosts:
            continue
        if name != "verifier":
            # The agent's sandbox is created public and a running sandbox's egress policy cannot be switched, so
            # only the separate verifier's own sandbox can carry a phase policy (multi-step tasks verify in the
            # agent's sandbox).
            raise HarborTaskError(f"{path}: {name} network restrictions are not supported yet")
        # The task-wide [verifier] restriction applies to every grader that runs: for a multi-step task those are the
        # steps' verifiers in their resolved modes (the task-level default is never run by itself), for a task without
        # steps the task's own verifier.
        if config.steps:
            shared_grader = any(step_verifier_mode(config, step) == "shared" for step in config.steps)
        else:
            shared_grader = config.is_shared_verifier
        if shared_grader:
            raise HarborTaskError(
                f"{path}: [verifier] network restrictions are not supported in Harbor's shared verifier mode: a "
                "running sandbox's network policy cannot be changed after the agent's turn, so the restricted "
                "policy is applied to the separate verifier's sandbox when it is created (declare "
                '[verifier].environment_mode = "separate" or a [verifier.environment], on every step of a '
                "multi-step task too)"
            )
        if phase.allowed_hosts and phase.network_mode != "allowlist":
            raise HarborTaskError(f"{path}: [verifier] allowed_hosts is only valid when network_mode='allowlist'")
    environments = [("environment", config.environment), ("verifier.environment", config.verifier.environment)]
    environments += [(f"steps.{step.name}.verifier.environment", step.verifier.environment) for step in config.steps]
    for name, environment in environments:
        if environment is not None and environment.os != "linux":
            raise HarborTaskError(f"{path}: {name} Windows tasks are not supported yet")

    instruction_path = path / INSTRUCTION_FILE
    turns = _load_turns(path, config)
    seed_trajectory = _seed_trajectory(path, turns)
    # Harbor reads a multi-step task's instructions from its steps; the top-level file is then optional.
    if not turns and not instruction_path.is_file():
        raise HarborTaskError(f"{path} has no {INSTRUCTION_FILE}")
    python_verifier = python_verifier_for(path, root) if not turns else None
    if not turns and python_verifier is None and not (path / "tests").is_dir():
        raise HarborTaskError(f"{path} has no tests/ folder (and the dataset has no {VERIFIER_FILE} to grade it with)")
    try:
        instruction = read_instruction(instruction_path) if instruction_path.is_file() else ""
    except (OSError, UnicodeDecodeError) as exc:
        raise HarborTaskError(f"{instruction_path}: {exc}") from exc
    if not config.is_shared_verifier and not config.artifacts:
        # Harbor accepts this shape: the separate verifier still receives /logs/artifacts.
        logger.warning(
            "%s uses separate verification but declares no artifacts; the verifier grades a fresh container "
            "that receives only what the agent writes under /logs/artifacts",
            path / TASK_FILE,
        )

    environment = config.environment
    image = environment.docker_image
    workdir = environment.workdir
    env = dict(environment.env)
    user: str | None = None
    overlay: tuple[OverlayRun, ...] = ()
    overlay_workdirs: tuple[OverlayWorkdir, ...] = ()
    resolved = None
    base_known = False
    base_user: str | None = None
    dockerfile = path / DOCKERFILE
    built: str | None = None
    context_id: str | None = None
    built_images: dict[str, str] = {}
    forced_tasks: set[str] = set()

    def recorded_builds() -> dict[str, str]:
        # Harbor's hash of environment/ is paid once per task, here, off the server's event loop (load_task runs
        # in a thread there): the build-mode branches and the Compose loader both read the record through it.
        nonlocal context_id, built_images, forced_tasks
        if context_id is None and dockerfile.parent.is_dir():
            context_id = context_digest(dockerfile.parent)
            record = read_built_images(root).get(context_id)
            built_images = recorded_service_images(record)
            forced_tasks = recorded_forced_tasks(record)
        return built_images

    # What Docker starts the image with: the record of the image the sandbox runs (built image), or the
    # base image's record carried through the Dockerfile's final USER and WORKDIR (pull and overlay mode).
    entrypoint: tuple[str, ...] | None = None
    entrypoint_user: str | None = None
    entrypoint_workdir: str | None = None
    image_healthcheck: dict[str, Any] | None = None
    image_shell: tuple[str, ...] | None = None
    if dockerfile.is_file():
        try:
            dockerfile_text = dockerfile.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise HarborTaskError(f"{dockerfile}: {exc}") from exc
        try:
            parsed = parse_dockerfile(dockerfile_text)
        except DockerfileNeedsBuild as exc:
            # Build mode: prepare built the context once and recorded the image next to the tasks. A task that
            # declares docker_image runs it, unless prepare built its own Dockerfile for it under `builds.force`
            # (Harbor's --force-build) and recorded the task as forced: that build then wins, as it does in Harbor.
            # A record of main made for another task of the context, or before the override, is not this task's.
            built = recorded_builds().get(MAIN_SERVICE)
            if image is not None and path.name not in forced_tasks:
                built = None
            if built is None and image is None:
                raise _build_required(dockerfile, exc, root) from exc
            parsed = None
        if parsed is not None and image is not None and _recorded_as_forced(root, path.name):
            # A pull or overlay mode Dockerfile next to a declared image contributes its settings to that image,
            # unless prepare built it for this task under `builds.force` (Harbor's --force-build builds the file
            # whatever it holds): the recorded build then runs as is, its RUN lines, USER and WORKDIR baked in.
            built = recorded_builds().get(MAIN_SERVICE)
            if path.name in forced_tasks and built is not None:
                parsed = None
            else:
                built = None
        if built is not None:
            # The built image is pulled like a prebuilt one; its recorded configuration gives the
            # Dockerfile's final WORKDIR, ENV and USER.
            record = read_image_configs(root).get(built)
            base = BaseImageConfig.from_record(record) if record is not None else None
            image = built
            if base is not None:
                base_known = True
                base_user = base.user
                workdir = workdir or base.workdir
                env = base.env | env
                user = base.user
                entrypoint, entrypoint_user, entrypoint_workdir = base.entrypoint, base.user, base.workdir
                image_healthcheck, image_shell = base.healthcheck, base.shell
        if parsed is not None and (image is None or not parsed.has_runs):
            # A prebuilt image already holds whatever its Dockerfile's RUN lines did; only a
            # pull-mode Dockerfile contributes settings next to a declared image.
            configs = image_configs if image_configs is not None else read_image_configs(root)
            record = configs.get(parsed.image)
            try:
                base = BaseImageConfig.from_record(record) if record is not None else None
            except ValueError as exc:
                raise HarborTaskError(f"{root / IMAGE_CONFIGS_FILE}: {parsed.image!r}: {exc}") from exc
            base_known = base is not None
            base_user = base.user if base is not None else None
            # The immutable reference the configuration was recorded at: what a task without a declared
            # image starts, so a moved tag cannot change the image under the recorded settings.
            pinned_image = str(record.get("image") or "") if base is not None else ""
            try:
                resolved = resolve_dockerfile(parsed, base)
            except ImageConfigRequired as exc:
                raise HarborTaskError(
                    f"{dockerfile}: {exc}, and the configuration of base image {parsed.image!r} is not recorded "
                    f"in {root / IMAGE_CONFIGS_FILE}. Prepare the dataset again (`gym eval run` or "
                    "`gym dataset validate`) to record it."
                ) from exc
            except DockerfileNeedsBuild as exc:
                if image is not None:
                    # As for a parse-time build requirement above: the declared image already holds the
                    # build's result, so the task runs it with task.toml's own settings.
                    logger.debug("%s: %s; the declared image %s runs as is", dockerfile, exc, image)
                    base_known = False
                else:
                    # An expansion the overlay cannot resolve: a build, unless prepare already built it.
                    built = recorded_builds().get(MAIN_SERVICE)
                    if built is None:
                        raise _build_required(dockerfile, exc, root) from exc
                    record = read_image_configs(root).get(built)
                    base = BaseImageConfig.from_record(record) if record is not None else None
                    image = built
                    base_known = base is not None
                    base_user = base.user if base is not None else None
                    if base is not None:
                        workdir = workdir or base.workdir
                        env = base.env | env
                        user = base.user
                        entrypoint, entrypoint_user, entrypoint_workdir = base.entrypoint, base.user, base.workdir
                        image_healthcheck, image_shell = base.healthcheck, base.shell
            if resolved is not None:
                image = image or pinned_image or resolved.image
                # The FROM image's record only stands in for a WORKDIR the task starts without. A declared
                # docker_image that runs as is keeps its own directory (the provider reports it) unless the
                # Dockerfile sets one.
                if workdir is None and (environment.docker_image is None or resolved.workdirs):
                    workdir = resolved.workdir
                env = resolved.env | env
                user = resolved.user
                overlay = resolved.runs
                # Pull mode creates nothing: only an overlay makes the Dockerfile's directories at seed.
                overlay_workdirs = resolved.workdirs if resolved.runs else ()
                if base is not None:
                    # The image Docker would build from this Dockerfile keeps the base's entrypoint (unless
                    # the Dockerfile cleared it), healthcheck and shell (a SHELL instruction needs a build),
                    # started with its final USER and WORKDIR.
                    entrypoint = None if resolved.entrypoint_cleared else base.entrypoint
                    entrypoint_user, entrypoint_workdir = resolved.user, resolved.workdir
                    image_healthcheck, image_shell = base.healthcheck, base.shell
    dockerfile_user = user
    if config.agent.user is not None:
        user = str(config.agent.user)
    # The default user of the image the sandbox starts from (see HarborTask.sandbox_user). A build forced over a
    # declared docker_image is the image the sandbox starts from too: its own recorded User, never the override.
    if (environment.docker_image is None or built is not None) and base_known:
        sandbox_user = base_user
    elif environment.docker_image is not None and resolved is not None and (base_known or dockerfile_user):
        sandbox_user = dockerfile_user
    else:
        sandbox_user = user
    if dockerfile.parent.is_dir():
        # The Compose loader needs the context id (COMPOSE_PROJECT_NAME) and the built sidecars at every seed: for a
        # task with a Compose file, and for a plain image task when the deployment's job-level compose_overlay names
        # Compose files, a run-config value the loader cannot see here. So every task with an environment/ folder
        # pays the hash once, at load, as Harbor computes its environment id at every trial start (``environment_id``
        # in harbor/environments/base.py, the ``hb__<id>`` main image name of ``docker.py``), never per seed.
        recorded_builds()

    # The separate verifier's image, by Harbor's rules (effective_verifier_environment, verifier_build_context): a
    # separate-mode task loads with an image for its verifier or not at all, as Harbor runs it or refuses it.
    verifier_image: str | None = None
    verifier_image_config: BaseImageConfig | None = None
    verifier_environment = effective_verifier_environment(config)
    if verifier_environment is not None:
        # A file the two readers below cannot read as written (not YAML, not a mapping, or a top-level key other than
        # services, an include whose files Compose merges into Harbor's verifier project or a networks block that
        # reshapes its default network) is refused before they read the services mapping alone.
        refusal = verifier_compose_refusal(path)
        if refusal:
            raise HarborTaskError(refusal)
        sidecars = verifier_compose_sidecars(path)
        if sidecars:
            raise HarborTaskError(
                f"{path / TESTS_COMPOSE_FILE} declares service(s) {', '.join(sidecars)} beside main; Harbor starts them "
                "as the separate verifier's Compose project and this server starts a single verifier sandbox, so the "
                "task is not supported until it does"
            )
        overrides = verifier_compose_main_overrides(path, prebuilt=bool(verifier_environment.docker_image))
        if overrides:
            raise HarborTaskError(
                f"{path / TESTS_COMPOSE_FILE} sets {', '.join(overrides)} on main; Harbor applies the file to the "
                "separate verifier's container and this server starts that verifier from task.toml's environment and "
                "the verifier image alone (main's build section and image are read for a verifier built from "
                "tests/Dockerfile only), so the task is not supported until it does"
            )
        if verifier_environment.docker_image:
            # Harbor's prebuilt verifier: the declared block's image, or the task's own when the effective environment
            # is a copy of [environment] (verifier_image stays None; the server runs the task image then).
            if config.verifier.environment is not None:
                verifier_image = verifier_environment.docker_image
        else:
            verifier_context = verifier_build_context(path, config)
            if verifier_context is None:
                raise _verifier_definition_required(path)
            # Harbor's hash of tests/, paid once per task like environment/'s: the key of the verifier's record.
            verifier_record = read_built_images(root).get(context_digest(verifier_context))
            verifier_image = recorded_service_images(verifier_record).get(VERIFIER_SERVICE)
            if verifier_image is None:
                raise _verifier_build_required(path, root)
        if verifier_image is not None:
            # The verifier image's recorded configuration: the server starts its entrypoint and polls its HEALTHCHECK
            # as it does the task image's, since Harbor `up --wait`s the verifier's image whether it built it or
            # pulled it. A built verifier image is always recorded; a declared one is when its reference is a
            # recorded base image (this task's own Dockerfile's, overlay or not, or another task's). Otherwise
            # nothing is recorded, as for a prebuilt docker_image, and nothing starts in it.
            verifier_config_record = read_image_configs(root).get(verifier_image)
            if verifier_config_record is not None:
                try:
                    verifier_image_config = BaseImageConfig.from_record(verifier_config_record)
                except ValueError as exc:
                    raise HarborTaskError(f"{root / IMAGE_CONFIGS_FILE}: {verifier_image!r}: {exc}") from exc

    if any(turn.verifier_mode == "separate" for turn in turns):
        turns = tuple(_resolve_turn_verifiers(path, root, config, turns))

    try:
        folder_digest = content_hash(path)
    except (OSError, ValueError) as exc:
        raise HarborTaskError(f"{path}: {exc}") from exc
    # A dataset-level verifier grades the task from outside its folder, so its content enters the task's digest.
    shared_verifier = python_verifier if python_verifier is not None and python_verifier.parent == root else None
    try:
        verifier_digest = hashlib.sha256(shared_verifier.read_bytes()).hexdigest() if shared_verifier else None
    except OSError as exc:
        raise HarborTaskError(f"{shared_verifier}: {exc}") from exc
    digest = task_digest(folder_digest, dataset, path.name, shared_verifier_digest=verifier_digest)
    task_data = read_task_fields(path)

    return HarborTask(
        path=path,
        task_id=path.name,
        config=config,
        instruction=instruction,
        digest=digest,
        folder_digest=folder_digest,
        image=image,
        workdir=workdir,
        env=env,
        user=user,
        overlay=overlay,
        sandbox_user=sandbox_user,
        built_image=built,
        context_id=context_id,
        built_images=built_images,
        verifier_image=verifier_image,
        entrypoint=entrypoint,
        entrypoint_user=entrypoint_user,
        image_shell=image_shell,
        entrypoint_workdir=entrypoint_workdir,
        image_healthcheck=image_healthcheck,
        verifier_image_config=verifier_image_config,
        overlay_workdirs=overlay_workdirs,
        turns=turns,
        task_data=task_data,
        python_verifier=python_verifier,
        seed_trajectory=seed_trajectory,
    )


def task_digest(
    folder_digest: str, dataset: GymDatasetConfig, task_id: str, *, shared_verifier_digest: str | None = None
) -> str:
    """The digest a task is materialized and served under: its folder's content hash plus the dataset slice applied to it.

    ``dataset.toml``'s ``[gym]`` table can change the effective task without touching the folder, so the
    defaults and the task's own override are folded in whenever the table shapes tasks
    (:attr:`GymDatasetConfig.shapes_tasks`). A dataset whose table shapes none, because it has no table or
    declares only identity, run and catalog keys (``license``, ``type``, ``allowed_agents``, ``domain``,
    ``verified``, ``[gym.defaults].num_repeats``), keeps the plain folder hash, which is also what the
    package store checks: none of those keys says what the task is, so none enters its digest.
    ``shared_verifier_digest`` is the digest of the dataset root's ``verifier.py`` when it grades the task:
    grading code outside the folder is part of the task's identity too.
    """
    if not dataset.shapes_tasks and shared_verifier_digest is None:
        return folder_digest
    # `num_repeats` is how often a run plays the task, not what the task is: left out of the slice too, so a
    # shaping table with a repeat count hashes the same as one without.
    applied: dict[str, Any] = {
        "defaults": dataset.defaults.model_dump(mode="json", exclude={"num_repeats"}),
        "task": (dataset.tasks.get(task_id) or GymTaskOverride()).model_dump(mode="json"),
    }
    if shared_verifier_digest is not None:
        applied[VERIFIER_FILE] = shared_verifier_digest
    return hashlib.sha256(f"{folder_digest}\n{json.dumps(applied, sort_keys=True)}".encode()).hexdigest()


def iter_task_folders(root: Path, *, max_depth: int = MAX_DISCOVERY_DEPTH) -> list[Path]:
    """Every task folder under ``root``, sorted by path: a folder with ``task.toml`` is a task, any other
    folder is searched in turn, up to ``max_depth`` levels down. Hidden folders are skipped and symbolic
    links are never followed (a linked task folder is listed, and ``load_task`` refuses it).

    A task's ID is its folder name, so two task folders of the same name anywhere under ``root`` raise
    :class:`HarborTaskError` here, before any caller (base-image recording, Dockerfile builds, discovery)
    does work keyed by that name."""
    root = Path(root)
    found: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        for child in sorted(folder.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if is_task_folder(child):
                # A symlinked task folder is listed so that load_task reports it; the walk never follows links.
                found.append(child)
            elif depth < max_depth and not child.is_symlink():
                walk(child, depth + 1)

    walk(root, 1)
    seen: dict[str, Path] = {}
    for child in found:
        if child.name in seen:
            raise HarborTaskError(
                f"Duplicate Harbor task ID {child.name!r} under {root}: {seen[child.name]} and {child}"
            )
        seen[child.name] = child
    return found


def _seed_trajectory(path: Path, turns: tuple[HarborTurn, ...]) -> Path | None:
    """The seed trajectory the task ships, relative to the task folder, or ``None``; warns when there is one.

    Harbor 0.22.0 loads ``trajectory.json`` beside the instruction (``steps/<first step>/trajectory.json`` for a
    multi-step task, where a top-level one is ignored: ``Trial._task_trajectory_path`` picks one path or the other,
    never both) into the agent's session as prior context, as its ``--load-trajectory`` option does, for every
    agent but the oracle and nop (the oracle runs the task's own solution, so the prior context does not apply to
    it, and task authors validate with it); an agent that cannot load an ATIF trajectory fails the trial before any
    environment spend. This server hands no prior context to a harness yet, so running the task for a model would
    grade it without the context the task was written for: the task loads, the oracle runs it as Harbor's does,
    and the environment server refuses it at seed for any other agent, naming the file. The file stays part of
    the task's identity and its package (:func:`~nemo_gym.tasks.harbor.digest.content_hash` covers it), so a
    fetched task verifies against its advertised hash.
    """
    seed = path / STEPS_DIR / turns[0].name / TRAJECTORY_FILE if turns else path / TRAJECTORY_FILE
    if not seed.is_file():
        return None
    where = seed.relative_to(path)
    logger.warning(
        "%s: %s is Harbor's seed trajectory (the prior context Harbor loads into the agent's session before its first "
        "turn, as --load-trajectory does, for every agent but the oracle and nop); loading a trajectory into a harness "
        "is not supported yet, so the environment server runs the task for the oracle alone and refuses it for any "
        "other agent. Remove the file to run the task without its prior context",
        path,
        where.as_posix(),
    )
    return where


def _is_root_user(user: str | int | None) -> bool:
    """Whether an agent user runs with uid 0 (``None`` is the image's own user, root in most images)."""
    return user is None or str(user).split(":")[0] in ("root", "0")


def _load_turns(path: Path, config: HarborTaskConfig) -> tuple[HarborTurn, ...]:
    """Read every ``[[steps]]`` entry as a turn, refusing the per-step features the server does not run yet.

    Each turn needs ``steps/<name>/instruction.md`` and a ``test.sh``, its own under
    ``steps/<name>/tests/`` or the task's under ``tests/``, which is the script Harbor runs for the step.
    A step's own ``[steps.agent].user`` is read as written: the server switches the sandbox's default user,
    the working directory's owner and the harness's user to it for that turn (and refuses a ``name:group``
    pair at seed on a provider that switches user by name only, as it refuses the task's own). A step's
    ``workdir/``, ``workdir/setup.sh``, ``[steps.healthcheck]`` and ``artifacts`` are read by the server per turn
    (:attr:`HarborTurn.workdir_dir`, :attr:`HarborTurn.setup_script`, ``HarborTaskStep.healthcheck`` and
    ``HarborTaskStep.artifacts``, collected after the turn's agent with the task's); a ``setup.sh``
    beside ``instruction.md`` instead of inside ``workdir/`` is not a Harbor file and is warned about, since Harbor
    0.22.0 runs ``steps/<name>/workdir/setup.sh`` only. A step without its own ``[steps.verifier].timeout_sec`` is
    graded within Harbor's step default of 600 s (times the dataset's ``timeout_multiplier``, like every budget),
    not the task's ``[verifier].timeout_sec`` (:meth:`HarborTaskConfig.step_verifier_timeout_sec`); when the task's
    figure differs, the steps it does not reach are warned about once, since an author who set it for the steps
    would expect it to apply.
    """
    turns: list[HarborTurn] = []
    # The agent users of the steps so far that are not root (`None` is the image's own user, unknown here).
    earlier_unprivileged: list[tuple[str, str]] = []
    # The steps graded within Harbor's 600 s step default (under the dataset's multiplier, as the task's figure is)
    # while the task's [verifier].timeout_sec says otherwise.
    default_timeout_steps: list[str] = []
    step_default = HARBOR_STEP_VERIFIER_TIMEOUT_SEC
    for step in config.steps:
        where = f"{path / TASK_FILE} step {step.name!r}"
        step_dir = path / STEPS_DIR / step.name
        if not step_dir.is_dir():
            raise HarborTaskError(f"{where}: {step_dir} does not exist")
        if step.verifier.timeout_sec is None:
            step_default = config.step_verifier_timeout_sec(step)
            if config.verifier.timeout_sec != step_default:
                default_timeout_steps.append(step.name)
        user = step.agent.user if step.agent.user is not None else config.agent.user
        if (step_dir / STEP_WORKDIR / STEP_SETUP_SCRIPT).is_file() and _is_root_user(user) and earlier_unprivileged:
            # The script runs as the step's agent, root here, in the working directory an earlier step's agent
            # owned: a process that agent left running can replace the uploaded script before it runs. Harbor
            # shares the window (docker cp, then exec); the step's agent gets root right after anyway.
            previous, previous_user = earlier_unprivileged[-1]
            logger.warning(
                "%s: steps/%s/%s/%s runs as %s after step %r ran as %r; a process that step left running in the "
                "working directory it owned can replace the script between its upload and its run, so the script "
                "must not be relied on to undo what the earlier step did (Harbor runs it the same way)",
                where,
                step.name,
                STEP_WORKDIR,
                STEP_SETUP_SCRIPT,
                "root" if user is not None else "the image's own user (root unless the image sets USER)",
                previous,
                previous_user,
            )
        if not _is_root_user(user):
            earlier_unprivileged.append((step.name, str(user)))
        if (step_dir / STEP_SETUP_SCRIPT).exists():
            logger.warning(
                "%s: steps/%s/%s is not run; Harbor runs steps/<name>/%s/%s (uploaded with the step's workdir/)",
                where,
                step.name,
                STEP_SETUP_SCRIPT,
                STEP_WORKDIR,
                STEP_SETUP_SCRIPT,
            )
        if not (step_dir / "tests" / "test.sh").is_file() and not (path / "tests" / "test.sh").is_file():
            raise HarborTaskError(f"{where}: no test.sh in steps/{step.name}/tests/ or tests/")
        instruction_path = step_dir / INSTRUCTION_FILE
        if not instruction_path.is_file():
            raise HarborTaskError(f"{where}: steps/{step.name}/{INSTRUCTION_FILE} is missing")
        try:
            instruction = read_instruction(instruction_path)
        except (OSError, UnicodeDecodeError) as exc:
            raise HarborTaskError(f"{instruction_path}: {exc}") from exc
        turns.append(HarborTurn(instruction=instruction, config=step, verifier_mode=step_verifier_mode(config, step)))
    if default_timeout_steps:
        logger.warning(
            "%s: step(s) %s name no [steps.verifier].timeout_sec and are graded within Harbor's step default of %gs%s, "
            "not the task's [verifier].timeout_sec = %gs (Harbor's StepConfig.verifier is a VerifierConfig of its own, "
            "whose timeout never falls back to the task's); set [steps.verifier].timeout_sec on each step that needs "
            "another bound",
            path / TASK_FILE,
            ", ".join(repr(name) for name in default_timeout_steps),
            step_default,
            ""
            if step_default == HARBOR_STEP_VERIFIER_TIMEOUT_SEC
            else f" ({HARBOR_STEP_VERIFIER_TIMEOUT_SEC:g}s times the dataset's timeout_multiplier)",
            config.verifier.timeout_sec,
        )
    return tuple(turns)


def _resolve_turn_verifiers(
    path: Path, root: Path, config: HarborTaskConfig, turns: tuple[HarborTurn, ...]
) -> list[HarborTurn]:
    """The separate turns' verifier images, by the rules :func:`load_task` applies to a single-step task's.

    Harbor starts a step's separate verifier from ``steps/<name>/tests/`` when that folder exists, else from
    ``tests/`` (``_verifier_env_build_context``), with the step's effective environment
    (:func:`step_verifier_environment`): a declared ``docker_image`` is run as it is, otherwise the folder's
    ``Dockerfile`` is built. Prepare builds ``tests/Dockerfile`` only, so a step whose verifier would be built from
    its own ``steps/<name>/tests/`` is refused naming the image to declare; one built from ``tests/`` reads the
    recorded build, as the task's does. The ``docker-compose.yaml`` of the folder the step starts from (the step's
    own ``tests/`` or the task's, which Harbor stacks on the step verifier's template, ``_docker_compose_paths``) is
    refused as the task's verifier's is: one that is not read as written (not valid YAML, not a mapping, or carrying a
    top-level key other than ``services``, ``name``, ``version`` and ``x-`` extensions, an ``include`` or a ``networks``
    block among them: :func:`verifier_compose_refusal`), services beside ``main``, and a runtime setting on
    ``main`` (:func:`verifier_compose_main_overrides`; ``main``'s ``image``, its ``build`` section and a
    ``pull_policy`` of ``build`` are ones beside a step verifier with a ``docker_image``), which would reach Harbor's
    step verifier and never this server's.
    """
    resolved: list[HarborTurn] = []
    image_configs = None
    for turn in turns:
        if turn.verifier_mode != "separate":
            resolved.append(turn)
            continue
        where = f"{path / TASK_FILE} step {turn.name!r}"
        environment = step_verifier_environment(config, turn.config)
        assert environment is not None  # the turn is separate
        context = step_verifier_build_context(path, turn.config)
        refusal = verifier_compose_refusal(context.parent) if context.name == "tests" else None
        if refusal:
            raise HarborTaskError(f"{where}: {refusal}")
        sidecars = verifier_compose_sidecars(context.parent) if context.name == "tests" else []
        if sidecars:
            raise HarborTaskError(
                f"{where}: {context / TESTS_COMPOSE_FILE.name} declares service(s) {', '.join(sidecars)} beside main; "
                "Harbor starts them as the separate verifier's Compose project and this server starts a single "
                "verifier sandbox, so the task is not supported until it does"
            )
        overrides = (
            verifier_compose_main_overrides(context.parent, prebuilt=bool(environment.docker_image))
            if context.name == "tests"
            else []
        )
        if overrides:
            raise HarborTaskError(
                f"{where}: {context / TESTS_COMPOSE_FILE.name} sets {', '.join(overrides)} on main; Harbor applies the "
                "file to the step's separate verifier container and this server starts that verifier from task.toml's "
                "environment and the verifier image alone (main's build section and image are read for a verifier "
                "built from tests/Dockerfile only), so the task is not supported until it does"
            )
        verifier_image: str | None = None
        if environment.docker_image:
            # A declared image (the step's or the task's [verifier.environment]) runs as it is; a copy of [environment]
            # with a docker_image runs the task's own image (verifier_image stays None), as for a single-step task.
            if turn.config.verifier.environment is not None or config.verifier.environment is not None:
                verifier_image = environment.docker_image
        elif context != path / TESTS_DOCKERFILE.parent:
            raise HarborTaskError(
                f"{where}: the step's separate verifier names no docker_image, so Harbor builds it from "
                f"{context / TESTS_DOCKERFILE.name} (the step's own tests/ folder is its environment dir); this server "
                "builds tests/Dockerfile only. Declare [steps.verifier.environment].docker_image (or "
                "[verifier.environment].docker_image) with a prebuilt image"
            )
        else:
            if not (path / TESTS_DOCKERFILE).is_file():
                raise _verifier_definition_required(path)
            verifier_record = read_built_images(root).get(context_digest(context))
            verifier_image = recorded_service_images(verifier_record).get(VERIFIER_SERVICE)
            if verifier_image is None:
                raise _verifier_build_required(path, root)
        verifier_image_config: BaseImageConfig | None = None
        if verifier_image is not None:
            if image_configs is None:
                image_configs = read_image_configs(root)
            record = image_configs.get(verifier_image)
            if record is not None:
                try:
                    verifier_image_config = BaseImageConfig.from_record(record)
                except ValueError as exc:
                    raise HarborTaskError(f"{root / IMAGE_CONFIGS_FILE}: {verifier_image!r}: {exc}") from exc
        resolved.append(replace(turn, verifier_image=verifier_image, verifier_image_config=verifier_image_config))
    return resolved


def discover_tasks(root: Path, *, skipped: dict[str, HarborTaskError] | None = None) -> list[HarborTask]:
    """A folder with ``task.toml`` is one task; otherwise its subfolders are searched for tasks.

    The search is recursive (see :func:`iter_task_folders`), so a dataset may group its tasks in
    folders; ``task_id`` stays the task folder's own name and must be unique across the dataset.
    Dataset-level files (``dataset.toml``, the recorded image configurations) are read from ``root``.

    With ``skipped``, a child that fails to load is recorded there under its folder name
    and the others still load; without it the first failure raises. A root that is
    itself a task always raises.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise HarborTaskError(f"{root} is not a directory")
    if is_task_folder(root):
        return [load_task(root)]
    children = iter_task_folders(root)
    if not children:
        raise HarborTaskError(f"{root} is neither a task folder nor a folder of task folders (no {TASK_FILE} found)")
    # The recorded image configurations are shared by every child: read once, not once per task.
    # An unreadable file is left for each load to report under its own task, as before.
    try:
        image_configs: Mapping[str, Any] | None = read_image_configs(root)
    except HarborTaskError:
        image_configs = None
    if skipped is None:
        return [load_task(child, root=root, image_configs=image_configs) for child in children]
    tasks: list[HarborTask] = []
    for child in children:
        try:
            tasks.append(load_task(child, root=root, image_configs=image_configs))
        except HarborTaskError as exc:
            skipped[child.name] = exc
    if not tasks:
        raise HarborTaskError(f"No task under {root} loads; first error: {next(iter(skipped.values()))}")
    return tasks
