# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import hashlib
import json
import logging
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
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
from nemo_gym.tasks.harbor.models import HarborTaskConfig, HarborTaskStep
from nemo_gym.tasks.harbor.records import BUILT_IMAGES_FILE, IMAGE_CONFIGS_FILE, read_built_images, read_image_configs


logger = logging.getLogger(__name__)

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
STEPS_DIR = "steps"
# Per-step Harbor features this loader does not run yet; a task using one is refused by name.
UNSUPPORTED_STEP_FILES = ("workdir", "setup.sh")
DOCKERFILE = Path("environment") / "Dockerfile"


@dataclass(frozen=True)
class HarborTurn:
    """One task step loaded with its ``steps/<name>/`` folder, which the harbor server runs as one agent turn.

    ``config`` is the step as ``task.toml`` writes it (:class:`HarborTaskStep`) and ``instruction`` is
    ``steps/<name>/instruction.md``, so ``HarborTaskStep`` is to ``HarborTurn`` what ``HarborTaskConfig`` is to
    ``HarborTask``. One turn per task step is this server's choice, as in Harbor's multi-step trial; a task
    step does not imply a single turn in general.
    """

    instruction: str
    config: HarborTaskStep

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tests_dir(self) -> Path:
        """Relative to the task folder; may be absent, in which case the task's ``tests/`` is used alone."""
        return Path(STEPS_DIR) / self.name / "tests"


@dataclass(frozen=True)
class HarborTask:
    """A task folder read into memory.

    ``task_id`` is the folder name: ``[task].name`` is an ``org/name`` label that is
    not unique across a dataset. ``image`` is the prebuilt image when the task
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
    ``[agent].user``; for a declared ``docker_image`` built from a pull-mode Dockerfile it is that
    Dockerfile's final ``USER``. When neither is known the task's ``user`` stands in for it.


    ``built_image`` is the image prepare built from the task's ``environment/`` folder when the
    Dockerfile needed a build (``built-images.json``); ``image`` is then that image and the task
    runs in pull mode from it. ``None`` in the other two modes and for a prebuilt image.

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

    ``turns`` holds the task's ``[[steps]]`` in order; empty for a single-turn task. A multi-turn task
    never sends ``instruction`` to the agent: each turn sends its own, and the top-level file may hold
    material the agent must not see (SWE-Interact puts the full hidden specification there).
    """

    path: Path
    task_id: str
    config: HarborTaskConfig
    instruction: str
    digest: str
    image: str | None
    workdir: str | None
    env: dict[str, str]
    user: str | None
    overlay: tuple[OverlayRun, ...] = ()
    sandbox_user: str | None = None
    built_image: str | None = None
    entrypoint: tuple[str, ...] | None = None
    entrypoint_user: str | None = None
    image_shell: tuple[str, ...] | None = None
    entrypoint_workdir: str | None = None
    image_healthcheck: dict[str, Any] | None = None
    overlay_workdirs: tuple[OverlayWorkdir, ...] = ()
    turns: tuple[HarborTurn, ...] = ()

    @property
    def needs_sandbox(self) -> bool:
        """A task runs in a sandbox exactly when it declares an image or a Dockerfile."""
        return self.image is not None or (self.path / DOCKERFILE).is_file()

    @property
    def needs_compose(self) -> bool:
        """The task's environment is a Compose group: ``main`` plus the sidecars its overlay declares."""
        return any(
            (self.path / "environment" / name).is_file()
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
        )

    @property
    def has_solution(self) -> bool:
        return (self.path / "solution" / "solve.sh").is_file()


def is_task_folder(path: Path) -> bool:
    return (Path(path) / TASK_FILE).is_file()


def built_image(task_folder: Path) -> str | None:
    """The image prepare built for the task's Dockerfile, when ``built-images.json`` records one."""
    task_folder = Path(task_folder)
    context = task_folder / DOCKERFILE.parent
    if not context.is_dir():
        return None
    entry = read_built_images(task_folder.parent).get(context_digest(context))
    image = entry.get("image") if isinstance(entry, dict) else None
    return str(image) if image else None


def _build_required(dockerfile: Path, reason: Exception) -> HarborTaskError:
    return HarborTaskError(
        f"{dockerfile} needs a build ({reason}) and no built image is recorded in "
        f"{dockerfile.parent.parent.parent / BUILT_IMAGES_FILE}. `gym eval run` and `gym dataset validate` build it "
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


def load_task(path: Path, *, image_configs: Mapping[str, Any] | None = None) -> HarborTask:
    """Read ``path`` as one task. Raises :class:`HarborTaskError` when it is not one.

    ``image_configs`` is the parent folder's recorded image configurations when the caller has
    read them already (``discover_tasks`` reads the shared file once); otherwise they are read here.
    """
    path = Path(path)
    if path.is_symlink():
        raise HarborTaskError(f"Task folder must not be a symlink: {path}")
    path = path.resolve()
    if not is_task_folder(path):
        raise HarborTaskError(f"{path} has no {TASK_FILE}")
    try:
        data = tomllib.loads((path / TASK_FILE).read_text())
        for key in HarborTaskConfig.unknown_keys(data):
            logger.warning("%s: ignoring unknown key `%s`", path / TASK_FILE, key)
        config = HarborTaskConfig.model_validate(data)
    except (OSError, ValueError) as exc:
        raise HarborTaskError(f"{path / TASK_FILE}: {exc}") from exc
    # The dataset's own settings sit beside the task folders and shape the effective task.toml.
    try:
        dataset = read_dataset_config(path.parent)
        config = apply_dataset_config(config, path.name, dataset)
    except ValueError as exc:
        raise HarborTaskError(str(exc)) from exc

    phases = {"environment": config.environment, "agent": config.agent, "verifier": config.verifier}
    if config.verifier.environment is not None:
        phases["verifier.environment"] = config.verifier.environment
    for step in config.steps:
        phases[f"steps.{step.name}.agent"] = step.agent
        phases[f"steps.{step.name}.verifier"] = step.verifier
    for name, phase in phases.items():
        if phase.network_mode not in (None, "public") or phase.allowed_hosts:
            raise HarborTaskError(f"{path}: {name} network restrictions are not supported yet")
    for name, environment in (
        ("environment", config.environment),
        ("verifier.environment", config.verifier.environment),
    ):
        if environment is not None and environment.os != "linux":
            raise HarborTaskError(f"{path}: {name} Windows tasks are not supported yet")

    instruction_path = path / INSTRUCTION_FILE
    turns = _load_turns(path, config)
    # Harbor reads a multi-step task's instructions from its steps; the top-level file is then optional.
    if not turns and not instruction_path.is_file():
        raise HarborTaskError(f"{path} has no {INSTRUCTION_FILE}")
    if not turns and not (path / "tests").is_dir():
        raise HarborTaskError(f"{path} has no tests/ folder")
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
            # declares docker_image runs it, unless a build of its own Dockerfile is recorded (`builds.force`,
            # Harbor's --force-build), which then wins as it does in Harbor.
            built = built_image(path)
            if built is None and image is None:
                raise _build_required(dockerfile, exc) from exc
            parsed = None
        if built is not None:
            # The built image is pulled like a prebuilt one; its recorded configuration gives the
            # Dockerfile's final WORKDIR, ENV and USER.
            record = read_image_configs(path.parent).get(built)
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
            configs = image_configs if image_configs is not None else read_image_configs(path.parent)
            record = configs.get(parsed.image)
            try:
                base = BaseImageConfig.from_record(record) if record is not None else None
            except ValueError as exc:
                raise HarborTaskError(f"{path.parent / IMAGE_CONFIGS_FILE}: {parsed.image!r}: {exc}") from exc
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
                    f"in {path.parent / IMAGE_CONFIGS_FILE}. Prepare the dataset again (`gym eval run` or "
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
                    built = built_image(path)
                    if built is None:
                        raise _build_required(dockerfile, exc) from exc
                    record = read_image_configs(path.parent).get(built)
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
    # The default user of the image the sandbox starts from (see HarborTask.sandbox_user).
    if environment.docker_image is None and base_known:
        sandbox_user = base_user
    elif environment.docker_image is not None and resolved is not None and (base_known or dockerfile_user):
        sandbox_user = dockerfile_user
    else:
        sandbox_user = user

    try:
        digest = task_digest(content_hash(path), dataset, path.name)
    except (OSError, ValueError) as exc:
        raise HarborTaskError(f"{path}: {exc}") from exc

    return HarborTask(
        path=path,
        task_id=path.name,
        config=config,
        instruction=instruction,
        digest=digest,
        image=image,
        workdir=workdir,
        env=env,
        user=user,
        overlay=overlay,
        sandbox_user=sandbox_user,
        built_image=built,
        entrypoint=entrypoint,
        entrypoint_user=entrypoint_user,
        image_shell=image_shell,
        entrypoint_workdir=entrypoint_workdir,
        image_healthcheck=image_healthcheck,
        overlay_workdirs=overlay_workdirs,
        turns=turns,
    )


def _load_turns(path: Path, config: HarborTaskConfig) -> tuple[HarborTurn, ...]:
    """Read every ``[[steps]]`` entry as a turn, refusing the per-step features the server does not run yet.

    Each turn needs ``steps/<name>/instruction.md`` and a ``test.sh``, its own under
    ``steps/<name>/tests/`` or the task's under ``tests/``, which is the script Harbor runs for the step.
    """
    turns: list[HarborTurn] = []
    for step in config.steps:
        where = f"{path / TASK_FILE} step {step.name!r}"
        step_dir = path / STEPS_DIR / step.name
        if not step_dir.is_dir():
            raise HarborTaskError(f"{where}: {step_dir} does not exist")
        for name in UNSUPPORTED_STEP_FILES:
            if (step_dir / name).exists():
                raise HarborTaskError(f"{where}: steps/{step.name}/{name} is not supported yet")
        if step.healthcheck is not None or step.artifacts:
            raise HarborTaskError(f"{where}: per-step healthcheck and artifacts are not supported yet")
        if step.verifier.environment_mode == "separate" or step.verifier.environment is not None:
            raise HarborTaskError(f"{where}: separate per-step verification is not supported yet")
        if not (step_dir / "tests" / "test.sh").is_file() and not (path / "tests" / "test.sh").is_file():
            raise HarborTaskError(f"{where}: no test.sh in steps/{step.name}/tests/ or tests/")
        instruction_path = step_dir / INSTRUCTION_FILE
        if not instruction_path.is_file():
            raise HarborTaskError(f"{where}: steps/{step.name}/{INSTRUCTION_FILE} is missing")
        try:
            instruction = read_instruction(instruction_path)
        except (OSError, UnicodeDecodeError) as exc:
            raise HarborTaskError(f"{instruction_path}: {exc}") from exc
        turns.append(HarborTurn(instruction=instruction, config=step))
    if turns and not config.is_shared_verifier:
        raise HarborTaskError(f"{path / TASK_FILE}: multi-step tasks with a separate verifier are not supported yet")
    return tuple(turns)


def task_digest(folder_digest: str, dataset: GymDatasetConfig, task_id: str) -> str:
    """The digest a task is materialized and served under: its folder's content hash plus the dataset slice applied to it.

    ``dataset.toml``'s ``[gym]`` table changes the effective task without touching the folder, so the
    defaults and the task's own override are folded in whenever the table is not empty. A dataset
    without a ``[gym]`` table keeps the plain folder hash, which is also what the package store checks.
    """
    if dataset.is_empty:
        return folder_digest
    applied = {
        "defaults": dataset.defaults.model_dump(mode="json"),
        "task": (dataset.tasks.get(task_id) or GymTaskOverride()).model_dump(mode="json"),
    }
    return hashlib.sha256(f"{folder_digest}\n{json.dumps(applied, sort_keys=True)}".encode()).hexdigest()


def discover_tasks(root: Path, *, skipped: dict[str, HarborTaskError] | None = None) -> list[HarborTask]:
    """A folder with ``task.toml`` is one task; otherwise its direct children are the tasks.

    With ``skipped``, a child that fails to load is recorded there under its folder name
    and the others still load; without it the first failure raises. A root that is
    itself a task always raises.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise HarborTaskError(f"{root} is not a directory")
    if is_task_folder(root):
        return [load_task(root)]
    children = sorted(child for child in root.iterdir() if child.is_dir() and is_task_folder(child))
    if not children:
        raise HarborTaskError(f"{root} is neither a task folder nor a folder of task folders (no {TASK_FILE} found)")
    # The recorded image configurations are shared by every child: read once, not once per task.
    # An unreadable file is left for each load to report under its own task, as before.
    try:
        image_configs: Mapping[str, Any] | None = read_image_configs(root)
    except HarborTaskError:
        image_configs = None
    if skipped is None:
        return [load_task(child, image_configs=image_configs) for child in children]
    tasks: list[HarborTask] = []
    for child in children:
        try:
            tasks.append(load_task(child, image_configs=image_configs))
        except HarborTaskError as exc:
            skipped[child.name] = exc
    if not tasks:
        raise HarborTaskError(f"No task under {root} loads; first error: {next(iter(skipped.values()))}")
    return tasks
