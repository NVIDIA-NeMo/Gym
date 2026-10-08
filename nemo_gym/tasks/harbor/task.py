# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import json
import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    OverlayRun,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.tasks.harbor.models import HarborTaskConfig


logger = logging.getLogger(__name__)

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
DOCKERFILE = Path("environment") / "Dockerfile"
# The OCI configurations recorded at prepare next to the task folders, keyed by image reference as the
# tasks wrote it (see ``nemo_gym.tasks.harbor.image_configs``): Compose sidecar images and the base image
# of every Dockerfile the loader resolves.
IMAGE_CONFIGS_FILE = "compose-images.json"


class HarborTaskError(ValueError):
    """The folder is not a runnable Harbor task."""


@dataclass(frozen=True)
class HarborTask:
    """A task folder read into memory.

    ``task_id`` is the folder name: ``[task].name`` is an ``org/name`` label that is
    not unique across a dataset. ``image`` is the prebuilt image when the task
    declares one, else the base image of a pull-mode or overlay-mode Dockerfile, else
    ``None`` (a build is required). ``workdir``, ``env`` and ``user`` merge ``task.toml``
    with what the Dockerfile recorded; ``task.toml`` wins.

    Dockerfile ``ENV``, ``WORKDIR`` and ``USER`` are resolved to literals as ``docker build``
    resolves them, starting from the base image's configuration recorded in
    :data:`IMAGE_CONFIGS_FILE` next to the task folders (see
    :mod:`nemo_gym.tasks.harbor.dockerfile`).

    ``overlay`` is Gym-derived data, not a Harbor field: the Dockerfile's ``RUN`` lines
    (with the ``WORKDIR``, ``ENV`` and ``USER`` in effect for each) that the server runs
    in the pulled base image at seed. Empty for a prebuilt image or a pull-mode Dockerfile.

    ``sandbox_user`` is the user the sandbox runs commands as when no user is asked for: the
    image the sandbox starts from is the Dockerfile's base image, so it is that image's recorded
    ``User`` (``None`` is root), not the Dockerfile's final ``USER`` or ``[agent].user``. When the
    base image's configuration is not known (a prebuilt image, or a Dockerfile resolved without
    it), the task's ``user`` stands in for it.
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


def read_image_configs(folder: Path) -> dict[str, Any]:
    """The image configurations recorded in ``folder`` (the tasks' parent), or ``{}`` when none were."""
    path = Path(folder) / IMAGE_CONFIGS_FILE
    if not path.is_file():
        return {}
    try:
        recorded = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise HarborTaskError(f"{path} is not valid JSON: {exc}") from exc
    return recorded if isinstance(recorded, dict) else {}


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


def load_task(path: Path) -> HarborTask:
    """Read ``path`` as one task. Raises :class:`HarborTaskError` when it is not one."""
    path = Path(path).resolve()
    if not is_task_folder(path):
        raise HarborTaskError(f"{path} has no {TASK_FILE}")
    try:
        data = tomllib.loads((path / TASK_FILE).read_text())
        for key in HarborTaskConfig.unknown_keys(data):
            logger.warning("%s: ignoring unknown key `%s`", path / TASK_FILE, key)
        config = HarborTaskConfig.model_validate(data)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise HarborTaskError(f"{path / TASK_FILE}: {exc}") from exc
    # The dataset's own settings sit beside the task folders and shape the effective task.toml.
    try:
        config = apply_dataset_config(config, path.name, read_dataset_config(path.parent))
    except ValueError as exc:
        raise HarborTaskError(str(exc)) from exc
    instruction_path = path / INSTRUCTION_FILE
    if not instruction_path.is_file():
        raise HarborTaskError(f"{path} has no {INSTRUCTION_FILE}")
    if not (path / "tests").is_dir():
        raise HarborTaskError(f"{path} has no tests/ folder")
    try:
        instruction = read_instruction(instruction_path)
    except UnicodeDecodeError as exc:
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
    base_known = False
    base_user: str | None = None
    dockerfile = path / DOCKERFILE
    if dockerfile.is_file():
        try:
            parsed = parse_dockerfile(dockerfile.read_text())
        except DockerfileNeedsBuild as exc:
            if image is None:
                raise HarborTaskError(
                    f"{dockerfile} needs a build ({exc}); building Dockerfiles is not supported yet. "
                    "Declare [environment].docker_image with a prebuilt image to run this task."
                ) from exc
            parsed = None
        if parsed is not None and (image is None or not parsed.has_runs):
            # A prebuilt image already holds whatever its Dockerfile's RUN lines did; only a
            # pull-mode Dockerfile contributes settings next to a declared image.
            record = read_image_configs(path.parent).get(parsed.image)
            base = BaseImageConfig.from_record(record) if record is not None else None
            base_known = base is not None
            base_user = base.user if base is not None else None
            try:
                resolved = resolve_dockerfile(parsed, base)
            except ImageConfigRequired as exc:
                raise HarborTaskError(
                    f"{dockerfile}: {exc}, and the configuration of base image {parsed.image!r} is not recorded "
                    f"in {path.parent / IMAGE_CONFIGS_FILE}. Prepare the dataset again (`gym eval run` or "
                    "`gym dataset validate`) to record it."
                ) from exc
            except DockerfileNeedsBuild as exc:
                raise HarborTaskError(f"{dockerfile} needs a build ({exc})") from exc
            image = image or resolved.image
            workdir = workdir or resolved.workdir
            env = resolved.env | env
            user = resolved.user
            overlay = resolved.runs
    if config.agent.user is not None:
        user = str(config.agent.user)

    return HarborTask(
        path=path,
        task_id=path.name,
        config=config,
        instruction=instruction,
        digest=content_hash(path),
        image=image,
        workdir=workdir,
        env=env,
        user=user,
        overlay=overlay,
        sandbox_user=base_user if base_known else user,
    )


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
    if skipped is None:
        return [load_task(child) for child in children]
    tasks: list[HarborTask] = []
    for child in children:
        try:
            tasks.append(load_task(child))
        except HarborTaskError as exc:
            skipped[child.name] = exc
    if not tasks:
        raise HarborTaskError(f"No task under {root} loads; first error: {next(iter(skipped.values()))}")
    return tasks
