# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.dockerfile import base_image_only
from nemo_gym.tasks.harbor.models import HarborTaskConfig


logger = logging.getLogger(__name__)

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"


class HarborTaskError(ValueError):
    """The folder is not a runnable Harbor task."""


@dataclass(frozen=True)
class HarborTask:
    """A task folder read into memory.

    ``task_id`` is the folder name: ``[task].name`` is an ``org/name`` label that is
    not unique across a dataset. ``image`` is the prebuilt image when the task
    declares one, else the base image of a base-image-only Dockerfile, else ``None``
    (a build is required). Without a prebuilt image, ``workdir``, ``env`` and ``user``
    merge ``task.toml`` with what the Dockerfile recorded; ``task.toml`` wins.
    With a prebuilt image, the Dockerfile contributes no settings.
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

    @property
    def needs_sandbox(self) -> bool:
        """A task runs in a sandbox exactly when it declares an image or a Dockerfile."""
        return self.image is not None or (self.path / "environment" / "Dockerfile").is_file()

    @property
    def has_solution(self) -> bool:
        return (self.path / "solution" / "solve.sh").is_file()


def is_task_folder(path: Path) -> bool:
    return (Path(path) / TASK_FILE).is_file()


def load_task(path: Path) -> HarborTask:
    """Read ``path`` as one task. Raises :class:`HarborTaskError` when it is not one."""
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
    phases = {"environment": config.environment, "agent": config.agent, "verifier": config.verifier}
    if config.verifier.environment is not None:
        phases["verifier.environment"] = config.verifier.environment
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
    if not instruction_path.is_file():
        raise HarborTaskError(f"{path} has no {INSTRUCTION_FILE}")
    if not (path / "tests").is_dir():
        raise HarborTaskError(f"{path} has no tests/ folder")
    try:
        instruction = instruction_path.read_text().strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise HarborTaskError(f"{instruction_path}: {exc}") from exc

    environment = config.environment
    image = environment.docker_image
    workdir = environment.workdir
    env = dict(environment.env)
    user: str | None = None
    dockerfile = path / "environment" / "Dockerfile"
    if image is None and dockerfile.is_file():
        try:
            base = base_image_only(dockerfile.read_text())
        except (OSError, ValueError) as exc:
            raise HarborTaskError(f"{dockerfile}: {exc}") from exc
        if base is None:
            raise HarborTaskError(
                f"{dockerfile} needs a build (unsupported instructions or settings); "
                "building Dockerfiles is not supported yet. Declare [environment].docker_image "
                "with a prebuilt image to run this task."
            )
        image = base.image
        workdir = workdir or base.workdir
        env = base.env | env
        user = base.user
    if config.agent.user is not None:
        user = str(config.agent.user)

    try:
        digest = content_hash(path)
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
