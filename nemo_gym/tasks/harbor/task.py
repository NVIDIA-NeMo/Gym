# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.dockerfile import base_image_only
from nemo_gym.tasks.harbor.models import HarborTaskConfig


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
    (a build is required). ``workdir``, ``env`` and ``user`` merge ``task.toml``
    with what the Dockerfile recorded; ``task.toml`` wins.
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
    path = Path(path).resolve()
    if not is_task_folder(path):
        raise HarborTaskError(f"{path} has no {TASK_FILE}")
    try:
        config = HarborTaskConfig.model_validate(tomllib.loads((path / TASK_FILE).read_text()))
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise HarborTaskError(f"{path / TASK_FILE}: {exc}") from exc
    instruction_path = path / INSTRUCTION_FILE
    if not instruction_path.is_file():
        raise HarborTaskError(f"{path} has no {INSTRUCTION_FILE}")
    if not (path / "tests").is_dir():
        raise HarborTaskError(f"{path} has no tests/ folder")

    environment = config.environment
    image = environment.docker_image
    workdir = environment.workdir
    env = dict(environment.env)
    user: str | None = None
    dockerfile = path / "environment" / "Dockerfile"
    if dockerfile.is_file():
        base = base_image_only(dockerfile.read_text())
        if base is None:
            if image is None:
                raise HarborTaskError(
                    f"{dockerfile} needs a build (RUN/COPY/ADD or a multi-stage FROM); "
                    "building Dockerfiles is not supported yet. Declare [environment].docker_image "
                    "with a prebuilt image to run this task."
                )
        else:
            image = image or base.image
            workdir = workdir or base.workdir
            env = base.env | env
            user = base.user
    if config.agent.user is not None:
        user = str(config.agent.user)

    return HarborTask(
        path=path,
        task_id=path.name,
        config=config,
        instruction=instruction_path.read_text().strip(),
        digest=content_hash(path),
        image=image,
        workdir=workdir,
        env=env,
        user=user,
    )


def discover_tasks(root: Path) -> list[HarborTask]:
    """A folder with ``task.toml`` is one task; otherwise its direct children are the tasks."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise HarborTaskError(f"{root} is not a directory")
    if is_task_folder(root):
        return [load_task(root)]
    children = sorted(child for child in root.iterdir() if child.is_dir() and is_task_folder(child))
    if not children:
        raise HarborTaskError(f"{root} is neither a task folder nor a folder of task folders (no {TASK_FILE} found)")
    return [load_task(child) for child in children]
