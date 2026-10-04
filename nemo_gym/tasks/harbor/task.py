# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one task folder, or a folder of task folders, from disk."""

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
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
        instruction=instruction,
        digest=content_hash(path),
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
