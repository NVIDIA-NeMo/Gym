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
from nemo_gym.tasks.harbor.digest import content_hash
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
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from nemo_gym.tasks.harbor.records import IMAGE_CONFIGS_FILE, read_image_configs


logger = logging.getLogger(__name__)

TASK_FILE = "task.toml"
INSTRUCTION_FILE = "instruction.md"
DOCKERFILE = Path("environment") / "Dockerfile"


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
    overlay_workdirs: tuple[OverlayWorkdir, ...] = ()

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
        instruction = read_instruction(instruction_path)
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
    if dockerfile.is_file():
        try:
            dockerfile_text = dockerfile.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            raise HarborTaskError(f"{dockerfile}: {exc}") from exc
        try:
            parsed = parse_dockerfile(dockerfile_text)
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
                if image is None:
                    raise HarborTaskError(f"{dockerfile} needs a build ({exc})") from exc
                # As for a parse-time build requirement above: the declared image already holds the
                # build's result, so the task runs it with task.toml's own settings.
                logger.debug("%s: %s; the declared image %s runs as is", dockerfile, exc, image)
                base_known = False
            if resolved is not None:
                image = image or pinned_image or resolved.image
                workdir = workdir or resolved.workdir
                env = resolved.env | env
                user = resolved.user
                overlay = resolved.runs
                # Pull mode creates nothing: only an overlay makes the Dockerfile's directories at seed.
                overlay_workdirs = resolved.workdirs if resolved.runs else ()
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
        overlay_workdirs=overlay_workdirs,
    )


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
