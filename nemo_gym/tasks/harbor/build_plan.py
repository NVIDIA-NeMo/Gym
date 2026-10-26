# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan the image builds of a task folder.

Reads task folders and dataset.toml, decides which Dockerfiles need a build rather than a pull
or an overlay, finds the sidecars with a build section and the separate verifier's tests
context, and groups the work by context digest into BuildPlanEntry values with BuildResult as
the record of one finished build. Tool detection and the storage root live here because the
plan names the tool. The module runs no subprocess and reads no Compose variables. It is the
root of the build modules: build_compose, build_records, build_tools and image_build import
it and it imports none of them.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.tasks.harbor.records import read_image_configs
from nemo_gym.tasks.harbor.registry import PLATFORM
from nemo_gym.tasks.harbor.task import DOCKERFILE, TASK_FILE, HarborTaskConfig, is_task_folder


LOGGER = logging.getLogger(__name__)

BUILD_TOOLS = ("buildah", "podman", "docker")
DEFAULT_PLATFORM = "/".join(PLATFORM)
REPOSITORY_ENV = "NEMO_GYM_BUILD_REPOSITORY"

# The one Compose file Harbor's Docker environment reads from a task (``_environment_docker_compose_path``,
# https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L312).
HARBOR_COMPOSE_FILE = "docker-compose.yaml"

# The default Dockerfile name of a build context, Compose's and the tools' alike.
DOCKERFILE_NAME = "Dockerfile"

# The seam tests replace: run one tool command and return its completed process (text mode). ``env`` is
# the full environment of the command when the caller sets one (the docker backend passes Harbor's
# Compose variables that way); ``None`` inherits the process environment.
Runner = Callable[..., subprocess.CompletedProcess]


class BuildError(RuntimeError):
    """A build, push or inspect command failed; nothing was recorded for the image."""


def default_storage_root() -> Path:
    """Where buildah keeps its layers: node-local when the cluster scratch disk exists, else the temp dir."""
    scratch = Path("/raid/scratch")
    if scratch.is_dir():
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "nemo-gym"
        return scratch / user / "nemo-gym-buildah"
    return Path(tempfile.gettempdir()) / "nemo-gym-buildah"


def detect_tool(repository: str | None = None) -> str | None:
    """The builder to use, or ``None`` when none is installed.

    With a repository the image is pushed, so any tool serves: buildah, then podman, then docker.
    Without one the image has to stay where a sandbox provider can start it, and only the Docker
    daemon is such a place (buildah and podman keep images in a store no provider reads), so only
    docker qualifies.
    """
    candidates = BUILD_TOOLS if repository else ("docker",)
    for tool in candidates:
        if shutil.which(tool):
            return tool
    return None


def needs_build(dockerfile_text: str, image_configs: dict[str, Any] | None = None) -> bool:
    """Whether a Dockerfile is outside what pull and overlay mode can run.

    A file whose instructions can only be resolved against its base image's configuration is
    resolved against the record in ``image_configs`` (``compose-images.json``, as prepare records
    it before planning builds). Without a record the file is classed as a build, so a context is
    never left unbuilt because the base image was unknown.
    """
    try:
        parsed = parse_dockerfile(dockerfile_text)
    except DockerfileNeedsBuild:
        return True
    try:
        resolve_dockerfile(parsed, None)
        return False
    except DockerfileNeedsBuild:
        return True
    except ImageConfigRequired:
        pass
    record = (image_configs or {}).get(parsed.image)
    if record is None:
        return True
    try:
        resolve_dockerfile(parsed, BaseImageConfig.from_record(record))
    except (DockerfileNeedsBuild, ImageConfigRequired):
        return True
    return False


@dataclass(frozen=True)
class BuildPlanEntry:
    """One distinct build context and the tasks that share it.

    ``digest`` is Harbor's environment id of the context (32 hex characters), the key of the build
    record and the body of the image name. ``env``, ``cpus`` and ``memory_mb`` are the first task's
    effective ``[environment]`` settings, the variables Harbor's ``docker compose build`` sees
    (:func:`compose_build_env`); ``env`` is empty when the context ships no ``docker-compose.yaml``, as
    Harbor only resolves the task's env for a Compose task (``_uses_compose``). Tasks sharing a context
    but declaring different settings share this image anyway, as they do in Harbor (the name is the
    content hash); :func:`plan_builds` logs them.

    ``build_context``, ``dockerfile``, ``build_args``, ``target`` and ``dockerfile_inline`` are ``main``'s ``build``
    section of the task's ``docker-compose.yaml`` (:func:`main_build_settings`), the settings ``docker compose
    build`` merges over Harbor's template: the direct build of buildah and podman applies them through
    :meth:`ImageBuilder.build_argv`. :func:`plan_builds` leaves them at their defaults (the context and its
    ``Dockerfile``); :meth:`ImageBuilder.build` fills them once the file has been checked, so a file that does not
    interpolate or is refused fails that context's build, not the planning of the whole dataset.
    """

    digest: str
    context: Path
    tasks: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cpus: int | None = None
    memory_mb: int | None = None
    build_context: Path | None = None
    dockerfile: str = DOCKERFILE_NAME
    build_args: dict[str, str] = field(default_factory=dict)
    compose_build: dict[str, Any] | None = None  # the build section as interpolated, see build_inputs_digest
    target: str | None = None
    dockerfile_inline: bool = False

    @property
    def build_dir(self) -> Path:
        """The folder the tool builds: ``main.build.context`` resolved, else the context itself."""
        return self.build_context if self.build_context is not None else self.context

    @property
    def dockerfile_path(self) -> Path:
        """The Dockerfile the tool builds, ``main.build.dockerfile`` relative to :attr:`build_dir`."""
        return self.build_dir / self.dockerfile


def plan_builds(folder: Path, *, select: Callable[[str], bool] | None = None) -> list[BuildPlanEntry]:
    """The distinct build contexts among the tasks under ``folder`` whose Dockerfile needs a build.

    ``folder`` is a task folder or a folder of task folders; ``select`` limits the scan to the
    tasks whose folder name it accepts. A task counts when it has ``environment/Dockerfile``,
    declares no ``[environment].docker_image`` after the dataset's overrides, and the Dockerfile
    is outside pull and overlay mode. Tasks whose ``task.toml`` does not load are left for the
    loader to report. Entries are ordered by the first task that uses them. The task's Compose file is
    not read here: nothing in it changes whether a context is built, and reading it is the build's job
    (:meth:`ImageBuilder.build`), so a file one task gets wrong fails that task's build and nothing else.
    """
    folder = Path(folder).resolve()
    single = is_task_folder(folder)
    children = [folder] if single else sorted(child for child in folder.iterdir() if child.is_dir())
    parent = folder.parent if single else folder
    try:
        # dataset.toml sits beside the task folders, or one level up for datasets/<name>/tasks/.
        dataset = read_dataset_config(parent)
    except ValueError:
        return []
    image_configs = read_image_configs(parent)
    entries: dict[str, list[str]] = {}
    contexts: dict[str, Path] = {}
    settings: dict[str, tuple[dict[str, str], int | None, int | None]] = {}
    for child in children:
        if not is_task_folder(child) or (select is not None and not select(child.name)):
            continue
        dockerfile = child / DOCKERFILE
        if not dockerfile.is_file():
            continue
        try:
            config = HarborTaskConfig.model_validate(tomllib.loads((child / TASK_FILE).read_text()))
            config = apply_dataset_config(config, child.name, dataset)
            text = dockerfile.read_text()
        except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
            continue
        if config.environment.docker_image or not needs_build(text, image_configs):
            continue
        digest = context_digest(dockerfile.parent)
        environment = config.environment
        env = dict(environment.env or {}) if (dockerfile.parent / HARBOR_COMPOSE_FILE).is_file() else {}
        declared = (env, environment.cpus, environment.memory_mb)
        if digest in entries and settings[digest] != declared:
            first_env, first_cpus, first_memory = settings[digest]
            differing = sorted(key for key in set(env) | set(first_env) if env.get(key) != first_env.get(key))
            if (environment.cpus, environment.memory_mb) != (first_cpus, first_memory):
                differing.append("cpus/memory_mb")
            # Values are not logged: a task's env may carry secrets.
            LOGGER.warning(
                "Tasks %s and %s share build context %s but declare different [environment] settings (%s); the "
                "shared image hb__%s is built with %s's values, as Harbor builds one image per context",
                entries[digest][0],
                child.name,
                dockerfile.parent,
                ", ".join(differing),
                digest,
                entries[digest][0],
            )
        entries.setdefault(digest, []).append(child.name)
        contexts.setdefault(digest, dockerfile.parent)
        settings.setdefault(digest, declared)
    return [
        BuildPlanEntry(digest, contexts[digest], tuple(tasks), *settings[digest]) for digest, tasks in entries.items()
    ]


@dataclass(frozen=True)
class BuildResult:
    """A built image, ready to record."""

    digest: str
    tasks: tuple[str, ...]
    tag: str
    image: str
    tool: str
    pushed: bool
    record: dict[str, Any] = field(default_factory=dict)


# The name Harbor's build template and task Compose files give the task's service.
MAIN_SERVICE = "main"
