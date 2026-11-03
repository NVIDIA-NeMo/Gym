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
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from nemo_gym.data.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.data.tasks.harbor.digest import context_digest, inputs_differences, inputs_digest
from nemo_gym.data.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.data.tasks.harbor.models import HarborEnvironment
from nemo_gym.data.tasks.harbor.records import MAIN_SERVICE, VERIFIER_SERVICE, read_image_configs
from nemo_gym.data.tasks.harbor.registry import PLATFORM
from nemo_gym.data.tasks.harbor.task import DOCKERFILE, TASK_FILE, HarborTaskConfig, is_task_folder, iter_task_folders
from nemo_gym.data.tasks.harbor.verifier_env import verifier_build_context, verifier_build_environment


if TYPE_CHECKING:
    # The type of what a Runner returns only: this module spawns nothing (see tests/unit_tests/test_sandbox_policy.py).
    import subprocess


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
type Runner = Callable[..., subprocess.CompletedProcess]


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
    """One service to build of one distinct build context, and the tasks that share the context.

    ``digest`` is Harbor's environment id of the context (32 hex characters), the key of the build
    record and the body of the image name; ``context`` is the task's ``environment/`` folder, the hash root,
    Compose's project directory and ``CONTEXT_DIR``, whatever the service's own build context is. ``service``
    is the Compose service the entry builds: ``main`` (the task's own image, ``hb__<digest>``) or a sidecar of
    the task's ``docker-compose.yaml`` with a ``build:`` section (``hb__<digest>-<service>``). The entries of one
    context share ``digest``, ``context``, ``tasks`` and the settings below, and differ in ``service`` and the
    build section. ``env``, ``cpus`` and ``memory_mb`` are the first task's
    effective ``[environment]`` settings, the variables Harbor's ``docker compose build`` sees
    (:func:`compose_build_env`); ``env`` is empty when the context ships no ``docker-compose.yaml``, as
    Harbor only resolves the task's env for a Compose task (``_uses_compose``). Tasks sharing a context
    but declaring different settings share this image anyway, as they do in Harbor (the name is the
    content hash); :func:`plan_builds` logs them. Tasks whose folders share the hash but not what a build
    reads (a mode, a link, an empty folder, a name the hash skips) do not share the entry: the first task's
    folder is the context and the others are left out, see :func:`plan_builds`.

    ``build_context``, ``dockerfile``, ``build_args``, ``target`` and ``dockerfile_inline`` are the service's ``build``
    section of the task's ``docker-compose.yaml`` (:func:`service_build_settings`), the settings ``docker compose
    build`` merges over Harbor's template: the direct build of buildah and podman applies them through
    :meth:`ImageBuilder.build_argv`. ``compose_build`` is that section whole, as interpolated, for the inputs digest
    of the image (:func:`build_inputs_digest`), and ``dependency_digests`` the inputs digests of the services this one
    builds from (the ``service:`` references of its ``additional_contexts``), by service, part of that digest too.
    :func:`plan_builds` leaves them at their defaults (the context and its ``Dockerfile``);
    :meth:`ImageBuilder.build_many` fills them once the file has been checked (``build_records.settle_with_document``),
    so a file that does not interpolate or is refused fails that context's build, not the planning of the whole
    dataset.

    ``verifier`` marks the separate verifier's image (service ``verifier``, see the module docstring): ``context`` and
    ``digest`` are then the task's ``tests/`` folder and Harbor's hash of it, ``env``, ``cpus`` and ``memory_mb`` the
    effective verifier environment's, and the entry builds Harbor's template service ``main`` of that context
    (:attr:`compose_service`), so a ``tests/docker-compose.yaml`` is read for ``main``'s build section, as Harbor's
    verifier environment reads it. A Compose sidecar that happens to be named ``verifier`` (kumo's) is an ordinary
    service entry of the ``environment/`` context with this flag unset.
    """

    digest: str
    context: Path
    tasks: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cpus: int | None = None
    memory_mb: int | None = None
    service: str = MAIN_SERVICE
    build_context: Path | None = None
    dockerfile: str = DOCKERFILE_NAME
    build_args: dict[str, str] = field(default_factory=dict)
    compose_build: dict[str, Any] | None = None  # the build section as interpolated, see build_inputs_digest
    target: str | None = None
    dockerfile_inline: bool = False
    # ``(service, inputs digest)`` of each service this one builds from, see build_records._dependency_digests.
    dependency_digests: tuple[tuple[str, str], ...] = ()
    verifier: bool = False
    # The tasks of the context that declare ``[environment].docker_image`` and whose ``main`` is planned anyway, under
    # ``builds.force`` (Harbor's ``--force-build``); recorded so the loader runs the build for them (``forced_tasks``).
    forced_tasks: tuple[str, ...] = ()

    @property
    def compose_service(self) -> str:
        """The service of the context's Compose project this entry builds: ``main`` for the separate verifier's image
        (Harbor's template service of the ``tests/`` context), else :attr:`service` itself."""
        return MAIN_SERVICE if self.verifier else self.service

    @property
    def build_dir(self) -> Path:
        """The folder the tool builds: the service's ``build.context`` resolved, else the context itself."""
        return self.build_context if self.build_context is not None else self.context

    @property
    def dockerfile_path(self) -> Path:
        """The Dockerfile the tool builds, the service's ``build.dockerfile`` relative to :attr:`build_dir`."""
        return self.build_dir / self.dockerfile

    @property
    def is_main(self) -> bool:
        return self.service == MAIN_SERVICE


def sidecar_build_services(context: Path) -> list[str]:
    """The services other than ``main`` of ``context``'s ``docker-compose.yaml`` that carry a ``build`` section, in
    file order; ``[]`` without the file, or for one that does not parse or has no services mapping (the loader
    reports such a file). Only the presence of the key is read here, nothing is interpolated."""
    path = Path(context) / HARBOR_COMPOSE_FILE
    if not path.is_file():
        return []
    try:
        document = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, UnicodeDecodeError, OSError):
        return []
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return []
    return [
        str(name)
        for name, service in services.items()
        if str(name) != MAIN_SERVICE and isinstance(service, dict) and service.get("build") is not None
    ]


@dataclass(frozen=True)
class BuildConflict:
    """Why :func:`plan_builds` planned a task for nothing: one of its folders has the Harbor hash of ``first``'s (the
    task planned first under it) but not its build inputs. ``reason`` spells the difference out for the task's skip."""

    first: str
    reason: str


def plan_builds(
    folder: Path,
    *,
    select: Callable[[str], bool] | None = None,
    conflicts: dict[str, BuildConflict] | None = None,
    force: bool = False,
) -> list[BuildPlanEntry]:
    """The services to build among the tasks under ``folder``, one entry per distinct build context and service.

    ``folder`` is a task folder or a folder of task folders; ``select`` limits the scan to the
    tasks whose folder name it accepts. ``conflicts``, when given, receives by task id the :class:`BuildConflict` of
    each task planned for nothing: one of its folders (``environment/`` or ``tests/``) has the Harbor hash of a task
    planned before it (the same file names and contents, so the same ``hb__<hash>`` image names and build record) but
    not the same build inputs (:func:`~nemo_gym.data.tasks.harbor.digest.inputs_digest`: a mode, a symbolic link, an empty
    folder or a name the hash skips differs). One image cannot be built from both folders under the one name, so the
    entry keeps the first task's folder and tasks, and the task is in no entry at all, its other folder's included,
    since all of its folders are checked before any is added: prepare leaves it out with the reason, as it leaves out
    a task whose build fails, or plans again without the first task when that one is left out itself.
    A task's ``main`` counts when it has ``environment/Dockerfile``,
    declares no ``[environment].docker_image`` after the dataset's overrides, and the Dockerfile
    is outside pull and overlay mode; a prebuilt ``main`` is never built, unless ``force`` is set: Harbor's
    ``--force-build`` builds the Dockerfile instead of pulling the declared image, whatever the file holds (one
    pull or overlay mode could run too), and the loader then runs the recorded build. Every other service of the task's
    ``environment/docker-compose.yaml`` (the one Compose file Harbor reads) with a ``build`` section counts
    whatever ``main``'s mode, as Harbor's ``docker compose build`` builds it (:func:`sidecar_build_services`).
    Tasks whose ``task.toml`` does not load are left for the loader to report. Entries are grouped by context in
    the order of the first task that uses it, ``main`` first and the sidecars in file order. The Compose file is
    not interpolated here, only scanned for ``build`` keys: resolving each section is the build's job
    (:meth:`ImageBuilder.build_many`), so a file one task gets wrong fails that task's build and nothing else.

    A task in separate verifier mode whose effective verifier environment declares no ``docker_image`` and that ships
    ``tests/Dockerfile`` (:func:`~nemo_gym.data.tasks.harbor.task.verifier_build_context`) plans one more entry, service
    ``verifier`` with ``tests/`` as its context and digest (its own group, after the task's ``environment/`` entries),
    whatever the file holds, as Harbor builds it whatever it holds. A shared verifier, a declared or inherited
    ``docker_image`` or a task without the file plans none (the loader refuses the last shape when no image is
    declared either, as Harbor does). A ``tests/docker-compose.yaml`` is read for ``main``'s build section only; one
    that declares other services fails the verifier's build before any command (:meth:`ImageBuilder.build_many`).
    """
    folder = Path(folder).resolve()
    single = is_task_folder(folder)
    children = [folder] if single else iter_task_folders(folder)
    parent = folder.parent if single else folder
    try:
        # dataset.toml sits at the dataset root, which is also where the tasks were discovered from.
        dataset = read_dataset_config(parent)
    except ValueError:
        return []
    image_configs = read_image_configs(parent)
    entries: dict[str, list[str]] = {}
    contexts: dict[str, Path] = {}
    services: dict[str, list[str]] = {}
    settings: dict[str, tuple[dict[str, str], int | None, int | None]] = {}
    # The inputs digest of each context's folder, computed once a second task claims the same Harbor hash.
    trees: dict[str, str] = {}
    if conflicts is None:
        conflicts = {}
    # The digests of tests/ contexts: their `verifier` service is the separate verifier's image, not a sidecar.
    verifier_digests: set[str] = set()
    # By context digest, the tasks whose main is planned over a declared docker_image (``force``).
    forced: dict[str, list[str]] = {}
    for child in children:
        if select is not None and not select(child.name):
            continue
        dockerfile = child / DOCKERFILE
        context = dockerfile.parent
        try:
            config = HarborTaskConfig.model_validate(tomllib.loads((child / TASK_FILE).read_text()))
            config = apply_dataset_config(config, child.name, dataset)
            text = dockerfile.read_text() if dockerfile.is_file() else None
        except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
            continue
        planned: list[str] = []
        if text is not None and (
            # Harbor's --force-build builds a declared image's Dockerfile whatever it holds; the loader runs a pull
            # or overlay mode file over the declared image only while nothing was built for the task.
            (force and config.environment.docker_image)
            or (not config.environment.docker_image and needs_build(text, image_configs))
        ):
            planned.append(MAIN_SERVICE)
        planned += sidecar_build_services(context)
        # The task's contexts, each with the services planned of it and the settings its build sees.
        folders: list[tuple[Path, list[str], HarborEnvironment]] = []
        if planned:
            folders.append((context, planned, config.environment))
        verifier_context = verifier_build_context(child, config)
        if verifier_context is not None:
            # Harbor builds the verifier's environment from tests/ with the effective verifier config's settings
            # (the task's, or the first separate step's that starts from tests/ without an image).
            verifier_environment = verifier_build_environment(child, config)
            assert verifier_environment is not None  # verifier_build_context returned a context
            folders.append((verifier_context, [VERIFIER_SERVICE], verifier_environment))
        # Every folder is checked before any is added, so a task left out is in no entry, not even its other folder's.
        digests = [context_digest(folder_path) for folder_path, _, _ in folders]
        conflict = next(
            (
                found
                for digest, (folder_path, _, _) in zip(digests, folders)
                if (found := _context_conflict(child.name, folder_path, digest, entries, contexts, trees)) is not None
            ),
            None,
        )
        if conflict is not None:
            conflicts[child.name] = conflict
            continue
        for digest, (folder_path, folder_services, environment) in zip(digests, folders):
            _plan_context(
                child.name, folder_path, digest, folder_services, environment, entries, contexts, services, settings
            )
            if folder_path == verifier_context:
                verifier_digests.add(digest)
            elif force and config.environment.docker_image and MAIN_SERVICE in folder_services:
                forced.setdefault(digest, []).append(child.name)
    return [
        BuildPlanEntry(
            digest,
            contexts[digest],
            tuple(tasks),
            *settings[digest],
            service=service,
            verifier=service == VERIFIER_SERVICE and digest in verifier_digests,
            forced_tasks=tuple(forced.get(digest, ())) if service == MAIN_SERVICE else (),
        )
        for digest, tasks in entries.items()
        for service in services[digest]
    ]


def _context_conflict(
    task_id: str,
    context: Path,
    digest: str,
    entries: dict[str, list[str]],
    contexts: dict[str, Path],
    trees: dict[str, str],
) -> BuildConflict | None:
    """The :class:`BuildConflict` of ``task_id``'s folder ``context`` with the folder :func:`plan_builds` already holds
    under its Harbor hash ``digest``, or ``None`` when no folder is held yet or the two read the same to a build.
    Harbor's hash covers names and bytes; the build reads more. The held folder's inputs digest is computed once."""
    if digest not in entries:
        return None
    first = entries[digest][0]
    if inputs_digest(context) == trees.setdefault(digest, inputs_digest(contexts[digest])):
        return None
    return BuildConflict(first, _conflict_reason(task_id, context, first, contexts[digest], digest))


def _plan_context(
    task_id: str,
    context: Path,
    digest: str,
    planned: list[str],
    environment: HarborEnvironment,
    entries: dict[str, list[str]],
    contexts: dict[str, Path],
    services: dict[str, list[str]],
    settings: dict[str, tuple[dict[str, str], int | None, int | None]],
) -> None:
    """Add ``task_id``'s services ``planned`` of ``context`` to :func:`plan_builds`'s tables, under the context's Harbor
    hash ``digest``, with ``environment``'s settings (``[environment]``'s for the task's own context, the effective
    verifier environment's for ``tests/``) when the context is new, logging a task that declares others for a known
    one. The folder has passed :func:`_context_conflict`."""
    env = dict(environment.env or {}) if (context / HARBOR_COMPOSE_FILE).is_file() else {}
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
            task_id,
            context,
            ", ".join(differing),
            digest,
            entries[digest][0],
        )
    entries.setdefault(digest, []).append(task_id)
    contexts.setdefault(digest, context)
    # The Compose file is part of the hash, so every task of a context plans the same sidecars; main may differ
    # when one task's dataset override declares a docker_image, so the lists are merged, main first.
    merged = services.setdefault(digest, [])
    merged += [service for service in planned if service not in merged]
    if MAIN_SERVICE in merged[1:]:
        merged.remove(MAIN_SERVICE)
        merged.insert(0, MAIN_SERVICE)
    settings.setdefault(digest, declared)


def _conflict_reason(task_id: str, context: Path, first: str, first_context: Path, digest: str) -> str:
    """Why ``task_id``'s ``context`` is left out of the entry ``first``'s ``first_context`` holds under ``digest``:
    the differences :func:`~nemo_gym.data.tasks.harbor.digest.inputs_differences` finds, the first three spelled out."""
    differences = inputs_differences(context, first_context)
    described = []
    for relative, ours, theirs in differences[:3]:
        if ours is None:
            described.append(f"{relative} is only in {first}'s")
        elif theirs is None:
            described.append(f"{relative} is only in {task_id}'s")
        else:
            described.append(f"{relative} is {ours} in {task_id}'s and {theirs} in {first}'s")
    if len(differences) > 3:
        described.append(f"and {len(differences) - 3} more")
    return (
        f"{task_id}: {context.name}/ has the same Harbor environment id as {first}'s, {digest} (the same file names "
        f"and contents, so the one record and the images named hb__{digest} would serve both), but not the same "
        f"build inputs: {', '.join(described)}; one image cannot be built from both folders, so the task is left "
        "out (make the two folders identical, or change a file's content so each gets its own image)"
    )


@dataclass(frozen=True)
class BuildResult:
    """A built image, ready to record: the image of ``service`` for the context ``digest``."""

    digest: str
    tasks: tuple[str, ...]
    tag: str
    image: str
    tool: str
    pushed: bool
    record: dict[str, Any] = field(default_factory=dict)
    service: str = MAIN_SERVICE
    forced_tasks: tuple[str, ...] = ()
