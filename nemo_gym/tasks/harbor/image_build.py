# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the task images a Dockerfile overlay cannot express (image mode three of three).

Prepare decides one of three modes per distinct Dockerfile. Pull, for a base-image-only file.
Overlay, for a pullable ``FROM`` plus ``RUN``, ``WORKDIR``, ``ENV`` and ``USER`` lines the
server applies at seed (see ``nemo_gym.tasks.harbor.dockerfile``). Build, for everything else
(``COPY``, ``ADD``, heredocs, BuildKit flags, multi-stage files): the task's ``environment/``
folder is the build context, built once with docker, buildah or podman, pushed to a registry
when one is named, and recorded in ``built-images.json`` next to the task folders, keyed by
Harbor's content hash of the build context. The loader then runs the task in pull mode from the
recorded image, with ``WORKDIR``, ``ENV`` and ``USER`` read from the image's recorded configuration
in ``compose-images.json`` like any other pulled image.

The hash and the image name are Harbor's (``nemo_gym.tasks.harbor.harbor_compat``, vendored from
Harbor v0.22.0): the hash is Harbor's environment id of the folder and the image is ``hb__<hash>``, the
value Harbor exports to every Compose command as ``MAIN_IMAGE_NAME``, under the repository's registry
path when one is named. Harbor's own build template sets no ``image`` (Compose then tags
``<session>-main``); Gym sets ``image: hb__<hash>`` so the result can be inspected, pushed and recorded,
and identical contexts get the same name on every prepare.

The docker backend is Harbor's build step. Harbor's Docker environment builds a task by running
``docker compose build`` over its own ``docker-compose-build.yaml`` (service ``main`` with
``build.context: ${CONTEXT_DIR}``) followed by the task's ``environment/docker-compose.yaml`` when
there is one, with the project directory set to the ``environment/`` folder and its infrastructure
variables (``MAIN_IMAGE_NAME``, ``CONTEXT_DIR`` and friends) in the process environment:

- ``DockerEnvironment.start()``, the build under the per-environment lock:
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L868
  (``compose build`` at L889-L896)
- ``_run_docker_compose_command``, the ``docker compose --project-name --project-directory -f ...`` line:
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L606
- ``_docker_compose_paths``, the file order (build template, then the task's Compose file):
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L316
- ``ComposeInfraEnvVars`` and ``_compose_infra_env_vars``, the variables:
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L238
  and #L552
- the template itself:
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker-compose-build.yaml

Gym writes the same file per build (:func:`harbor_compose_document`, with ``CONTEXT_DIR`` resolved and
``image: hb__<hash>`` added), appends the task's ``environment/docker-compose.yaml`` the way Harbor does
(that one file name; the other Compose spellings the loader accepts are refused at build time, see
:func:`task_compose_file`), and runs ``docker compose build`` with Harbor's environment
(:func:`compose_build_env`: the process environment, the task's ``[environment].env``, then Harbor's
infrastructure variables), so ``main``'s own build args and target in the task file apply. The task
file is third-party content run on the build host, so it is checked first (:func:`validate_task_compose`),
interpolated against the build's own environment the way Compose interpolates it (``environment/.env``
included, its values expanded as Compose's dotenv reader expands them), so a refused value
cannot hide behind ``${X:-host}``: build secrets, ssh, extra contexts and hosts, host networking, cache
credentials and any context or Dockerfile outside ``environment/`` are refused, on every service that builds.
Every service of the task file with a ``build:`` section is built, as Harbor's ``docker compose build`` (run with
no service argument) builds it: ``main`` from ``environment/Dockerfile`` and each sidecar from its own context under
``environment/``, planned as one :class:`BuildPlanEntry` per ``(context digest, service)``. A sidecar's image is
named ``hb__<hash>-<service>`` (Compose's own ``<project>-<service>`` form under the project name Gym passes, with
Harbor's hash as the project) and recorded next to ``main``'s under the same digest. At seed the loader substitutes
the recorded image for every built service and drops their ``build:`` and ``pull_policy``
(``nemo_gym.tasks.harbor.image_configs.with_built_images``), so prepare and the resources server read one document.
``docker build`` is never called. Harbor serialises concurrent builds of one environment with an
in-process lock; prepare builds each distinct context once, sequentially, so no lock is needed here.
buildah and podman are the OpenSandbox path: they build each service's Dockerfile directly and push it to a registry.

The separate verifier's image is a fourth kind of entry. In separate mode Harbor starts a second Docker environment
for the verifier with ``tests/`` as its environment dir and the effective verifier config (``[verifier.environment]``,
else a copy of ``[environment]``; ``_separate_verifier_env`` and ``_verifier_env_build_context`` in trial.py,
``resolve_effective_verifier_env_config`` in verifier_mode.py at v0.22.0), so without a ``docker_image`` it runs the
same ``docker compose build`` over its template with ``CONTEXT_DIR`` = ``tests/`` and builds ``tests/Dockerfile``,
and it uploads no ``tests/`` into that image (``skip_tests_upload``). Prepare plans that build as one entry per task
with ``tests/`` as its context (``nemo_gym.tasks.harbor.task.verifier_build_context``), the verifier config's env and
resources, and service name ``verifier``: the context is hashed on its own, as Harbor's ``environment_id`` hashes the
verifier environment's dir, so the digest is Harbor's and ``hb__<digest>`` is the ``MAIN_IMAGE_NAME`` Harbor exports
for the verifier env; Gym tags the image ``hb__<digest>-verifier`` (its sidecar form, naming the role; Harbor's own
tag of it is Compose's per-session ``<session>-main``, never published) and records it under the ``tests/`` digest as
service ``verifier``. The loader reads it into ``HarborTask.verifier_image`` and the resources server grades in it
with no ``tests/`` upload when the image owns ``/tests/test.sh``. On the docker backend the entry builds Harbor's
template service ``main`` of the ``tests/`` context (:attr:`BuildPlanEntry.compose_service`); buildah and podman
build ``tests/Dockerfile`` directly. A verifier build that fails leaves its task out of the preparation like a failed
``main`` build.

A build is cancel-safe: nothing is recorded until the tool has built, pushed and inspected the
image, and each image is recorded as soon as it is done, so an interrupted prepare resumes
where it stopped. The record files are replaced atomically under a sidecar lock
(``nemo_gym.tasks.harbor.task.record_lock``), so two Gym processes recording into one datasets
folder keep each other's entries and a reader never sees a half-written file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from nemo_gym.sandbox.compose_config import ComposeSubstitutionError, env_file_entries, substitute_compose
from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import (
    BaseImageConfig,
    DockerfileNeedsBuild,
    ImageConfigRequired,
    dockerfile_has_heredoc,
    parse_dockerfile,
    resolve_dockerfile,
)
from nemo_gym.tasks.harbor.harbor_compat import sanitize_docker_image_name
from nemo_gym.tasks.harbor.image_configs import (
    COMPOSE_FILE_NAMES,
    LOG_MOUNT_PATHS,
    PLATFORM,
    RECORDED_CONFIG_KEYS,
    TEST_DIR,
    substitute_with_empty,
)
from nemo_gym.tasks.harbor.models import HarborEnvironment
from nemo_gym.tasks.harbor.task import (
    BUILT_IMAGES_FILE,
    DOCKERFILE,
    IMAGE_CONFIGS_FILE,
    MAIN_SERVICE,
    TASK_FILE,
    VERIFIER_SERVICE,
    HarborTaskConfig,
    effective_verifier_environment,
    is_task_folder,
    read_built_images,
    read_image_configs,
    record_lock,
    recorded_service_images,
    recorded_service_records,
    verifier_build_context,
    verifier_compose_sidecars,
    write_json_record,
)


LOGGER = logging.getLogger(__name__)

BUILD_TOOLS = ("buildah", "podman", "docker")
DEFAULT_PLATFORM = "/".join(PLATFORM)
REPOSITORY_ENV = "NEMO_GYM_BUILD_REPOSITORY"
# The variable Harbor exports to every Compose command with the task image's name (docker.py:238-243 in v0.22.0).
MAIN_IMAGE_VARIABLE = "MAIN_IMAGE_NAME"
# The file name Harbor ships its build template under; Gym writes its copy under the same name.
COMPOSE_BUILD_FILE = "docker-compose-build.yaml"
# Gym's own override, passed to ``docker compose build`` after the task's file: the image name of every service built.
COMPOSE_IMAGES_FILE = "docker-compose-images.yaml"
# The one Compose file Harbor's Docker environment reads from a task (``_environment_docker_compose_path``,
# https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L312).
HARBOR_COMPOSE_FILE = "docker-compose.yaml"
# Build keys of a task's Compose file that reach the build host and are refused (see validate_task_compose).
REFUSED_BUILD_KEYS = ("secrets", "ssh", "extra_hosts")
# Keys of a ``cache_from`` / ``cache_to`` entry (``type=...,ref=...,key=value``) that carry credentials.
CACHE_CREDENTIAL_KEYS = frozenset(
    {"token", "secret", "password", "credentials", "access_key_id", "secret_access_key", "session_token"}
)
# Harbor's ``[environment].env`` template: a whole value ``${VAR}`` or ``${VAR:-default}`` (``_TEMPLATE_PATTERN``,
# https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/utils/env.py#L4).
HARBOR_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
# A 403 in a tool's output, not the digits inside a hex digest.
HTTP_403 = re.compile(r"(?<![0-9a-fA-F])403(?![0-9a-fA-F])")
# The Compose plugin the docker backend needs (`docker compose`, Compose v2), by its package name.
COMPOSE_PLUGIN = "docker-compose-plugin"
# The file Compose reads interpolation variables from in the project directory (Harbor's: the ``environment/``
# folder) when the variable is not in the command's environment.
COMPOSE_DOTENV = ".env"
# Points Compose at env files other than the project directory's ``.env``; refused for a build.
COMPOSE_ENV_FILES = "COMPOSE_ENV_FILES"
# buildah and podman read this variable as a containers.conf layered over the host's; see ImageBuilder.tool_env.
CONTAINERS_CONF_OVERRIDE = "CONTAINERS_CONF_OVERRIDE"
CONTAINERS_CONF_FILE = "containers.conf"
# No default volumes: the host's containers.conf may bind-mount files such as /etc/localtime into every container,
# build steps included, and a Dockerfile that installs tzdata then fails replacing the mounted file.
CONTAINERS_CONF_TEXT = "[containers]\nvolumes = []\n"
# buildah and podman read this variable as their registries.conf (it replaces the host's main file; the drop-ins
# under registries.conf.d still load); see ImageBuilder.tool_env.
CONTAINERS_REGISTRIES_CONF = "CONTAINERS_REGISTRIES_CONF"
REGISTRIES_CONF_FILE = "registries.conf"
# An unqualified FROM (``FROM oven/bun:1``) resolves to docker.io, as it does under ``docker compose build``. Debian
# and Ubuntu ship a registries.conf with no unqualified-search registry, so buildah and podman otherwise refuse it.
REGISTRIES_CONF_TEXT = 'unqualified-search-registries = ["docker.io"]\n'
# The image format buildah and podman build with: docker, the format BuildKit builds. Under their default OCI
# format SHELL is ignored by the build's RUN steps and HEALTHCHECK and ONBUILD are dropped from the image.
IMAGE_FORMAT = "docker"
# The heredoc preflight of buildah and podman: a two-line build that pulls no base image and parses only when the
# tool was built with openshift/imagebuilder 1.2.6 or later (Ubuntu 24.04's packages were not).
HEREDOC_PROBE_DOCKERFILE = "FROM scratch\nCOPY <<EOF /probe\nprobe\nEOF\n"
HEREDOC_PROBE_TAG = "localhost/nemo-gym/heredoc-probe"
HEREDOC_REMEDY = (
    "Dockerfile heredocs need a buildah or podman built with openshift/imagebuilder 1.2.6 or later (upstream buildah "
    "1.33.0 or later; Ubuntu 24.04's packaged buildah 1.33.7 and podman 4.9.3 do not qualify), named with "
    "builds.tool, or builds.tool: docker."
)
# Configuration keys recorded for a built image: the pulled-image keys plus the docker-format fields a build sets.
BUILT_CONFIG_KEYS = (*RECORDED_CONFIG_KEYS, "OnBuild", "Shell")
# The default Dockerfile name of a build context, Compose's and the tools' alike.
DOCKERFILE_NAME = "Dockerfile"
# ``(context, variable)`` pairs already warned about as unset: a context's file is interpolated more than once per build.
_WARNED_UNSET: set[tuple[str, str]] = set()

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
    content hash); :func:`plan_builds` logs them.

    ``build_context``, ``dockerfile``, ``build_args``, ``target`` and ``dockerfile_inline`` are the service's ``build``
    section of the task's ``docker-compose.yaml`` (:func:`service_build_settings`), the settings ``docker compose
    build`` merges over Harbor's template: the direct build of buildah and podman applies them through
    :meth:`ImageBuilder.build_argv`. :func:`plan_builds` leaves them at their defaults (the context and its
    ``Dockerfile``); :meth:`ImageBuilder.build_many` fills them once the file has been checked, so a file that does
    not interpolate or is refused fails that context's build, not the planning of the whole dataset.

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
    target: str | None = None
    dockerfile_inline: bool = False
    verifier: bool = False

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


def plan_builds(folder: Path, *, select: Callable[[str], bool] | None = None) -> list[BuildPlanEntry]:
    """The services to build among the tasks under ``folder``, one entry per distinct build context and service.

    ``folder`` is a task folder or a folder of task folders; ``select`` limits the scan to the
    tasks whose folder name it accepts. A task's ``main`` counts when it has ``environment/Dockerfile``,
    declares no ``[environment].docker_image`` after the dataset's overrides, and the Dockerfile
    is outside pull and overlay mode; a prebuilt ``main`` is never built. Every other service of the task's
    ``environment/docker-compose.yaml`` (the one Compose file Harbor reads) with a ``build`` section counts
    whatever ``main``'s mode, as Harbor's ``docker compose build`` builds it (:func:`sidecar_build_services`).
    Tasks whose ``task.toml`` does not load are left for the loader to report. Entries are grouped by context in
    the order of the first task that uses it, ``main`` first and the sidecars in file order. The Compose file is
    not interpolated here, only scanned for ``build`` keys: resolving each section is the build's job
    (:meth:`ImageBuilder.build_many`), so a file one task gets wrong fails that task's build and nothing else.

    A task in separate verifier mode whose effective verifier environment declares no ``docker_image`` and that ships
    ``tests/Dockerfile`` (:func:`~nemo_gym.tasks.harbor.task.verifier_build_context`) plans one more entry, service
    ``verifier`` with ``tests/`` as its context and digest (its own group, after the task's ``environment/`` entries),
    whatever the file holds, as Harbor builds it whatever it holds. A shared verifier, a declared or inherited
    ``docker_image`` or a task without the file plans none (the loader refuses the last shape when no image is
    declared either, as Harbor does). A ``tests/docker-compose.yaml`` is read for ``main``'s build section only; one
    that declares other services fails the verifier's build before any command (:meth:`ImageBuilder.build_many`).
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
    services: dict[str, list[str]] = {}
    settings: dict[str, tuple[dict[str, str], int | None, int | None]] = {}
    # The digests of tests/ contexts: their `verifier` service is the separate verifier's image, not a sidecar.
    verifier_digests: set[str] = set()
    for child in children:
        if not is_task_folder(child) or (select is not None and not select(child.name)):
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
        if text is not None and not config.environment.docker_image and needs_build(text, image_configs):
            planned.append(MAIN_SERVICE)
        planned += sidecar_build_services(context)
        if planned:
            _plan_context(child.name, context, planned, config.environment, entries, contexts, services, settings)
        verifier_context = verifier_build_context(child, config)
        if verifier_context is not None:
            # Harbor builds the verifier's environment from tests/ with the effective verifier config's settings.
            verifier_environment = effective_verifier_environment(config)
            assert verifier_environment is not None  # verifier_build_context returned a context
            verifier_digests.add(
                _plan_context(
                    child.name,
                    verifier_context,
                    [VERIFIER_SERVICE],
                    verifier_environment,
                    entries,
                    contexts,
                    services,
                    settings,
                )
            )
    return [
        BuildPlanEntry(
            digest,
            contexts[digest],
            tuple(tasks),
            *settings[digest],
            service=service,
            verifier=service == VERIFIER_SERVICE and digest in verifier_digests,
        )
        for digest, tasks in entries.items()
        for service in services[digest]
    ]


def _plan_context(
    task_id: str,
    context: Path,
    planned: list[str],
    environment: HarborEnvironment,
    entries: dict[str, list[str]],
    contexts: dict[str, Path],
    services: dict[str, list[str]],
    settings: dict[str, tuple[dict[str, str], int | None, int | None]],
) -> str:
    """Add ``task_id``'s services ``planned`` of ``context`` to :func:`plan_builds`'s tables, under the context's digest
    (returned), with ``environment``'s settings (``[environment]``'s for the task's own context, the effective verifier
    environment's for ``tests/``) when the context is new, logging a task that declares others for a known one."""
    digest = context_digest(context)
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
    return digest


def build_section(document: dict[str, Any] | None, service: str) -> dict[str, Any] | None:
    """``services.<service>.build`` of a Compose document as a mapping (the string form read as its ``context``),
    or ``None`` when the service has none."""
    services = document.get("services") if isinstance(document, dict) else None
    entry = services.get(service) if isinstance(services, dict) else None
    build = entry.get("build") if isinstance(entry, dict) else None
    if isinstance(build, str):
        build = {"context": build}
    return build if isinstance(build, dict) else None


def service_build_settings(
    entry: BuildPlanEntry, document: dict[str, Any] | None, env: dict[str, str]
) -> dict[str, Any]:
    """The entry's service's ``build`` section of the task's Compose file as :class:`BuildPlanEntry` fields, or
    ``{}`` without one.

    ``document`` is the file as :func:`validate_task_compose` returned it for ``env``, the build's environment
    (:func:`compose_build_env`): interpolated as Compose interpolates it, its paths checked to stay under
    ``environment/``. This is what ``docker compose build`` applies from the task file (over Harbor's template for
    ``main``), read here so the direct build of buildah and podman applies the same: the build ``context`` (a path,
    or the string form of the section) and ``dockerfile``, resolved under ``environment/``
    (:func:`_contained_build_path` again, so a caller handing in an unchecked document still cannot escape it);
    ``args`` as a mapping or Compose's ``KEY=value`` list, where a mapping value of ``null`` and a bare list ``KEY``
    alike take the variable's value from the build's environment and are dropped when it is unset, as Compose
    drops them, so the Dockerfile's own ``ARG`` default stands; ``target``; and whether a ``dockerfile_inline`` is
    set. ``args`` of another shape is a :class:`BuildError`.
    """
    path = entry.context / HARBOR_COMPOSE_FILE
    service = entry.compose_service
    build = build_section(document, service)
    if build is None:
        return {}
    build_context = _contained_build_path(
        build.get("context", "."),
        base=entry.context,
        context=entry.context,
        path=path,
        service=service,
        key="build.context",
    ).resolve()
    dockerfile = DOCKERFILE_NAME
    if build.get("dockerfile") is not None:
        _contained_build_path(
            build["dockerfile"],
            base=build_context,
            context=entry.context,
            path=path,
            service=service,
            key="build.dockerfile",
        )
        dockerfile = str(build["dockerfile"])
    raw_args = build.get("args") or {}
    pairs: list[tuple[str, Any]] = []
    if isinstance(raw_args, dict):
        pairs = [(str(key), value) for key, value in raw_args.items()]
    elif isinstance(raw_args, list):
        for item in raw_args:
            key, separator, value = str(item).partition("=")
            pairs.append((key, value if separator else None))
    else:
        raise _refuse(path, service, "build.args", "it must be a mapping or a list of KEY=value")
    args: dict[str, str] = {}
    variables: dict[str, str] | None = None
    for key, value in pairs:
        if value is None:
            # Compose: a null mapping value or a bare list KEY is read from the build's environment, dropped when unset.
            if variables is None:
                variables = compose_interpolation_variables(entry.context, env, tasks=entry.tasks)
            if key in variables:
                args[key] = variables[key]
            continue
        args[key] = _compose_scalar(value)
    target = build.get("target")
    return {
        "build_context": build_context,
        "dockerfile": dockerfile,
        "build_args": args,
        "target": _compose_scalar(target) if target not in (None, "") else None,
        "dockerfile_inline": build.get("dockerfile_inline") is not None,
    }


def _compose_scalar(value: Any) -> str:
    """A YAML scalar as Compose passes it to the build: ``true``/``false`` lower-case, else its text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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


def _run_default(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, errors="replace", env=env)


# -- Harbor's Compose build step ----------------------------------------------------------------


def harbor_compose_document(context: Path, image: str) -> dict[str, Any]:
    """The Compose file Harbor's Docker environment builds a task with, resolved for one build context.

    Harbor's ``docker-compose-build.yaml`` (v0.22.0, quoted in full; it is the first ``-f`` of every
    ``docker compose`` command Harbor runs for a task, see ``_docker_compose_paths`` at
    https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L316)::

        services:
          main:
            build:
              context: ${CONTEXT_DIR}
            pull_policy: build
            command: [ "sh", "-c", "sleep infinity" ]

    This is that file with ``CONTEXT_DIR`` resolved to the task's ``environment/`` folder and one addition:
    ``image`` set to ``hb__<hash>``, the value Harbor exports as ``MAIN_IMAGE_NAME``
    (``ComposeInfraEnvVars.main_image_name`` at
    https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L238).
    Harbor's template sets no ``image``: its ``compose build`` tags ``<session>-main`` unless the task's own
    Compose file says ``image: ${MAIN_IMAGE_NAME}``. Gym needs a stable name to inspect, push and record
    the result under, so it sets that value itself. No ``dockerfile`` is named, as in Harbor's file: Compose
    defaults to ``Dockerfile``, and naming it would clash with a task file's ``dockerfile_inline`` (Compose
    refuses a service declaring both). Every other field is Harbor's, verbatim.
    """
    return {
        "services": {
            MAIN_SERVICE: {
                "build": {"context": str(context)},
                "image": image,
                "pull_policy": "build",
                "command": ["sh", "-c", "sleep infinity"],
            }
        }
    }


def images_compose_document(images: Mapping[str, str]) -> dict[str, Any]:
    """Gym's images override for ``docker compose build``: ``image`` per built service, ``main`` included.

    Harbor passes its template, then the task's file, then its own generated overrides (``docker.py#L363-L371`` at
    v0.22.0). Gym's one override, passed last, names every service it builds: ``main`` as ``hb__<hash>`` (the
    template already says so) and each sidecar as ``hb__<hash>-<service>``. Last wins in Compose, so a sidecar's
    own ``image:`` in the task file (kumo's ``${MAIN_IMAGE_NAME}_verifier``) is overridden without a ``docker tag``
    step, and a sidecar without one gets Gym's name instead of Compose's ``<project>-<service>`` default, which
    under the project name Gym passes is the same string.
    """
    return {"services": {service: {"image": image} for service, image in images.items()}}


def write_compose_files(folder: Path, context: Path, main_image: str, images: Mapping[str, str]) -> tuple[Path, Path]:
    """Write Harbor's build template (:func:`harbor_compose_document`) and Gym's images override
    (:func:`images_compose_document`) under ``folder`` and return their paths, in the order they are passed."""
    build_path = Path(folder) / COMPOSE_BUILD_FILE
    build_path.write_text(yaml.safe_dump(harbor_compose_document(context, main_image), sort_keys=False))
    images_path = Path(folder) / COMPOSE_IMAGES_FILE
    images_path.write_text(yaml.safe_dump(images_compose_document(images), sort_keys=False))
    return build_path, images_path


def task_compose_file(context: Path) -> Path | None:
    """The task's own Compose file in ``context``: ``docker-compose.yaml``, the one name Harbor reads, or ``None``.

    The loader accepts the other spellings (``docker-compose.yml``, ``compose.yaml``, ``compose.yml``) at run
    time; Harbor's Docker environment never appends them to a build, so a context shipping one is refused here
    rather than built from a file Harbor would ignore, under the same ``hb__<hash>`` as Harbor's build.
    """
    others = [name for name in COMPOSE_FILE_NAMES if name != HARBOR_COMPOSE_FILE and (context / name).is_file()]
    if others:
        raise BuildError(
            f"{context / others[0]}: Harbor's Docker environment reads a task's Compose file from "
            f"environment/{HARBOR_COMPOSE_FILE} only (docker.py#L312 at v0.22.0), and the docker backend builds "
            f"exactly what Harbor builds. Rename the file to {HARBOR_COMPOSE_FILE} so the task builds and runs "
            "with the same file."
        )
    path = context / HARBOR_COMPOSE_FILE
    return path if path.is_file() else None


def _refuse(path: Path, service: str, key: str, why: str) -> BuildError:
    return BuildError(f"{path}: services.{service}.{key} is refused for a Dockerfile build: {why}")


def _contained_build_path(value: Any, *, base: Path, context: Path, path: Path, service: str, key: str) -> Path:
    """``value`` (a Compose build path, already interpolated) resolved against ``base`` and checked to stay under
    ``context`` with no symlink escape; otherwise :class:`BuildError`."""
    text = str(value)
    if "://" in text or text.startswith("git@"):
        raise _refuse(path, service, key, f"{text!r} is a remote context; the build context is the environment folder")
    candidate = Path(text) if Path(text).is_absolute() else Path(base) / text
    root = Path(context).resolve()
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
        raise _refuse(path, service, key, f"{text!r} resolves outside the task's environment folder {root}")
    return candidate


def _cache_credentials(entry: Any) -> str | None:
    """The credential key a ``cache_from`` / ``cache_to`` entry carries, if any."""
    if isinstance(entry, dict):
        keys = {str(key).strip().lower() for key in entry}
    else:
        keys = {part.split("=", 1)[0].strip().lower() for part in str(entry).split(",")}
    found = sorted(keys & CACHE_CREDENTIAL_KEYS)
    return found[0] if found else None


def compose_interpolation_variables(
    context: Path, env: dict[str, str], *, tasks: tuple[str, ...] = ()
) -> dict[str, str]:
    """The variables ``docker compose`` interpolates the task's Compose file with for a command run with ``env``:
    that environment over the project directory's ``.env`` file (Compose reads it from ``--project-directory``,
    which Harbor sets to ``context``; a variable set in the environment wins over the file). ``.env`` values are
    read as Compose's dotenv reader reads them (``compose-go/dotenv``): an unquoted or double-quoted value is
    itself interpolated, in file order, against the environment and then the lines above it, with the document's
    ``${VAR}``, ``${VAR:-default}`` and ``$VAR`` forms and ``$$`` (or ``\\$`` in double quotes) for a literal
    dollar; a single-quoted value is literal. A reference is resolved as it is in the document
    (:func:`substitute_with_empty`, the loader's rule): ``NET=${X:-host}`` in ``.env`` is ``host`` to the check as
    it is to Compose, a bare ``${X}`` nobody sets is an empty string with one warning naming ``tasks``, and
    ``${X:?msg}`` is refused. ``COMPOSE_ENV_FILES`` set in ``env`` is refused: it points Compose at other files
    than the ``.env`` read here."""
    if COMPOSE_ENV_FILES in env:
        raise BuildError(
            f"{COMPOSE_ENV_FILES} is set in the build's environment; a Dockerfile build reads interpolation "
            f"variables from the task's environment/{COMPOSE_DOTENV} only, so unset it for the build"
        )
    variables: dict[str, str] = {}
    dotenv = Path(context) / COMPOSE_DOTENV
    unset: set[str] = set()
    if dotenv.is_file():
        try:
            for key, value, quote in env_file_entries(dotenv.read_text(), escaped_dollar="$$"):
                if quote != "'":
                    value = substitute_with_empty(value, {**variables, **env}, unset).replace("$$", "$")
                variables[key] = value
        except ComposeSubstitutionError as exc:
            raise BuildError(
                f"{dotenv}: {exc}; a value docker compose interpolates in .env (unquoted or double-quoted) can read "
                "only the build's environment, an earlier line of the file or its own default"
            ) from exc
    _warn_unset(context, tasks, unset)
    variables.update(env)
    return variables


def _warn_unset(context: Path, tasks: tuple[str, ...], unset: set[str]) -> None:
    """One warning per context and variable for a bare ``${VAR}`` the build resolved to an empty string."""
    for name in sorted(unset):
        if (str(context), name) in _WARNED_UNSET:
            continue
        _WARNED_UNSET.add((str(context), name))
        LOGGER.warning(
            "%s: Compose variable ${%s} has no value in the build's environment (the process environment, the "
            "task's [environment].env and Harbor's variables) or environment/%s for task(s) %s; it is an empty "
            "string, as docker compose makes it",
            context / HARBOR_COMPOSE_FILE,
            name,
            COMPOSE_DOTENV,
            ", ".join(tasks) or "?",
        )


def validate_task_compose(
    context: Path, env: dict[str, str], *, tasks: tuple[str, ...] = (), builds_main: bool = True
) -> dict[str, Any] | None:
    """Check the task's Compose file before ``docker compose build`` runs it on the build host, and return it
    interpolated (``None`` when the context ships none). Anything refused raises :class:`BuildError` naming the
    service and key.

    The file is third-party content and the build host often holds the push credentials, so what it may ask
    of the host is bounded to what a legitimate task build needs: a service with a ``build`` section may carry
    args, a target, a context and a Dockerfile under ``environment/`` (symlinks resolved, like the loader's
    ``env_file`` rule), or a ``dockerfile_inline``. Refused on any service with a ``build`` section: ``secrets``, ``ssh``,
    ``extra_hosts``, ``network: host``, ``additional_contexts`` other than ``service:`` references, and
    ``cache_from`` / ``cache_to`` entries carrying credentials; refused anywhere: a top-level ``include``
    and an ``extends`` with a ``file``, which pull further files in unchecked. ``main``'s ``image`` has to
    resolve to ``MAIN_IMAGE_NAME`` when set and ``main`` is among the services being built (``builds_main``):
    ``${MAIN_IMAGE_NAME}``, ``${MAIN_IMAGE_NAME:-...}`` and ``${MAIN_IMAGE_NAME-...}`` all do, since Harbor
    always exports the variable; otherwise the build would be tagged under another name than the one recorded.
    A sidecar-only build (a prebuilt, pull-mode or overlay ``main``, or one already recorded) neither reads nor
    tags ``main``'s image, so its spelling is not checked, as Harbor's prebuilt mode never reads it.

    Compose interpolates the file against the command's environment before it reads a single key, so the checks
    run on the document interpolated the same way: ``env`` is the environment the build runs with
    (:func:`compose_build_env`), the project directory's ``.env`` under it with its values expanded as Compose's
    dotenv reader expands them (:func:`compose_interpolation_variables`), with Compose's ``${VAR}``, ``${VAR:-default}``, ``${VAR-default}``,
    ``${VAR:?error}``, ``${VAR:+alt}`` and ``$VAR`` forms on every string. Otherwise ``network: ${X:-host}`` or
    ``cache_to: [type=gha,${X:-token}=abc]`` would pass a check on the raw text and resolve to the refused value at
    build time. A bare ``${VAR}`` nobody sets is an empty string with one warning naming ``tasks`` and the variable,
    as ``docker compose build`` makes it and as the loader resolves the same file at seed
    (:func:`substitute_with_empty`), so the document checked is the document built; ``${VAR:?message}`` is refused
    with the message. ``main``'s ``image`` is interpolated the same way before it is compared, so any spelling
    Compose resolves to the main image passes and a literal other image, or a reference to another variable, is
    refused with what it resolves to.

    A sidecar with a ``build`` section passes the same checks as ``main``'s and is built (:func:`plan_builds`),
    as Harbor builds it. Its ``image:``, when it has one, is accepted whatever it says: the build is tagged with
    Gym's name for it (:meth:`ImageBuilder.tag_for`) and the seed document is filled from the record by service
    name, so no rule on its spelling is needed.
    """
    path = task_compose_file(context)
    if path is None:
        return None
    try:
        document = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise BuildError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise BuildError(f"{path} is not a mapping")
    if "include" in document:
        raise BuildError(
            f"{path}: include is refused for a Dockerfile build: it pulls in files the build does not check"
        )
    raw_main = document.get("services", {}).get(MAIN_SERVICE) if isinstance(document.get("services"), dict) else None
    main_image = raw_main.get("image") if isinstance(raw_main, dict) else None
    unset: set[str] = set()
    try:
        variables = compose_interpolation_variables(context, env, tasks=tasks)
        if builds_main and main_image:
            expected = variables.get(MAIN_IMAGE_VARIABLE)
            try:
                resolved = str(substitute_compose(str(main_image), variables)).strip()
            except ComposeSubstitutionError:
                resolved = None
            if expected is None or resolved != expected:
                resolves_to = "" if resolved in (None, str(main_image).strip()) else f" (it resolves to {resolved!r})"
                raise BuildError(
                    f"{path}: services.{MAIN_SERVICE}.image must be ${{{MAIN_IMAGE_VARIABLE}}} (the value Harbor exports; "
                    f"${{{MAIN_IMAGE_VARIABLE}:-...}} and ${{{MAIN_IMAGE_VARIABLE}-...}} resolve to it too) for a "
                    f"Dockerfile build, not {main_image!r}{resolves_to}"
                )
        document = substitute_with_empty(document, variables, unset)
    except ComposeSubstitutionError as exc:
        who = f" of task(s) {', '.join(tasks)}" if tasks else ""
        raise BuildError(
            f"{path}: interpolation is refused for a Dockerfile build{who}: {exc}. Set the variable in the "
            "environment of this prepare, in the task's [environment].env or in environment/.env, or give it a "
            "default, so the file checked is the file docker compose builds."
        ) from exc
    _warn_unset(context, tasks, unset)
    services = document.get("services")
    if not isinstance(services, dict):
        return document
    for name, service in services.items():
        service_name = str(name)
        if not isinstance(service, dict):
            continue
        extends = service.get("extends")
        if isinstance(extends, dict) and extends.get("file"):
            raise _refuse(path, service_name, "extends.file", "it pulls in a file the build does not check")
        build = service.get("build")
        if build is None:
            continue
        if isinstance(build, str):
            build = {"context": build}
        if not isinstance(build, dict):
            raise _refuse(path, service_name, "build", "it must be a path or a mapping")
        for key in REFUSED_BUILD_KEYS:
            if key in build:
                raise _refuse(path, service_name, f"build.{key}", "it reaches the build host")
        if str(build.get("network") or "").strip() == "host":
            raise _refuse(path, service_name, "build.network", "host networking reaches the build host")
        extra = build.get("additional_contexts") or {}
        extra_values = (
            list(extra.values()) if isinstance(extra, dict) else [str(item).split("=", 1)[-1] for item in extra]
        )
        for value in extra_values:
            if not str(value).startswith("service:"):
                raise _refuse(
                    path, service_name, "build.additional_contexts", f"{value!r} is not a service: reference"
                )
        for key in ("cache_from", "cache_to"):
            for item in build.get(key) or []:
                credential = _cache_credentials(item)
                if credential is not None:
                    raise _refuse(path, service_name, f"build.{key}", f"an entry carries {credential}=")
        build_context = _contained_build_path(
            build.get("context", "."),
            base=context,
            context=context,
            path=path,
            service=service_name,
            key="build.context",
        )
        if build.get("dockerfile") is not None:
            _contained_build_path(
                build["dockerfile"],
                base=build_context,
                context=context,
                path=path,
                service=service_name,
                key="build.dockerfile",
            )
    return document


def resolve_task_env(env: dict[str, str], *, tasks: tuple[str, ...] = ()) -> dict[str, str]:
    """A task's ``[environment].env`` the way Harbor hands it to ``docker compose`` (``resolve_env_vars``,
    https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/utils/env.py#L94): a whole value of the
    form ``${VAR}`` or ``${VAR:-default}`` is read from the process environment, the default standing in when
    the variable is unset; any other value is literal. A ``${VAR}`` with no default and no value fails the
    build naming it, as Harbor fails the trial."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = HARBOR_ENV_TEMPLATE.fullmatch(str(value))
        if match is None:
            resolved[str(key)] = str(value)
            continue
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            resolved[str(key)] = os.environ[name]
        elif default is not None:
            resolved[str(key)] = default
        else:
            raise BuildError(
                f"[environment].env {key} of task(s) {', '.join(tasks) or '?'} reads ${{{name}}}, which is not set in "
                "the environment of this prepare and has no default"
            )
    return resolved


def compose_build_env(entry: BuildPlanEntry, image: str, platform: str | None) -> dict[str, str]:
    """The environment of ``docker compose build``, merged as Harbor's ``_compose_env_vars`` merges it
    (https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L579):
    the process environment, then the task's ``[environment].env`` (:func:`resolve_task_env`), then Harbor's
    infrastructure variables, which win: ``MAIN_IMAGE_NAME``, ``CONTEXT_DIR``, ``CPUS`` and ``MEMORY`` (as
    ``<n>M``) from the task's resources, and the legacy ``HOST_*_PATH`` / ``ENV_*_PATH`` log mounts, plus the
    two the loader defines for the same file (``nemo_gym.tasks.harbor.image_configs.compose_variables``):
    ``TEST_DIR`` and ``COMPOSE_PROJECT_NAME`` (``hb__<hash>``, the image name; a stable stand-in, not Harbor's
    per-trial project name), so the build and the seed interpolate a task's file from the same set. A task
    variable named like an infrastructure one is ignored with a warning, as ``merge_compose_env`` does.
    ``DOCKER_DEFAULT_PLATFORM`` carries ``builds.platform``, the one setting Harbor has no equivalent of.
    """
    env = dict(os.environ)
    user = resolve_task_env(entry.env, tasks=entry.tasks)
    infra = {
        "MAIN_IMAGE_NAME": image,
        "CONTEXT_DIR": str(entry.context),
        "TEST_DIR": TEST_DIR,
        "COMPOSE_PROJECT_NAME": f"hb__{entry.digest}",
    }
    if entry.cpus:
        infra["CPUS"] = str(entry.cpus)
    if entry.memory_mb:
        infra["MEMORY"] = f"{entry.memory_mb}M"
    for suffix, path in LOG_MOUNT_PATHS.items():
        infra[f"HOST_{suffix}_PATH"] = path
        infra[f"ENV_{suffix}_PATH"] = path
    collisions = sorted(set(user) & set(infra))
    if collisions:
        LOGGER.warning(
            "[environment].env of task(s) %s names variables reserved by Harbor's Compose infrastructure, ignored: %s",
            ", ".join(entry.tasks),
            ", ".join(collisions),
        )
    env.update(user)
    env.update(infra)
    if platform:
        env["DOCKER_DEFAULT_PLATFORM"] = platform
    return env


class ImageBuilder:
    """Build the services of one context into named images with docker, buildah or podman, and push them when asked.

    ``main``'s image is named as Harbor names it, ``hb__<hash>``, a sidecar's ``hb__<hash>-<service>`` (see
    :meth:`tag_for`). ``repository`` is the registry path the names go under (``registry/path/hb__<hash>``) and
    the images are pushed there. Without a repository an image stays local under its name in the Docker daemon,
    the only local store a sandbox provider starts images from, so buildah and podman need a repository. ``push``
    defaults to whether a repository was named. ``platform`` (``linux/amd64`` by default, the platform sandboxes
    run) is passed to the tool, and an image that comes out for another platform is refused before it is pushed
    or recorded. ``run`` is the command seam.

    docker builds the way Harbor's Docker environment does: one ``docker compose build`` over Harbor's build
    Compose file, the task's own and Gym's images override (:meth:`compose_build_argv`), never ``docker build``.
    buildah and podman build each service's Dockerfile directly (:meth:`build_argv`).
    """

    def __init__(
        self,
        *,
        tool: str | None = None,
        repository: str | None = None,
        push: bool | None = None,
        platform: str | None = None,
        storage_root: Path | None = None,
        run: Runner | None = None,
    ) -> None:
        self.repository = repository.rstrip("/") if repository else None
        self.tool = tool or detect_tool(self.repository)
        if self.tool is None:
            wanted = (
                ", ".join(BUILD_TOOLS) if self.repository else "docker (buildah and podman need builds.repository)"
            )
            raise BuildError(f"No image builder found: install {wanted}, or name one with builds.tool")
        if self.tool not in BUILD_TOOLS:
            raise BuildError(f"Unknown build tool {self.tool!r}; expected one of " + ", ".join(BUILD_TOOLS))
        if self.tool != "docker" and not self.repository:
            raise BuildError(
                f"{self.tool} keeps built images in a store no sandbox provider reads; set builds.repository "
                "so the image is pushed, or build with docker to keep it in the local daemon"
            )
        self.push = bool(self.repository) if push is None else push
        if self.push and not self.repository:
            raise BuildError("Pushing a built image needs builds.repository")
        self.platform = platform or DEFAULT_PLATFORM
        self.storage_root = Path(storage_root) if storage_root is not None else default_storage_root()
        self._run = run or _run_default
        self._compose_checked = False
        self._tool_env: dict[str, str] | None = None
        # The heredoc preflight's outcome: None until it ran, "" when the tool builds heredocs, else why it does not.
        self._heredoc_failure: str | None = None

    # -- commands ------------------------------------------------------------------------------

    def _storage_flags(self) -> list[str]:
        if self.tool != "buildah":
            return []
        return [
            "--storage-driver",
            "vfs",
            "--root",
            str(self.storage_root / "storage"),
            "--runroot",
            str(self.storage_root / "run"),
        ]

    def tag_for(self, entry: BuildPlanEntry) -> str:
        """The image's name under the repository when there is one: ``hb__<hash>`` for ``main``,
        ``hb__<hash>-<service>`` for a sidecar.

        The hash is the entry's digest, Harbor's environment id of the context, and ``hb__<hash>`` is the
        value Harbor exports as ``MAIN_IMAGE_NAME`` for it (the name a task Compose file that sets
        ``image: ${MAIN_IMAGE_NAME}`` builds under; Harbor's own template leaves the tag to Compose). A sidecar
        takes Compose's default form for a service without ``image:``, ``<project>-<service>``, under the project
        name Gym passes to ``docker compose build`` (``hb__<hash>``; Harbor's is the session id, so its sidecar tags
        change per run and are never published). The separate verifier's image takes the sidecar form too,
        ``hb__<hash>-verifier`` with Harbor's hash of ``tests/``: the hash is the one Harbor exports as
        ``MAIN_IMAGE_NAME`` for its verifier environment, and the suffix names the role where Harbor's own tag is the
        per-session ``<session>-main``. One flat name per image, sanitised like Harbor's, stays within the
        one repository path ``builds.repository`` promises; a slash would open a path per hash.
        """
        return self.tag_name(entry.digest, entry.service)

    def tag_name(self, digest: str, service: str = MAIN_SERVICE) -> str:
        """:meth:`tag_for` by digest and service."""
        name = f"hb__{digest}" if service == MAIN_SERVICE else f"hb__{digest}-{service}"
        name = sanitize_docker_image_name(name)
        return f"{self.repository}/{name}" if self.repository else name

    def build_argv(self, entry: BuildPlanEntry, tag: str) -> list[str]:
        """The direct Dockerfile build of buildah and podman (docker builds through :meth:`compose_build_argv`).

        ``--format docker`` builds the image format BuildKit builds, so ``SHELL`` applies to the ``RUN`` steps and
        ``HEALTHCHECK`` and ``ONBUILD`` reach the image (the tools' default OCI format has none of the three).
        The service's ``build`` section of the task's Compose file, which ``docker compose build`` would apply (over
        Harbor's template for ``main``), is applied the same way: its context and Dockerfile, ``--build-arg`` per arg
        and ``--target`` (:func:`service_build_settings`).
        """
        platform = ["--platform", self.platform] if self.platform else []
        options = [*platform]
        for key, value in entry.build_args.items():
            options += ["--build-arg", f"{key}={value}"]
        if entry.target:
            options += ["--target", entry.target]
        tail = ["-t", tag, "-f", str(entry.dockerfile_path), str(entry.build_dir)]
        if self.tool == "buildah":
            storage = self._storage_flags()
            return ["buildah", *storage, "bud", "--format", IMAGE_FORMAT, "--isolation", "chroot", *options, *tail]
        return [self.tool, "build", "--format", IMAGE_FORMAT, *options, *tail]

    def compose_build_argv(
        self,
        entry: BuildPlanEntry,
        compose_file: Path,
        images_file: Path | None = None,
        services: tuple[str, ...] = (),
    ) -> list[str]:
        """Harbor's ``docker compose build`` line for the context (``_run_docker_compose_command``,
        https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py#L606).

        The project name stands in for Harbor's session id (Compose's default image name is
        ``<project>-<service>``; ``main`` is named explicitly, see :func:`harbor_compose_document`); the
        project directory is the context, so relative paths in the task's Compose file resolve as Harbor
        resolves them; the files are Harbor's build template, then the task's ``docker-compose.yaml`` when
        it has one, in Harbor's order, then Gym's images override (:func:`images_compose_document`), last so its
        names win. ``services`` names the services to build; Harbor passes none, and so does a full build, so every
        service with a ``build`` section is built. A partial build (some services of the context recorded, or
        ``main`` not planned) names the ones it builds.
        """
        argv = [
            "docker",
            "compose",
            "--project-name",
            f"hb__{entry.digest}",
            "--project-directory",
            str(entry.context),
            "-f",
            str(compose_file),
        ]
        task_compose = task_compose_file(entry.context)
        if task_compose is not None:
            argv += ["-f", str(task_compose)]
        if images_file is not None:
            argv += ["-f", str(images_file)]
        return [*argv, "build", *services]

    def push_argv(self, tag: str, digest_file: Path) -> list[str]:
        if self.tool == "buildah":
            return [
                "buildah",
                *self._storage_flags(),
                "push",
                "--digestfile",
                str(digest_file),
                tag,
                f"docker://{tag}",
            ]
        if self.tool == "podman":
            return ["podman", "push", "--digestfile", str(digest_file), tag]
        return ["docker", "push", tag]

    def inspect_argv(self, tag: str) -> list[str]:
        if self.tool == "buildah":
            return ["buildah", *self._storage_flags(), "inspect", "--type", "image", tag]
        return [self.tool, "image", "inspect", tag]

    def tool_env(self) -> dict[str, str] | None:
        """The environment buildah and podman commands run with, or ``None`` (inherit) for docker.

        The host's ``containers.conf`` may give every container a default bind mount (this deployment mounts
        ``/etc/localtime`` read-only into each one through ``containers.conf.d``), and buildah and podman apply
        those mounts to build steps too, so a Dockerfile that installs ``tzdata`` (Ubuntu with ``python3`` pulls
        it in) fails with ``mv: cannot move '/etc/localtime.dpkg-new' to '/etc/localtime': Device or resource
        busy``. The builder writes a ``containers.conf`` with no default volumes under its storage root, once, and
        hands it to the tool as ``CONTAINERS_CONF_OVERRIDE``, the layer containers-common applies over the host
        files. A ``CONTAINERS_CONF_OVERRIDE`` the caller already set is left as it is.

        Both tools resolve an unqualified ``FROM`` (``FROM oven/bun:1``) through the host's ``registries.conf``, and
        Debian's and Ubuntu's stock file defines no unqualified-search registry, so the build fails with
        ``short-name "oven/bun:1" did not resolve to an alias`` where ``docker compose build`` pulls from docker.io.
        The builder writes a ``registries.conf`` naming docker.io as the one unqualified-search registry under its
        storage root and hands it to the tool as ``CONTAINERS_REGISTRIES_CONF``. That file replaces the host's main
        ``registries.conf`` (the drop-ins under ``registries.conf.d``, short-name aliases included, still load), so a
        site whose main file carries mirrors or blocked registries exports its own ``CONTAINERS_REGISTRIES_CONF``,
        which is left as it is. docker needs neither file and inherits the process environment.
        """
        if self.tool == "docker":
            return None
        if self._tool_env is None:
            env = dict(os.environ)
            self.storage_root.mkdir(parents=True, exist_ok=True)
            for variable, name, text in (
                (CONTAINERS_CONF_OVERRIDE, CONTAINERS_CONF_FILE, CONTAINERS_CONF_TEXT),
                (CONTAINERS_REGISTRIES_CONF, REGISTRIES_CONF_FILE, REGISTRIES_CONF_TEXT),
            ):
                if variable in os.environ:
                    continue
                path = self.storage_root / name
                if not path.is_file() or path.read_text() != text:
                    path.write_text(text)
                env[variable] = str(path)
            self._tool_env = env
        return self._tool_env

    # -- the pipeline --------------------------------------------------------------------------

    def _check(self, argv: list[str], what: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        try:
            completed = self._run(argv) if env is None else self._run(argv, env=env)
        except OSError as exc:
            raise BuildError(f"{what} failed to start ({argv[0]}): {exc}") from exc
        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-15:])
            message = f"{what} failed (exit {completed.returncode}): {' '.join(argv)}\n{tail}"
            note = _proxy_note(tail, env if env is not None else dict(os.environ))
            if note:
                LOGGER.warning("%s: %s", what, note)
                message += "\n" + note
            raise BuildError(message)
        return completed

    def build(self, entry: BuildPlanEntry) -> BuildResult:
        """:meth:`build_many` for one service, returning its result."""
        (result,) = list(self.build_many([entry]))
        return result

    def build_many(self, entries: list[BuildPlanEntry], *, partial: bool = False) -> Iterator[BuildResult]:
        """Build the services ``entries`` of one context, push when configured, inspect, and yield each result to
        record as soon as it is done, in order. Raises :class:`BuildError`; what was yielded before stays built.

        The task's Compose file is read here, not at planning: it is checked once for the context
        (:func:`validate_task_compose`) and each entry's service's ``build`` section fills it
        (:func:`service_build_settings`), so a file that does not interpolate, escapes ``environment/`` or carries a
        refused key fails this context's build and the caller leaves its tasks out, the way it leaves out a task
        whose build command fails. The build's environment is Harbor's (:func:`compose_build_env`) with
        ``MAIN_IMAGE_NAME`` the ``main`` tag, whether or not ``main`` is among the entries, as Harbor always exports
        it.

        docker runs one ``docker compose build`` for the context, Harbor's build step, and then inspects and pushes
        each image; ``partial`` says ``entries`` are not every service the context builds (the others are
        recorded), so the build names them rather than building the whole project. buildah and podman run one build
        per service (:meth:`build_argv`); on that path ``dockerfile_inline`` and ``additional_contexts`` (even
        ``service:`` references, which need ``--build-context`` and a build order) are refused naming the service,
        before the context's first build, since the checked file already says so, and ``no_cache`` and ``pull`` are
        ignored, as the direct tools have no Compose to honour them.
        """
        if not entries:
            return
        first = entries[0]
        tasks = ", ".join(first.tasks[:5])
        if first.verifier:
            # Harbor builds and starts every service of tests/docker-compose.yaml as the verifier's Compose project;
            # the server starts one verifier sandbox, so the loader refuses such a task. Refused here too, before
            # any command, so the docker backend builds no sidecar that nothing records or runs.
            sidecars = verifier_compose_sidecars(first.context.parent)
            if sidecars:
                raise BuildError(
                    f"{first.context / HARBOR_COMPOSE_FILE} of task(s) {tasks} declares service(s) "
                    f"{', '.join(sidecars)} beside main; Harbor starts them as the separate verifier's Compose "
                    "project and this server starts a single verifier sandbox"
                )
        main_tag = self.tag_name(first.digest)
        if self.tool == "docker":
            self._ensure_compose_plugin()
        env = compose_build_env(first, main_tag, self.platform)
        # Checked as docker compose will read it: interpolated against the build's own environment. On every tool:
        # a context the loader refuses is never built and pushed under the hb__ name.
        document = validate_task_compose(
            first.context,
            env,
            tasks=first.tasks,
            builds_main=any(entry.compose_service == MAIN_SERVICE for entry in entries),
        )
        settled = [replace(entry, **service_build_settings(entry, document, env)) for entry in entries]
        for entry in settled:
            if not entry.dockerfile_inline and not entry.dockerfile_path.is_file():
                where = "" if entry.compose_service == MAIN_SERVICE else f" for services.{entry.service}.build"
                raise BuildError(f"{entry.build_dir} has no {entry.dockerfile}{where}")
            if self.tool != "docker":
                # Knowable from the checked document: refused before this context builds or pushes anything.
                self._refuse_on_direct_path(entry, document, tasks)
        if self.tool == "buildah":
            self.storage_root.mkdir(parents=True, exist_ok=True)
        if self.tool == "docker":
            # By Compose service: the verifier entry is Harbor's `main` of the tests/ context, tagged with Gym's name
            # for it through the images override, which is passed last and wins over the template's hb__<hash>.
            images = {entry.compose_service: self.tag_for(entry) for entry in settled}
            # Harbor builds the whole project; a build of some of its services names them.
            services = () if not partial and MAIN_SERVICE in images else tuple(images)
            what = f"Building {', '.join(images.values())} for task(s) {tasks}"
            LOGGER.info("%s: docker compose build of service(s) %s", what, ", ".join(images))
            with tempfile.TemporaryDirectory(prefix="nemo-gym-compose-") as tmp:
                build_file, images_file = write_compose_files(Path(tmp), first.context, main_tag, images)
                self._check(self.compose_build_argv(first, build_file, images_file, services), what, env=env)
            for entry in settled:
                yield self._finish(entry, images[entry.compose_service])
            return
        for entry in settled:
            tag = self.tag_for(entry)
            what = f"Building {tag} for task(s) {tasks}"
            if dockerfile_has_heredoc(entry.dockerfile_path.read_text()):
                self._ensure_heredoc_support(entry.dockerfile_path)
            self._check(self.build_argv(entry, tag), what, env=self.tool_env())
            yield self._finish(entry, tag)

    def _refuse_on_direct_path(self, entry: BuildPlanEntry, document: dict[str, Any] | None, tasks: str) -> None:
        """The build section keys only ``docker compose build`` can honour, refused or ignored by buildah and podman."""
        path = entry.context / HARBOR_COMPOSE_FILE
        service = entry.compose_service
        if entry.dockerfile_inline:
            raise BuildError(
                f"{path}: services.{service}.build.dockerfile_inline is not supported by the direct "
                f"{self.tool} build of task(s) {tasks}; docker compose build reads it, so build with "
                "builds.tool: docker, or move the Dockerfile text into a file under environment/"
            )
        section = build_section(document, service) or {}
        if section.get("additional_contexts"):
            raise BuildError(
                f"{path}: services.{service}.build.additional_contexts is not supported by the direct "
                f"{self.tool} build of task(s) {tasks} (it needs --build-context and a build order); build with "
                "builds.tool: docker"
            )
        for key in ("no_cache", "pull"):
            if key in section:
                LOGGER.debug(
                    "%s: services.%s.build.%s is ignored by the direct %s build", path, service, key, self.tool
                )

    def _finish(self, entry: BuildPlanEntry, tag: str) -> BuildResult:
        """Inspect the built ``tag``, refuse another platform, push when configured, and return the result."""
        tasks = ", ".join(entry.tasks[:5])
        record = self.inspect(tag)
        built_platform = f"{record.get('os')}/{record.get('architecture')}"
        if platform_key(built_platform) != platform_key(self.platform):
            service = "" if entry.is_main else f" (service {entry.service})"
            raise BuildError(
                f"Building {tag} for task(s) {tasks}{service}: the image came out as {built_platform}, not "
                f"{self.platform}. Build on a host of that platform, or one with qemu/binfmt emulation; "
                "builds.platform overrides the target."
            )
        image = tag
        if self.push:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-push-") as tmp:
                digest_file = Path(tmp) / "digest"
                completed = self._check(self.push_argv(tag, digest_file), f"Pushing {tag}", env=self.tool_env())
                digest = _pushed_digest(self.tool, digest_file, completed, self._run, tag)
            image = f"{tag}@{digest}"
        record = {**record, "image": image}
        return BuildResult(
            digest=entry.digest,
            tasks=entry.tasks,
            tag=tag,
            image=image,
            tool=self.tool,
            pushed=self.push,
            record=record,
            service=entry.service,
        )

    def _ensure_compose_plugin(self) -> None:
        """Fail before anything is built when ``docker compose`` (the Compose v2 plugin) is missing."""
        if self._compose_checked:
            return
        argv = ["docker", "compose", "version"]
        try:
            completed = self._run(argv)
        except OSError as exc:
            raise BuildError(
                f"docker failed to start ({exc}); the docker backend needs docker and {COMPOSE_PLUGIN}"
            ) from exc
        if completed.returncode != 0:
            detail = [line for line in (completed.stderr or completed.stdout or "").splitlines() if line.strip()]
            raise BuildError(
                f"The docker backend builds with `docker compose build`, as Harbor does, and the Compose plugin "
                f"is not available: `{' '.join(argv)}` exited {completed.returncode}"
                + (f" ({detail[0].strip()})" if detail else "")
                + f". Install {COMPOSE_PLUGIN} (Docker Compose v2), or build with buildah or podman and a "
                "builds.repository."
            )
        self._compose_checked = True

    def _ensure_heredoc_support(self, dockerfile: Path) -> None:
        """Fail before a Dockerfile with a heredoc is built when this host's buildah or podman cannot parse one.

        Ubuntu 24.04's packaged buildah 1.33.7 and podman 4.9.3 are linked against openshift/imagebuilder 1.2.5,
        which reads the body of ``RUN <<EOF`` as instructions and fails with ``Unknown instruction: "ECHO"``. The
        preflight builds :data:`HEREDOC_PROBE_DOCKERFILE` (``FROM scratch`` plus a ``COPY`` heredoc; no base image
        is pulled) once per builder with the build's own flags and environment, and a failure is one
        :class:`BuildError` naming the tool, its version, the Dockerfile and the remedy (:data:`HEREDOC_REMEDY`)
        instead of one ``Unknown instruction`` per task. The outcome is cached either way.
        """
        if self._heredoc_failure is None:
            with tempfile.TemporaryDirectory(prefix="nemo-gym-heredoc-probe-") as tmp:
                probe = BuildPlanEntry("heredoc-probe", Path(tmp), ())
                probe.dockerfile_path.write_text(HEREDOC_PROBE_DOCKERFILE)
                argv = self.build_argv(probe, HEREDOC_PROBE_TAG)
                try:
                    completed = self._run(argv, env=self.tool_env())
                except OSError as exc:
                    self._heredoc_failure = f"{argv[0]} failed to start: {exc}"
                else:
                    tail = "\n".join((completed.stderr or completed.stdout or "").strip().splitlines()[-3:])
                    self._heredoc_failure = (
                        "" if completed.returncode == 0 else f"exit {completed.returncode}: {tail or 'no output'}"
                    )
            if self._heredoc_failure:
                self._heredoc_failure = f"{self._tool_version()}: {self._heredoc_failure}"
        if self._heredoc_failure:
            raise BuildError(
                f"{dockerfile} uses a Dockerfile heredoc (<<EOF) and this host's {self.tool} cannot build one: the "
                f"preflight build of `FROM scratch` + `COPY <<EOF /probe` failed ({self._heredoc_failure}). "
                + HEREDOC_REMEDY
            )

    def _tool_version(self) -> str:
        """``<tool> --version``'s first line, or the tool's name when it will not say."""
        try:
            completed = self._run([self.tool, "--version"])
        except (OSError, ValueError, StopIteration):
            return self.tool
        line = ((completed.stdout or "").strip() or (completed.stderr or "").strip()).splitlines()
        return line[0].strip() if line else self.tool

    def inspect(self, tag: str) -> dict[str, Any]:
        """The image's OCI configuration in the ``compose-images.json`` record shape (``image`` unset)."""
        completed = self._check(self.inspect_argv(tag), f"Inspecting {tag}", env=self.tool_env())
        try:
            data = json.loads(completed.stdout)
        except ValueError as exc:
            raise BuildError(f"Inspecting {tag}: {self.tool} returned no JSON ({exc})") from exc
        return normalize_inspect(self.tool, data)


def proxy_variable_names(env: dict[str, str]) -> list[str]:
    """The proxy-related variables in ``env`` (``http_proxy``, ``HTTPS_PROXY``, ``no_proxy``, ``all_proxy`` and the
    like), names only: their values can carry credentials."""
    return sorted(name for name in env if name.lower().endswith("_proxy"))


def _proxy_note(output: str, env: dict[str, str]) -> str:
    """Why a registry step may have answered 403, when the tool's output says it did: a proxy the build tool
    inherits (from its environment, or from its own configuration when the environment names none) can answer
    403 to the registry ping unless ``no_proxy`` carries the registry host. Names only, never values."""
    if not HTTP_403.search(output):
        return ""
    names = proxy_variable_names(env)
    if names:
        return (
            f"The registry answered 403 and the build tool's environment carries proxy variables {', '.join(names)}: "
            "a proxy inherited by the build tool can answer 403 to the registry ping; add the registry host to "
            "no_proxy (and NO_PROXY) or clear the proxy variables for the build."
        )
    return (
        "The registry answered 403 and the build tool's environment carries no proxy variables; a proxy the tool reads "
        "from its own configuration (containers.conf [engine].env, ~/.docker/config.json proxies, the Docker daemon) "
        "can still answer 403 to the registry ping unless its no_proxy carries the registry host."
    )


def _pushed_digest(tool: str, digest_file: Path, completed: subprocess.CompletedProcess, run: Runner, tag: str) -> str:
    """The manifest digest the registry assigned to the pushed image."""
    if tool in ("buildah", "podman"):
        if digest_file.is_file():
            digest = digest_file.read_text().strip()
            if digest.startswith("sha256:"):
                return digest
        raise BuildError(f"Pushing {tag}: {tool} wrote no digest file")
    # docker: the push output ends with `latest: digest: sha256:... size: ...`; RepoDigests holds it afterwards.
    for line in (completed.stdout or "").splitlines():
        if "digest: sha256:" in line:
            return "sha256:" + line.split("digest: sha256:", 1)[1].split()[0]
    inspected = run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", tag])
    if inspected.returncode == 0:
        for repo_digest in json.loads(inspected.stdout or "[]"):
            if "@sha256:" in repo_digest:
                return repo_digest.split("@", 1)[1]
    raise BuildError(f"Pushing {tag}: docker reported no digest")


def normalize_inspect(tool: str, data: Any) -> dict[str, Any]:
    """Turn a tool's inspect JSON into the ``compose-images.json`` record shape (without ``image``).

    buildah reports ``{"OCIv1": {"architecture", "os", "config"}, "Docker": {"config"}, "FromImageID"}``; podman and
    docker report a list with ``Architecture``, ``Os``, ``Config`` and ``Id``. The image is built in docker format
    (:data:`IMAGE_FORMAT`), whose ``Shell``, ``Healthcheck`` and ``OnBuild`` the OCI image configuration cannot
    hold: buildah reports them under ``Docker.config``, read here over the OCI config; docker reports them in
    ``Config``; podman's ``Config`` is the OCI image configuration (its inspect does not expose ``Shell`` and
    ``OnBuild``, although the image carries them) and ``Healthcheck`` sits beside ``Config`` as a top-level key,
    read here when ``Config`` has none.
    """
    if tool == "buildah":
        if not isinstance(data, dict) or "OCIv1" not in data:
            raise BuildError("buildah inspect returned no OCIv1 block")
        oci = data["OCIv1"]
        config = {**(oci.get("config") or {}), **_docker_config(data)}
        config_digest = data.get("FromImageID") or ""
        architecture, os_name = oci.get("architecture"), oci.get("os")
    else:
        entry = data[0] if isinstance(data, list) and data else data
        if not isinstance(entry, dict) or "Config" not in entry:
            raise BuildError(f"{tool} image inspect returned no Config block")
        config = {**(entry.get("Config") or {}), **_docker_config(entry)}
        # podman: the docker-format healthcheck is a top-level field of the inspect entry.
        config.setdefault("Healthcheck", entry.get("Healthcheck"))
        config_digest = str(entry.get("Id") or "")
        architecture, os_name = entry.get("Architecture"), entry.get("Os")
    if config_digest and not config_digest.startswith("sha256:"):
        config_digest = f"sha256:{config_digest}"
    if (os_name, architecture) != ("linux", "amd64"):
        LOGGER.warning(
            "Built image is %s/%s, not linux/amd64; it runs only on a sandbox host of that platform",
            os_name,
            architecture,
        )
    return {
        "architecture": architecture,
        "config": {key: config[key] for key in BUILT_CONFIG_KEYS if config.get(key)},
        "config_digest": config_digest,
        "os": os_name,
    }


def _docker_config(data: dict[str, Any]) -> dict[str, Any]:
    """The set fields of an inspect's ``Docker.config`` section (the docker-format image configuration), or ``{}``."""
    docker = data.get("Docker")
    config = docker.get("config") if isinstance(docker, dict) else None
    return {key: value for key, value in config.items() if value} if isinstance(config, dict) else {}


# -- recording -----------------------------------------------------------------------------------


def platform_key(platform: str | None) -> str:
    """``os/arch`` of a platform string, with a variant such as ``linux/arm64/v8`` dropped."""
    parts = (platform or "").split("/")
    return "/".join(parts[:2])


def recorded_image_mismatch(
    entry: dict[str, Any],
    image_configs: dict[str, Any],
    *,
    repository: str | None,
    platform: str,
    service: str = MAIN_SERVICE,
) -> str | None:
    """Why the image a ``built-images.json`` entry records for ``service`` cannot serve this builder, or ``None``
    when it can.

    A service the entry does not record is missing. A local (unpushed) image serves only local builds; a pushed
    one serves both, judged by that service's own ``pushed`` flag (:func:`recorded_service_records`: the entry's
    flag stands in for a record written before the flags were kept per service), since the services of one
    context may have been built by differently configured runs. The recorded platform has to be the one asked for.
    """
    service_record = recorded_service_records(entry).get(service)
    if service_record is None:
        return f"no image is recorded for service {service}"
    if not service_record.get("pushed") and repository is not None:
        return "it was built locally and this run pushes to a repository"
    image = service_record["image"]
    record = image_configs.get(image)
    if record is None:
        return "its configuration is missing from compose-images.json"
    recorded = f"{record.get('os')}/{record.get('architecture')}"
    if platform_key(recorded) != platform_key(platform):
        return f"it is {recorded}, not {platform}"
    return None


def record_built_image(folder: Path, result: BuildResult, *, overwrite: bool = False) -> Path:
    """Record one built image next to the task folders: under its context digest and service in
    ``built-images.json`` and by image in ``compose-images.json``. Existing entries are kept unless ``overwrite``.

    The digest's entry is merged, not replaced: ``services[<service>]`` is set to this build's ``image``, ``tool``
    and ``pushed`` (the other services keep theirs: one context's services may be built by differently configured
    runs, and a pushed ``main`` must not vouch for a local sidecar or the reverse), ``tasks`` unioned, and the
    top-level keys older readers use kept equal to ``main``'s: ``image`` (absent while ``main`` is not recorded)
    with ``tool`` and ``pushed`` (the first recorded service's while ``main`` is not recorded). A service recorded
    once keeps its image unless ``overwrite``, like ``compose-images.json`` keeps its entries.

    Both files are merged and replaced under their sidecar locks (:func:`record_lock`), the built-images
    lock first and the compose-images lock inside it, always in that order; :func:`record_image_configs`
    takes only the compose-images lock, so the two writers cannot deadlock and neither loses the other's
    entries. The configuration lands before the build entry, as before, so a reader that finds the entry
    finds its configuration.
    """
    folder = Path(folder)
    built_path = folder / BUILT_IMAGES_FILE
    configs_path = folder / IMAGE_CONFIGS_FILE
    with record_lock(built_path):
        built = read_built_images(folder)
        existing = built.get(result.digest) if isinstance(built.get(result.digest), dict) else {}
        services = recorded_service_records(existing)
        if result.service in services and not overwrite:
            # Pinned: a service recorded once keeps its image, like compose-images.json keeps its entries.
            return built_path
        services[result.service] = {"image": result.image, "tool": result.tool, "pushed": result.pushed}
        tasks = sorted(set(result.tasks) | {str(task) for task in existing.get("tasks") or []})
        lead = services.get(MAIN_SERVICE) or next(iter(services.values()))
        entry: dict[str, Any] = {
            "tasks": tasks,
            "tool": lead.get("tool", result.tool),
            "pushed": lead.get("pushed", result.pushed),
            "services": services,
        }
        if MAIN_SERVICE in services:
            entry["image"] = services[MAIN_SERVICE]["image"]
        built[result.digest] = entry
        with record_lock(configs_path):
            configs = read_image_configs(folder)
            if overwrite:
                configs[result.image] = result.record
            else:
                configs.setdefault(result.image, result.record)
            write_json_record(configs_path, configs)
        write_json_record(built_path, built)
    return built_path


def build_images(
    folder: Path,
    *,
    builder: ImageBuilder,
    select: Callable[[str], bool] | None = None,
    contexts: Collection[str] | None = None,
    log: Callable[[str], None] = print,
) -> list[BuildResult]:
    """Build and record every service of every context under ``folder`` that needs it and has no usable record.

    ``select`` limits the plan to the tasks whose folder name it accepts, ``contexts`` to the entries whose digest it
    holds (a task in separate verifier mode owns two contexts, its ``environment/`` and its verifier's ``tests/``;
    prepare builds one context per call so a failure is scoped to it).

    A recorded image is reused when it fits this builder: pushed, or local when nothing is pushed,
    and of the requested platform. Otherwise it is rebuilt and its record replaced, and the reason
    is logged. A context is keyed by its content hash, so a changed build context is a new build;
    deleting a context's entry from ``built-images.json`` forces one for an unchanged context. The services of
    one context are built together (:meth:`ImageBuilder.build_many`), the ones with a fitting record left out,
    and each is recorded as soon as it is done, so a prepare interrupted after a context's first service
    resumes at its second. Returns the images built by this call. Raises :class:`BuildError` on the first failure,
    after recording whatever was built before it.
    """
    folder = Path(folder).resolve()
    parent = folder.parent if is_task_folder(folder) else folder
    plan = plan_builds(folder, select=select)
    if contexts is not None:
        plan = [entry for entry in plan if entry.digest in contexts]
    if not plan:
        return []
    recorded = read_built_images(parent)
    image_configs = read_image_configs(parent)
    results: list[BuildResult] = []
    groups: dict[str, list[BuildPlanEntry]] = {}
    for entry in plan:
        groups.setdefault(entry.digest, []).append(entry)
    for digest, group in groups.items():
        names = ", ".join(group[0].tasks[:3])
        images = recorded_service_images(recorded.get(digest))
        pending: list[BuildPlanEntry] = []
        for entry in group:
            who = names if entry.is_main else f"{names} ({entry.service})"
            if entry.service in images:
                why = recorded_image_mismatch(
                    recorded[digest],
                    image_configs,
                    repository=builder.repository,
                    platform=builder.platform,
                    service=entry.service,
                )
                if why is None:
                    log(f"Built image recorded for {who}: {images[entry.service]}")
                    continue
                log(f"Rebuilding {images[entry.service]} for {who}: {why}")
            pending.append(entry)
        if not pending:
            continue
        for result in builder.build_many(pending, partial=len(pending) < len(group)):
            # The build just made is authoritative for its image: a configuration left behind under the same
            # name (by a deleted built-images.json entry, or a record that no longer fits) is replaced.
            record_built_image(parent, result, overwrite=True)
            where = "pushed" if result.pushed else "local"
            log(f"Built {result.image} ({where}, {result.tool}) for task(s) {', '.join(result.tasks[:5])}")
            results.append(result)
    return results
