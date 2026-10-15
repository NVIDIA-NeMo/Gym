# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the OCI configuration of the images a task runs.

Docker reads an image's environment, user, working directory, entrypoint, command and
exposed ports from the image itself. A sandbox platform without Docker cannot, so Gym
records them from the registry into ``compose-images.json`` next to the task folders,
one entry per image reference exactly as the task wrote it, pinned to the manifest digest it
resolved to for the platform the sandboxes run (linux/amd64 unless ``builds.platform`` names
another). Two readers depend on the file:

- the harbor resources server, for the sidecar images a Compose task starts (see
  ``nemo_gym.sandbox.compose_config.resolve_compose``);
- the task loader, for the base image of a pull-mode or overlay-mode Dockerfile, whose
  ``ENV``, ``WORKDIR`` and ``USER`` lines resolve from the base image's own configuration
  the way ``docker build`` resolves them (see ``nemo_gym.tasks.harbor.dockerfile``).
"""

import base64
import hashlib
import json
import logging
import os
import re
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections import ChainMap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from nemo_gym.sandbox.compose_config import (
    DEFAULT_PLATFORM,
    ComposeFieldError,
    ComposeSubstitutionError,
    ComposeVolumeError,
    check_compose_fields,
    check_sidecar_extra_hosts,
    env_file_entries,
    inline_env_files,
    plan_compose_volumes,
    residual_interpolation,
    substitute_compose,
)
from nemo_gym.sandbox.providers.registry import provider_service_option_fields
from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from nemo_gym.tasks.harbor.task import (
    COMPOSE_FILE_NAMES,
    DOCKERFILE,
    IMAGE_CONFIGS_FILE,
    MAIN_SERVICE,
    TASK_FILE,
    HarborTask,
    dockerfile_base_image,
    is_task_folder,
    iter_task_folders,
    read_image_configs,
    record_lock,
    write_json_record,
)


LOGGER = logging.getLogger(__name__)

COMPOSE_IMAGES_FILE = IMAGE_CONFIGS_FILE
# The fields Compose would read from the image to start it, plus the build-time state (environment,
# user, working directory) a Dockerfile's instructions resolve from; the rest of the config is not recorded.
RECORDED_CONFIG_KEYS = ("Cmd", "Entrypoint", "Env", "ExposedPorts", "Healthcheck", "User", "WorkingDir")
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
DOCKER_HUB_REGISTRY = "registry-1.docker.io"
PLATFORM = ("linux", "amd64")


def platform_pair(platform: str | None) -> tuple[str, str]:
    """``(os, architecture)`` of an ``os/arch[/variant]`` platform string such as ``builds.platform``; ``None``
    (or an empty string) is linux/amd64, the default platform sandboxes run."""
    parts = [part for part in (platform or "").lower().split("/") if part]
    if not parts:
        return PLATFORM
    if len(parts) < 2:
        raise ImageConfigError(f"Not a platform: {platform!r} (expected os/arch, for example linux/arm64)")
    return parts[0], parts[1]


class ImageConfigError(RuntimeError):
    """An image reference could not be resolved to a configuration for the platform the sandboxes run."""


@dataclass(frozen=True)
class ImageRef:
    """A parsed image reference: ``[registry/]repository[:tag][@digest]``."""

    registry: str
    repository: str
    tag: str | None
    digest: str | None

    @property
    def manifest_ref(self) -> str:
        return self.digest or self.tag or "latest"

    @property
    def pinned_repository(self) -> str:
        """The repository a digest pin is written against: as the registry names it, host included off Docker Hub."""
        if self.registry != DOCKER_HUB_REGISTRY:
            return f"{self.registry}/{self.repository}"
        return self.repository


def parse_image_ref(reference: str) -> ImageRef:
    """Split an image reference the way Docker does, defaulting to Docker Hub."""
    rest, _, digest = reference.partition("@")
    first, slash, remainder = rest.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        registry, path = first, remainder
    else:
        registry, path = DOCKER_HUB_REGISTRY, rest
    if registry == "docker.io":
        registry = DOCKER_HUB_REGISTRY
    repository, _, tag = path.partition(":")
    if registry == DOCKER_HUB_REGISTRY and "/" not in repository:
        repository = f"library/{repository}"
    if not repository:
        raise ImageConfigError(f"Not an image reference: {reference!r}")
    return ImageRef(registry=registry, repository=repository, tag=tag or None, digest=digest or None)


class RegistryClient:
    """Read manifests and configuration blobs over the OCI distribution API with the standard library.

    ``credentials`` maps a registry host to ``(username, password)`` for private repositories;
    public ones use the anonymous token the registry hands out. ``platform`` is the ``(os, architecture)``
    an image reference is resolved for (linux/amd64 by default): the manifest an index names for it, and
    the only one a single-platform reference may be.
    """

    def __init__(
        self,
        credentials: dict[str, tuple[str, str]] | None = None,
        *,
        timeout_s: float = 60,
        retries: int = 3,
        platform: tuple[str, str] = PLATFORM,
    ) -> None:
        self.credentials = credentials or {}
        self.timeout_s = timeout_s
        self.retries = retries
        self.platform = tuple(platform)
        self._tokens: dict[tuple[str, str], str] = {}

    def image_config(self, reference: str) -> dict[str, Any]:
        """One ``compose-images.json`` entry for ``reference``."""
        ref = parse_image_ref(reference)
        manifest, manifest_digest = self._manifest(ref, ref.manifest_ref)
        if manifest.get("mediaType") in INDEX_MEDIA_TYPES or "manifests" in manifest:
            manifest_digest = self._select_platform(ref, manifest)
            manifest, _ = self._manifest(ref, manifest_digest)
        config_digest = manifest["config"]["digest"]
        blob = json.loads(self._get(ref, f"blobs/{config_digest}", accept="*/*")[0])
        if (blob.get("os"), blob.get("architecture")) != self.platform:
            raise ImageConfigError(
                f"{reference}: resolved to {blob.get('os')}/{blob.get('architecture')}, not {'/'.join(self.platform)}"
            )
        config = blob.get("config") or {}
        return {
            "architecture": blob["architecture"],
            "config": {key: config[key] for key in RECORDED_CONFIG_KEYS if config.get(key)},
            "config_digest": config_digest,
            "image": f"{ref.pinned_repository}@{manifest_digest}",
            "os": blob["os"],
        }

    def _select_platform(self, ref: ImageRef, index: dict[str, Any]) -> str:
        for entry in index.get("manifests", []):
            platform = entry.get("platform") or {}
            if (entry.get("annotations") or {}).get("vnd.docker.reference.type") == "attestation-manifest":
                continue
            if (platform.get("os"), platform.get("architecture")) == self.platform and not platform.get("variant"):
                return entry["digest"]
        raise ImageConfigError(f"{ref.pinned_repository}: the image index has no {'/'.join(self.platform)} manifest")

    def _manifest(self, ref: ImageRef, manifest_ref: str) -> tuple[dict[str, Any], str]:
        body, headers = self._get(ref, f"manifests/{manifest_ref}", accept=MANIFEST_ACCEPT)
        digest = headers.get("Docker-Content-Digest") or "sha256:" + hashlib.sha256(body).hexdigest()
        return json.loads(body), digest

    def _get(self, ref: ImageRef, path: str, *, accept: str) -> tuple[bytes, dict[str, str]]:
        url = f"https://{ref.registry}/v2/{ref.repository}/{path}"
        headers = {"Accept": accept}
        token = self._tokens.get((ref.registry, ref.repository))
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            return self._open(url, headers)
        except urllib.error.HTTPError as exc:
            challenge = exc.headers.get("WWW-Authenticate", "") if exc.code == 401 else ""
            if not challenge:
                raise ImageConfigError(
                    f"{ref.pinned_repository}: registry returned HTTP {exc.code} for {path}"
                ) from exc
        token = self._token(ref, challenge)
        self._tokens[(ref.registry, ref.repository)] = token
        headers["Authorization"] = f"Bearer {token}"
        try:
            return self._open(url, headers)
        except urllib.error.HTTPError as exc:
            raise ImageConfigError(f"{ref.pinned_repository}: registry returned HTTP {exc.code} for {path}") from exc

    def _token(self, ref: ImageRef, challenge: str) -> str:
        params = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        realm = params.get("realm")
        if not challenge.lower().startswith("bearer") or not realm:
            raise ImageConfigError(f"{ref.pinned_repository}: unsupported registry challenge {challenge!r}")
        query = {key: params[key] for key in ("service", "scope") if key in params}
        query.setdefault("scope", f"repository:{ref.repository}:pull")
        headers = {}
        if ref.registry in self.credentials:
            username, password = self.credentials[ref.registry]
            headers["Authorization"] = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        try:
            body, _ = self._open(f"{realm}?{urllib.parse.urlencode(query)}", headers)
        except urllib.error.HTTPError as exc:
            raise ImageConfigError(f"{ref.pinned_repository}: token request failed with HTTP {exc.code}") from exc
        payload = json.loads(body)
        token = payload.get("token") or payload.get("access_token")
        if not token:
            raise ImageConfigError(f"{ref.pinned_repository}: token response carried no token")
        return token

    def _open(self, url: str, headers: dict[str, str]) -> tuple[bytes, dict[str, str]]:
        """One GET; connection-level failures (not HTTP errors) are retried with a short backoff."""
        request = urllib.request.Request(url, headers=headers)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return response.read(), dict(response.headers)
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise ImageConfigError(
                        f"registry unreachable after {attempt + 1} attempts: {url} ({exc})"
                    ) from exc
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def compose_file(task: HarborTask) -> Path | None:
    """The task's Compose overlay, if it has one."""
    for name in COMPOSE_FILE_NAMES:
        path = task.path / "environment" / name
        if path.is_file():
            return path
    return None


# In-sandbox paths Harbor's legacy ``${HOST_*_PATH}`` and ``${ENV_*_PATH}`` log-mount variables stand for.
LOG_MOUNT_PATHS = {"VERIFIER_LOGS": "/logs/verifier", "AGENT_LOGS": "/logs/agent", "ARTIFACTS": "/logs/artifacts"}
# Where a task's ``tests/`` folder lands in the sandbox, the value of ``TEST_DIR``. Harbor's terminal-bench mapper
# writes ``TEST_DIR=${TEST_DIR}`` into its Compose files and Harbor never sets the variable, so it is empty under
# Harbor; Gym defines it, a deliberate improvement.
TEST_DIR = "/tests"
# Harbor's ``[environment].env`` template (``_TEMPLATE_PATTERN`` in ``harbor/utils/env.py`` at v0.22.0): a whole
# value ``${VAR}`` or ``${VAR:-default}``, read from the host environment.
HARBOR_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
# The project directory's env file Compose interpolates with, below everything the command's environment sets.
COMPOSE_DOTENV = ".env"
# ``(task id, variable)`` pairs already warned about: the server loads a task's document once per episode.
_WARNED_UNSET: set[tuple[str, str]] = set()


def compose_variables(task: HarborTask, *, unset: set[str] | None = None) -> dict[str, str]:
    """The variables a task's Compose file may interpolate, resolved in the order Harbor's ``docker compose`` sees.

    Harbor runs ``docker compose`` with the process environment, the task's ``[environment].env`` over it and
    its own infrastructure variables over both (``merge_compose_env`` in
    ``harbor/environments/docker/compose_env.py`` and ``_compose_env_vars`` in ``docker.py`` at v0.22.0, called
    with ``include_os_env=True``), and Compose reads the project directory's ``.env`` below everything the
    command's environment sets. Highest first:

    1. Gym's infrastructure set: ``MAIN_IMAGE_NAME`` (the resolved task image), ``CONTEXT_DIR`` (the task's
       ``environment/`` folder), ``TEST_DIR`` (:data:`TEST_DIR`), ``COMPOSE_PROJECT_NAME`` (``hb__<hash>``,
       Harbor's environment id of the context and the ``main`` image name, a stable placeholder: compose-go
       injects the real value from ``--project-name``, the trial session id under Harbor and ``compose-<uuid>``
       per episode in Gym's adapter, which has no Compose-generated names for it to match), ``PREBUILT_IMAGE_NAME``
       when ``docker_image`` is declared, ``CPUS`` and ``MEMORY`` (as ``<n>M``, Harbor's spelling) from the
       task's resources, and the ``HOST_*_PATH`` / ``ENV_*_PATH`` log mounts as the ``/logs`` tree;
    2. the task's ``[environment].env``, each value read as Harbor's ``resolve_env_vars`` reads it
       (``harbor/utils/env.py``): a whole value ``${VAR}`` or ``${VAR:-default}`` comes from the process
       environment, the default standing in when the variable is unset, and one with neither raises
       :class:`ImageConfigError` naming the task and the variable, as Harbor fails the trial; any other value
       is literal;
    3. the process environment;
    4. ``environment/.env``, its unquoted and double-quoted values interpolated against the levels above and
       the lines before them, as Compose's dotenv reader does; single-quoted values are literal.

    A reference none of them sets takes its default (``${VAR:-x}``, ``${VAR-x}``), else it is an empty string,
    as docker compose makes it, with one warning naming the task and the variable (added to ``unset`` when
    given, so the caller warns once for the whole document, else logged here). ``${VAR:?msg}`` and
    ``${VAR?msg}`` raise :class:`ImageConfigError` with the message. The process environment is not part of
    the task's digest: prepare and the environment server must run with the same variables, as Harbor's one
    process does.

    Every value is literal to the document, as it is to Compose (which splices a variable's value in verbatim and
    never re-scans it): a ``$`` in a host, task or infrastructure value is doubled here, the convention
    :func:`parse_env_file` and :func:`_dotenv_variables` keep for a literal dollar, so a password such as
    ``$2b$12$...`` neither trips the document's residual check nor loses a ``$$`` when the sandbox starts.
    """
    environment = task.config.environment
    context = task.path / "environment"
    infra: dict[str, str] = {"MAIN_IMAGE_NAME": task.image or "", "CONTEXT_DIR": str(context), "TEST_DIR": TEST_DIR}
    if context.is_dir():
        # Not the runtime project name (Harbor: the trial session id; Gym: the adapter's compose-<uuid>). A stable
        # per-task value, so files like o11y-bench's `o11y-stack-${COMPOSE_PROJECT_NAME}` alias and URL agree.
        infra["COMPOSE_PROJECT_NAME"] = f"hb__{task.context_id or context_digest(context)}"
    if environment.docker_image:
        infra["PREBUILT_IMAGE_NAME"] = environment.docker_image
    if environment.cpus:
        infra["CPUS"] = str(environment.cpus)
    if environment.memory_mb:
        infra["MEMORY"] = f"{environment.memory_mb}M"
    for suffix, path in LOG_MOUNT_PATHS.items():
        infra[f"HOST_{suffix}_PATH"] = path
        infra[f"ENV_{suffix}_PATH"] = path
    host = {key: _literal(value) for key, value in os.environ.items()}
    above = {
        **host,
        **_task_env(task, environment.env or {}),
        **{key: _literal(value) for key, value in infra.items()},
    }
    collected: set[str] = set() if unset is None else unset
    variables = {**_dotenv_variables(task, context, above, collected), **above}
    if unset is None:
        _warn_unset(task, collected)
    return variables


def _literal(value: str) -> str:
    """``value`` as the document's interpolation and literal checks read a literal: every ``$`` doubled."""
    return value.replace("$", "$$")


def _task_env(task: HarborTask, env: Mapping[str, Any]) -> dict[str, str]:
    """``[environment].env`` as Harbor's ``resolve_env_vars`` hands it to ``docker compose`` (see
    :func:`compose_variables`), each value literal to the document (:func:`_literal`)."""
    resolved: dict[str, str] = {}
    for key, value in env.items():
        match = HARBOR_ENV_TEMPLATE.fullmatch(str(value))
        if match is None:
            resolved[str(key)] = _literal(str(value))
            continue
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            resolved[str(key)] = _literal(os.environ[name])
        elif default is not None:
            resolved[str(key)] = _literal(default)
        else:
            raise ImageConfigError(
                f"{task.task_id}: [environment].env {key} reads ${{{name}}}, which is not set in the process "
                "environment and has no default"
            )
    return resolved


def _dotenv_variables(task: HarborTask, context: Path, above: Mapping[str, str], unset: set[str]) -> dict[str, str]:
    """``environment/.env`` read as Compose's dotenv reader reads it (``compose-go/dotenv``): an unquoted or
    double-quoted value is interpolated against ``above`` and then the lines before it, a single-quoted value is
    literal. A dollar that stays literal is kept doubled, as :func:`parse_env_file` keeps it, so the document's
    literal check leaves it alone."""
    dotenv = context / COMPOSE_DOTENV
    if not dotenv.is_file():
        return {}
    variables: dict[str, str] = {}
    try:
        for key, value, quote in env_file_entries(dotenv.read_text(), escaped_dollar="$$"):
            if quote == "'":
                variables[key] = value.replace("$", "$$")
            else:
                variables[key] = substitute_with_empty(value, ChainMap(above, variables), unset)
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: environment/{COMPOSE_DOTENV}: {exc}") from exc
    return variables


def substitute_with_empty(value: Any, variables: Mapping[str, str], unset: set[str]) -> Any:
    """:func:`substitute_compose` with Compose's rule for a bare ``${VAR}`` or ``$VAR`` nobody sets: an empty
    string, the name added to ``unset`` for the caller to warn about once. Only the bare form raises the resolver's
    default message; ``${VAR:-x}`` keeps its default, ``${VAR:?msg}`` its refusal, and every other failure passes
    through unchanged. The loader and the Dockerfile build (``nemo_gym.tasks.harbor.image_build``) interpolate a
    task's Compose file through this one rule, so the two never disagree on the same file."""
    if isinstance(value, list):
        return [substitute_with_empty(item, variables, unset) for item in value]
    if isinstance(value, dict):
        return {key: substitute_with_empty(item, variables, unset) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    empty: dict[str, str] = {}
    while True:
        try:
            return substitute_compose(value, ChainMap(variables, empty))
        except ComposeSubstitutionError as exc:
            if str(exc) != str(ComposeSubstitutionError(exc.name)) or exc.name in empty:
                raise
            empty[exc.name] = ""
            unset.add(exc.name)


def _warn_unset(task: HarborTask, unset: set[str]) -> None:
    for name in sorted(unset):
        if (task.task_id, name) in _WARNED_UNSET:
            continue
        _WARNED_UNSET.add((task.task_id, name))
        LOGGER.warning(
            "%s: Compose variable ${%s} has no value from the task, the process environment or environment/.env; "
            "it is an empty string, as docker compose makes it",
            task.task_id,
            name,
        )


def load_compose_document(task: HarborTask) -> dict[str, Any] | None:
    """The task's Compose overlay with ``env_file`` entries inlined, ``${VAR}`` references resolved and the images
    prepare built standing in for the services it built (:func:`with_built_images`), ``main`` only when the loader
    resolved the task to its built image.

    Returns ``None`` for a task without a Compose file. Prepare and the resources server both read the
    file through this function, so the document the sandbox starts is the one prepare validated, provided
    both run with the same process environment (see :func:`compose_variables`). A required variable
    (``${VAR:?msg}``) without a value, a task variable the host does not set or a missing ``env_file``
    raises :class:`ImageConfigError` naming it; a variable nobody sets is an empty string, with a warning.
    The record is the one :func:`~nemo_gym.tasks.harbor.task.load_task` read (``task.built_images``): nothing is
    hashed here, which runs once per episode on the server.
    """
    path = compose_file(task)
    if path is None:
        return None
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        raise ImageConfigError(f"{task.task_id}: {path.name} is not a mapping")
    unset: set[str] = set()
    try:
        # Interpolate first, as Compose does, so an env_file path may use ${CONTEXT_DIR}; env_file values are then
        # literal. Anything still carrying a ``$NAME`` afterwards fails here, at prepare, not at seed.
        variables = compose_variables(task, unset=unset)
        document = substitute_with_empty(document, variables, unset)
        document = inline_env_files(document, path.parent, variables)
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: {path.name}: {exc}") from exc
    _warn_unset(task, unset)
    residual = residual_interpolation(document)
    if residual is not None:
        # The key path only: the value next to it may be a host secret.
        raise ImageConfigError(f"{task.task_id}: {path.name}: unresolved interpolation left at {residual}")
    images = dict(task.built_images)
    if task.built_image is None:
        # The loader resolved main elsewhere (docker_image declared, or pull/overlay mode): a recorded main is
        # another task's build of the same context, or a build that predates the override. Harbor runs the
        # prebuilt image then, so main keeps ${MAIN_IMAGE_NAME} = task.image. Sidecars are built whatever main's
        # mode and are always substituted.
        images.pop(MAIN_SERVICE, None)
    return with_built_images(document, images)


# The keys of a service that describe how Harbor builds its image, meaningless once the image is built and recorded.
BUILD_KEYS = ("build", "pull_policy")


def with_built_images(document: dict[str, Any], images: Mapping[str, str]) -> dict[str, Any]:
    """``document`` with the recorded image of every service in ``images`` (service name to image, as
    ``built-images.json`` records them) set as its ``image`` and its ``build`` and ``pull_policy`` removed.

    Harbor builds every service with a ``build:`` section from the task's Compose file at every start (``main``
    with args, a target or a ``dockerfile_inline``, and the sidecars from their own contexts), and Harbor's own
    template sets ``pull_policy: build``. Gym builds those images once at prepare, with the same file, and
    records them by service name; here each recorded image stands in for its service's build, so a sidecar's own
    ``image:`` spelling (kumo's ``${MAIN_IMAGE_NAME}_verifier``, a local tag under Harbor) is replaced by the name
    Gym built and recorded. The two keys describe a build that already happened; the sandbox Compose adapter
    starts images only and refuses them as unsupported fields, so they are dropped here rather than failing every
    episode of a task prepare accepted. Services not in ``images`` are untouched: a task in pull or overlay mode
    keeps whatever its file gives ``main``, so an unsupported field there is still reported, and a sidecar
    prepare recorded no build for (with or without an ``image:`` of its own) is reported by
    :func:`compose_image_references`.
    """
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    for name, image in images.items():
        if name not in services:
            continue
        service = services[name] if isinstance(services[name], dict) else {}
        services[name] = {**{key: value for key, value in service.items() if key not in BUILD_KEYS}, "image": image}
    return document


# The keys of ``main``'s ``build:`` that only restate what Harbor's own build template supplies for the task's
# ``environment/`` folder; any other key configures a build Gym does not run in pull or overlay mode.
RESTATED_BUILD_KEYS = frozenset({"context", "dockerfile", "pull"})


def without_restated_main_build(task: HarborTask, document: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """``document`` with ``main``'s ``build:`` dropped when it only restates Harbor's own build of the task's
    Dockerfile, for a task in pull or overlay mode, and a reason instead when that section carries a key Gym
    cannot apply without a build. Any other task is returned as it is.

    Harbor builds ``main`` from ``environment/Dockerfile`` through its own template, which already sets
    ``build.context: ${CONTEXT_DIR}`` (``docker-compose-build.yaml`` at v0.22.0, appended before the task file),
    so a task file that repeats ``context``, ``dockerfile: Dockerfile`` or ``pull`` changes nothing there;
    orca-bench, terminal_bench_aai and data-eng-bench do, some 600 hub tasks. Gym runs such a Dockerfile without
    a build (the loader folded it into the task in pull mode, or runs its ``RUN`` lines at seed in overlay mode),
    so the restatement is dropped as a built task's section is (:func:`with_built_images`). ``args``,
    ``target``, ``dockerfile_inline``, another ``dockerfile`` or another ``context`` would change the image
    Harbor builds and have no equivalent without one, so the task is reported, not started on the wrong image.
    """
    if task.built_image is not None or task.config.environment.docker_image or not (task.path / DOCKERFILE).is_file():
        return document, None
    services = document.get("services") if isinstance(document, dict) else None
    main = services.get("main") if isinstance(services, dict) else None
    if not isinstance(main, dict) or "build" not in main:
        return document, None
    build = main["build"]
    if isinstance(build, str):
        build = {"context": build}
    unsupported = "cannot be applied in pull or overlay mode (Gym does not build this Dockerfile)"
    if not isinstance(build, dict):
        return document, f"main.build {unsupported}"
    offending = sorted(str(key) for key in build if key not in RESTATED_BUILD_KEYS)
    context = build.get("context")
    if context is not None and context not in (".", "./") and Path(str(context)) != task.path / "environment":
        offending.append("context")
    if build.get("dockerfile") not in (None, "Dockerfile", "./Dockerfile"):
        offending.append("dockerfile")
    if offending:
        return document, f"{', '.join(f'main.build.{key}' for key in offending)} {unsupported}"
    services["main"] = {key: value for key, value in main.items() if key != "build"}
    return document, None


def without_pull_policy(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with ``pull_policy`` removed from every service, for the sandbox Compose adapter.

    The key tells ``docker compose`` when to pull or build an image before starting it (Harbor's template sets
    ``pull_policy: build`` on ``main``; a task file may give a sidecar ``always`` or ``missing``). The sandbox
    starts the recorded image it is given and has no pull step of its own to configure, so the key is dropped
    from the document the sandbox receives, in every image mode, instead of being refused as an unsupported
    field. ``build:`` is not touched here: a sidecar's is still refused (no provider starts a locally built
    sidecar), a built service's is dropped by :func:`with_built_images` (the loader applies it for every service
    prepare built, ``pull_policy`` included) and ``main``'s by :func:`without_restated_main_build` when it only
    restates Harbor's build.
    """
    services = document.get("services") if isinstance(document, dict) else None
    if isinstance(services, dict):
        for name, service in services.items():
            if isinstance(service, dict) and "pull_policy" in service:
                services[name] = {key: value for key, value in service.items() if key != "pull_policy"}
    return document


def plan_task_compose_volumes(
    task: HarborTask, document: dict[str, Any], *, volume_sources: dict[str, str] | None = None
) -> dict[str, list[dict[str, str]]]:
    """Resolve the ``volumes`` of a task's Compose document for a sandbox group; mutates ``document``.

    Bind mounts of the task's own files become ``x-sandbox.uploads`` entries (performed into the service
    once it starts), named volumes stay for the provider's shared storage, and external volumes or bind
    sources outside the task resolve through ``volume_sources`` (the sandbox block's mapping). Anything
    the group cannot run raises :class:`ImageConfigError` naming the volume, path or field, so prepare
    fails before a sandbox starts. Returns the uploads per service.
    """
    path = compose_file(task)
    base_dir = path.parent if path is not None else task.path / "environment"
    try:
        return plan_compose_volumes(document, base_dir, root=task.path, volume_sources=volume_sources)
    except ComposeVolumeError as exc:
        raise ImageConfigError(f"{task.task_id}: {base_dir.name}/{path.name if path else ''}: {exc}") from exc


def compose_image_references(task: HarborTask, document: dict[str, Any] | None = None) -> list[str]:
    """Every image a Compose task starts: the task image for ``main`` and each sidecar's image."""
    if document is None:
        document = load_compose_document(task)
    if document is None:
        return []
    services = document.get("services") or {}
    references: list[str] = []
    main_image = (services.get(MAIN_SERVICE) or {}).get("image") or task.image
    if main_image:
        references.append(main_image)
    for name, service in services.items():
        if name == MAIN_SERVICE:
            continue
        service = service or {}
        if "build" in service:
            # Still carrying its build section after with_built_images: no record. Under Compose and Harbor a
            # service with build: is built, never pulled, so its image: (when it has one) is only a tag and is not
            # resolved from the registry.
            raise ImageConfigError(
                f"{task.task_id}: Compose service {name!r} has a build section and prepare recorded no build for it"
            )
        image = service.get("image")
        if not image:
            raise ImageConfigError(
                f"{task.task_id}: Compose service {name!r} names no image and prepare recorded no build for it"
            )
        references.append(image)
    return list(dict.fromkeys(references))


def task_declares_gpus(task: HarborTask) -> bool:
    """Whether ``task.toml`` asks for GPUs (agent or verifier environment): the task then runs on the GPU block."""
    verifier_environment = task.config.verifier.environment
    return bool(task.config.environment.gpus or (verifier_environment is not None and verifier_environment.gpus))


def _compose_field_error(task: HarborTask, exc: ComposeFieldError) -> ImageConfigError:
    path = compose_file(task)
    base_dir = path.parent if path is not None else task.path / "environment"
    return ImageConfigError(f"{task.task_id}: {base_dir.name}/{path.name if path else ''}: {exc}")


def check_task_compose_fields(
    task: HarborTask,
    document: dict[str, Any],
    *,
    provider: str | None = None,
    gpu_provider: str | None = None,
    platform: str = DEFAULT_PLATFORM,
) -> dict[str, Any]:
    """Give the service fields of a task's Compose document their meaning for a sandbox group; mutates ``document``.

    See :func:`nemo_gym.sandbox.compose_config.check_compose_fields`: ``deploy.resources`` become sandbox
    resources (GPUs included), ``restart`` and ``pull_policy`` take their recorded forms, ``platform`` must
    be ``platform``, the one the deployment runs (``sandbox_provider_options.platform`` when set, else
    linux/amd64, the one prepare records images for; see ``compose_config.deployment_platform``), ``networks``
    and ``extra_hosts`` their mapping forms, ``post_start`` hooks are checked, and ``tmpfs``, ``ulimits``,
    ``sysctls`` and ``security_opt`` must be among what the service's provider can apply. ``provider`` and
    ``gpu_provider`` are sandbox provider names (such as ``opensandbox``) of the task's block and of the
    ``gpu_sandbox_provider`` block; each service is checked against the one it runs on, as the server routes
    it: the task's services run on ``gpu_provider`` when ``task.toml`` declares GPUs, and a sidecar that
    reserves GPUs through ``deploy.resources`` runs there as well. Anything else raises
    :class:`ImageConfigError` naming the task, the file and the field, so prepare leaves the task out before a
    sandbox starts. Without a provider name the capability check runs at seed instead, against the live provider.

    The document checked is the one the server hands the adapter (``environment_servers/harbor/app.py``,
    ``_compose_document``): the services prepare built carrying their recorded images without ``build:`` or
    ``pull_policy`` (:func:`with_built_images`, applied by the loader), ``pull_policy`` dropped from every other
    service (the sandbox has no pull step to configure), and ``main``'s ``build:`` dropped when it only restates
    Harbor's own build of the task's Dockerfile (:func:`without_restated_main_build`; a build key Gym cannot apply
    without a build is this task's reason). A sidecar ``build:`` prepare recorded no build for and any other key
    the adapter does not start are then refused by the check's allow-list.
    """
    document = without_pull_policy(document)
    if task.built_image is None:
        document, reason = without_restated_main_build(task, document)
        if reason is not None:
            raise _compose_field_error(task, ComposeFieldError("build", reason))
    if gpu_provider is not None and task_declares_gpus(task):
        provider = gpu_provider
    fields = provider_service_option_fields(provider) if provider is not None else None
    gpu_fields = provider_service_option_fields(gpu_provider) if gpu_provider is not None else None
    try:
        return check_compose_fields(
            document,
            platform=platform,
            provider=provider,
            service_option_fields=fields,
            gpu_provider=gpu_provider,
            gpu_service_option_fields=gpu_fields,
        )
    except ComposeFieldError as exc:
        raise _compose_field_error(task, exc) from exc


def dockerfile_base_images(folder: Path, *, select: Callable[[str], bool] | None = None) -> dict[str, list[str]]:
    """The base image of every task Dockerfile the loader resolves under ``folder``, with the tasks using it.

    ``folder`` is a task folder or a folder of task folders, as ``discover_tasks`` reads it; with
    ``select``, only the tasks whose folder name it accepts count. The image a task declares is the
    effective one, exactly as ``load_task`` sees it: ``task.toml`` after the dataset's ``[gym]``
    overrides. A per-task ``docker_image`` override
    thus turns a Dockerfile with ``RUN`` lines into a prebuilt image whose ``FROM`` is never
    resolved, so there is nothing to record for it. Tasks whose ``task.toml`` does not load, and a
    dataset whose ``dataset.toml`` does not load (which fails every task), are left for the loader
    to report.
    """
    # Resolved like load_task does: overrides are keyed by the task folder's name and dataset.toml
    # sits at the dataset root (the task's parent for a single task folder, else the folder itself).
    folder = Path(folder).resolve()
    single = is_task_folder(folder)
    children = [folder] if single else iter_task_folders(folder)
    try:
        dataset = read_dataset_config(folder.parent if single else folder)
    except ValueError:
        return {}
    images: dict[str, list[str]] = {}
    for child in children:
        if select is not None and not select(child.name):
            continue
        try:
            config = HarborTaskConfig.model_validate(tomllib.loads((child / TASK_FILE).read_text()))
            config = apply_dataset_config(config, child.name, dataset)
        except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError):
            continue
        image = dockerfile_base_image(child, config.environment.docker_image or None)
        if image is not None:
            images.setdefault(image, []).append(child.name)
    return images


def record_image_configs(
    references: dict[str, list[str]],
    folder: Path,
    *,
    client: RegistryClient | None = None,
    what: str = "image",
    platform: str | None = None,
) -> Path | None:
    """Record the configuration of ``references`` (image reference to the tasks using it) into ``folder``.

    Entries already recorded are kept as they are, so a pinned recording never changes
    under a completed run; only references missing from the file are resolved. Returns
    the file's path, or ``None`` when there is nothing to record. A reference the registry
    cannot resolve raises :class:`ImageConfigError` naming the image, the tasks and the
    registry's answer. ``platform`` (``os/arch``, ``builds.platform``; linux/amd64 when ``None``) is
    the platform the default client resolves references for; a ``client`` given carries its own.

    The registry is asked outside the file's sidecar lock (:func:`record_lock`), so a slow
    registry never holds up another process; the file is then read again, merged and replaced
    under the lock, and an entry another process recorded in the meantime is kept over the
    one resolved here, as a pinned entry always is.
    """
    if not references:
        return None
    folder = Path(folder)
    path = folder / IMAGE_CONFIGS_FILE
    missing = [ref for ref in references if ref not in read_image_configs(folder)]
    if not missing:
        return path
    if client is None:
        client = RegistryClient() if platform is None else RegistryClient(platform=platform_pair(platform))
    LOGGER.info(f"Recording the configuration of {len(missing)} {what}(s) into {path}")
    resolved: dict[str, Any] = {}
    for reference in missing:
        try:
            resolved[reference] = client.image_config(reference)
        except ImageConfigError as exc:
            tasks = ", ".join(references[reference][:5]) or "-"
            raise ImageConfigError(
                f"Could not record the configuration of {what} {reference!r} (used by task(s) {tasks}): {exc}"
            ) from exc
    with record_lock(path):
        recorded = read_image_configs(folder)
        for reference, record in resolved.items():
            recorded.setdefault(reference, record)
        write_json_record(path, recorded)
    return path


def record_compose_images(
    tasks: list[HarborTask],
    folder: Path,
    *,
    client: RegistryClient | None = None,
    skipped: dict[str, str] | None = None,
    platform: str | None = None,
    sandbox_provider: str | None = None,
    gpu_sandbox_provider: str | None = None,
    compose_platform: str = DEFAULT_PLATFORM,
    volume_sources: dict[str, str] | None = None,
    gpu_volume_sources: dict[str, str] | None = None,
) -> Path | None:
    """Write ``compose-images.json`` in ``folder`` for the Compose tasks among ``tasks``.

    Returns the file's path, or ``None`` when no task uses Compose. Each Compose document's service fields are
    checked on the way (see :func:`check_task_compose_fields`), with ``sandbox_provider`` and
    ``gpu_sandbox_provider`` naming the providers of the ``sandbox:`` and ``gpu_sandbox_provider`` blocks for the
    capability check and ``compose_platform`` the deployment's, which a service's ``platform`` must match;
    ``platform`` is the one the images are recorded for, as :func:`record_image_configs` takes it (the built images'
    ``builds.platform``). Each document's ``volumes`` are planned next (see :func:`plan_task_compose_volumes`), with
    ``volume_sources`` standing for the deployment's external volumes; a task that declares GPUs is checked against
    ``gpu_volume_sources`` instead when that is given (the ``gpu_sandbox_provider`` block's mapping, which is what
    the server resolves such a task against; an empty mapping when that block has none). A task whose Compose file
    does not resolve (an :class:`ImageConfigError` from :func:`load_compose_document`), carries a field the group
    cannot run, or declares volumes it cannot run (the planner's :class:`ImageConfigError`) does not stop the others: with ``skipped`` given it is recorded there by task id with the reason and left out,
    otherwise one :class:`ImageConfigError` at the end names every such task. Its images are still recorded, so
    a fixed task file needs no new registry round-trip. Once the image configurations are recorded, a sidecar
    whose image runs as a non-root user (which the group starts without host injection) must declare no
    ``extra_hosts``, the way the adapter decides it at seed. An image the registry cannot resolve still raises
    for the whole set, as :func:`record_image_configs` says.
    """
    references: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    documents: list[tuple[HarborTask, dict[str, Any]]] = []
    for task in tasks:
        try:
            document = load_compose_document(task)
            if document is None:
                continue
            for reference in compose_image_references(task, document):
                references.setdefault(reference, []).append(task.task_id)
            check_task_compose_fields(
                task, document, provider=sandbox_provider, gpu_provider=gpu_sandbox_provider, platform=compose_platform
            )
            sources = (
                gpu_volume_sources if gpu_volume_sources is not None and task_declares_gpus(task) else volume_sources
            )
            plan_task_compose_volumes(task, document, volume_sources=sources)
        except ImageConfigError as exc:
            failures[task.task_id] = str(exc)
            continue
        documents.append((task, document))
    if failures and skipped is None:
        raise ImageConfigError(f"{len(failures)} Compose task(s) do not resolve: " + "; ".join(failures.values()))
    if failures:
        skipped.update(failures)
    path = record_image_configs(references, folder, client=client, what="Compose image", platform=platform)
    if path is not None:
        recorded = json.loads(path.read_text())
        for task, document in documents:
            try:
                check_sidecar_extra_hosts(document, recorded)
            except ComposeFieldError as exc:
                if skipped is None:
                    raise _compose_field_error(task, exc) from exc
                skipped[task.task_id] = str(_compose_field_error(task, exc))
    return path


def record_base_images(
    folder: Path,
    *,
    client: RegistryClient | None = None,
    select: Callable[[str], bool] | None = None,
    platform: str | None = None,
) -> Path | None:
    """Record the base image of every Dockerfile under ``folder`` next to the task folders, before they load.

    ``select`` limits the scan to the tasks whose folder name it accepts (see
    :func:`dockerfile_base_images`). Returns the file's path, or ``None`` when no task has a
    Dockerfile the loader resolves. ``platform`` is as :func:`record_image_configs` takes it.
    """
    folder = Path(folder).resolve()
    parent = folder.parent if is_task_folder(folder) else folder
    return record_image_configs(
        dockerfile_base_images(folder, select=select), parent, client=client, what="base image", platform=platform
    )
