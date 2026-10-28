# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The task's docker-compose.yaml as docker compose build reads it.

Harbor builds with docker compose build over its own template followed by the task's Compose
file, with Harbor's infrastructure variables in the environment. This module writes that
template and Gym's images override, resolves the build environment in Harbor's order,
interpolates the task file the way Compose does, reads each service's build section into the
plan, and refuses what must not run on the build host: secrets, ssh, extra contexts and hosts,
host networking, BuildKit entitlements, cache credentials and paths outside the environment folder. The build-side
variable helpers deliberately stay separate from the loader's copies in compose_variables.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from nemo_gym.sandbox.compose_fields import ComposeSubstitutionError
from nemo_gym.tasks.harbor.build_plan import DOCKERFILE_NAME, HARBOR_COMPOSE_FILE, BuildError, BuildPlanEntry
from nemo_gym.tasks.harbor.compose_variables import (
    LOG_MOUNT_PATHS,
    TEST_DIR,
    env_file_entries,
    substitute_compose,
    substitute_with_empty,
)
from nemo_gym.tasks.harbor.harbor_compat import sanitize_docker_image_name
from nemo_gym.tasks.harbor.image_configs import COMPOSE_FILE_NAMES
from nemo_gym.tasks.harbor.records import MAIN_SERVICE


LOGGER = logging.getLogger(__name__)

# The variable Harbor exports to every Compose command with the task image's name (docker.py:238-243 in v0.22.0).
MAIN_IMAGE_VARIABLE = "MAIN_IMAGE_NAME"
# The file name Harbor ships its build template under; Gym writes its copy under the same name.
COMPOSE_BUILD_FILE = "docker-compose-build.yaml"
# Gym's own override, passed to ``docker compose build`` after the task's file: the image name of every service built.
COMPOSE_IMAGES_FILE = "docker-compose-images.yaml"

# Build keys of a task's Compose file that reach the build host and are refused (see validate_task_compose).
# ``entitlements`` is Compose's spelling of BuildKit's ``--allow`` (``network.host`` lets ``RUN --network=host``
# use the build host's network, ``security.insecure`` lifts the sandbox) and ``privileged: true`` grants both, so
# the two are refused whatever their value, like ``secrets``: the main service and a built sidecar alike.
REFUSED_BUILD_KEYS = ("secrets", "ssh", "extra_hosts", "entitlements", "privileged")
# Keys of a ``cache_from`` / ``cache_to`` entry (``type=...,ref=...,key=value``) that carry credentials.
CACHE_CREDENTIAL_KEYS = frozenset(
    {"token", "secret", "password", "credentials", "access_key_id", "secret_access_key", "session_token"}
)
# Harbor's ``[environment].env`` template: a whole value ``${VAR}`` or ``${VAR:-default}`` (``_TEMPLATE_PATTERN``,
# https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/utils/env.py#L4).
HARBOR_ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")

# The file Compose reads interpolation variables from in the project directory (Harbor's: the ``environment/``
# folder) when the variable is not in the command's environment.
COMPOSE_DOTENV = ".env"
# Points Compose at env files other than the project directory's ``.env``; refused for a build.
COMPOSE_ENV_FILES = "COMPOSE_ENV_FILES"

# ``(context, variable)`` pairs already warned about as unset: a context's file is interpolated more than once per build.
_WARNED_UNSET: set[tuple[str, str]] = set()


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
        "compose_build": dict(build),
    }


def _compose_scalar(value: Any) -> str:
    """A YAML scalar as Compose passes it to the build: ``true``/``false`` lower-case, else its text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def image_name(digest: str, repository: str | None) -> str:
    """``hb__<digest>`` sanitised as Harbor sanitises it, under ``repository`` when there is one: the name
    :meth:`ImageBuilder.tag_for` builds under and ``MAIN_IMAGE_NAME`` in the build's environment."""
    name = sanitize_docker_image_name(f"hb__{digest}")
    return f"{repository.rstrip('/')}/{name}" if repository else name


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
    ``extra_hosts``, ``entitlements`` and ``privileged`` (BuildKit's ``--allow network.host`` / ``security.insecure``,
    which let a ``RUN --network=host`` line reach the build host's network), ``network: host``,
    ``additional_contexts`` other than ``service:`` references, and
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
