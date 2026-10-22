# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the OCI configuration of the images a task runs.

Docker reads an image's environment, user, working directory, entrypoint, command and
exposed ports from the image itself. A sandbox platform without Docker cannot, so Gym
records them from the registry into ``compose-images.json`` next to the task folders,
one entry per image reference exactly as the task wrote it, pinned to the linux/amd64
manifest digest it resolved to. Two readers depend on the file:

- the harbor resources server, for the sidecar images a Compose task starts (see
  ``nemo_gym.sandbox.compose_config.resolve_compose``);
- the task loader, for the base image of a pull-mode or overlay-mode Dockerfile, whose
  ``ENV``, ``WORKDIR`` and ``USER`` lines resolve from the base image's own configuration
  the way ``docker build`` resolves them (see ``nemo_gym.tasks.harbor.dockerfile``).

The ``image`` an entry pins is the digest of the manifest the written reference named: the
multi-arch index behind a tag, from which a sandbox of any platform selects its own manifest,
or the single manifest of a single-platform reference. The loader starts a pull-mode or
overlay-mode task from that pinned reference rather than the tag the Dockerfile wrote, so a
moved tag cannot change the image under a recorded configuration; the entry stays keyed by
the written reference, which is how the task loader and the Compose resolver look it up.
"""

import logging
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import yaml

from nemo_gym.sandbox.compose_fields import ComposeSubstitutionError
from nemo_gym.tasks.harbor.compose_variables import (
    _warn_unset,
    compose_variables,
    inline_env_files,
    residual_interpolation,
    restore_compose_types,
    substitute_with_empty,
)
from nemo_gym.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.tasks.harbor.digest import context_digest
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from nemo_gym.tasks.harbor.records import (
    BUILT_IMAGES_FILE,
    IMAGE_CONFIGS_FILE,
    read_built_images,
    read_image_configs,
    record_lock,
    write_json_record,
)
from nemo_gym.tasks.harbor.registry import ImageConfigError, RegistryClient
from nemo_gym.tasks.harbor.task import DOCKERFILE, TASK_FILE, HarborTask, dockerfile_base_image, is_task_folder


LOGGER = logging.getLogger(__name__)

COMPOSE_IMAGES_FILE = IMAGE_CONFIGS_FILE
COMPOSE_FILE_NAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")


def compose_file(task: HarborTask) -> Path | None:
    """The task's Compose overlay, if it has one."""
    for name in COMPOSE_FILE_NAMES:
        path = task.path / "environment" / name
        if path.is_file():
            return path
    return None


def load_compose_document(
    task: HarborTask, *, deployment_variables: Mapping[str, str] | None = None
) -> dict[str, Any] | None:
    """The task's Compose overlay with ``env_file`` entries inlined and ``${VAR}`` references resolved.

    Returns ``None`` for a task without a Compose file. Prepare and the resources server both read the
    file through this function, so the document the sandbox starts is the one prepare validated, provided
    both run with the same process environment (see :func:`compose_variables`). A required variable
    (``${VAR:?msg}``) without a value, a task variable the host does not set or a missing ``env_file``
    raises :class:`ImageConfigError` naming it; a variable nobody sets is an empty string, with a warning.

    ``deployment_variables`` are the environment server's ``compose_variables`` for one episode, at the process
    environment's level (see :func:`compose_variables`). Prepare reads the file without them.
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
        variables = compose_variables(task, unset=unset, deployment=deployment_variables)
        document = substitute_with_empty(document, variables, unset)
        document = inline_env_files(document, path.parent, variables)
        # Interpolation hands back strings; the adapter reads healthcheck ``disable`` and ``retries`` by type.
        document = restore_compose_types(document)
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: {path.name}: {exc}") from exc
    _warn_unset(task, unset)
    residual = residual_interpolation(document)
    if residual is not None:
        # The key path only: the value next to it may be a host secret.
        raise ImageConfigError(f"{task.task_id}: {path.name}: unresolved interpolation left at {residual}")
    return with_built_sidecars(document, task)


# The keys of ``main`` that describe how Harbor builds its image, meaningless once the image is built and recorded.
MAIN_BUILD_KEYS = ("build", "pull_policy")


def sidecar_build_context(service: str, build: Any, environment_dir: Path) -> Path:
    """The folder a sidecar's ``build:`` builds: a context inside the task's ``environment/`` with a ``Dockerfile``.

    Only ``context`` and the default ``dockerfile`` are supported for a sidecar; anything else (args, a target,
    another Dockerfile, a context outside ``environment/``) raises :class:`ImageConfigError` naming it.
    """
    if isinstance(build, str):
        build = {"context": build}
    if not isinstance(build, dict) or set(build) - {"context", "dockerfile"}:
        raise ImageConfigError(
            f"Compose service {service!r}: a sidecar build takes only build.context (and the default dockerfile)"
        )
    if build.get("dockerfile", "Dockerfile") != "Dockerfile":
        raise ImageConfigError(f"Compose service {service!r}: a sidecar builds from its context's Dockerfile only")
    root = Path(environment_dir).resolve()
    context = (root / str(build.get("context", "."))).resolve()
    if context != root and root not in context.parents:
        raise ImageConfigError(f"Compose service {service!r}: build.context {context} is outside {root}")
    if not (context / "Dockerfile").is_file():
        raise ImageConfigError(f"Compose service {service!r}: no Dockerfile in {context}")
    return context


def sidecar_build_contexts(document: dict[str, Any], environment_dir: Path) -> dict[str, Path]:
    """The build context of every sidecar (a service other than ``main``) with a ``build:`` section."""
    services = document.get("services") if isinstance(document, dict) else None
    return {
        str(name): sidecar_build_context(str(name), service["build"], environment_dir)
        for name, service in (services.items() if isinstance(services, dict) else ())
        if name != "main" and isinstance(service, dict) and service.get("build") is not None
    }


def with_built_sidecars(document: dict[str, Any], task: HarborTask) -> dict[str, Any]:
    """``document`` with each sidecar's ``build:`` replaced by the image prepare built from that context.

    Prepare builds a sidecar's context once, like ``main``'s Dockerfile, and records it in
    ``built-images.json`` by Harbor's hash of the context; the sandbox Compose adapter starts images only. A
    sidecar whose build is not recorded raises :class:`ImageConfigError`.
    """
    contexts = sidecar_build_contexts(document, task.path / "environment")
    if not contexts:
        return document
    recorded = read_built_images(task.path.parent)
    for name, context in contexts.items():
        entry = recorded.get(context_digest(context))
        image = entry.get("image") if isinstance(entry, dict) else None
        if not image:
            raise ImageConfigError(
                f"{task.task_id}: Compose service {name!r} builds {context} and no built image is recorded in "
                f"{task.path.parent / BUILT_IMAGES_FILE}; prepare builds it when buildah, podman or docker is "
                "configured (the builds: block)"
            )
        service = {key: value for key, value in document["services"][name].items() if key not in MAIN_BUILD_KEYS}
        document["services"][name] = service | {"image": str(image)}
    return document


def without_sidecar_builds(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with every sidecar's ``build`` and ``pull_policy`` removed, for a check of the raw file that runs
    before prepare builds them: each such sidecar then starts from its recorded image (:func:`with_built_sidecars`).
    """
    services = document.get("services") if isinstance(document, dict) else None
    for name, service in services.items() if isinstance(services, dict) else ():
        if name != "main" and isinstance(service, dict) and service.get("build") is not None:
            services[name] = {key: value for key, value in service.items() if key not in MAIN_BUILD_KEYS}
    return document


def without_main_build(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with ``main``'s ``build`` and ``pull_policy`` removed, for a task that runs from the image
    prepare built and recorded.

    Harbor builds ``main`` from the task's Compose file at every start, so the file may give ``main`` a
    ``build:`` (args, a target, a ``dockerfile_inline``) and Harbor's own template sets ``pull_policy: build``.
    Gym builds that image once at prepare, with the same file, and the loader then substitutes the recorded
    ``hb__<hash>`` image for ``main`` (``MAIN_IMAGE_NAME`` resolves to it). The two keys describe a build that
    already happened; the sandbox Compose adapter starts images only and refuses them as unsupported fields,
    so they are dropped here rather than failing every episode of a task prepare accepted. Nothing else is
    touched: a sidecar's ``build:`` gives way to its own recorded image (:func:`with_built_sidecars`), and a task
    in pull or overlay mode goes through :func:`without_restated_main_build` instead.
    """
    services = document.get("services") if isinstance(document, dict) else None
    main = services.get("main") if isinstance(services, dict) else None
    if isinstance(main, dict) and any(key in main for key in MAIN_BUILD_KEYS):
        services["main"] = {key: value for key, value in main.items() if key not in MAIN_BUILD_KEYS}
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
    so the restatement is dropped as a built task's section is (:func:`without_main_build`). ``args``,
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
    field. ``build:`` is not touched here: a sidecar's gives way to its recorded image (:func:`with_built_sidecars`)
    and ``main``'s is dropped by :func:`without_main_build` for a task that runs from its recorded
    built image, or by :func:`without_restated_main_build` when it only restates Harbor's build.
    """
    services = document.get("services") if isinstance(document, dict) else None
    if isinstance(services, dict):
        for name, service in services.items():
            if isinstance(service, dict) and "pull_policy" in service:
                services[name] = {key: value for key, value in service.items() if key != "pull_policy"}
    return document


def compose_image_references(task: HarborTask) -> list[str]:
    """Every image a Compose task starts: the task image for ``main`` and each sidecar's image."""
    document = load_compose_document(task)
    if document is None:
        return []
    services = document.get("services") or {}
    references: list[str] = []
    main_image = (services.get("main") or {}).get("image") or task.image
    if main_image:
        references.append(main_image)
    for name, service in services.items():
        if name == "main":
            continue
        image = (service or {}).get("image")
        if not image:
            raise ImageConfigError(f"{task.task_id}: Compose service {name!r} names no image and no build")
        references.append(image)
    return list(dict.fromkeys(references))


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
    # sits beside the task folders, or one level up for datasets/<name>/tasks/.
    folder = Path(folder).resolve()
    single = is_task_folder(folder)
    children = [folder] if single else sorted(child for child in folder.iterdir() if child.is_dir())
    try:
        dataset = read_dataset_config(folder.parent if single else folder)
    except ValueError:
        return {}
    images: dict[str, list[str]] = {}
    for child in children:
        if not is_task_folder(child) or (select is not None and not select(child.name)):
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
) -> Path | None:
    """Record the configuration of ``references`` (image reference to the tasks using it) into ``folder``.

    Entries already recorded are kept as they are, so a pinned recording never changes
    under a completed run; only references missing from the file are resolved. An entry an
    older Gym wrote (no current ``record_version``: it lacks fields the Dockerfile resolver
    reads) is read again, at the digest it was pinned to rather than the tag, so the pin does
    not move either. Returns the file's path, or ``None`` when there is nothing to record. A
    reference the registry cannot resolve raises :class:`ImageConfigError` naming the image,
    the tasks and the registry's answer.

    The registry is asked outside the file's sidecar lock (:func:`record_lock`), so a slow
    registry never holds up another process; the file is then read again, merged and replaced
    under the lock, and an entry another process recorded in the meantime is kept over the
    one resolved here, as a pinned entry always is (a legacy entry is replaced by its re-read).
    """
    if not references:
        return None
    folder = Path(folder)
    path = folder / IMAGE_CONFIGS_FILE
    recorded = read_image_configs(folder)
    missing = [ref for ref in references if (recorded.get(ref) or {}).get("record_version") != RECORD_VERSION]
    if not missing:
        return path
    if client is None:
        client = RegistryClient()
    LOGGER.info(f"Recording the configuration of {len(missing)} {what}(s) into {path}")
    resolved: dict[str, Any] = {}
    for reference in missing:
        pinned = (recorded.get(reference) or {}).get("image")
        try:
            resolved[reference] = client.image_config(pinned or reference)
        except ImageConfigError as exc:
            tasks = ", ".join(references[reference][:5]) or "-"
            raise ImageConfigError(
                f"Could not record the configuration of {what} {reference!r} (used by task(s) {tasks}): {exc}"
            ) from exc
    with record_lock(path):
        recorded = read_image_configs(folder)
        for reference, record in resolved.items():
            if (recorded.get(reference) or {}).get("record_version") != RECORD_VERSION:
                recorded[reference] = record
        write_json_record(path, recorded)
    return path


def record_compose_images(
    tasks: list[HarborTask],
    folder: Path,
    *,
    client: RegistryClient | None = None,
    skipped: dict[str, str] | None = None,
) -> Path | None:
    """Write ``compose-images.json`` in ``folder`` for the Compose tasks among ``tasks``.

    Returns the file's path, or ``None`` when no task uses Compose. A task whose Compose file does not
    resolve (an :class:`ImageConfigError` from :func:`load_compose_document`) does not stop the others:
    with ``skipped`` given it is recorded there by task id with the reason and left out, otherwise one
    :class:`ImageConfigError` at the end names every such task. An image the registry cannot resolve
    still raises for the whole set, as :func:`record_image_configs` says.
    """
    references: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    for task in tasks:
        try:
            task_references = compose_image_references(task)
        except ImageConfigError as exc:
            failures[task.task_id] = str(exc)
            continue
        for reference in task_references:
            references.setdefault(reference, []).append(task.task_id)
    if failures and skipped is None:
        raise ImageConfigError(f"{len(failures)} Compose task(s) do not resolve: " + "; ".join(failures.values()))
    if failures:
        skipped.update(failures)
    return record_image_configs(references, folder, client=client, what="Compose image")


def record_base_images(
    folder: Path, *, client: RegistryClient | None = None, select: Callable[[str], bool] | None = None
) -> Path | None:
    """Record the base image of every Dockerfile under ``folder`` next to the task folders, before they load.

    ``select`` limits the scan to the tasks whose folder name it accepts (see
    :func:`dockerfile_base_images`). Returns the file's path, or ``None`` when no task has a
    Dockerfile the loader resolves.
    """
    folder = Path(folder).resolve()
    parent = folder.parent if is_task_folder(folder) else folder
    return record_image_configs(
        dockerfile_base_images(folder, select=select), parent, client=client, what="base image"
    )
