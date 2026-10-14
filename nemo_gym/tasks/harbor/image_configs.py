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

import json
import logging
import tomllib
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
from nemo_gym.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.tasks.harbor.models import HarborTaskConfig
from nemo_gym.tasks.harbor.records import IMAGE_CONFIGS_FILE
from nemo_gym.tasks.harbor.registry import ImageConfigError, RegistryClient
from nemo_gym.tasks.harbor.task import TASK_FILE, HarborTask, dockerfile_base_image, is_task_folder


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


def load_compose_document(task: HarborTask) -> dict[str, Any] | None:
    """The task's Compose overlay with ``env_file`` entries inlined and ``${VAR}`` references resolved.

    Returns ``None`` for a task without a Compose file. Prepare and the resources server both read the
    file through this function, so the document the sandbox starts is the one prepare validated, provided
    both run with the same process environment (see :func:`compose_variables`). A required variable
    (``${VAR:?msg}``) without a value, a task variable the host does not set or a missing ``env_file``
    raises :class:`ImageConfigError` naming it; a variable nobody sets is an empty string, with a warning.
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
        # Interpolation hands back strings; the adapter reads healthcheck ``disable`` and ``retries`` by type.
        document = restore_compose_types(document)
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: {path.name}: {exc}") from exc
    _warn_unset(task, unset)
    residual = residual_interpolation(document)
    if residual is not None:
        # The key path only: the value next to it may be a host secret.
        raise ImageConfigError(f"{task.task_id}: {path.name}: unresolved interpolation left at {residual}")
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
            raise ImageConfigError(
                f"{task.task_id}: Compose service {name!r} names no image; builds are not supported"
            )
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
    # sits beside the task folders.
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
    references: dict[str, list[str]], folder: Path, *, client: RegistryClient | None = None, what: str = "image"
) -> Path | None:
    """Record the configuration of ``references`` (image reference to the tasks using it) into ``folder``.

    Entries already recorded are kept as they are, so a pinned recording never changes
    under a completed run; only references missing from the file are resolved. An entry an
    older Gym wrote (no current ``record_version``: it lacks fields the Dockerfile resolver
    reads) is read again, at the digest it was pinned to rather than the tag, so the pin does
    not move either. Returns the file's path, or ``None`` when there is nothing to record. A
    reference the registry cannot resolve raises :class:`ImageConfigError` naming the image,
    the tasks and the registry's answer.
    """
    if not references:
        return None
    path = Path(folder) / IMAGE_CONFIGS_FILE
    recorded: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    missing = [ref for ref in references if (recorded.get(ref) or {}).get("record_version") != RECORD_VERSION]
    if not missing:
        return path
    client = client or RegistryClient()
    LOGGER.info(f"Recording the configuration of {len(missing)} {what}(s) into {path}")
    for reference in missing:
        pinned = (recorded.get(reference) or {}).get("image")
        try:
            recorded[reference] = client.image_config(pinned or reference)
        except ImageConfigError as exc:
            tasks = ", ".join(references[reference][:5]) or "-"
            raise ImageConfigError(
                f"Could not record the configuration of {what} {reference!r} (used by task(s) {tasks}): {exc}"
            ) from exc
    path.write_text(json.dumps(dict(sorted(recorded.items())), indent=2, sort_keys=True) + "\n")
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
