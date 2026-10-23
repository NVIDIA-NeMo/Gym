# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the OCI configuration of the images a task runs.

Docker reads an image's environment, user, working directory, entrypoint, command and
exposed ports from the image itself. A sandbox platform without Docker cannot, so Gym
records them from the registry into ``compose-images.json`` next to the task folders,
one entry per image reference exactly as the task wrote it, pinned to the digest of the manifest
the reference named (see below), with the configuration read for the platform the sandboxes run
(linux/amd64 unless ``builds.platform`` names another). Two readers depend on the file:

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import yaml

from nemo_gym.sandbox.compose_config import check_sidecar_extra_hosts, plan_compose_volumes
from nemo_gym.sandbox.compose_fields import (
    DEFAULT_PLATFORM,
    ComposeFieldError,
    ComposeSubstitutionError,
    check_compose_fields,
)
from nemo_gym.sandbox.compose_volumes import ComposeVolumeError
from nemo_gym.sandbox.providers.registry import provider_service_option_fields
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
from nemo_gym.tasks.harbor.records import (
    IMAGE_CONFIGS_FILE,
    MAIN_SERVICE,
    read_image_configs,
    record_lock,
    write_json_record,
)
from nemo_gym.tasks.harbor.registry import ImageConfigError, PlatformMismatchError, RegistryClient, platform_pair
from nemo_gym.tasks.harbor.task import (
    COMPOSE_FILE_NAMES,
    DOCKERFILE,
    TASK_FILE,
    HarborTask,
    dockerfile_base_image,
    is_task_folder,
    iter_task_folders,
)


LOGGER = logging.getLogger(__name__)

COMPOSE_IMAGES_FILE = IMAGE_CONFIGS_FILE


def compose_file(task: HarborTask) -> Path | None:
    """The task's Compose overlay, if it has one."""
    for name in COMPOSE_FILE_NAMES:
        path = task.path / "environment" / name
        if path.is_file():
            return path
    return None


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
        # Interpolation hands back strings; the adapter reads healthcheck ``disable`` and ``retries`` by type.
        document = restore_compose_types(document)
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


def with_built_images(document: dict[str, Any], images: Mapping[str, str]) -> dict[str, Any]:
    """``document`` with the recorded image of every service in ``images`` (service name to image, as
    ``built-images.json`` records them) set as its ``image`` and its ``build`` removed, along with a
    ``pull_policy: build``, the policy that names that build.

    Harbor builds every service with a ``build:`` section from the task's Compose file at every start (``main``
    with args, a target or a ``dockerfile_inline``, and the sidecars from their own contexts), and Harbor's own
    template sets ``pull_policy: build``. Gym builds those images once at prepare, with the same file, and
    records them by service name; here each recorded image stands in for its service's build, so a sidecar's own
    ``image:`` spelling (kumo's ``${MAIN_IMAGE_NAME}_verifier``, a local tag under Harbor) is replaced by the name
    Gym built and recorded. The section and the ``build`` policy describe a build that already happened; the
    sandbox Compose adapter starts images only and refuses the section as an unsupported field, so they are
    dropped here rather than failing every episode of a task prepare accepted. Any other ``pull_policy``
    (``always``, ``missing``, ``never``) is the service's runtime pull policy, not part of its build, and stays
    for the field check as on any pulled service: it now governs the pull of the recorded image. Services not in
    ``images`` are untouched: a task in pull or overlay mode keeps whatever its file gives ``main``, so an
    unsupported field there is still reported, and a sidecar prepare recorded no build for (with or without an
    ``image:`` of its own) is reported by :func:`compose_image_references`.
    """
    services = document.get("services") if isinstance(document, dict) else None
    if not isinstance(services, dict):
        return document
    for name, image in images.items():
        if name not in services:
            continue
        service = services[name] if isinstance(services[name], dict) else {}
        services[name] = {
            **{
                key: value
                for key, value in service.items()
                if key != "build" and not (key == "pull_policy" and is_build_pull_policy(value))
            },
            "image": image,
        }
    return document


# The keys of ``main``'s ``build:`` that only restate what Harbor's own build template supplies for the task's
# ``environment/`` folder; any other key configures a build Gym does not run in pull or overlay mode.
RESTATED_BUILD_KEYS = frozenset({"context", "dockerfile", "pull"})
UNBUILT_MAIN_NOTE = "cannot be applied in pull or overlay mode (Gym does not build this Dockerfile)"


def restated_main_build_reason(build: Any, *, context_dir: Path, literal_contexts: tuple[str, ...] = ()) -> str | None:
    """Why ``main``'s ``build`` section (as the Compose file spells it) is more than a restatement of Harbor's own
    build of ``context_dir``'s Dockerfile, as a reason naming the keys, or ``None`` when it only restates it and can
    be dropped. The rule of :func:`without_restated_main_build`, on one section, for the loader's interpolated
    document (``context_dir`` is the task's ``environment/`` folder) and for the raw file prepare screens before a
    build (``literal_contexts``: the spellings of that folder an uninterpolated file carries, ``${CONTEXT_DIR}``).
    """
    if isinstance(build, str):
        build = {"context": build}
    if not isinstance(build, dict):
        return f"main.build {UNBUILT_MAIN_NOTE}"
    offending = sorted(str(key) for key in build if key not in RESTATED_BUILD_KEYS)
    context = build.get("context")
    if (
        context is not None
        and context not in (".", "./")
        and context not in literal_contexts
        and Path(str(context)) != context_dir
    ):
        offending.append("context")
    if build.get("dockerfile") not in (None, "Dockerfile", "./Dockerfile"):
        offending.append("dockerfile")
    if offending:
        return f"{', '.join(f'main.build.{key}' for key in offending)} {UNBUILT_MAIN_NOTE}"
    return None


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
    if not isinstance(main, dict):
        return document, None
    if "build" in main:
        reason = restated_main_build_reason(main["build"], context_dir=task.path / "environment")
        if reason is not None:
            return document, reason
    # Harbor's template gives that build ``pull_policy: build`` as well, so a file repeating it restates the template
    # too; any other policy is main's runtime pull policy and stays for the field check and the adapter.
    restated = {"build"} | ({"pull_policy"} if is_build_pull_policy(main.get("pull_policy")) else set())
    if restated & set(main):
        services["main"] = {key: value for key, value in main.items() if key not in restated}
    return document, None


def is_build_pull_policy(value: Any) -> bool:
    """Whether a service's ``pull_policy`` is Compose's ``build``, the one that names a build step, not a pull."""
    return isinstance(value, str) and value.strip().lower() == "build"


def without_pull_policy(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` with ``pull_policy`` removed from every service.

    For the raw-file screen prepare runs before a build (``prepare_builds.refused_before_build``), which reads
    keys only and accepts this one whatever its value. The documents the loader and the server hand the field
    check keep it: a service's ``always``, ``missing`` or ``never`` is the sandbox's pull policy, which the
    adapter applies per sandbox where the provider can (``check_compose_fields`` normalizes the spelling), and
    only the ``build`` policy, which names a build step rather than a pull, is dropped with the build it
    belongs to, by :func:`with_built_images` for a service prepare built and by
    :func:`without_restated_main_build` for a pull or overlay ``main`` restating Harbor's template. ``build:``
    is not touched here either: a sidecar's is still refused (no provider starts a locally built sidecar).
    """
    services = document.get("services") if isinstance(document, dict) else None
    if isinstance(services, dict):
        for name, service in services.items():
            if isinstance(service, dict) and "pull_policy" in service:
                services[name] = {key: value for key, value in service.items() if key != "pull_policy"}
    return document


def plan_task_compose_volumes(
    task: HarborTask,
    document: dict[str, Any],
    *,
    volume_sources: dict[str, str] | None = None,
    gpu_volume_sources: dict[str, str] | None = None,
    shared_storage: bool | None = None,
    gpu_shared_storage: bool | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Resolve the ``volumes`` of a task's Compose document for a sandbox group; mutates ``document``.

    Bind mounts of the task's own files become ``x-sandbox.uploads`` entries (performed into the service
    once it starts), named volumes stay for the provider's shared storage, and external volumes or bind
    sources outside the task resolve through ``volume_sources`` (the sandbox block's mapping). Each service
    resolves through the mapping of the block it runs on, the way :func:`check_task_compose_fields` routes
    the capability check: when ``gpu_volume_sources`` is given (the ``gpu_sandbox_provider`` block's mapping),
    every service of a task that declares GPUs and a GPU-reserving sidecar of any task resolve through it.
    ``shared_storage`` and ``gpu_shared_storage`` say whether each block's provider has shared storage, routed
    the same way: ``False`` leaves a task whose named volumes or shared logs binds need it out at prepare,
    naming the setting, rather than failing every seed; ``None`` (unknown) leaves that to the adapter.
    Anything the group cannot run raises :class:`ImageConfigError` naming the volume, path or field, so
    prepare fails before a sandbox starts. Returns the uploads per service.
    """
    path = compose_file(task)
    base_dir = path.parent if path is not None else task.path / "environment"
    if (gpu_volume_sources, gpu_shared_storage) != (None, None) and task_declares_gpus(task):
        volume_sources, shared_storage = gpu_volume_sources, gpu_shared_storage
    try:
        return plan_compose_volumes(
            document,
            base_dir,
            root=task.path,
            volume_sources=volume_sources,
            gpu_volume_sources=gpu_volume_sources,
            shared_storage=shared_storage,
            gpu_shared_storage=gpu_shared_storage,
        )
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
    ``pull_policy: build`` (:func:`with_built_images`, applied by the loader), and ``main``'s ``build:`` and
    ``pull_policy: build`` dropped when they only restate Harbor's own build of the task's Dockerfile
    (:func:`without_restated_main_build`; a build key Gym cannot apply without a build is this task's reason).
    Every other ``pull_policy`` stays: it is the service's runtime pull policy, normalized here and applied by the
    adapter where the provider can, and ``build`` on a service nothing built fails the check. A sidecar
    ``build:`` prepare recorded no build for and any other key the adapter does not start are then refused by
    the check's allow-list.
    """
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

    Entries already recorded for the platform of this run are kept as they are, so a pinned
    recording never changes under a completed run; only references missing from the file are
    resolved. An entry an older Gym wrote (no current ``record_version``: it lacks fields the
    Dockerfile resolver reads), or one recorded for another platform (``builds.platform`` was
    another when the file was written; the sandbox adapter refuses such a record at seed), is read
    again, at the digest it was pinned to rather than the tag, so the pin does not move either:
    behind a tag that digest is the multi-arch index, which names a manifest for the platform now
    asked for, so the configuration follows ``builds.platform`` under the same pin. A pin that is a
    single manifest of another platform cannot, and is reported with the remedy. Returns the file's
    path, or ``None`` when there is nothing to record. A reference the registry cannot resolve raises
    :class:`ImageConfigError` naming the image, the tasks and the registry's answer. ``platform``
    (``os/arch``, ``builds.platform``; linux/amd64 when ``None``) is the platform the default client
    resolves references for, and the one a record must be of to be kept; a ``client`` given carries
    its own.

    The registry is asked outside the file's sidecar lock (:func:`record_lock`), so a slow
    registry never holds up another process; the file is then read again, merged and replaced
    under the lock, and an entry another process recorded in the meantime is kept over the
    one resolved here, as a pinned entry of this platform always is (a legacy or other-platform
    entry is replaced by its re-read).
    """
    if not references:
        return None
    folder = Path(folder)
    path = folder / IMAGE_CONFIGS_FILE
    # The platform a record is kept for: the client's when it says (the default client's is `platform`'s), else
    # `platform`'s, linux/amd64 unless set. A record of another platform is read again at its pin, like a legacy one.
    requested = tuple(getattr(client, "platform", None) or platform_pair(platform))[:2]
    recorded = read_image_configs(folder)
    missing = [ref for ref in references if not _reusable(recorded.get(ref), requested)]
    if not missing:
        return path
    if client is None:
        client = RegistryClient() if platform is None else RegistryClient(platform=platform_pair(platform))
    LOGGER.info(f"Recording the configuration of {len(missing)} {what}(s) into {path}")
    resolved: dict[str, Any] = {}
    for reference in missing:
        pinned = (recorded.get(reference) or {}).get("image")
        try:
            resolved[reference] = client.image_config(pinned or reference)
        except PlatformMismatchError as exc:
            # The pin is a single manifest of another platform: no manifest for this one is behind it.
            tasks = ", ".join(references[reference][:5]) or "-"
            raise ImageConfigError(
                f"Could not record the configuration of {what} {reference!r} (used by task(s) {tasks}): {exc}; pin a "
                f"multi-arch tag or an image of {exc.requested}, or prepare this dataset for {exc.resolved}"
            ) from exc
        except ImageConfigError as exc:
            tasks = ", ".join(references[reference][:5]) or "-"
            raise ImageConfigError(
                f"Could not record the configuration of {what} {reference!r} (used by task(s) {tasks}): {exc}"
            ) from exc
    with record_lock(path):
        recorded = read_image_configs(folder)
        for reference, record in resolved.items():
            if not _reusable(recorded.get(reference), requested):
                recorded[reference] = record
        write_json_record(path, recorded)
    return path


def _reusable(record: Any, platform: tuple[str, ...]) -> bool:
    """A ``compose-images.json`` entry this run reads as it is: of the current record version and of ``platform``
    (``os``, ``architecture``), the one the run records images for. Anything else is resolved again at its pin."""
    if not isinstance(record, dict) or record.get("record_version") != RECORD_VERSION:
        return False
    recorded = (str(record.get("os") or "").lower(), str(record.get("architecture") or "").lower())
    return recorded == tuple(platform)[:2]


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
    shared_storage: bool | None = None,
    gpu_shared_storage: bool | None = None,
) -> Path | None:
    """Write ``compose-images.json`` in ``folder`` for the Compose tasks among ``tasks``.

    Returns the file's path, or ``None`` when no task uses Compose. Each Compose document's service fields are
    checked on the way (see :func:`check_task_compose_fields`), with ``sandbox_provider`` and
    ``gpu_sandbox_provider`` naming the providers of the ``sandbox:`` and ``gpu_sandbox_provider`` blocks for the
    capability check and ``compose_platform`` the deployment's, which a service's ``platform`` must match;
    ``platform`` is the one the images are recorded for, as :func:`record_image_configs` takes it (the built images'
    ``builds.platform``). Each document's ``volumes`` are planned next (see :func:`plan_task_compose_volumes`), with
    ``volume_sources`` standing for the deployment's external volumes; a task that declares GPUs, and a GPU-reserving
    sidecar of any task, are checked against ``gpu_volume_sources`` instead when that is given (the
    ``gpu_sandbox_provider`` block's mapping, which is what the server resolves them against; an empty mapping when
    that block has none), and ``shared_storage``/``gpu_shared_storage`` say whether each block's provider has the
    shared storage named volumes and shared logs binds need (``False`` leaves such a task out, naming the
    setting). A task whose Compose file
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
            plan_task_compose_volumes(
                task,
                document,
                volume_sources=volume_sources,
                gpu_volume_sources=gpu_volume_sources,
                shared_storage=shared_storage,
                gpu_shared_storage=gpu_shared_storage,
            )
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
