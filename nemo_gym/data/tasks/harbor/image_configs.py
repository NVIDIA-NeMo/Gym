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
  the way ``docker build`` resolves them (see ``nemo_gym.data.tasks.harbor.dockerfile``).

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
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Callable

import yaml

from nemo_gym.data.tasks.harbor.compose_overlay import (
    RESET,
    ComposeOverlayError,
    HarborComposeOverlay,
    OverrideMapping,
    absolute_env_files,
    interpolate_overlay,
    load_overlay_file,
    merge_compose,
    overlay_files,
    resolve_passthrough_environment,
    synthesized_group,
)
from nemo_gym.data.tasks.harbor.compose_variables import (
    _warn_unset,
    compose_variables,
    infrastructure_variables,
    inline_env_files,
    literal_variables,
    residual_interpolation,
    resolve_env_files,
    restore_compose_types,
    substitute_compose,
    substitute_with_empty,
    under_task_folder,
)
from nemo_gym.data.tasks.harbor.dataset_config import apply_dataset_config, read_dataset_config
from nemo_gym.data.tasks.harbor.dockerfile import RECORD_VERSION
from nemo_gym.data.tasks.harbor.models import HarborTaskConfig
from nemo_gym.data.tasks.harbor.records import (
    IMAGE_CONFIGS_FILE,
    MAIN_SERVICE,
    read_image_configs,
    record_lock,
    write_json_record,
)
from nemo_gym.data.tasks.harbor.registry import ImageConfigError, PlatformMismatchError, RegistryClient, platform_pair
from nemo_gym.data.tasks.harbor.task import (
    COMPOSE_FILE_NAMES,
    DOCKERFILE,
    TASK_FILE,
    HarborTask,
    dockerfile_base_image,
    is_task_folder,
    iter_task_folders,
)
from nemo_gym.sandbox.compose_config import check_sidecar_extra_hosts, plan_compose_volumes
from nemo_gym.sandbox.compose_fields import (
    DEFAULT_PLATFORM,
    ComposeFieldError,
    ComposeSubstitutionError,
    check_compose_fields,
)
from nemo_gym.sandbox.compose_volumes import ComposeVolumeError
from nemo_gym.sandbox.providers.registry import provider_service_option_fields


LOGGER = logging.getLogger(__name__)

COMPOSE_IMAGES_FILE = IMAGE_CONFIGS_FILE


def compose_file(task: HarborTask) -> Path | None:
    """The task's Compose overlay, if it has one."""
    for name in COMPOSE_FILE_NAMES:
        path = task.path / "environment" / name
        if path.is_file():
            return path
    return None


def compose_source(task: HarborTask, overlay: HarborComposeOverlay | None = None) -> str:
    """What a message about the task's Compose document names as its source: ``environment/<file>`` for the task's own
    file; with a job-level ``overlay`` that names Compose files, that file ``merged with Compose overlay files [...]``
    (the files in merge order), or the overlay files alone for a task that declares no Compose file, since a field, a
    bind source or a service the group cannot run may then come from any of them."""
    path = compose_file(task)
    own = f"environment/{path.name}" if path is not None else None
    extra = overlay_files(overlay)
    if not extra:
        return own or "environment/"
    files = "[" + ", ".join(str(file) for file in extra) + "]"
    if own is None:
        return f"Compose overlay files {files} (the task declares no Compose file of its own)"
    return f"{own} merged with Compose overlay files {files}"


# ``(task id, service)`` pairs already warned about: the server loads a task's document twice per episode.
_WARNED_SHARED: set[tuple[str, str]] = set()


def shared_services(own_services: Collection[str], extra: Mapping[str, Any]) -> list[str]:
    """The services, ``main`` aside, that a task's own Compose file declares (``own_services``) and the overlay
    document ``extra`` merges into rather than replaces.

    Compose merges ``-f`` files by service name, the task's file included, so a task that declares a name an overlay
    file declares keeps its definition (image, ``entrypoint``, ``command``, ...) under the overlay's fields, the
    overlay's ``environment`` among them. A service ``extra`` tags ``!override`` starts the overlay's definition alone,
    one it tags ``!reset`` is dropped and one it writes with no value adds nothing, so none of those is shared; ``main``
    is the task's own service, which the glue files extend by design (``depends_on``, ``networks``).
    """
    services = extra.get("services")
    if not isinstance(services, Mapping) or isinstance(services, OverrideMapping):
        return []
    return sorted(
        str(name)
        for name, value in services.items()
        if str(name) in own_services
        and str(name) != MAIN_SERVICE
        and isinstance(value, Mapping)
        and not isinstance(value, OverrideMapping)
    )


def remaining_own_services(own_services: Collection[str], extra: Mapping[str, Any]) -> set[str]:
    """``own_services`` without the services the overlay document ``extra`` replaced or dropped, for the next overlay
    file's :func:`shared_services`: a service ``extra`` tags ``!override`` is the overlay's definition from then on,
    one it tags ``!reset`` is gone, and a ``services`` mapping tagged either leaves none of the task's. The next file
    then merges into the overlay's definition, not the task's (the standalone file plus the glue file of the README's
    example), and is not warned about; a service an earlier overlay file only merged into keeps the task's fields
    under it and stays shared.
    """
    services = extra.get("services")
    if services is RESET or isinstance(services, OverrideMapping):
        return set()
    if not isinstance(services, Mapping):
        return set(own_services)
    return {
        name
        for name in own_services
        if not (name in services and (services[name] is RESET or isinstance(services[name], OverrideMapping)))
    }


def _warn_shared_services(
    task: HarborTask, extra_path: Path, extra: Mapping[str, Any], own_services: Collection[str]
) -> None:
    """One warning per task and service (:func:`shared_services`), at prepare and on the server alike: the merge is
    Compose's and not refused (an overlay written for a dataset may customize a service it knows by name, as Harbor's
    extra ``-f`` files do), but an overlay service that carries the deployment's values should not merge onto a
    definition a third-party task wrote under the same name."""
    for service in shared_services(own_services, extra):
        if (task.task_id, service) in _WARNED_SHARED:
            continue
        _WARNED_SHARED.add((task.task_id, service))
        LOGGER.warning(
            "%s: Compose overlay %s declares service %r, which the task's own Compose file also declares: the "
            "overlay's fields merge onto the task's definition, whose image, entrypoint, command and the rest stand "
            "where the overlay sets none, as Compose merges -f files; tag the overlay service !override to start the "
            "overlay's definition alone, or !reset to drop the task's",
            task.task_id,
            extra_path,
            service,
        )


def load_compose_document(
    task: HarborTask,
    *,
    operator_variables: Mapping[str, str] | None = None,
    overlay: HarborComposeOverlay | None = None,
) -> dict[str, Any] | None:
    """The task's Compose overlay with ``env_file`` entries inlined, ``${VAR}`` references resolved and the images
    prepare built standing in for the services it built (:func:`with_built_images`), ``main`` only when the loader
    resolved the task to its built image.

    Returns ``None`` for a task without a Compose file, unless the job-level ``overlay`` names Compose files
    (:class:`~nemo_gym.data.tasks.harbor.compose_overlay.HarborComposeOverlay`): then a task without a file gets
    Harbor's one-service group (``main`` alone, :func:`~nemo_gym.data.tasks.harbor.compose_overlay.synthesized_group`)
    and every task's document takes the overlay files merged onto it, in order, as Compose merges the extra
    ``-f`` files Harbor passes after the task's (:func:`~nemo_gym.data.tasks.harbor.compose_overlay.merge_compose`).
    Each overlay file is interpolated with the task's variables (the deployment's, when given, above the task's own:
    see ``operator_variables`` below); its relative paths resolve against the project directory, the task's
    ``environment`` folder, as Compose resolves every ``-f`` file's and Harbor's ``--project-directory`` names it,
    never against the overlay file's own folder: its ``env_file`` entries are made absolute against it before the
    merge (an overlay's ``env_file`` may be any readable file; a task's stays under the task) and its bind sources
    are left for the planner, which resolves them against the same
    folder as the task's. The env files of the merged document are read once after the merge, the merged
    ``environment`` winning over them, as Compose reads them after merging the ``-f`` files.
    An overlay file that does not parse or merge raises :class:`ImageConfigError` naming it; an error in the merged
    document (an env file it names, an interpolation left over, a ``services`` key the overlay removed) names the
    task's file and the overlay files merged onto it (:func:`compose_source`).

    Prepare and the resources server both read the
    file through this function, so the document the sandbox starts is the one prepare validated, provided
    both run with the same process environment (see :func:`compose_variables`). A required variable
    (``${VAR:?msg}``) without a value, a task variable the host does not set or a missing ``env_file``
    raises :class:`ImageConfigError` naming it; a variable nobody sets is an empty string, with a warning.
    The record is the one :func:`~nemo_gym.data.tasks.harbor.task.load_task` read (``task.built_images``): nothing is
    hashed here, which runs once per episode on the server.

    ``operator_variables`` are the deployment's values (the environment server's ``compose_variables``), which
    the server resolves at seed: they fill the names neither the task, the process environment nor Harbor's own
    variables set (SWE-Interact's simulated user reads its model endpoint and key that way). In the overlay files
    they stand above the task's own sources instead (its ``[environment].env``, ``environment/.env`` and the process
    environment), below Harbor's infrastructure set: the overlay files are the operator's like the values, so a task
    cannot redirect what they do with the deployment's values by setting one of their names itself. The pass-through
    spellings of an overlay service's ``environment`` (a bare ``NAME`` in the list form, ``NAME:`` with no value in
    the mapping form) follow the same order as its ``${NAME}``: they are resolved before the merge
    (:func:`~nemo_gym.data.tasks.harbor.compose_overlay.resolve_passthrough_environment`), where the task's own are read
    after it, in the task's order. With them a name
    still unset fails here, before anything is provisioned; without them, as at prepare, it is an empty string
    with a warning, the images and structure being all prepare reads.
    """
    path = compose_file(task)
    extra_files = overlay_files(overlay)
    if path is None and not extra_files:
        return None
    if path is None:
        # Harbor starts a plain image task through Compose too (its own files define ``main``), so the job's
        # extra files extend it the same way; the adapter fills ``main`` from the task as it does for a task
        # file that leaves it out, and ``main`` starts as the task's single sandbox would (its resolved entrypoint,
        # none for a Dockerfile that cleared it).
        document: dict[str, Any] = synthesized_group(task)
    else:
        document = yaml.safe_load(path.read_text()) or {}
        if not isinstance(document, dict):
            raise ImageConfigError(f"{task.task_id}: {path.name} is not a mapping")
    name = path.name if path is not None else "(no Compose file)"
    base_dir = path.parent if path is not None else task.path / "environment"
    unset: set[str] = set()
    try:
        # Interpolate first, as Compose does, so an env_file path may use ${CONTEXT_DIR}; env_file values are then
        # literal. Anything still carrying a ``$NAME`` afterwards fails here, at prepare, not at seed.
        variables = compose_variables(task, unset=unset)
        # The deployment's values are literal to the document like every other source: a key or password may
        # carry `$`, and nothing of the deployment's may trip the residual check or lose a `$$`.
        operator_variables = literal_variables(operator_variables)
        if operator_variables is not None:
            variables = dict(operator_variables) | variables
            # The overlay files are the operator's, as the deployment's values are: in them a value the deployment
            # sets stands above the task's sources (its ``[environment].env``, ``environment/.env`` and the process
            # environment the task's env overrides), below Harbor's infrastructure set as under Harbor's
            # ``merge_compose_env``. A task cannot then redirect what the operator's file does with the operator's
            # values (the endpoint an overlay sidecar posts the deployment's key to) by setting the name itself; in
            # the task's own file the deployment's values keep filling only the names the task leaves unset.
            overlay_variables = {
                **variables,
                **operator_variables,
                **literal_variables(infrastructure_variables(task)),
            }

            def substitute(value: Any) -> Any:
                return substitute_compose(value, variables)

            def substitute_overlay(value: Any) -> Any:
                return substitute_compose(value, overlay_variables)

        else:

            def substitute(value: Any) -> Any:
                return substitute_with_empty(value, variables, unset)

            substitute_overlay = substitute
            overlay_variables = variables

        document = substitute(document)
        # The task's env_file paths resolve under its environment folder now; the files are read after the merge.
        document = resolve_env_files(document, under_task_folder(base_dir))
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: {name}: {exc}") from exc
    # The services the task's own file declares that the files merged so far have not replaced or dropped: an overlay
    # file that declares one merges into the task's definition (``shared_services``), which is warned about once per
    # task and service; after each file the set follows what it replaced (``remaining_own_services``).
    own_services: set[str] = (
        set(document["services"]) if path is not None and isinstance(document.get("services"), dict) else set()
    )
    for extra_path in extra_files:
        # The task's variables, the deployment's values above the task's own (``substitute_overlay``): Harbor's one
        # ``docker compose`` command interpolates every ``-f`` file against one environment, resolves every file's
        # relative paths against the project directory (``--project-directory``, the task's environment folder:
        # ``base_dir``), then merges them in order. The overlay's env files land where the task's do, by the
        # operator's rule (any readable file); its bind sources are the planner's, which resolves them against
        # ``base_dir`` as it resolves the task's. The overlay's pass-through ``environment`` entries (a bare ``NAME``)
        # are resolved with the same variables now: the merged document's are read after the merge in the task's
        # order (``inline_env_files`` below), which would put the task's sources back above the deployment's values.
        try:
            extra = load_overlay_file(extra_path)
            _warn_shared_services(task, extra_path, extra, own_services)
            extra = interpolate_overlay(extra, substitute_overlay)
            extra = resolve_passthrough_environment(extra, overlay_variables)
            extra = absolute_env_files(extra, base_dir)
            document = merge_compose(document, extra)
            own_services = remaining_own_services(own_services, extra)
        except (ComposeOverlayError, ComposeSubstitutionError) as exc:
            raise ImageConfigError(f"{task.task_id}: Compose overlay {extra_path}: {exc}") from exc
    if extra_files:
        # The merged document is what the rest reads: its errors name every file it came from.
        name = compose_source(task, overlay)
        if not isinstance(document.get("services"), dict):
            # ``services: !reset`` (or an ``!override`` scalar) in an overlay file: the merge removed the mapping the
            # field check, the planner and the adapter index. Refused with the reason, not a KeyError at prepare and a
            # retried seed.
            raise ImageConfigError(f"{task.task_id}: {name}: the merged document has no services mapping")
    try:
        # The env files are read once, after the merge, the merged ``environment`` winning over them, as Compose
        # reads them after merging the ``-f`` files (so a later file's ``env_file: !reset []`` dropped the earlier
        # lists above); every service's ``environment`` becomes a mapping of strings here.
        document = inline_env_files(document, None, variables)
        # Interpolation hands back strings; the adapter reads healthcheck ``disable`` and ``retries`` by type.
        document = restore_compose_types(document)
    except ComposeSubstitutionError as exc:
        raise ImageConfigError(f"{task.task_id}: {name}: {exc}") from exc
    _warn_unset(task, unset)
    residual = residual_interpolation(document)
    if residual is not None:
        # The key path only: the value next to it may be a host secret.
        raise ImageConfigError(f"{task.task_id}: {name}: unresolved interpolation left at {residual}")
    images = dict(task.built_images)
    if task.built_image is None:
        # The loader resolved main elsewhere (docker_image declared, or pull/overlay mode): a recorded main is
        # another task's build of the same context, or a build that predates the override. Harbor runs the
        # prebuilt image then, so main keeps ${MAIN_IMAGE_NAME} = task.image. Sidecars are built whatever main's
        # mode and are always substituted.
        images.pop(MAIN_SERVICE, None)
    services = document.get("services") if isinstance(document.get("services"), dict) else {}
    for service_name in list(images):
        service = services.get(service_name)
        if service_name != MAIN_SERVICE and isinstance(service, dict) and "build" not in service:
            # A recorded sidecar image stands in for the ``build:`` prepare ran, and the merged document no longer
            # has one: an overlay file replaced the build with an image of its own (``build: !reset null`` next to
            # ``image: org/replacement:2``, or a service tagged ``!override``). Compose builds nothing it is not
            # asked for and starts the image named, so the record does not undo the replacement; the service's
            # image, like any pulled one, is what the group starts and what prepare records.
            images.pop(service_name)
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
    overlay: HarborComposeOverlay | None = None,
    warn_read_only: bool = False,
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
    prepare fails before a sandbox starts; with the job-level ``overlay`` the document came from, the message names
    the overlay files merged onto the task's (:func:`compose_source`). Returns the uploads per service.
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
            warn_read_only=warn_read_only,
        )
    except ComposeVolumeError as exc:
        raise ImageConfigError(f"{task.task_id}: {compose_source(task, overlay)}: {exc}") from exc


def compose_image_references(
    task: HarborTask, document: dict[str, Any] | None = None, *, overlay: HarborComposeOverlay | None = None
) -> list[str]:
    """Every image a Compose task starts: the task image for ``main`` and each sidecar's image. A service without an
    image and without a recorded build is refused by name; with the job-level ``overlay`` the document came from,
    naming the overlay files merged onto the task's (:func:`compose_source`), since the service may be theirs."""
    if document is None:
        document = load_compose_document(task, overlay=overlay)
    if document is None:
        return []
    source = f"{compose_source(task, overlay)}: " if overlay_files(overlay) else ""
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
                f"{task.task_id}: {source}Compose service {name!r} has a build section and prepare recorded no "
                "build for it"
            )
        image = service.get("image")
        if not image:
            raise ImageConfigError(
                f"{task.task_id}: {source}Compose service {name!r} names no image and prepare recorded no build for it"
            )
        references.append(image)
    return list(dict.fromkeys(references))


def task_declares_gpus(task: HarborTask) -> bool:
    """Whether ``task.toml`` asks for GPUs (agent or verifier environment): the task then runs on the GPU block."""
    verifier_environment = task.config.verifier.environment
    return bool(task.config.environment.gpus or (verifier_environment is not None and verifier_environment.gpus))


def _compose_field_error(
    task: HarborTask, exc: ComposeFieldError, overlay: HarborComposeOverlay | None = None
) -> ImageConfigError:
    return ImageConfigError(f"{task.task_id}: {compose_source(task, overlay)}: {exc}")


def check_task_sidecar_extra_hosts(
    task: HarborTask,
    document: dict[str, Any],
    image_configs: Mapping[str, Any],
    *,
    overlay: HarborComposeOverlay | None = None,
) -> None:
    """The check that needs the recorded images: a sidecar whose image runs as a non-root user, which the group
    starts without host injection, must declare no ``extra_hosts``
    (:func:`~nemo_gym.sandbox.compose_config.check_sidecar_extra_hosts`). Prepare runs it once the images are
    recorded; the seed runs it on the merged document before anything is provisioned, where ``adapt_non_root_sidecars``
    would otherwise raise it inside the group's start. Raises :class:`ImageConfigError` naming the task and its
    Compose source (with the overlay files merged onto it, :func:`compose_source`)."""
    try:
        check_sidecar_extra_hosts(document, image_configs)
    except ComposeFieldError as exc:
        raise _compose_field_error(task, exc, overlay) from exc


def check_task_compose_services(
    task: HarborTask, document: Mapping[str, Any], *, overlay: HarborComposeOverlay | None = None
) -> None:
    """Every service the task's ``artifacts`` entries and ``[[verifier.collect]]`` hooks name must be in the group
    (``main`` always is: the group starts it from the task image). A task's own file names them in practice; a
    job-level overlay file can reset one away (``kafka: !reset`` against a hook with ``service = "kafka"``), which
    only the merged document shows, so prepare skips such a task with the reason and the seed refuses it terminally,
    where the group's start would fail and be retried for every episode. Raises :class:`ImageConfigError` naming the
    task, its Compose source (:func:`compose_source`) and the services."""
    named = {artifact.service for artifact in task.config.artifacts}
    named |= {hook.service for hook in task.config.verifier.collect}
    missing = sorted(named - {None, MAIN_SERVICE} - set(document.get("services") or {}))
    if missing:
        raise ImageConfigError(
            f"{task.task_id}: {compose_source(task, overlay)}: artifacts or collect hooks name Compose services the "
            f"group does not have: {missing}"
        )


def check_task_compose_fields(
    task: HarborTask,
    document: dict[str, Any],
    *,
    provider: str | None = None,
    gpu_provider: str | None = None,
    platform: str = DEFAULT_PLATFORM,
    overlay: HarborComposeOverlay | None = None,
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
    sandbox starts; with the job-level ``overlay`` the document came from, the file named is the task's merged with
    the overlay files (:func:`compose_source`), since the service or the field may be theirs. Without a provider
    name the capability check runs at seed instead, against the live provider.

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
            raise _compose_field_error(task, ComposeFieldError("build", reason), overlay)
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
        raise _compose_field_error(task, exc, overlay) from exc


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
    missing = [ref for ref in references if not reusable_record(recorded.get(ref), requested)]
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
            if not reusable_record(recorded.get(reference), requested):
                recorded[reference] = record
        write_json_record(path, recorded)
    return path


def reusable_record(record: Any, platform: tuple[str, ...]) -> bool:
    """A ``compose-images.json`` entry this run reads as it is: of the current record version and of ``platform``
    (``os``, ``architecture``), the one the run records images for. Anything else is resolved again at its pin by
    prepare, and refused at seed by the server, which cannot resolve it (``_check_compose_images``)."""
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
    overlay: HarborComposeOverlay | None = None,
) -> Path | None:
    """Write ``compose-images.json`` in ``folder`` for the Compose tasks among ``tasks``.

    Returns the file's path, or ``None`` when no task uses Compose. With a job-level ``overlay`` that names Compose
    files every task is a Compose task (:func:`load_compose_document`): the overlay's images are recorded with the
    tasks', and a plain image task's own image too, since the group starts it as ``main``. Each Compose document's service fields are
    checked on the way (see :func:`check_task_compose_fields`), with ``sandbox_provider`` and
    ``gpu_sandbox_provider`` naming the providers of the ``sandbox:`` and ``gpu_sandbox_provider`` blocks for the
    capability check and ``compose_platform`` the deployment's, which a service's ``platform`` must match;
    ``platform`` is the one the images are recorded for, as :func:`record_image_configs` takes it (the built images'
    ``builds.platform``). Each document's ``volumes`` are planned next (see :func:`plan_task_compose_volumes`), with
    ``volume_sources`` standing for the deployment's external volumes; a task that declares GPUs, and a GPU-reserving
    sidecar of any task, are checked against ``gpu_volume_sources`` instead when that is given (the
    ``gpu_sandbox_provider`` block's mapping, which is what the server resolves them against; an empty mapping when
    that block has none), ``shared_storage``/``gpu_shared_storage`` say whether each block's provider has the
    shared storage named volumes and shared logs binds need (``False`` leaves such a task out, naming the
    setting), and the services the task's ``artifacts`` and ``[[verifier.collect]]`` hooks name must be
    in the merged document (:func:`check_task_compose_services`; an overlay file can reset one away). A task whose
    Compose file does not resolve (an :class:`ImageConfigError` from :func:`load_compose_document`), carries a field
    the group cannot run, declares volumes it cannot run (the planner's :class:`ImageConfigError`), or lacks a
    service its hooks name does not stop the others
    (with an overlay, the reason names the overlay files merged onto the task's, :func:`compose_source`): with
    ``skipped`` given it is recorded there by task id with the reason and left out,
    otherwise one :class:`ImageConfigError` at the end names every such task. Its images are still recorded, so
    a fixed task file needs no new registry round-trip. Once the image configurations are recorded, a sidecar
    whose image runs as a non-root user (which the group starts without host injection) must declare no
    ``extra_hosts`` (:func:`check_task_sidecar_extra_hosts`), the way the adapter decides it at seed. An image the
    registry cannot resolve still raises for the whole set, as :func:`record_image_configs` says.
    """
    references: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    documents: list[tuple[HarborTask, dict[str, Any]]] = []
    for task in tasks:
        try:
            document = load_compose_document(task, overlay=overlay)
            if document is None:
                continue
            for reference in compose_image_references(task, document, overlay=overlay):
                references.setdefault(reference, []).append(task.task_id)
            check_task_compose_fields(
                task,
                document,
                provider=sandbox_provider,
                gpu_provider=gpu_sandbox_provider,
                platform=compose_platform,
                overlay=overlay,
            )
            plan_task_compose_volumes(
                task,
                document,
                volume_sources=volume_sources,
                gpu_volume_sources=gpu_volume_sources,
                shared_storage=shared_storage,
                gpu_shared_storage=gpu_shared_storage,
                overlay=overlay,
                warn_read_only=True,  # prepare says it once; the server plans the same document again at every seed
            )
            check_task_compose_services(task, document, overlay=overlay)
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
                check_task_sidecar_extra_hosts(task, document, recorded, overlay=overlay)
            except ImageConfigError as exc:
                if skipped is None:
                    raise
                skipped[task.task_id] = str(exc)
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
