# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build step of prepare and the two prepare-time Compose field checks.

Before any build runs, each build context's raw Compose file is checked against the sandbox
adapter's field allow-list, and after the tasks load each task's resolved document is checked
the same way, so a task the adapter would refuse at seed is refused at prepare with the same
message. The build step plans the builds, drops refused contexts, reuses recorded images that
fit, honours the builds table, constructs the ImageBuilder and collects per-task failure
reasons. cli.py calls it from prepare_target.
"""

from pathlib import Path

import yaml

from nemo_gym.sandbox.compose_fields import unsupported_compose_fields
from nemo_gym.tasks.harbor.build_options import BUILDS_KEY, BuildOptions
from nemo_gym.tasks.harbor.build_plan import DEFAULT_PLATFORM, BuildError, BuildPlanEntry, plan_builds
from nemo_gym.tasks.harbor.build_records import pop_stale_build_inputs, recorded_image_mismatch
from nemo_gym.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_build import build_images
from nemo_gym.tasks.harbor.image_configs import (
    COMPOSE_FILE_NAMES,
    compose_file,
    load_compose_document,
    without_main_build,
    without_pull_policy,
    without_restated_main_build,
    without_sidecar_builds,
)
from nemo_gym.tasks.harbor.records import read_built_images, read_image_configs
from nemo_gym.tasks.harbor.task import HarborTask, is_task_folder


REFUSED_NOTE = " (the sandbox Compose adapter does not start them)"


def unsupported_compose_fields_error(task: HarborTask) -> HarborTaskError | None:
    """Why the sandbox Compose adapter would refuse ``task``'s document at seed, or ``None`` when it starts it.

    The document checked is the one the server hands the adapter (``environment_servers/harbor/app.py``,
    ``_compose_document``): interpolated and ``env_file``-inlined by the loader, ``pull_policy`` dropped from
    every service, ``main``'s ``build:`` dropped when the task runs from its recorded built image or when it
    only restates Harbor's own build of the task's Dockerfile (:func:`without_restated_main_build`; a build
    key Gym cannot apply without a build is this task's reason), and a sidecar's ``build:`` replaced by its
    recorded image. A field no provider starts (``deploy``, ``extra_hosts``, ``platform``) is then a per-task
    reason naming the service and the field, recorded in ``skipped.json`` like a task that does not load, instead of a failure of every episode
    at seed.
    """
    path = compose_file(task)
    document = load_compose_document(task)
    if path is None or document is None:
        return None
    document = without_pull_policy(document)
    if task.built_image is not None:
        document = without_main_build(document)
    else:
        document, reason = without_restated_main_build(task, document)
        if reason is not None:
            return HarborTaskError(f"{path.name}: {reason}")
    refused = unsupported_compose_fields(document)
    if not refused:
        return None
    return HarborTaskError(f"{path.name}: {'; '.join(refused)}{REFUSED_NOTE}")


def refused_before_build(plan: list[BuildPlanEntry]) -> dict[str, HarborTaskError]:
    """The field refusals of the tasks ``plan`` would build, read from their Compose files before any build runs.

    :func:`unsupported_compose_fields_error` is the authoritative check, on the loader's document once the
    tasks load. This one needs only key names, which interpolation and ``env_file`` inlining do not add, so it
    reads the raw file the way the loader's document will look: ``pull_policy`` dropped, ``main``'s ``build:``
    dropped (the task is about to be built and will run from its recorded image) and ``env_file`` set aside,
    as the loader folds it into ``environment``; a sidecar's ``build:`` is dropped too, as its recorded image
    stands in for it. A task it names is neither built nor pushed for an image no
    episode would start, and a build that would fail for it cannot stop the preparation. A file that does not
    parse is left to the loader to report in its own words.
    """
    refused: dict[str, HarborTaskError] = {}
    for entry in plan:
        path = next((entry.context / name for name in COMPOSE_FILE_NAMES if (entry.context / name).is_file()), None)
        if path is None:
            continue
        try:
            document = yaml.safe_load(path.read_text()) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        document = without_sidecar_builds(without_main_build(without_pull_policy(document)))
        services = document.get("services")
        if isinstance(services, dict):
            for name, service in services.items():
                if isinstance(service, dict) and "env_file" in service:
                    services[name] = {key: value for key, value in service.items() if key != "env_file"}
        messages = unsupported_compose_fields(document)
        if messages:
            refused.update(
                dict.fromkeys(entry.tasks, HarborTaskError(f"{path.name}: {'; '.join(messages)}{REFUSED_NOTE}"))
            )
    return refused


def _build_dockerfiles(folder: Path, build: BuildOptions, *, select) -> dict[str, str]:
    """Build and record the images of the selected tasks whose Dockerfile needs a build.

    With builds disabled, the tasks that need one are left for the loader to skip with its
    message, and a recorded image that does not fit the run (another platform, or local when the run
    pushes) leaves its tasks out with the reason, as the enabled path would rebuild it. When every context
    has a usable record nothing is built and no build tool is looked for. Otherwise a missing builder stops
    the preparation, naming the tasks. A failed build does not:
    each context is built on its own, and the tasks of one that fails are returned by id with the
    reason, for the caller to leave out the way it leaves out a task that does not load. A planned task
    whose Compose file the sandbox adapter would refuse whatever the build gave (:func:`refused_before_build`)
    is returned the same way, and its context is neither built nor pushed.
    """
    plan = plan_builds(folder, select=select, force=build.force)
    refused = {task_id: str(error) for task_id, error in refused_before_build(plan).items()}
    plan = [entry for entry in plan if not any(task_id in refused for task_id in entry.tasks)]
    if not plan:
        return refused
    # A recorded image is reused when it fits this run (pushed when the run pushes, of the requested platform),
    # with builds enabled or not; the check is build_images' own.
    parent = folder.parent if is_task_folder(folder) else folder
    recorded, image_configs = read_built_images(parent), read_image_configs(parent)
    # A record built from other inputs than the context holds now is no record (build_inputs_mismatch): with builds
    # enabled the context is rebuilt under the same name (build_images drops the record the same way and says why);
    # disabled, its tasks are left out with the reason, since the loader would run the stale image otherwise.
    stale = pop_stale_build_inputs(
        recorded, plan, image_configs, repository=build.repository, platform=build.platform or DEFAULT_PLATFORM
    )
    if not build.enabled:
        for entry, record, why in stale:
            reason = f"recorded image {record['image']} cannot be used: {why}; builds are disabled"
            print(f"Skipping {', '.join(entry.tasks[:3])}: {reason}")
            refused.update(dict.fromkeys(entry.tasks, reason))
        plan = [entry for entry in plan if not any(task_id in refused for task_id in entry.tasks)]
    platform = build.platform or DEFAULT_PLATFORM
    reusable: dict[str, str] = {}
    unusable: dict[str, str] = {}
    for entry in plan:
        if entry.digest not in recorded:
            continue
        why = recorded_image_mismatch(
            recorded[entry.digest], image_configs, repository=build.repository, platform=platform
        )
        (reusable if why is None else unusable)[entry.digest] = why or recorded[entry.digest]["image"]
    if not build.enabled:
        unbuilt = [task for entry in plan if entry.digest not in recorded for task in entry.tasks[:3]]
        if unbuilt:
            print(
                f"Skipping Dockerfile builds ({BUILDS_KEY}.enabled is false); tasks that need one are left out: "
                + ", ".join(unbuilt)
            )
        # A record that does not fit the run is no better than none: its tasks are left out with the reason.
        mismatched: dict[str, str] = {}
        for entry in plan:
            if entry.digest in unusable:
                image = recorded[entry.digest]["image"]
                reason = f"recorded image {image} cannot be used: {unusable[entry.digest]}; builds are disabled"
                print(f"Skipping {', '.join(entry.tasks[:3])}: {reason}")
                mismatched.update(dict.fromkeys(entry.tasks, reason))
        return {**refused, **mismatched}
    needs_build = [entry for entry in plan if entry.digest not in reusable]
    if not needs_build:
        # Nothing to build, so no build tool is needed: a host with every image recorded prepares without one.
        for entry in plan:
            print(f"Built image recorded for {', '.join(entry.tasks[:3])}: {reusable[entry.digest]}")
        return refused
    names = ", ".join(task for entry in needs_build for task in entry.tasks[:3])
    try:
        builder = ImageBuilder(tool=build.tool, repository=build.repository, platform=build.platform)
    except BuildError as exc:
        raise ValueError(
            f"Could not build the Dockerfile image(s) for task(s) {names}: {exc} (set {BUILDS_KEY}.enabled to false "
            "to run with recorded images only)"
        ) from exc
    results = []
    failures: dict[str, str] = {}
    for entry in plan:
        # One build_images call per context, so a failure stops that context only and the rest still build.
        tasks = set(entry.tasks)
        try:
            results += build_images(folder, builder=builder, select=lambda name: name in tasks, force=build.force)
        except BuildError as exc:
            reason = f"Could not build the Dockerfile image(s) for task(s) {', '.join(entry.tasks[:3])}: {exc}"
            failures.update(dict.fromkeys(entry.tasks, reason))
    if results:
        pushed = sum(1 for result in results if result.pushed)
        print(
            f"Built {len(results)} image(s) with {results[0].tool}"
            + (f", pushed {pushed}" if pushed else ", kept local")
        )
    return {**refused, **failures}
