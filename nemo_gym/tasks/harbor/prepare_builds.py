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

from nemo_gym.sandbox.compose_fields import DEFAULT_PLATFORM, unsupported_compose_fields
from nemo_gym.tasks.harbor.build_options import BUILDS_KEY, BuildOptions
from nemo_gym.tasks.harbor.build_plan import DOCKERFILE_NAME, BuildError, BuildPlanEntry, plan_builds
from nemo_gym.tasks.harbor.build_records import build_inputs_mismatch, recorded_image_mismatch
from nemo_gym.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.image_build import build_images
from nemo_gym.tasks.harbor.image_configs import (
    COMPOSE_FILE_NAMES,
    restated_main_build_reason,
    with_built_images,
    without_pull_policy,
)
from nemo_gym.tasks.harbor.records import MAIN_SERVICE, read_built_images, read_image_configs, recorded_service_images
from nemo_gym.tasks.harbor.task import is_task_folder


def refused_before_build(plan: list[BuildPlanEntry]) -> dict[str, HarborTaskError]:
    """The key refusals of the tasks ``plan`` would build, read from their Compose files before any build runs.

    The authoritative check is ``check_task_compose_fields`` (:func:`record_compose_images` runs it on the loader's
    document once the tasks load); this screen is its first step, the key allow-list
    (:func:`~nemo_gym.sandbox.compose_config.unsupported_compose_fields`), which needs only key names, which
    interpolation and ``env_file`` inlining do not add. It reads the raw file the way the loader's document will
    look: ``pull_policy`` dropped, the ``build:`` of every service the plan builds from the context dropped (they
    are about to be built and will run from their recorded images) and ``env_file`` set aside, as the loader folds
    it into ``environment``. A task it names is neither built nor pushed for an image no episode would start, and a
    build that would fail for it cannot stop the preparation, with the reason spelled as the loader-time check
    spells it. A file that does not parse is left to the loader to report in its own words. The separate verifier's
    ``tests/`` context is not checked: its image starts as a single sandbox, never through the adapter.
    Its entry is the one with ``BuildPlanEntry.verifier`` set; an ordinary sidecar that happens to be named
    ``verifier`` (kumo's) is a service of the ``environment/`` document like any other, built and substituted here.

    When the plan does not build ``main`` (a pull or overlay Dockerfile beside a built sidecar), a ``main`` ``build:``
    that only restates Harbor's own build of that Dockerfile is dropped as the loader drops it
    (:func:`restated_main_build_reason`, the rule of ``without_restated_main_build``, with ``${CONTEXT_DIR}`` accepted
    as the raw file spells the context), and a key Gym cannot apply without a build is the task's reason here too,
    spelled as the loader spells it.
    """
    refused: dict[str, HarborTaskError] = {}
    planned: dict[str, dict[str, str]] = {}
    for entry in plan:
        if not entry.verifier:
            planned.setdefault(entry.digest, {})[entry.service] = "built"
    for entry in plan:
        if entry.digest not in planned or entry.service != next(iter(planned[entry.digest])):
            continue  # once per context, on its first planned service
        path = next((entry.context / name for name in COMPOSE_FILE_NAMES if (entry.context / name).is_file()), None)
        if path is None:
            continue
        try:
            document = yaml.safe_load(path.read_text()) or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue
        document = with_built_images(without_pull_policy(document), planned[entry.digest])
        services = document.get("services")
        if isinstance(services, dict):
            main = services.get(MAIN_SERVICE)
            if (
                MAIN_SERVICE not in planned[entry.digest]
                and isinstance(main, dict)
                and "build" in main
                and (entry.context / DOCKERFILE_NAME).is_file()
            ):
                # A pull or overlay main beside a built sidecar: Harbor's template already builds main from this
                # Dockerfile, so a section that only restates it is dropped as the loader drops it; the raw file is
                # not interpolated, so ${CONTEXT_DIR} spells the context here. (A prebuilt main with a Dockerfile is
                # not told apart here; the loader still refuses its section as today.)
                reason = restated_main_build_reason(
                    main["build"], context_dir=entry.context, literal_contexts=("${CONTEXT_DIR}", "${CONTEXT_DIR}/")
                )
                if reason is not None:
                    where = f"{entry.context.name}/{path.name}"
                    refused.update(
                        {task_id: HarborTaskError(f"{task_id}: {where}: {reason}") for task_id in entry.tasks}
                    )
                    continue
                services[MAIN_SERVICE] = {key: value for key, value in main.items() if key != "build"}
            for name, service in services.items():
                if isinstance(service, dict) and "env_file" in service:
                    services[name] = {key: value for key, value in service.items() if key != "env_file"}
        messages = unsupported_compose_fields(document)
        if messages:
            where = f"{entry.context.name}/{path.name}"
            refused.update({task_id: HarborTaskError(f"{task_id}: {where}: {messages[0]}") for task_id in entry.tasks})
    return refused


def _build_dockerfiles(folder: Path, build: BuildOptions, *, select) -> dict[str, str]:
    """Build and record the images of the selected tasks whose Dockerfile needs a build.

    With builds disabled, the tasks that need one are left for the loader to skip with its message, a task whose
    separate verifier's ``tests/Dockerfile`` is unbuilt included (the loader refuses an unrecorded verifier image
    the way it refuses an unrecorded ``main``), and a recorded image that does not fit the run (another platform,
    or local when the run pushes) leaves its tasks out with the reason, as the enabled path would rebuild it. When
    every context has a usable record nothing is built and no build tool is looked for. Otherwise a missing builder
    stops the preparation, naming the tasks. A failed build does not:
    each context (every service it builds) is built on its own, and the tasks of one that fails are returned
    by id with the reason, for the caller to leave out the way it leaves out a task that does not load. A task in
    separate verifier mode may own two contexts, ``environment/`` and ``tests/`` (its verifier's image); a failure of
    either leaves the task out. A planned task whose Compose file the sandbox adapter would refuse whatever the
    build gave (:func:`refused_before_build`) is returned the same way, and its contexts are neither built nor pushed
    for it; a context it shares with an eligible task (a separate verifier's ``tests/``, byte-identical across tasks
    whose ``environment/`` differ) is still built for that task. A task whose folder has another task's Harbor
    hash but other build inputs (``plan_builds``' ``conflicts``) is left out with the reason while that other task
    stays in: when the other task is itself refused in the same round, the plan is made again without it, and this
    task's folder is the one built under the name if it is first then.
    """
    refused: dict[str, str] = {}
    while True:
        conflicts = {}  # task id -> BuildConflict, filled by plan_builds
        services = plan_builds(
            folder, select=lambda name: name not in refused and (select is None or select(name)), conflicts=conflicts
        )
        found = {task_id: str(error) for task_id, error in refused_before_build(services).items()}
        # A conflict with a task refused this round is not a refusal: without that task the plan may hold this one.
        found |= {task_id: conflict.reason for task_id, conflict in conflicts.items() if conflict.first not in found}
        if not found:
            break
        # Planned again without the refused tasks rather than filtered by them: a context only they use is gone, a
        # context an eligible task shares with them is kept with its tasks narrowed to the eligible ones, and its
        # settings (env, cpus, memory) come from the first of those, by plan_builds' own first-task rule. Another
        # task may then be the first of a context, so the plan is repeated until nothing new is refused.
        refused |= found
    # One entry per context: the plan has one per (context, service), with the same tasks.
    plan = list({entry.digest: entry for entry in services}.values())
    if not plan:
        return refused
    # A recorded image is reused when it fits this run (pushed when the run pushes, of the requested platform),
    # with builds enabled or not; the check is build_images' own, per planned service of the context. A context
    # is unbuilt while any of its planned services has no record: a legacy record of main alone, or a prepare
    # interrupted after a sidecar, leaves the digest present and the task still unrunnable.
    parent = folder.parent if is_task_folder(folder) else folder
    recorded, image_configs = read_built_images(parent), read_image_configs(parent)
    platform = build.platform or DEFAULT_PLATFORM
    unbuilt_digests: set[str] = set()
    unusable: dict[str, tuple[str, str]] = {}  # context digest -> (image, why) of the first record that does not fit
    for entry in services:
        images = recorded_service_images(recorded.get(entry.digest))
        if entry.service not in images:
            unbuilt_digests.add(entry.digest)
            continue
        # A record that does not fit the run, or was built from other inputs than the context holds now, is no better
        # than none: the enabled path rebuilds the service under the same name, the disabled path leaves the task out.
        why = recorded_image_mismatch(
            recorded[entry.digest],
            image_configs,
            repository=build.repository,
            platform=platform,
            service=entry.service,
        ) or build_inputs_mismatch(
            entry, recorded[entry.digest], image_configs, repository=build.repository, platform=platform
        )
        if why is not None:
            unusable.setdefault(entry.digest, (images[entry.service], why))
    reusable = {
        entry.digest: ", ".join(sorted(set(recorded_service_images(recorded[entry.digest]).values())))
        for entry in plan
        if entry.digest in recorded and entry.digest not in unbuilt_digests and entry.digest not in unusable
    }
    if not build.enabled:
        unbuilt = [task for entry in plan if entry.digest in unbuilt_digests for task in entry.tasks[:3]]
        if unbuilt:
            print(
                f"Skipping Dockerfile builds ({BUILDS_KEY}.enabled is false); tasks that need one are left out: "
                + ", ".join(unbuilt)
            )
        # A record that does not fit the run is no better than none: its tasks are left out with the reason, a
        # context whose other service is unrecorded included (the loader refuses the task that needs that service,
        # but a task whose main is pinned by the dataset needs only the sidecar and would run the stale record).
        mismatched: dict[str, str] = {}
        for entry in plan:
            if entry.digest in unusable:
                image, why = unusable[entry.digest]
                reason = f"recorded image {image} cannot be used: {why}; builds are disabled"
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
        if tasks <= set(failures):
            # Every task of this context is already out (its other context failed): nothing to build it for.
            continue
        try:
            results += build_images(
                folder, builder=builder, select=lambda name: name in tasks, contexts={entry.digest}
            )
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
