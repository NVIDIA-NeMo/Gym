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
the recorded image for every built service and drops their ``build:`` and a ``pull_policy: build``
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

The hash names the image; it is not the whole build. Harbor's hash covers file names and bytes only, so the
configuration record of each built image in ``compose-images.json`` also carries an inputs digest
(:func:`build_inputs_digest`, over every entry of the context tree with its mode, content or link target, the
names Harbor's hash skips included, plus the service's ``build`` section of the task's Compose file as the build
reads it: interpolated against the build's environment, its paths resolved, its arguments as the tool is given
them, and the inputs digest of each service it builds from through a ``service:`` reference of its
``additional_contexts``, transitively), beside the ``config_digest`` of what came out. A recorded image is reused
only when the hash and the digest both match: a service whose inputs changed without changing the context's hash is
rebuilt under the same name and its records replaced, the context's other services left as recorded
(:func:`build_inputs_mismatch`); a consumer of a changed service is rebuilt with it.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from pathlib import Path
from typing import Callable

from nemo_gym.tasks.harbor.build_plan import BuildConflict, BuildPlanEntry, BuildResult, plan_builds
from nemo_gym.tasks.harbor.build_records import build_inputs_mismatch, record_built_image, recorded_image_mismatch
from nemo_gym.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.tasks.harbor.records import (
    read_built_images,
    read_image_configs,
    recorded_service_images,
)
from nemo_gym.tasks.harbor.task import is_task_folder


LOGGER = logging.getLogger(__name__)


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
    and of the requested platform, and built from the inputs the context holds now
    (:func:`build_inputs_mismatch`). Otherwise it is rebuilt and its record replaced, and the reason
    is logged. A context is keyed by its content hash, so a changed build context is a new build;
    deleting a context's entry from ``built-images.json`` forces one for an unchanged context. The services of
    one context are built together (:meth:`ImageBuilder.build_many`), the ones with a fitting record left out,
    and each is recorded as soon as it is done, so a prepare interrupted after a context's first service
    resumes at its second. Returns the images built by this call. Raises :class:`BuildError` on the first failure,
    after recording whatever was built before it.
    """
    folder = Path(folder).resolve()
    parent = folder.parent if is_task_folder(folder) else folder
    conflicts: dict[str, BuildConflict] = {}
    plan = plan_builds(folder, select=select, conflicts=conflicts)
    for conflict in conflicts.values():
        # A folder with another task's Harbor hash and other build inputs: in no entry (prepare leaves the task out
        # with this reason), the entry built for the task whose folder it holds.
        log(f"Skipping {conflict.reason}")
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
                ) or build_inputs_mismatch(
                    entry, recorded[digest], image_configs, repository=builder.repository, platform=builder.platform
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
