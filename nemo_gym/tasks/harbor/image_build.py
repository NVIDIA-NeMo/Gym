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
credentials and any context or Dockerfile outside ``environment/`` are refused, and so is a sidecar with a
``build:`` section, since no Gym provider starts a locally built sidecar (the loader records sidecars from
their ``image:``). At seed the loader runs the task from the recorded image, so ``main``'s ``build:`` and
``pull_policy`` in the task file are dropped there (``nemo_gym.tasks.harbor.image_configs.without_main_build``).
``docker build`` is never called. Harbor serialises concurrent builds of one environment with an
in-process lock; prepare builds each distinct context once, sequentially, so no lock is needed here.
buildah and podman are the OpenSandbox path: they build the Dockerfile directly and push it to a registry.

A build is cancel-safe: nothing is recorded until the tool has built, pushed and inspected the
image, and each image is recorded as soon as it is done, so an interrupted prepare resumes
where it stopped. The record files are replaced atomically under a sidecar lock
(``nemo_gym.tasks.harbor.task.record_lock``), so two Gym processes recording into one datasets
folder keep each other's entries and a reader never sees a half-written file.

The hash names the image; it is not the whole build. Harbor's hash covers file names and bytes only, so the
configuration record of each built image in ``compose-images.json`` also carries an inputs digest
(:func:`build_inputs_digest`, over every entry of the context tree with its mode, content or link target, the
names Harbor's hash skips included, plus ``main``'s ``build`` section of the task's Compose file as the build reads
it: interpolated against the build's environment, its paths resolved, its arguments as the tool is given them),
beside the ``config_digest`` of what came out. A recorded image is reused only when the hash and the digest both
match: a context whose inputs changed without changing its hash is rebuilt under the same name and its records
replaced (:func:`build_inputs_mismatch`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from nemo_gym.tasks.harbor.build_plan import BuildResult, plan_builds
from nemo_gym.tasks.harbor.build_records import pop_stale_build_inputs, record_built_image, recorded_image_mismatch
from nemo_gym.tasks.harbor.build_tools import ImageBuilder
from nemo_gym.tasks.harbor.records import (
    read_built_images,
    read_image_configs,
)
from nemo_gym.tasks.harbor.task import is_task_folder


LOGGER = logging.getLogger(__name__)


def build_images(
    folder: Path,
    *,
    builder: ImageBuilder,
    select: Callable[[str], bool] | None = None,
    log: Callable[[str], None] = print,
) -> list[BuildResult]:
    """Build and record every context under ``folder`` that needs it and has no usable record.

    A recorded image is reused when it fits this builder: pushed, or local when nothing is pushed,
    and of the requested platform. Otherwise it is rebuilt and its record replaced, and the reason
    is logged. A context is keyed by its content hash, so a changed build context is a new build;
    deleting a context's entry from ``built-images.json`` forces one for an unchanged context.
    Returns the images built by this call. Raises :class:`BuildError` on the first failure, after
    recording whatever was built before it.
    """
    folder = Path(folder).resolve()
    parent = folder.parent if is_task_folder(folder) else folder
    plan = plan_builds(folder, select=select)
    if not plan:
        return []
    recorded = read_built_images(parent)
    image_configs = read_image_configs(parent)
    for entry, stale, why in pop_stale_build_inputs(
        recorded, plan, image_configs, repository=builder.repository, platform=builder.platform
    ):
        log(f"Rebuilding {stale['image']} for {', '.join(entry.tasks[:3])}: {why}")
    results: list[BuildResult] = []
    for entry in plan:
        names = ", ".join(entry.tasks[:3])
        if entry.digest in recorded:
            why = recorded_image_mismatch(
                recorded[entry.digest], image_configs, repository=builder.repository, platform=builder.platform
            )
            if why is None:
                log(f"Built image recorded for {names}: {recorded[entry.digest]['image']}")
                continue
            log(f"Rebuilding {recorded[entry.digest]['image']} for {names}: {why}")
        result = builder.build(entry)
        # The build just made is authoritative for its image: a configuration left behind under the same
        # name (by a deleted built-images.json entry, or a record that no longer fits) is replaced.
        record_built_image(parent, result, overwrite=True)
        where = "pushed" if result.pushed else "local"
        log(f"Built {result.image} ({where}, {result.tool}) for task(s) {', '.join(entry.tasks[:5])}")
        results.append(result)
    return results
