# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record the OCI configuration of every image a Compose task starts.

Docker Compose reads a service's entrypoint, command, user, working directory and
exposed ports from the image itself. A sandbox platform without Docker cannot, so
the harbor resources server reads them from ``compose-images.json`` next to the task
folders (see ``nemo_gym.sandbox.compose_config.resolve_compose``). This module writes
that file from the registry, one entry per image reference exactly as the task wrote
it, pinned to the linux/amd64 manifest digest it resolved to.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from nemo_gym.tasks.harbor.registry import ImageConfigError, RegistryClient
from nemo_gym.tasks.harbor.task import HarborTask


LOGGER = logging.getLogger(__name__)

COMPOSE_IMAGES_FILE = "compose-images.json"
COMPOSE_FILE_NAMES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")


def compose_file(task: HarborTask) -> Path | None:
    """The task's Compose overlay, if it has one."""
    for name in COMPOSE_FILE_NAMES:
        path = task.path / "environment" / name
        if path.is_file():
            return path
    return None


def compose_image_references(task: HarborTask) -> list[str]:
    """Every image a Compose task starts: the task image for ``main`` and each sidecar's image."""
    path = compose_file(task)
    if path is None:
        return []
    document = yaml.safe_load(path.read_text()) or {}
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


def record_compose_images(
    tasks: list[HarborTask], folder: Path, *, client: RegistryClient | None = None
) -> Path | None:
    """Write ``compose-images.json`` in ``folder`` for the Compose tasks among ``tasks``.

    Entries already recorded are kept as they are, so a pinned recording never changes
    under a completed run; only references missing from the file are resolved. Returns
    the file's path, or ``None`` when no task uses Compose.
    """
    references = list(dict.fromkeys(ref for task in tasks for ref in compose_image_references(task)))
    if not references:
        return None
    path = Path(folder) / COMPOSE_IMAGES_FILE
    recorded: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    missing = [ref for ref in references if ref not in recorded]
    if not missing:
        return path
    client = client or RegistryClient()
    LOGGER.info(f"Recording the configuration of {len(missing)} Compose image(s) into {path}")
    for reference in missing:
        recorded[reference] = client.image_config(reference)
    path.write_text(json.dumps(dict(sorted(recorded.items())), indent=2, sort_keys=True) + "\n")
    return path
