# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two record files prepare writes next to the task folders, and how they are read.

compose-images.json records image configurations, built-images.json records the images built
per context digest and service. This module names the files and the service keys, takes the
sidecar lock, replaces a file atomically, reads it with one retry against a torn write, and
reads the per-service entries with the legacy main-only fallback. The loader, the recorders,
the build modules and the environment server all read through it.
"""

import json
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.errors import HarborTaskError


# The OCI configurations recorded at prepare next to the task folders, keyed by image reference as the
# tasks wrote it (see ``nemo_gym.tasks.harbor.image_configs``): Compose sidecar images and the base image
# of every Dockerfile the loader resolves.
IMAGE_CONFIGS_FILE = "compose-images.json"


def read_image_configs(folder: Path) -> dict[str, Any]:
    """The image configurations recorded in ``folder`` (the tasks' parent), or ``{}`` when none were."""
    path = Path(folder) / IMAGE_CONFIGS_FILE
    if not path.is_file():
        return {}
    try:
        recorded = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise HarborTaskError(f"{path} is not valid JSON: {exc}") from exc
    return recorded if isinstance(recorded, dict) else {}
