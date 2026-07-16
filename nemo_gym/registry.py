# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Registry of co-located environments under ``environments/<name>/``.

An *environment* is a directory ``environments/<name>/`` whose ``config.yaml`` wires together a
resources server, an agent, and datasets (and references a model server). This module maps an
environment's short ``<name>`` to its config so it can be enumerated by name — the foundation for
``gym list environments``. Resolving a name to a config path for *running* is handled by the CLI's
generic ``--environment`` asset selector, so this module is intentionally discovery-only.

Discovery only reads config files and never starts servers; ``domain``/``description`` come from the
shared :func:`~nemo_gym.discovery.read_config_metadata` reader, which tolerates unset secrets/API keys,
so it's safe to call even when those aren't set.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from nemo_gym import PARENT_DIR
from nemo_gym.discovery import discover_components, read_config_metadata


ENVIRONMENTS_SUBDIR = "environments"
ENVIRONMENTS_DIR = PARENT_DIR / ENVIRONMENTS_SUBDIR
ENVIRONMENT_CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class EnvironmentEntry:
    """A discovered environment: its name, where it lives, and lightweight metadata."""

    name: str
    config_path: Path
    path: Path
    description: Optional[str] = None
    domain: Optional[str] = None


def _discover_environments_in_dir(environments_dir: Path) -> Dict[str, EnvironmentEntry]:
    """Map environment name -> :class:`EnvironmentEntry` for every ``<name>/config.yaml`` under one dir.

    The name is the directory name. Returns an empty dict if the directory is missing.
    """
    environments: Dict[str, EnvironmentEntry] = {}
    if not environments_dir.is_dir():
        return environments

    for child in sorted(environments_dir.iterdir()):
        config_path = child / ENVIRONMENT_CONFIG_FILENAME
        if not (child.is_dir() and config_path.is_file()):
            continue

        domain, description = read_config_metadata(config_path)
        environments[child.name] = EnvironmentEntry(
            name=child.name,
            config_path=config_path,
            path=child,
            description=description,
            domain=domain,
        )

    return environments


def discover_environments() -> Dict[str, EnvironmentEntry]:
    """Map environment name -> :class:`EnvironmentEntry` for every discoverable ``<name>/config.yaml``.

    Scans the ``environments/`` subdir of every :func:`~nemo_gym.discovery.component_search_roots` root
    (``NEMO_GYM_EXTRA_ROOTS`` + cwd + built-ins), merged so user environments shadow same-named built-ins.
    """
    return discover_components(ENVIRONMENTS_SUBDIR, _discover_environments_in_dir)
