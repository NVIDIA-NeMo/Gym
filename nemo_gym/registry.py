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

Discovery only reads config files; it never resolves interpolations or starts servers, so it is
safe to call even when secrets/API keys referenced by a config are not set in the environment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from omegaconf import OmegaConf

from nemo_gym import PARENT_DIR


ENVIRONMENTS_DIR = PARENT_DIR / "environments"
ENVIRONMENT_CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class EnvironmentEntry:
    """A discovered environment: its name, where it lives, and lightweight metadata."""

    name: str
    config_path: Path
    path: Path
    description: Optional[str] = None
    domain: Optional[str] = None


def _read_metadata(config_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort ``(description, domain)`` from the config's resources_servers entry.

    Reads without resolving interpolations or missing values so a config that references an unset
    key (e.g. an API key) still yields metadata instead of raising.
    """
    try:
        container = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        return None, None

    if not isinstance(container, dict):
        return None, None

    for top_level_value in container.values():
        if not isinstance(top_level_value, dict):
            continue
        resources_servers = top_level_value.get("resources_servers")
        if not isinstance(resources_servers, dict):
            continue
        for server_config in resources_servers.values():
            if isinstance(server_config, dict):
                description = server_config.get("description")
                domain = server_config.get("domain")
                return (
                    description if isinstance(description, str) else None,
                    domain if isinstance(domain, str) else None,
                )
    return None, None


def discover_environments(environments_dir: Path = ENVIRONMENTS_DIR) -> Dict[str, EnvironmentEntry]:
    """Map environment name -> :class:`EnvironmentEntry` for every ``<name>/config.yaml``.

    The name is the directory name. Returns an empty dict if the directory is missing.
    """
    environments: Dict[str, EnvironmentEntry] = {}
    if not environments_dir.is_dir():
        return environments

    for child in sorted(environments_dir.iterdir()):
        config_path = child / ENVIRONMENT_CONFIG_FILENAME
        if not (child.is_dir() and config_path.is_file()):
            continue

        description, domain = _read_metadata(config_path)
        environments[child.name] = EnvironmentEntry(
            name=child.name,
            config_path=config_path,
            path=child,
            description=description,
            domain=domain,
        )

    return environments
