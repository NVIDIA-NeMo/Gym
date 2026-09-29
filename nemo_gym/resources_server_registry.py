# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Registry of resources servers under ``resources_servers/<name>/``.

A resources server (verifier + per-task state) is one *component* of an environment, selected by name
with ``--resources-server``. This module maps each config flavor to its ``(domain, description)`` — read
the same way ``gym list environments``/``benchmarks`` read theirs (via
:func:`~nemo_gym.discovery.read_config_metadata`) — so they can be enumerated by name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from omegaconf import OmegaConf

from nemo_gym import PARENT_DIR
from nemo_gym.discovery import discover_components, iter_server_configs, read_config_metadata


RESOURCES_SERVERS_SUBDIR = "resources_servers"
RESOURCES_SERVERS_DIR = PARENT_DIR / RESOURCES_SERVERS_SUBDIR
RESOURCES_SERVER_CONFIGS_SUBDIR = "configs"


@dataclass(frozen=True)
class ResourcesServerEntry:
    """A discovered resources server: its name, where it lives, and lightweight metadata."""

    name: str
    config_path: Path
    path: Path
    description: Optional[str] = None
    domain: Optional[str] = None


def _config_defines_resources_server(config_path: Path) -> bool:
    """True if a config declares a ``resources_servers`` block (vs a helper like a judge model). Never raises."""
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    return any(
        isinstance(instance, dict) and isinstance(instance.get("resources_servers"), dict) for instance in raw.values()
    )


def _discover_resources_servers_in_dir(resources_servers_dir: Path) -> Dict[str, ResourcesServerEntry]:
    """Map resources-server name -> :class:`ResourcesServerEntry` for every flavor under one dir.

    One entry per config flavor that declares a `resources_servers` block: `<dir>` for the default config
    (`<dir>.yaml`) and `<dir>/<flavor>` for the rest (`<flavor>.yaml`).
    Helper configs (no `resources_servers` block) are skipped. Empty dict if the dir is missing.
    """
    servers: Dict[str, ResourcesServerEntry] = {}
    if not resources_servers_dir.is_dir():
        return servers

    for child in sorted(resources_servers_dir.iterdir()):
        if not child.is_dir():
            continue
        configs_dir = child / RESOURCES_SERVER_CONFIGS_SUBDIR
        config_files = sorted(configs_dir.glob("*.yaml")) if configs_dir.is_dir() else []
        if not config_files:
            continue
        for config in config_files:
            if not _config_defines_resources_server(config):
                continue
            name = child.name if config.stem == child.name else f"{child.name}/{config.stem}"
            domain, description = read_config_metadata(config)
            servers[name] = ResourcesServerEntry(
                name=name,
                config_path=config,
                path=child,
                description=description,
                domain=domain,
            )

    return servers


def discover_resources_servers() -> Dict[str, ResourcesServerEntry]:
    """Map resources-server name -> :class:`ResourcesServerEntry` for every discoverable server.

    Scans the ``resources_servers/`` subdir of every :func:`~nemo_gym.discovery.component_search_roots`
    root (``NEMO_GYM_EXTRA_ROOTS`` + cwd + built-ins), merged so user servers shadow same-named built-ins.
    """
    return discover_components(RESOURCES_SERVERS_SUBDIR, _discover_resources_servers_in_dir)


def read_resources_server_value(config_path: Path) -> Optional[str]:
    """The ``value`` field declared on a resources server config (for the inspect view). Never raises."""
    try:
        raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        return None
    for group_key, _server_name, server_config in iter_server_configs(raw):
        if group_key == "resources_servers" and server_config.get("value"):
            return str(server_config["value"])
    return None
