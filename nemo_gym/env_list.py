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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# Directories to scan for environment configs.
# Agent directories are included so agent-only environments (e.g. verifiers) can be listed,
# but an agent config only appears if it has `domain` set. this prevents other agents
# (simple_agent, langgraph_agent, etc.) from showing up even if they gain dataset entries.
SCAN_FOLDERS = ["resources_servers", "responses_api_agents"]
KNOWN_SERVER_TYPES = {"resources_servers", "responses_api_agents"}


@dataclass
class EnvInfo:
    name: str
    server_type: str
    domain: Optional[str]
    description: Optional[str]
    has_train: bool
    has_validation: bool
    config_path: str
    is_example: bool = False


def _visit_server_metadata(data: dict, level: int = 1) -> dict:
    if level == 4 and isinstance(data, dict):
        return {k: data.get(k) for k in ("domain", "description", "verified", "value")}
    if isinstance(data, dict):
        for k, v in data.items():
            if level == 2 and k not in KNOWN_SERVER_TYPES:
                continue
            return _visit_server_metadata(v, level + 1)
    return {}


def _visit_datasets(data: dict) -> list[str]:
    """Return all dataset type strings found anywhere in the YAML."""
    types: list[str] = []
    for v1 in data.values():
        if not isinstance(v1, dict):
            continue
        for server_type in KNOWN_SERVER_TYPES:
            v2 = v1.get(server_type)
            if not isinstance(v2, dict):
                continue
            for v3 in v2.values():
                if not isinstance(v3, dict):
                    continue
                for entry in v3.get("datasets") or []:
                    if isinstance(entry, dict) and entry.get("type"):
                        types.append(entry["type"])
    return types


def get_envs(parent_dir: Path) -> list[EnvInfo]:
    """Scan server type directories and return environments.

    For resources_servers: included if datasets are present.
    For responses_api_agents: also requires `domain` to be set, so infrastructure agents
    (simple_agent, langgraph_agent, etc.) are never listed unintentionally.
    """
    envs: list[EnvInfo] = []
    for folder_name in SCAN_FOLDERS:
        folder = parent_dir / folder_name
        if not folder.exists():
            continue
        for subdir in sorted(folder.iterdir()):
            if not subdir.is_dir():
                continue
            configs_folder = subdir / "configs"
            if not (configs_folder.exists() and configs_folder.is_dir()):
                continue
            for yaml_file in sorted(configs_folder.glob("*.yaml")):
                with yaml_file.open() as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    continue
                types = _visit_datasets(data)
                if not types:
                    continue
                metadata = _visit_server_metadata(data)
                # skip if no domain
                if folder_name == "responses_api_agents" and not metadata.get("domain"):
                    continue
                envs.append(
                    EnvInfo(
                        name=subdir.name,
                        server_type=folder_name,
                        domain=metadata.get("domain"),
                        description=metadata.get("description"),
                        has_train="train" in types,
                        has_validation="validation" in types,
                        config_path=f"{folder_name}/{subdir.name}/configs/{yaml_file.name}",
                        is_example=subdir.name.startswith("example_"),
                    )
                )
    return envs
