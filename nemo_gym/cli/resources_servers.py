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
import json

from rich.table import Table

from nemo_gym.cli.utils import (
    exit_unknown_component,
    fuzzy_matches,
    print_no_matches,
    print_rich_table,
    render_component_inspection,
)
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
    COMPONENT_NAME_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    QUERY_KEY_NAME,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.resources_server_registry import discover_resources_servers, read_resources_server_value


def _inspect_resources_server(name: str, servers: dict, global_config_dict) -> None:
    """Render the ``gym list resources-servers <name>`` inspect view for one server."""
    entry = servers.get(name)
    if entry is None:
        exit_unknown_component(name, servers, "resources server")
        return

    value = read_resources_server_value(entry.config_path)
    description = entry.description
    if value:  # surface `value` as a trailing line of the description
        description = f"{description}\nValue: {value}" if description else f"Value: {value}"

    render_component_inspection(
        json_output=global_config_dict.get(JSON_OUTPUT_KEY_NAME, False),
        name=name,
        type_noun="resources server",
        domain=entry.domain,
        description=description,
        details={"config": str(entry.config_path.resolve())},
        usage=f"gym env start --resources-server {name} --model-type vllm_model",
    )


def list_resources_servers() -> None:
    """List the resources servers selectable with ``--resources-server``, or inspect one by name
    (``gym list resources-servers <name>``). Optionally filtered by a `query` (the
    `gym search resources-servers` entry point). ``--search-dir`` adds extra roots on top of the cwd and built-ins.
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    servers = discover_resources_servers()

    name = global_config_dict.get(COMPONENT_NAME_KEY_NAME)
    if name:
        _inspect_resources_server(name, servers, global_config_dict)
        return

    # `gym search resources-servers <query>` reuses this command, narrowing to fuzzy matches on
    # name + domain + description.
    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        servers = {
            name: s for name, s in servers.items() if fuzzy_matches(query, name, s.domain or "", s.description or "")
        }

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(
            json.dumps(
                [{"name": name, "domain": s.domain, "description": s.description} for name, s in servers.items()]
            )
        )
        return

    if not servers:
        print_no_matches("resources servers", query)
        return

    title = (
        f"Resources servers matching '{query}' ({len(servers)})"
        if query
        else f"Available resources servers in NeMo Gym ({len(servers)})"
    )
    table = Table(title=title)
    table.add_column("Name")
    table.add_column("Domain")
    table.add_column("Description")
    for name, server in servers.items():
        table.add_row(name, server.domain or "", server.description or "")
    print_rich_table(table)
