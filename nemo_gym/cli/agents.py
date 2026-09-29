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
import json

from rich.table import Table

from nemo_gym.agent_registry import discover_agents
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


def _inspect_agent(name: str, agents: dict, global_config_dict) -> None:
    """Render the ``gym list agents <name>`` inspect view for one agent (thin: no usage example)."""
    entry = agents.get(name)
    if entry is None:
        exit_unknown_component(name, agents, "agent")
        return

    details = {
        "path": str(entry.path.resolve()),
        "composition": "self-contained (B)" if entry.self_contained else "composable (A)",
    }
    if entry.variants:
        details["variants"] = ", ".join(sorted(entry.variants))

    render_component_inspection(
        json_output=global_config_dict.get(JSON_OUTPUT_KEY_NAME, False),
        name=name,
        type_noun="agent",
        description=entry.description,
        details=details,
    )


def list_agents() -> None:
    """List discovered agent harnesses and how each composes (Pattern A vs. self-contained B), or inspect one
    by name (``gym list agents <name>``). Optionally filtered by a `query` (the `gym search agents` entry
    point). ``--search-dir`` adds extra roots on top of the cwd and built-ins.
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    agents = discover_agents()

    name = global_config_dict.get(COMPONENT_NAME_KEY_NAME)
    if name:
        _inspect_agent(name, agents, global_config_dict)
        return

    # `gym search agents <query>` reuses this command, narrowing to fuzzy matches on
    # name + description + variant names.
    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        agents = {
            name: entry
            for name, entry in agents.items()
            if fuzzy_matches(query, name, entry.description or "", *entry.variants)
        }

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        payload = [
            {
                "name": name,
                "pattern": "B (self-contained)" if entry.self_contained else "A (composable)",
                "self_contained": entry.self_contained,
                "variants": sorted(entry.variants),
                "description": entry.description,
            }
            for name, entry in agents.items()
        ]
        print(json.dumps(payload))
        return

    if not agents:
        print_no_matches("agents", query)
        return

    table = Table(title=f"Agents matching '{query}'" if query else "NeMo Gym agents")
    table.add_column("agent", style="bold")
    table.add_column("composition")
    table.add_column("variants")
    table.add_column("description")
    for name, entry in agents.items():
        composition = "self-contained (B)" if entry.self_contained else "composable (A)"
        table.add_row(
            name,
            composition,
            ", ".join(sorted(entry.variants)) or "—",
            entry.description or "",
        )
    print_rich_table(table)
