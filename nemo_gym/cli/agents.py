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

import rich
from rich.table import Table

from nemo_gym.agent_registry import discover_agents
from nemo_gym.config_types import BaseNeMoGymCLIConfig
from nemo_gym.global_config import (
    JSON_OUTPUT_KEY_NAME,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)


def list_agents() -> None:
    """CLI command: list discovered agent harnesses and how each composes (Pattern A vs B).

    Complements ``gym list benchmarks``: the asset selectors resolve a component *by name*, but only
    this listing surfaces which agents are freely wireable into a separate environment (Pattern A)
    versus self-contained harnesses that run with their own config (Pattern B) — the distinction the
    config composer's compatibility guard relies on.
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    agents = discover_agents()

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
        rich.print("No agents found.")
        return

    table = Table(title="NeMo Gym agents")
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
    rich.print(table)
