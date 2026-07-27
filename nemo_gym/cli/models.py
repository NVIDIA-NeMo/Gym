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
from nemo_gym.model_registry import discover_models


def _inspect_model(name: str, models: dict, global_config_dict) -> None:
    """Render the ``gym list models <name>`` inspect view for one model (thin: no usage example).

    ``name`` may be a bare model or a ``<model>/<flavor>`` token; a valid flavor renders the model's
    (main) inspection.
    """

    entry = models.get(name)
    if entry is None:
        exit_unknown_component(name, models, "model")
        return

    render_component_inspection(
        json_output=global_config_dict.get(JSON_OUTPUT_KEY_NAME, False),
        name=name,
        type_noun="model",
        details={"config": str(entry.config_path.resolve())},
    )


def list_models() -> None:
    """List model servers (one row per ``--model-type`` value: ``Model`` is the token to pass, ``Model group``
    its model), or inspect one by name (``gym list models <name>``). Optionally filtered by a `query` (the
    `gym search models` entry point). ``--search-dir`` adds extra roots on top of the cwd and built-ins.
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    models = discover_models()

    name = global_config_dict.get(COMPONENT_NAME_KEY_NAME)
    if name:
        _inspect_model(name, models, global_config_dict)
        return

    # One row per passable `--model-type` value: `model` is the token, `model_group` its model.
    rows = [{"model": entry.name, "model_group": entry.model_group} for entry in models.values()]

    # `gym search models <query>` reuses this command, narrowing to rows matching the token or its model.
    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        rows = [row for row in rows if fuzzy_matches(query, row["model"], row["model_group"])]

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps(rows))
        return

    if not rows:
        print_no_matches("models", query)
        return

    table = Table(title=f"Models matching '{query}'" if query else "NeMo Gym models")
    table.add_column("Model", style="bold")
    table.add_column("Model group")
    for row in rows:
        table.add_row(row["model"], row["model_group"])
    print_rich_table(table)
