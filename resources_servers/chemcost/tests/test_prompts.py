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
"""The transcribed prompts in prompts.py against the pinned upstream package, on every released record."""

import json

from chemcost.baselines.react_agent import SYSTEM_PROMPT, build_task_prompt
from chemcost.evaluation.evaluator import prepare_agent_input
from chemcost.tools.agent_tools import TOOL_REGISTRY

from resources_servers.chemcost import prompts
from resources_servers.chemcost.pricing_data import RECORDS_FILE, ensure_release_file


def _records() -> list[dict]:
    with ensure_release_file(RECORDS_FILE).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_system_prompt_is_upstream() -> None:
    assert prompts.SYSTEM_PROMPT == SYSTEM_PROMPT


def test_tools_are_upstream_responses_api_definitions() -> None:
    # As ReActAgent._run_openai_responses builds them from TOOL_REGISTRY.
    expected = [
        {
            "type": "function",
            "name": name,
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "description": d} for p, d in info["parameters"].items()},
                "required": list(info["parameters"].keys()),
                "additionalProperties": False,
            },
            "strict": True,
        }
        for name, info in TOOL_REGISTRY.items()
    ]
    assert prompts.TOOLS == expected


def test_every_row_shows_the_upstream_view_and_prompt() -> None:
    records = _records()
    assert len(records) == 1427
    for record in records:
        view = prepare_agent_input(record)
        assert prompts.agent_view(record) == view, record["reaction_id"]
        row = prompts.build_row(record)
        system, user = row["responses_create_params"]["input"]
        assert (system["role"], system["content"]) == ("system", SYSTEM_PROMPT)
        assert (user["role"], user["content"]) == ("user", build_task_prompt(view)), record["reaction_id"]


def test_rows_hide_structures_and_labels_from_the_model() -> None:
    record = _records()[0]
    row = prompts.build_row(record)
    shown = json.dumps(row["responses_create_params"])
    for component in record["components"]:
        if len(component.get("smiles") or "") >= 8:
            assert component["smiles"] not in shown
    assert str(record["procurement_cost_usd_per_g_product"]) not in shown
    assert (
        row["verifier_metadata"]["procurement_cost_usd_per_g_product"] == record["procurement_cost_usd_per_g_product"]
    )
