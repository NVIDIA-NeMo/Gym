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

"""Tool schemas and prompts for synthetic tasks.

The tools are the upstream GeoBenchX tools. Each task offers only the catalog
entries of its own state and one other state from the same split, so dataset
names stay inside their split.
"""

from __future__ import annotations

import copy
from typing import Any

from resources_servers.geobenchx.catalog import TOOL_DEFINITIONS


UNUSED_TOOLS = {"make_bivariate_map"}

SYSTEM_PROMPT = """You are a geographer who answers questions with the supplied GIS tools and data catalog.
Do not provide Python code. Compute results with the tools; do not estimate.

Rules:
- If the task is not geospatial and needs no supplied data, answer directly without tools.
- If a task cannot be solved with the available datasets and tools, call reject_task.
- Reject when the datasets cover only part of the requested geography or period.
- Do not ask for clarification.
- You may make at most 25 tool calls.
- End your final message with a line of the form "Answer: <number>", giving the number as the tools report it,
  without units or thousands separators. Do not give an answer line after calling reject_task.
"""


def response_tools(vector_names: list[str], raster_names: list[str], table_names: list[str]) -> list[dict[str, Any]]:
    tools = []
    for name, (description, properties, required) in TOOL_DEFINITIONS.items():
        if name in UNUSED_TOOLS:
            continue
        properties = copy.deepcopy(properties)
        if name == "load_data":
            properties["dataset"]["enum"] = table_names
        elif name == "load_geodata":
            properties["geodataset"]["enum"] = vector_names
        elif name == "get_raster_path":
            properties["rasterdataset"]["enum"] = raster_names
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        )
    return tools


def partner_state(state: str, split_states: list[str]) -> str | None:
    others = [s for s in split_states if s != state]
    return others[split_states.index(state) % len(others)] if others else None


def offered_entries(entries: list[dict[str, Any]], state: str, split_states: list[str]) -> list[dict[str, Any]]:
    partner = partner_state(state, split_states)
    return [e for e in entries if e["state"] in (state, partner)]


def dataset_row(task: dict[str, Any], offered: list[dict[str, Any]]) -> dict[str, Any]:
    vectors = sorted(e["name"] for e in offered if e["kind"] not in ("cdl", "table"))
    rasters = sorted(e["name"] for e in offered if e["kind"] == "cdl")
    tables = sorted(e["name"] for e in offered if e["kind"] == "table")
    return {
        "task_id": task["task_id"],
        "activity": "geobenchx_synth",
        "template": task["template"],
        "task_kind": task["task_kind"],
        "state": task["state"],
        "county": task["county"],
        "split": task["split"],
        "answers": task["answers"],
        "rel_tol": task["rel_tol"],
        "abs_tol": task["abs_tol"],
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]},
            ],
            "tools": response_tools(vectors, rasters, tables),
            "parallel_tool_calls": False,
            "temperature": 0,
        },
    }
