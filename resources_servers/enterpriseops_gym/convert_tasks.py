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
"""Convert EnterpriseOps-Gym task JSONs into NeMo Gym dataset JSONL rows.

Sources: a local folder of EOG task configs (e.g. data/revised/csm in the EOG checkout) or
the ServiceNow-AI/EnterpriseOps-Gym HuggingFace dataset (config=mode, split=domain).

Tool schemas are baked in from per-domain tools/list snapshots (see snapshot_tools.py),
filtered by each task's selected_tools (empty = all tools, as upstream) minus
restricted_tools, preserving snapshot (server) order and cleaning schemas with the EOG
schema cleaner — matching the tool set the upstream harness would advertise to the model.

Output row shape:
    {
      "responses_create_params": {"input": [system, user], "tools": [...]},
      "verifier_metadata": {task_id, domain, mode, gym_servers_config, verifiers,
                            selected_tools, restricted_tools, tool_to_gym}
    }
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from resources_servers.enterpriseops_gym.schema_utils import clean_json_schema


logger = logging.getLogger(__name__)

# Fields the HF dataset stores as JSON strings (mirrors evaluate.py).
HF_JSON_STRING_FIELDS = {"gym_servers_config", "verifiers"}


def strip_private_keys(task: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror EOG's load_config: drop top-level and per-verifier keys starting with '_'."""
    task = {k: v for k, v in task.items() if not k.startswith("_")}
    if task.get("verifiers"):
        task["verifiers"] = [
            {k: v for k, v in verifier.items() if not k.startswith("_")} for verifier in task["verifiers"]
        ]
    return task


def load_snapshots(snapshot_paths: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Load tool snapshots. Returns {gym_name: [raw tool dicts]} — merging happens PER TASK
    (see build_tools_for_task) because EOG discovers tools from each task's own gyms in the
    task's gym_servers_config order, and duplicate tool names resolve first-wins in that order.
    """
    gym_tools: Dict[str, List[Dict[str, Any]]] = {}
    for snapshot_path in snapshot_paths:
        with open(snapshot_path) as f:
            snapshot = json.load(f)
        gym_tools[snapshot["gym_name"]] = snapshot["tools"]
    return gym_tools


def build_tools_for_task(
    task: Dict[str, Any], gym_tools: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Merge + filter + convert snapshot tools for one task, mirroring the upstream executor:
    tools are discovered from the TASK's gyms in gym_servers_config order (duplicate names
    keep the first occurrence, as executor._discover_and_merge_tools warns and does), then
    the selected_tools filter applies (empty = all tools, discovery order), then
    restricted_tools exclusion. Schemas go through the EOG cleaner. Output is Responses API
    tool format.
    """
    selected = task.get("selected_tools") or []
    restricted = set(task.get("restricted_tools") or [])

    # Per-task merge over the task's own gyms, in the task's order (EOG parity).
    merged_tools: List[Dict[str, Any]] = []
    tool_to_gym: Dict[str, str] = {}
    for server_config in task.get("gym_servers_config") or []:
        gym_name = server_config.get("mcp_server_name")
        if gym_name not in gym_tools:
            logger.warning(f"No tools snapshot provided for gym '{gym_name}'; its tools will be missing.")
            continue
        for tool in gym_tools[gym_name]:
            tool_name = tool.get("name", "unknown")
            if tool_name in tool_to_gym:
                logger.debug(
                    f"Tool '{tool_name}' duplicated across gyms "
                    f"('{tool_to_gym[tool_name]}' and '{gym_name}'); keeping first occurrence."
                )
                continue
            merged_tools.append(tool)
            tool_to_gym[tool_name] = gym_name

    filtered = [t for t in merged_tools if not selected or t.get("name") in selected]
    if selected:
        missing = set(selected) - {t.get("name") for t in filtered}
        if missing:
            logger.warning(f"Task selected_tools not found in snapshots: {sorted(missing)}")
    filtered = [t for t in filtered if t.get("name") not in restricted]

    tools = [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": clean_json_schema(
                tool.get("inputSchema", {"type": "object", "properties": {}, "required": []})
            ),
            # EOG's LangChain path binds tools non-strict; strict mode would also reject
            # the EOG-cleaned schemas (optional params, no additionalProperties: false).
            "strict": False,
        }
        for tool in filtered
    ]
    task_tool_to_gym = {t["name"]: tool_to_gym[t["name"]] for t in tools}
    return tools, task_tool_to_gym


def convert_task(
    task: Dict[str, Any],
    task_id: str,
    domain: str,
    mode: str,
    gym_tools: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    task = strip_private_keys(task)

    for required_field in ("system_prompt", "user_prompt"):
        if required_field not in task:
            raise ValueError(f"Task {task_id} missing required field: {required_field}")
    if not task.get("gym_servers_config"):
        raise ValueError(f"Task {task_id} missing gym_servers_config (legacy single-gym tasks are not supported)")

    tools, task_tool_to_gym = build_tools_for_task(task, gym_tools)

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_prompt"]},
            ],
            "tools": tools,
        },
        "verifier_metadata": {
            "task_id": task_id,
            "domain": domain,
            "mode": mode,
            "gym_servers_config": task["gym_servers_config"],
            "verifiers": task.get("verifiers") or [],
            "selected_tools": task.get("selected_tools") or [],
            "restricted_tools": task.get("restricted_tools") or [],
            "tool_to_gym": task_tool_to_gym,
        },
    }


def load_tasks_from_dir(tasks_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    tasks = []
    for task_path in sorted(tasks_dir.glob("*.json")):
        with open(task_path) as f:
            tasks.append((task_path.stem, json.load(f)))
    return tasks


def load_tasks_from_hf(hf_dataset: str, mode: str, domain: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Load tasks from the EOG HuggingFace dataset (config=mode, split=domain), as evaluate.py does."""
    from datasets import load_dataset as hf_load_dataset

    tasks = []
    hf_ds = hf_load_dataset(hf_dataset, mode, split=domain)
    for i, row in enumerate(hf_ds):
        task_id = row.get("task_id", f"task_{i}")
        task = {}
        for k, v in row.items():
            if k in ("task_id", "domain"):
                continue
            if k in HF_JSON_STRING_FIELDS and isinstance(v, str):
                v = json.loads(v)
            task[k] = v
        tasks.append((task_id, task))
    return tasks


def convert(
    tasks: List[Tuple[str, Dict[str, Any]]],
    snapshot_paths: List[Path],
    domain: str,
    mode: str,
    output: Path,
    limit: Optional[int] = None,
) -> int:
    gym_tools = load_snapshots(snapshot_paths)

    output.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    with open(output, "w") as f:
        for task_id, task in tasks[:limit]:
            row = convert_task(task, task_id, domain, mode, gym_tools)
            f.write(json.dumps(row) + "\n")
            num_written += 1
    return num_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks-dir", type=Path, help="Folder of EOG task JSON configs")
    source.add_argument("--hf-dataset", help="HF dataset repo id (e.g. ServiceNow-AI/EnterpriseOps-Gym)")
    parser.add_argument(
        "--tools-snapshot",
        type=Path,
        action="append",
        required=True,
        help="Tools snapshot JSON from snapshot_tools.py (repeat for hybrid/multi-gym domains)",
    )
    parser.add_argument("--domain", required=True, help="EOG domain tag (e.g. csm); HF split when --hf-dataset")
    parser.add_argument(
        "--mode", default="oracle", help="EOG tool-set mode (e.g. oracle); HF config when --hf-dataset"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.tasks_dir:
        tasks = load_tasks_from_dir(args.tasks_dir)
    else:
        tasks = load_tasks_from_hf(args.hf_dataset, args.mode, args.domain)

    num_written = convert(tasks, args.tools_snapshot, args.domain, args.mode, args.output, args.limit)
    print(f"Wrote {num_written} rows to {args.output}")


if __name__ == "__main__":
    main()
