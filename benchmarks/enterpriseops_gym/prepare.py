# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the EnterpriseOps-Gym benchmark JSONL: the public oracle-mode tasks of all eight domains.

Downloads ServiceNow's task set and the gym servers' tool schemas, both pinned, and writes one row per task. The row's
``responses_create_params`` holds the task's system and user prompts and its oracle tools; the other fields are what
the ``enterpriseops_gym`` resources server reads: the gym servers to start, each with its database dump and context
headers, the gym server of each tool, and the SQL verifiers.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "enterpriseops_gym_benchmark.jsonl"

DATASET = "ServiceNow-AI/EnterpriseOps-Gym"
DATASET_REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"  # pragma: allowlist secret
DOMAINS = ("calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams")
EXPECTED_TASK_COUNT = 649
# tools/list of each gym server image, captured from the images the resources server config pins.
TOOLS_REPO = "nvidia/NeMo-Gym-EnterpriseOps-Assets"
TOOLS_REVISION = "8918dc64b8575d5ff476e62e1cc3687523ab59c2"  # pragma: allowlist secret
TOOL_SNAPSHOTS = ("calendar", "csm", "drive", "email", "hr", "itsm", "teams")


def context_headers(context: dict[str, Any]) -> dict[str, str]:
    """The headers that carry a gym server's task context, such as the acting user's token.

    The upstream harness sends each key as a header, prefixing ``x-`` (and turning ``_`` into ``-``) when the key does
    not already start with ``x-``. The tasks were validated that way, so it is kept even where it misnames a header:
    ``x_auth_token`` becomes ``x-x-auth-token``, which the drive server ignores, so those tasks act as its default
    user. Surrounding whitespace is stripped, as upstream's revised task files do, since no header name can hold it.
    """
    headers: dict[str, str] = {}
    for key, value in context.items():
        name = key.strip().lower()
        name = name if name.startswith("x-") else f"x-{name.replace('_', '-')}"
        if headers.get(name, str(value)) != str(value):
            raise ValueError(f"Context keys for header {name!r} disagree: {context}")
        headers[name] = str(value)
    return headers


def load_tool_snapshots() -> dict[str, list[dict[str, Any]]]:
    """Each gym server's tools, in the order its tools/list returns them."""
    from huggingface_hub import hf_hub_download

    from nemo_gym.global_config import get_hf_token

    tools: dict[str, list[dict[str, Any]]] = {}
    for name in TOOL_SNAPSHOTS:
        path = hf_hub_download(
            TOOLS_REPO,
            f"enterpriseops_gym/tools/{name}.json",
            repo_type="dataset",
            revision=TOOLS_REVISION,
            token=get_hf_token(),
        )
        snapshot = json.loads(Path(path).read_text())
        tools[snapshot["gym_name"]] = snapshot["tools"]
    return tools


def convert_task(
    row: dict[str, Any], tools_by_gym: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[str]]:
    """One benchmark row, and the selected tools that no gym server of the task serves."""
    gym_configs = json.loads(row["gym_servers_config"])
    allowed = set(row["selected_tools"]) - set(row["restricted_tools"] or [])
    # Upstream offers the selected tools in the gym servers' order, each from the first gym server that serves it.
    tool_servers: dict[str, str] = {}
    tools: list[dict[str, Any]] = []
    for gym in gym_configs:
        for tool in tools_by_gym[gym["mcp_server_name"]]:
            if tool["name"] in allowed and tool["name"] not in tool_servers:
                tool_servers[tool["name"]] = gym["mcp_server_name"]
                tools.append(
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool.get("description") or "",
                        "parameters": tool["inputSchema"],
                        "strict": False,
                    }
                )
    missing = sorted(allowed - set(tool_servers))
    record = {
        "task_id": row["task_id"],
        "domain": row["domain"],
        "gym_servers": [
            {
                "name": gym["mcp_server_name"],
                "seed_database_file": gym["seed_database_file"],
                "headers": context_headers(gym.get("context") or {}),
            }
            for gym in gym_configs
        ],
        "tool_servers": tool_servers,
        "verifiers": json.loads(row["verifiers"]),
        "responses_create_params": {
            "input": [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            "tools": tools,
        },
    }
    return record, missing


def prepare() -> Path:
    from datasets import load_dataset

    from nemo_gym.global_config import get_hf_token

    tools_by_gym = load_tool_snapshots()
    splits = load_dataset(DATASET, "oracle", revision=DATASET_REVISION, token=get_hf_token())
    if set(splits) != set(DOMAINS):
        raise ValueError(f"{DATASET}@{DATASET_REVISION} oracle splits are {sorted(splits)}, expected {DOMAINS}")
    records, missing_tools = [], Counter()
    for domain in DOMAINS:
        for row in splits[domain]:
            record, missing = convert_task(row, tools_by_gym)
            records.append(record)
            missing_tools.update(missing)
    if len(records) != EXPECTED_TASK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TASK_COUNT} tasks, got {len(records)}")
    if missing_tools:
        # Upstream drops them too: the gym images serve no tool of that name.
        print(f"Selected tools no gym server serves, left out of their tasks: {dict(missing_tools)}", file=sys.stderr)

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUTPUT_FPATH.with_suffix(".jsonl.partial")
    with partial.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    partial.replace(OUTPUT_FPATH)
    print(f"Wrote {len(records)} tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
