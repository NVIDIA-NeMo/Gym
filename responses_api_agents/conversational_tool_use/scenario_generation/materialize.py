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

"""Materialize scenario-generation Gym rollouts for conversational_tool_use_agent."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.conversational_tool_use.simulation.prompt import (
    agent_system_message,
    responses_api_tools,
)


AGENT_NAME = "conversational_tool_use_agent"
SCENARIO_FIELDS = (
    "customer_persona",
    "reason_for_contact",
    "customer_details",
    "unknown_info",
    "task_instructions",
    "representative_domain",
    "outside_policy_scope",
)
GYM_IDENTITY_KEYS = (
    "id",
    TASK_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
)


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if line.strip():
                yield line_number, json.loads(line)


def _simulator_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "doc": tool["doc"],
            "params": tool["params"],
            "returns": tool["returns"],
        }
        for tool in tools
    ]


def _customer_scenario(
    raw_scenario: dict[str, Any],
    *,
    rollout_line: int,
    scenario_index: int,
) -> dict[str, Any] | None:
    if "unknown_info" not in raw_scenario:
        return None

    required_fields = tuple(field for field in SCENARIO_FIELDS if field != "unknown_info")
    missing = [field for field in required_fields if field not in raw_scenario]
    if missing:
        raise ValueError(
            f"rollout line {rollout_line} scenario {scenario_index} is missing fields: {', '.join(missing)}"
        )
    return {
        "customer_persona": raw_scenario["customer_persona"],
        "reason_for_contact": raw_scenario["reason_for_contact"],
        "customer_details": raw_scenario["customer_details"],
        "unknown_info": raw_scenario["unknown_info"],
        "task_instructions": raw_scenario["task_instructions"],
        "representative_domain": raw_scenario["representative_domain"],
        "outside_policy_scope": raw_scenario["outside_policy_scope"],
    }


def _derived_rollout_id(
    rollout: dict[str, Any],
    *,
    fallback: str,
) -> str:
    base = str(rollout.get("id") or fallback)
    identity_parts = []
    if rollout.get(TASK_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"t{rollout[TASK_INDEX_KEY_NAME]}")
    if rollout.get(ROLLOUT_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"r{rollout[ROLLOUT_INDEX_KEY_NAME]}")
    if rollout.get(ATTEMPT_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"a{rollout[ATTEMPT_INDEX_KEY_NAME]}")
    if identity_parts:
        return f"{base}_ng_{'_'.join(identity_parts)}"
    return base


def _materialized_rows(
    rollout: dict[str, Any],
    *,
    rollout_line: int,
) -> Iterable[dict[str, Any]]:
    result = rollout.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"rollout line {rollout_line} has no typed result")
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError(f"rollout line {rollout_line} result has no scenarios list")

    policy = rollout["policy"]
    domain_name = rollout["domain_name"]
    profile = rollout["profile"]
    tools = _simulator_tools(rollout.get("tools", []))
    rollout_id = _derived_rollout_id(
        rollout,
        fallback=f"scenario_generation_rollout_{rollout_line:06d}",
    )
    existing_artifacts = rollout.get("source_artifacts", {})
    if not isinstance(existing_artifacts, dict):
        raise ValueError(f"rollout line {rollout_line} source_artifacts must be a JSON object")
    responses_create_params = {
        "input": [
            {
                "role": "system",
                "content": agent_system_message(policy),
            }
        ],
        "parallel_tool_calls": False,
        "tools": responses_api_tools(tools),
    }
    NeMoGymResponseCreateParamsNonStreaming.model_validate(responses_create_params)

    for scenario_index, scenario in enumerate(scenarios):
        materialized_scenario = _customer_scenario(
            dict(scenario),
            rollout_line=rollout_line,
            scenario_index=scenario_index,
        )
        if materialized_scenario is None:
            continue
        scenario = materialized_scenario
        stage_identity = {key: rollout[key] for key in GYM_IDENTITY_KEYS if rollout.get(key) is not None}
        stage_identity["scenario_index"] = scenario_index
        source_artifacts = copy.deepcopy(existing_artifacts)
        source_artifacts["scenario_generation"] = stage_identity
        row = {
            "id": f"{rollout_id}_scenario_{scenario_index:06d}",
            "domain_name": domain_name,
            "profile": profile,
            "policy": policy,
            "tools": tools,
            "customer_scenario": scenario,
            "source_artifacts": source_artifacts,
            "responses_create_params": responses_create_params,
            "agent_ref": {
                "type": "responses_api_agents",
                "name": AGENT_NAME,
            },
        }
        yield row


def materialize_rollouts(input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    seen_ids: set[str] = set()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            for rollout_line, rollout in _read_jsonl(input_path):
                for row in _materialized_rows(
                    rollout,
                    rollout_line=rollout_line,
                ):
                    row_id = row["id"]
                    if row_id in seen_ids:
                        raise ValueError(f"duplicate materialized row id on rollout line {rollout_line}: {row_id}")
                    seen_ids.add(row_id)
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Convert scenario-generation Gym rollout JSONL into conversational_tool_use_agent input JSONL.")
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows_written = materialize_rollouts(args.input, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows_written": rows_written,
            }
        )
    )


if __name__ == "__main__":
    main()
