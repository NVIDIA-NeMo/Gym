# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize scenario-generation Gym rollouts for conversational_tool_use_agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.conversational_tool_use_agent.app import (
    ConversationalToolUseAgent,
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

AGENT_SYSTEM_MESSAGE_TEMPLATE = ConversationalToolUseAgent.AGENT_SYSTEM_MESSAGE_TEMPLATE


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if line.strip():
                yield line_number, json.loads(line)


def _responses_api_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["doc"],
            "parameters": tool["params"],
            "strict": True,
        }
        for tool in tools
    ]


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
    tools = _simulator_tools(rollout.get("tools", []))
    rollout_id = str(rollout.get("id") or f"scenario_generation_rollout_{rollout_line:06d}")
    responses_create_params = {
        "input": [
            {
                "role": "system",
                "content": AGENT_SYSTEM_MESSAGE_TEMPLATE.format(domain_policy=policy),
            }
        ],
        "parallel_tool_calls": False,
        "tools": _responses_api_tools(tools),
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
        yield {
            "id": f"{rollout_id}_scenario_{scenario_index:06d}",
            "policy": policy,
            "tools": tools,
            "customer_scenario": scenario,
            "responses_create_params": responses_create_params,
            "agent_ref": {
                "type": "responses_api_agents",
                "name": AGENT_NAME,
            },
        }


def materialize_rollouts(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for rollout_line, rollout in _read_jsonl(input_path):
            for row in _materialized_rows(
                rollout,
                rollout_line=rollout_line,
            ):
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
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
