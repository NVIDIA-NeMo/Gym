# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize accepted policy/tool rollouts as scenario-generation Gym inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from responses_api_agents.conversational_tool_use_policy_tool_generation.compat import format_domain_name
from responses_api_agents.conversational_tool_use_policy_tool_generation.models import PolicyToolGenerationResult


class ScenarioGenerationAgentRef(BaseModel):
    type: str = "responses_api_agents"
    name: str = "conversational_tool_use_scenario_generation"


class ScenarioGenerationInput(BaseModel):
    agent_ref: ScenarioGenerationAgentRef = Field(default_factory=ScenarioGenerationAgentRef)
    responses_create_params: dict[str, Any]
    domain_name: str
    policy: str
    tools: list[dict[str, Any]]


def _accepted_container(row: dict[str, Any]) -> dict[str, Any] | None:
    result = row.get("result")
    if row.get("reward") == 1.0 and isinstance(result, dict) and result.get("accepted") is True:
        return row
    return None


def scenario_input_from_rollout(row: dict[str, Any]) -> ScenarioGenerationInput | None:
    accepted = _accepted_container(row)
    if accepted is None:
        return None
    result = PolicyToolGenerationResult.model_validate(accepted["result"])
    return ScenarioGenerationInput(
        responses_create_params={"input": []},
        domain_name=format_domain_name(result.domain.name),
        policy=result.policy_md,
        tools=result.tools,
    )


def materialize(input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
    with input_path.open("r", encoding="utf-8") as input_file, output_path.open("w", encoding="utf-8") as target:
        for rollout_line, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on input line {rollout_line}: {exc}") from exc
            item = scenario_input_from_rollout(row)
            if item is None:
                continue
            target.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
            accepted_count += 1
    return accepted_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    count = materialize(args.input_path, args.output_path)
    print(f"Materialized {count} accepted policy/tool rollouts to {args.output_path}")


if __name__ == "__main__":
    main()
