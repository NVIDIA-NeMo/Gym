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

"""Materialize accepted policy/tool rollouts as scenario-generation Gym inputs."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from responses_api_agents.conversational_tool_use.policy_tool_generation.compat import format_domain_name
from responses_api_agents.conversational_tool_use.policy_tool_generation.models import (
    PolicyToolGenerationResult,
    PolicyToolProfile,
)


GYM_IDENTITY_KEYS = (
    "id",
    TASK_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
)


class ScenarioGenerationAgentRef(BaseModel):
    type: str = "responses_api_agents"
    name: str = "conversational_tool_use_scenario_generation"


class ScenarioGenerationInput(BaseModel):
    agent_ref: ScenarioGenerationAgentRef = Field(default_factory=ScenarioGenerationAgentRef)
    id: str | None = None
    responses_create_params: dict[str, Any]
    profile: PolicyToolProfile
    domain_name: str
    policy: str
    tools: list[dict[str, Any]]
    source_artifacts: dict[str, Any] = Field(default_factory=dict)


def _accepted_container(row: dict[str, Any]) -> dict[str, Any] | None:
    result = row.get("result")
    if row.get("reward") == 1.0 and isinstance(result, dict) and result.get("accepted") is True:
        return row
    return None


def _derived_rollout_id(row: dict[str, Any]) -> str | None:
    if row.get("id") is None:
        return None
    base = str(row["id"])
    identity_parts = []
    if row.get(TASK_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"t{row[TASK_INDEX_KEY_NAME]}")
    if row.get(ROLLOUT_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"r{row[ROLLOUT_INDEX_KEY_NAME]}")
    if row.get(ATTEMPT_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"a{row[ATTEMPT_INDEX_KEY_NAME]}")
    if identity_parts:
        return f"{base}_ng_{'_'.join(identity_parts)}"
    return base


def scenario_input_from_rollout(
    row: dict[str, Any],
    *,
    fallback_id: str | None = None,
) -> ScenarioGenerationInput | None:
    accepted = _accepted_container(row)
    if accepted is None:
        return None
    result = PolicyToolGenerationResult.model_validate(accepted["result"])
    existing_artifacts = accepted.get("source_artifacts", {})
    if not isinstance(existing_artifacts, dict):
        raise ValueError("source_artifacts must be a JSON object when present")
    stage_identity = {key: accepted[key] for key in GYM_IDENTITY_KEYS if accepted.get(key) is not None}
    stage_identity["attempt_count"] = result.attempt_count
    source_artifacts = copy.deepcopy(existing_artifacts)
    source_artifacts["policy_tool_generation"] = stage_identity
    return ScenarioGenerationInput(
        id=_derived_rollout_id(accepted) or fallback_id,
        responses_create_params={"input": []},
        profile=result.profile,
        domain_name=format_domain_name(result.domain.name),
        policy=result.policy_md,
        tools=result.tools,
        source_artifacts=source_artifacts,
    )


def materialize(input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
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
        ) as target:
            temporary_path = Path(target.name)
            with input_path.open("r", encoding="utf-8") as input_file:
                for rollout_line, line in enumerate(input_file, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSON on input line {rollout_line}: {exc}") from exc
                    item = scenario_input_from_rollout(
                        row,
                        fallback_id=f"policy_tool_rollout_{rollout_line:06d}",
                    )
                    if item is None:
                        continue
                    assert item.id is not None
                    if item.id in seen_ids:
                        raise ValueError(f"duplicate materialized row id on input line {rollout_line}: {item.id}")
                    seen_ids.add(item.id)
                    target.write(
                        json.dumps(item.model_dump(mode="json", exclude_none=True), ensure_ascii=False) + "\n"
                    )
                    accepted_count += 1
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
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
