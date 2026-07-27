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

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.conversational_tool_use_agent.prompt import (
    AGENT_SYSTEM_MESSAGE_TEMPLATE,
)
from responses_api_agents.conversational_tool_use_scenario_generation.materialize import (
    main,
    materialize_rollouts,
)


def scenario(reason: str) -> dict:
    return {
        "customer_persona": "A customer",
        "reason_for_contact": reason,
        "customer_details": "Order O-123",
        "unknown_info": None,
        "task_instructions": "Ask for help.",
        "representative_domain": "order support",
        "outside_policy_scope": False,
    }


def test_materialize_preserves_rollout_and_scenario_order(tmp_path: Path) -> None:
    input_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "agent-inputs.jsonl"
    rollouts = [
        {
            "id": "rollout-a",
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "profile": "general",
            "domain_name": "order support",
            "policy": "Authenticate first.",
            "tools": [
                {
                    "name": "lookup_order",
                    "doc": "Look up an order.",
                    "params": {"type": "object", "properties": {}},
                    "returns": {"type": "object", "properties": {}},
                    "ignored": "not part of the simulator tool contract",
                }
            ],
            "result": {"scenarios": [scenario("first"), scenario("second")]},
            "source_artifacts": {"policy_tool_generation": {"attempt_count": 3}},
        },
        {
            "id": "rollout-b",
            "profile": "proactive",
            "domain_name": "billing support",
            "policy": "Explain charges.",
            "tools": [],
            "result": {"scenarios": [scenario("third")]},
        },
    ]
    input_path.write_text(
        "".join(json.dumps(rollout) + "\n" for rollout in rollouts),
        encoding="utf-8",
    )

    assert materialize_rollouts(input_path, output_path) == 3
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [row["customer_scenario"]["reason_for_contact"] for row in rows] == ["first", "second", "third"]
    assert set(rows[0]) == {
        "id",
        "profile",
        "domain_name",
        "policy",
        "tools",
        "customer_scenario",
        "source_artifacts",
        "responses_create_params",
        "agent_ref",
    }
    assert rows[0]["tools"] == [
        {
            "name": "lookup_order",
            "doc": "Look up an order.",
            "params": {"type": "object", "properties": {}},
            "returns": {"type": "object", "properties": {}},
        }
    ]
    assert list(rows[0]["customer_scenario"]) == [
        "customer_persona",
        "reason_for_contact",
        "customer_details",
        "unknown_info",
        "task_instructions",
        "representative_domain",
        "outside_policy_scope",
    ]
    assert rows[0]["customer_scenario"]["unknown_info"] is None
    assert [row["id"] for row in rows] == [
        "rollout-a_ng_t4_r2_a1_scenario_000000",
        "rollout-a_ng_t4_r2_a1_scenario_000001",
        "rollout-b_scenario_000000",
    ]
    assert rows[0]["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "conversational_tool_use_agent",
    }
    assert rows[0]["profile"] == "general"
    assert rows[0]["domain_name"] == "order support"
    assert rows[0]["source_artifacts"] == {
        "policy_tool_generation": {"attempt_count": 3},
        "scenario_generation": {
            "id": "rollout-a",
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "scenario_index": 0,
        },
    }
    assert rows[0]["responses_create_params"]["input"] == [
        {
            "role": "system",
            "content": AGENT_SYSTEM_MESSAGE_TEMPLATE.format(domain_policy="Authenticate first."),
        }
    ]
    assert set(rows[0]["responses_create_params"]) == {
        "input",
        "parallel_tool_calls",
        "tools",
    }
    assert rows[0]["responses_create_params"]["parallel_tool_calls"] is False
    assert rows[0]["responses_create_params"]["tools"] == [
        {
            "type": "function",
            "name": "lookup_order",
            "description": "Look up an order.",
            "parameters": {"type": "object", "properties": {}},
            "strict": True,
        }
    ]
    NeMoGymResponseCreateParamsNonStreaming.model_validate(rows[0]["responses_create_params"])
    assert "initial_user_message" not in rows[0]


def test_materialize_drops_omitted_unknown_info_and_retains_explicit_null(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "agent-inputs.jsonl"
    omitted = scenario("omitted")
    omitted.pop("unknown_info")
    input_path.write_text(
        json.dumps(
            {
                "id": "rollout",
                "profile": "general",
                "domain_name": "order support",
                "policy": "Authenticate first.",
                "tools": [],
                "result": {
                    "scenarios": [
                        omitted,
                        scenario("explicit null"),
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert materialize_rollouts(input_path, output_path) == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "rollout_scenario_000001"
    assert rows[0]["customer_scenario"]["reason_for_contact"] == "explicit null"
    assert rows[0]["customer_scenario"]["unknown_info"] is None


def test_materializer_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must differ"):
        materialize_rollouts(path, path)

    assert path.read_text(encoding="utf-8") == "{}\n"


def test_materializer_rejects_duplicate_ids_without_replacing_output(tmp_path: Path) -> None:
    input_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "agent-inputs.jsonl"
    rollout = {
        "id": "duplicate",
        "profile": "general",
        "domain_name": "order support",
        "policy": "Authenticate first.",
        "tools": [],
        "result": {"scenarios": [scenario("help")]},
    }
    input_path.write_text(
        json.dumps(rollout) + "\n" + json.dumps(rollout) + "\n",
        encoding="utf-8",
    )
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate materialized row id"):
        materialize_rollouts(input_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "existing\n"


def test_materialize_cli_uses_explicit_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": "rollout",
                "profile": "general",
                "domain_name": "order support",
                "policy": "Authenticate first.",
                "tools": [],
                "result": {"scenarios": [scenario("help")]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "input": str(input_path),
        "output": str(output_path),
        "rows_written": 1,
    }
    assert output_path.is_file()
