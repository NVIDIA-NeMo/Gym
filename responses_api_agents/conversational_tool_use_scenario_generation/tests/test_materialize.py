# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.conversational_tool_use_agent.app import (
    ConversationalToolUseAgent,
)
from responses_api_agents.conversational_tool_use_scenario_generation.materialize import (
    AGENT_SYSTEM_MESSAGE_TEMPLATE,
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
        },
        {
            "id": "rollout-b",
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
        "policy",
        "tools",
        "customer_scenario",
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
        "rollout-a_scenario_000000",
        "rollout-a_scenario_000001",
        "rollout-b_scenario_000000",
    ]
    assert rows[0]["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "conversational_tool_use_agent",
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
    assert AGENT_SYSTEM_MESSAGE_TEMPLATE == ConversationalToolUseAgent.AGENT_SYSTEM_MESSAGE_TEMPLATE
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
