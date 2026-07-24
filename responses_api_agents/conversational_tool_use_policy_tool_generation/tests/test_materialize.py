# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from responses_api_agents.conversational_tool_use_policy_tool_generation.materialize import (
    main,
    materialize,
)
from responses_api_agents.conversational_tool_use_scenario_generation.app import (
    ScenarioGenerationRunRequest,
)


def accepted_rollout(
    artifact_id: str,
    *,
    domain: str,
    policy: str,
    tools_jsonl: str,
) -> dict:
    tools = [json.loads(line) for line in tools_jsonl.splitlines() if line]
    return {
        "id": artifact_id,
        "_ng_task_index": 4,
        "_ng_rollout_index": 2,
        "reward": 1.0,
        "result": {
            "accepted": True,
            "profile": "general",
            "domain": {"name": domain, "applications": [{"raw": "kept"}]},
            "attempt_count": 3,
            "policy_md": policy,
            "tools": tools,
            "tools_jsonl": tools_jsonl,
        },
        "generation_trace": {"domain_name": domain.replace(" ", "_")},
    }


def test_materializer_keeps_accepted_order_and_next_agent_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "scenario-inputs.jsonl"
    first_policy = "Policy line one.\nPolicy café."
    first_tools = '{"name":"a","doc":"A","params":null,"returns":null}\n'
    second_policy = "Second policy."
    second_tools = '{"name":"b","doc":"B","params":null,"returns":null}\n'
    rows = [
        accepted_rollout(
            "first",
            domain="Order Support",
            policy=first_policy,
            tools_jsonl=first_tools,
        ),
        {"reward": 0.0, "result": {"accepted": False}},
        accepted_rollout(
            "second",
            domain="Billing Support",
            policy=second_policy,
            tools_jsonl=second_tools,
        ),
    ]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert materialize(input_path, output_path) == 2
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["policy"] for row in written] == [first_policy, second_policy]
    assert [row["domain_name"] for row in written] == ["Order_Support", "Billing_Support"]
    assert written[0]["tools"] == [{"name": "a", "doc": "A", "params": None, "returns": None}]
    assert written[0]["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "conversational_tool_use_scenario_generation",
    }
    assert set(written[0]) == {
        "agent_ref",
        "responses_create_params",
        "domain_name",
        "policy",
        "tools",
    }
    for row in written:
        ScenarioGenerationRunRequest.model_validate(row)


def test_cli_requires_explicit_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    row = accepted_rollout(
        "artifact",
        domain="Order Support",
        policy="Policy.",
        tools_jsonl='{"name":"a","doc":"A","params":null,"returns":null}\n',
    )
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
        ],
    )

    main()

    assert "Materialized 1 accepted policy/tool rollouts" in capsys.readouterr().out
    assert output_path.is_file()
    with pytest.raises(ValueError, match="must differ"):
        materialize(input_path, input_path)
