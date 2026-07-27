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

from pathlib import Path

from responses_api_agents.conversational_tool_use_agent.app import ConversationalToolUseAgentRunRequest
from responses_api_agents.conversational_tool_use_domain_generation.materialize import materialize_policy_tool_rows
from responses_api_agents.conversational_tool_use_policy_tool_generation.materialize import (
    scenario_input_from_rollout,
)
from responses_api_agents.conversational_tool_use_scenario_generation.materialize import _materialized_rows


def test_generation_materializers_preserve_profile_lineage_and_runtime_contract(tmp_path: Path) -> None:
    domain_rollout = {
        "id": "domain-rollout",
        "_ng_task_index": 1,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "result": {
            "candidates": [
                {
                    "name": "Order Support",
                    "applications": [{"function": "Look up an order"}],
                }
            ]
        },
    }
    [policy_input] = materialize_policy_tool_rows(
        [(1, domain_rollout)],
        source=tmp_path / "domain-rollouts.jsonl",
        profile="general",
    )

    tool = {
        "name": "lookup_order",
        "doc": "Look up an order.",
        "params": {"type": "object", "properties": {}},
        "returns": {"type": "object", "properties": {}},
    }
    policy_rollout = policy_input | {
        "_ng_task_index": 3,
        "_ng_rollout_index": 4,
        "_ng_attempt_index": 2,
        "reward": 1.0,
        "result": {
            "accepted": True,
            "profile": "general",
            "domain": policy_input["domain"],
            "attempt_count": 5,
            "policy_md": "Authenticate before looking up an order.",
            "tools": [tool],
            "tools_jsonl": "",
        },
    }
    scenario_input = scenario_input_from_rollout(policy_rollout)
    assert scenario_input is not None

    scenario_rollout = scenario_input.model_dump(mode="json") | {
        "_ng_task_index": 6,
        "_ng_rollout_index": 7,
        "_ng_attempt_index": 3,
        "result": {
            "scenarios": [
                {
                    "customer_persona": "A customer",
                    "reason_for_contact": "Find order O-1.",
                    "customer_details": "Order O-1",
                    "unknown_info": None,
                    "task_instructions": "Ask for order status.",
                    "representative_domain": "order support",
                    "outside_policy_scope": False,
                }
            ]
        },
    }
    [conversation_input] = list(_materialized_rows(scenario_rollout, rollout_line=1))
    parsed = ConversationalToolUseAgentRunRequest.model_validate(conversation_input)

    assert parsed.policy == "Authenticate before looking up an order."
    assert parsed.profile == "general"
    assert conversation_input["source_artifacts"] == {
        "domain_generation": {
            "id": "domain-rollout",
            "_ng_task_index": 1,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "candidate_index": 0,
        },
        "policy_tool_generation": {
            "id": "domain-rollout_ng_t1_r2_a1_candidate_000000",
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
            "_ng_attempt_index": 2,
            "attempt_count": 5,
        },
        "scenario_generation": {
            "id": "domain-rollout_ng_t1_r2_a1_candidate_000000_ng_t3_r4_a2",
            "_ng_task_index": 6,
            "_ng_rollout_index": 7,
            "_ng_attempt_index": 3,
            "scenario_index": 0,
        },
    }
