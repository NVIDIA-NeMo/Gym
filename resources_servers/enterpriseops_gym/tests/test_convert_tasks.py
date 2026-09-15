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
import json

import pytest

from resources_servers.enterpriseops_gym.convert_tasks import (
    build_tools_for_task,
    convert,
    convert_task,
    load_snapshots,
    strip_private_keys,
)


SNAPSHOT = {
    "gym_name": "sn-csm-server",
    "gym_url": "http://localhost:8001",
    "snapshot_at": "2026-07-07T00:00:00+00:00",
    "tools": [
        {
            "name": "find_user",
            "description": "Find a user",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": ""}},
                "required": ["name"],
            },
        },
        {
            "name": "update_entitlement",
            "description": "Update an entitlement",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entitlement_id": {"type": "integer", "description": ""},
                    "coverage_hours": {"type": ["string", "null"], "description": "optional"},
                },
                "required": ["entitlement_id", "coverage_hours"],
            },
        },
        {
            "name": "create_new_case",
            "description": "Create a case",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        },
    ],
}

TASK = {
    "_comment": "should be stripped",
    "system_prompt": "CSM Agent Policy ...",
    "user_prompt": "Upgrade the entitlement.",
    "mcp_endpoint": "/mcp",
    "number_of_runs": 1,
    "gym_servers_config": [
        {
            "mcp_server_name": "sn-csm-server",
            "mcp_server_url": "http://localhost:8001",
            "seed_database_file": "Domain Wise DBs and Task-DB Mappings/csm/dbs/db_x.sql",
            "context": {"x-user-email": "joanne@example.com"},
        }
    ],
    "selected_tools": ["update_entitlement", "find_user"],
    "restricted_tools": ["find_user"],
    "verifiers": [
        {
            "verifier_type": "database_state",
            "name": "update_entitlement",
            "_description": "should be stripped",
            "gym_name": "sn-csm-server",
            "validation_config": {"query": "SELECT 1;", "expected_value": 1, "comparison_type": "equals"},
        }
    ],
}


@pytest.fixture
def snapshot_path(tmp_path):
    path = tmp_path / "csm.json"
    path.write_text(json.dumps(SNAPSHOT))
    return path


class TestConvertTask:
    def test_full_conversion(self, snapshot_path) -> None:
        gym_tools = load_snapshots([snapshot_path])
        row = convert_task(dict(TASK), "task_1", "csm", "oracle", gym_tools)

        # Messages
        messages = row["responses_create_params"]["input"]
        assert messages[0] == {"role": "system", "content": TASK["system_prompt"]}
        assert messages[1] == {"role": "user", "content": TASK["user_prompt"]}

        # selected minus restricted -> only update_entitlement
        tools = row["responses_create_params"]["tools"]
        assert [t["name"] for t in tools] == ["update_entitlement"]
        assert tools[0]["type"] == "function"
        assert tools[0]["strict"] is False  # EOG parity: non-strict tool binding

        # EOG schema cleaning: nullable type array collapsed and demoted from required
        parameters = tools[0]["parameters"]
        assert parameters["properties"]["coverage_hours"]["type"] == "string"
        assert parameters["required"] == ["entitlement_id"]

        # Metadata
        metadata = row["verifier_metadata"]
        assert metadata["task_id"] == "task_1"
        assert metadata["domain"] == "csm"
        assert metadata["mode"] == "oracle"
        assert metadata["tool_to_gym"] == {"update_entitlement": "sn-csm-server"}
        assert metadata["gym_servers_config"] == TASK["gym_servers_config"]
        # Private keys stripped
        assert "_description" not in metadata["verifiers"][0]

    def test_empty_selected_tools_means_all_in_snapshot_order(self, snapshot_path) -> None:
        gym_tools = load_snapshots([snapshot_path])
        task = dict(TASK) | {"selected_tools": [], "restricted_tools": []}
        tools, task_tool_to_gym = build_tools_for_task(task, gym_tools)
        assert [t["name"] for t in tools] == ["find_user", "update_entitlement", "create_new_case"]
        assert set(task_tool_to_gym.values()) == {"sn-csm-server"}

    def test_missing_required_field_raises(self, snapshot_path) -> None:
        gym_tools = load_snapshots([snapshot_path])
        with pytest.raises(ValueError, match="system_prompt"):
            convert_task({"user_prompt": "x", "gym_servers_config": [{}]}, "t", "csm", "oracle", gym_tools)

    def test_strip_private_keys(self) -> None:
        stripped = strip_private_keys(dict(TASK))
        assert "_comment" not in stripped
        assert "_description" not in stripped["verifiers"][0]

    def test_convert_writes_jsonl(self, snapshot_path, tmp_path) -> None:
        output = tmp_path / "out.jsonl"
        num_written = convert([("task_1", dict(TASK))], [snapshot_path], "csm", "oracle", output)
        assert num_written == 1
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert rows[0]["verifier_metadata"]["task_id"] == "task_1"

    def test_duplicate_tools_resolve_in_task_gym_order(self, tmp_path) -> None:
        """EOG parity: tools merge per task in the TASK's gym_servers_config order, so a
        duplicate-named tool maps to the task's FIRST gym that provides it — regardless of
        snapshot load order. (A fixed global merge would route hybrid tool calls to gyms
        that aren't even part of the task's session.)"""
        snapshot_a = tmp_path / "a.json"
        snapshot_b = tmp_path / "b.json"
        snapshot_a.write_text(json.dumps({"gym_name": "gym-a", "tools": [{"name": "shared", "inputSchema": {}}]}))
        snapshot_b.write_text(json.dumps({"gym_name": "gym-b", "tools": [{"name": "shared", "inputSchema": {}}]}))
        gym_tools = load_snapshots([snapshot_a, snapshot_b])  # load order: a, b

        task = {
            "selected_tools": ["shared"],
            "restricted_tools": [],
            # Task uses gym-b FIRST -> shared must map to gym-b, not snapshot-order gym-a.
            "gym_servers_config": [
                {"mcp_server_name": "gym-b", "mcp_server_url": "http://b"},
                {"mcp_server_name": "gym-a", "mcp_server_url": "http://a"},
            ],
        }
        tools, task_tool_to_gym = build_tools_for_task(task, gym_tools)
        assert [t["name"] for t in tools] == ["shared"]
        assert task_tool_to_gym == {"shared": "gym-b"}

    def test_task_gym_without_snapshot_warns_and_skips(self, snapshot_path) -> None:
        gym_tools = load_snapshots([snapshot_path])
        task = dict(TASK) | {
            "gym_servers_config": TASK["gym_servers_config"]
            + [{"mcp_server_name": "unknown-gym", "mcp_server_url": "http://x"}]
        }
        tools, task_tool_to_gym = build_tools_for_task(task, gym_tools)
        # Known gym's tools still resolve; unknown gym contributes nothing.
        assert task_tool_to_gym == {"update_entitlement": "sn-csm-server"}
