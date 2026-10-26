# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conversion of upstream task rows into benchmark rows, which the resources server's task model must accept."""

import json

import datasets
import pytest

from benchmarks.enterpriseops_gym import prepare as prepare_module
from benchmarks.enterpriseops_gym.prepare import context_headers, convert_task
from resources_servers.enterpriseops_gym.task_data import TaskData


def _tool(name: str) -> dict:
    return {"name": name, "description": f"{name} tool", "inputSchema": {"type": "object", "properties": {}}}


TOOLS_BY_GYM = {
    "gym-calendar": [_tool("list_users"), _tool("create_event"), _tool("delete_event")],
    "sn-csm-server": [_tool("list_users"), _tool("create_case")],
}


def _row(**overrides) -> dict:
    row = {
        "task_id": "task_1",
        "domain": "hybrid",
        "system_prompt": "You are an assistant.",
        "user_prompt": "Book the review.",
        "selected_tools": ["create_case", "list_users", "create_event", "get_missing"],
        "restricted_tools": [],
        "gym_servers_config": json.dumps(
            [
                {
                    "mcp_server_name": "gym-calendar",
                    "mcp_server_url": "http://localhost:8003",
                    "seed_database_file": "dbs/calendar.sql",
                    "context": {"x-access-token": "tok"},
                    "user_info": "",
                },
                {
                    "mcp_server_name": "sn-csm-server",
                    "mcp_server_url": "http://localhost:8001",
                    "seed_database_file": "dbs/csm.sql",
                    "context": {"x-user-email": "a@example.com"},
                },
            ]
        ),
        "verifiers": json.dumps(
            [
                {
                    "verifier_type": "database_state",
                    "name": "case created",
                    "description": "",
                    "gym_name": "sn-csm-server",
                    "validation_config": {"query": "SELECT 1", "expected_value": 1, "comparison_type": "equals"},
                }
            ]
        ),
    }
    return row | overrides


def test_context_keys_become_headers_as_the_upstream_harness_sends_them():
    headers = context_headers(
        {"X-Auth-Token": "a", "x_auth_token": "a", "user_id": "u", "primary_email": "e", " x-user-email": "m"}
    )

    # Upstream only prefixes keys that lack an x- prefix, so x_auth_token is sent as x-x-auth-token.
    assert headers == {
        "x-auth-token": "a",
        "x-x-auth-token": "a",
        "x-user-id": "u",
        "x-primary-email": "e",
        "x-user-email": "m",
    }


def test_context_keys_for_one_header_must_agree():
    with pytest.raises(ValueError, match="disagree"):
        context_headers({"user_id": "1", "user-id": "2"})


def test_a_row_offers_the_selected_tools_in_gym_order_each_from_the_first_gym_that_serves_it():
    record, missing = convert_task(_row(), TOOLS_BY_GYM)

    assert [tool["name"] for tool in record["responses_create_params"]["tools"]] == [
        "list_users",
        "create_event",
        "create_case",
    ]
    assert record["tool_servers"] == {
        "list_users": "gym-calendar",
        "create_event": "gym-calendar",
        "create_case": "sn-csm-server",
    }
    assert missing == ["get_missing"]
    assert record["responses_create_params"]["tools"][0] == {
        "type": "function",
        "name": "list_users",
        "description": "list_users tool",
        "parameters": {"type": "object", "properties": {}},
        "strict": False,
    }
    assert record["responses_create_params"]["input"] == [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Book the review."},
    ]
    assert record["gym_servers"] == [
        {"name": "gym-calendar", "seed_database_file": "dbs/calendar.sql", "headers": {"x-access-token": "tok"}},
        {"name": "sn-csm-server", "seed_database_file": "dbs/csm.sql", "headers": {"x-user-email": "a@example.com"}},
    ]
    # The resources server accepts what preparation writes.
    task_fields = {key: value for key, value in record.items() if key != "responses_create_params"}
    assert TaskData.model_validate(task_fields).verifiers[0].gym_name == "sn-csm-server"


def test_restricted_tools_are_left_out():
    record, _ = convert_task(_row(restricted_tools=["create_event"]), TOOLS_BY_GYM)

    assert "create_event" not in record["tool_servers"]


def test_prepare_writes_one_row_per_task_of_every_domain(tmp_path, monkeypatch):
    output = tmp_path / "data" / "benchmark.jsonl"
    splits = {domain: [_row(task_id=f"{domain}_task")] for domain in prepare_module.DOMAINS}
    requested = {}

    def load_dataset(name, config, *, revision, token):
        requested.update(name=name, config=config, revision=revision)
        return splits

    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output)
    monkeypatch.setattr(prepare_module, "EXPECTED_TASK_COUNT", len(prepare_module.DOMAINS))
    monkeypatch.setattr(prepare_module, "load_tool_snapshots", lambda: TOOLS_BY_GYM)
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    monkeypatch.setattr("nemo_gym.global_config.get_hf_token", lambda: None)

    assert prepare_module.prepare() == output

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == [f"{domain}_task" for domain in prepare_module.DOMAINS]
    assert requested == {
        "name": prepare_module.DATASET,
        "config": "oracle",
        "revision": prepare_module.DATASET_REVISION,
    }


def test_prepare_refuses_a_task_set_of_another_size(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "benchmark.jsonl")
    monkeypatch.setattr(prepare_module, "load_tool_snapshots", lambda: TOOLS_BY_GYM)
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: {d: [] for d in prepare_module.DOMAINS})
    monkeypatch.setattr("nemo_gym.global_config.get_hf_token", lambda: None)

    with pytest.raises(ValueError, match="Expected 649 tasks, got 0"):
        prepare_module.prepare()
    assert not (tmp_path / "benchmark.jsonl").exists()
