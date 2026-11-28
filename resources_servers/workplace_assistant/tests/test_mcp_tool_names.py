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
"""/verify scores the tool-call names that harnesses report for this server's MCP tools."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import DictConfig

from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import maybe_auto_expose
from resources_servers.workplace_assistant.app import WorkbenchResourcesServer, WorkbenchResourcesServerConfig


ROW = json.loads((Path(__file__).resolve().parents[1] / "data" / "example.jsonl").read_text().splitlines()[0])
(CALL,) = ROW["ground_truth"]


def _server(name: str = "workplace_assistant") -> WorkbenchResourcesServer:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({})
    config = WorkbenchResourcesServerConfig(
        host="127.0.0.1", port=0, entrypoint="", name=name, expose_tools_over_mcp=True
    )
    return WorkbenchResourcesServer(config=config, server_client=server_client)


def _verify_body(tool_name: str) -> dict[str, Any]:
    return {
        "responses_create_params": ROW["responses_create_params"],
        **{key: ROW[key] for key in ("ground_truth", "id", "category", "environment_name")},
        "response": {
            "id": "resp",
            "created_at": 0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "function_call",
                    "name": tool_name,
                    "arguments": CALL["arguments"],
                    "call_id": "call-1",
                    "id": "fc-1",
                    "status": "completed",
                }
            ],
        },
    }


@pytest.mark.parametrize(
    ("tool_name", "reward"),
    [
        (CALL["name"], 1.0),
        (f"mcp__workplace_assistant__{CALL['name']}", 1.0),
        (f"workplace_assistant_{CALL['name']}", 1.0),
        (f"workplace_assistant__{CALL['name']}", 1.0),
        (f"other_server_{CALL['name']}", 0.0),
    ],
    ids=["bare", "claude_code", "opencode", "openclaw", "other_server"],
)
def test_verify_scores_the_harness_tool_name(tool_name: str, reward: float) -> None:
    server = _server()
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        seeded = client.post("/seed_session", json={"responses_create_params": ROW["responses_create_params"]})
        assert seeded.status_code == 200, seeded.text
        verified = client.post("/verify", json=_verify_body(tool_name))

    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == reward
    assert verified.json()["response"]["output"][0]["name"] == tool_name


def test_normalize_tool_name_strips_only_this_server_prefix_before_its_own_tools() -> None:
    named = _server()
    assert named.normalize_tool_name("workplace_assistant_not_a_tool") == "workplace_assistant_not_a_tool"
    assert named.normalize_tool_name("other_server__email_reply_email") == "other_server__email_reply_email"
    assert named.normalize_tool_name("workplace_assistant__email_reply_email") == "email_reply_email"

    unnamed = _server("")
    assert unnamed.normalize_tool_name("WorkbenchResourcesServer_email_reply_email") == "email_reply_email"
    assert (
        unnamed.normalize_tool_name("workplace_assistant_email_reply_email") == "workplace_assistant_email_reply_email"
    )
