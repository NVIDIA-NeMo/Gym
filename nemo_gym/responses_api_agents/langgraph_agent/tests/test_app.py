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
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from nemo_gym.responses_api_agents.langgraph_agent.reflection_agent import (
    ReflectionAgent,
    ReflectionAgentConfig,
)


MOCK_RESPONSE = {
    "id": "resp_test123",
    "created_at": 1770000000.0,
    "model": "test-model",
    "object": "response",
    "output": [
        {
            "id": "msg_test123",
            "content": [
                {
                    "annotations": [],
                    "text": "The answer is <answer>42</answer>.",
                    "type": "output_text",
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


def _make_config():
    return ReflectionAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        max_reflections=2,
    )


def _mock_model_response():
    mock = AsyncMock()
    mock.json.return_value = MOCK_RESPONSE
    mock.read.return_value = json.dumps(MOCK_RESPONSE)
    mock.cookies = MagicMock()
    mock.cookies.items.return_value = []
    mock.ok = True
    return mock


class TestReflectionAgent:
    def test_sanity(self) -> None:
        ReflectionAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

    def test_graph_builds(self) -> None:
        agent = ReflectionAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        assert agent.graph is not None

    async def test_responses_stops_on_answer_tag(self) -> None:
        agent = ReflectionAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        app = agent.setup_webserver()
        client = TestClient(app)

        agent.server_client.post.return_value = _mock_model_response()

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "What is 6 * 7?"}]})
        assert res.status_code == 200

        output = res.json()["output"]
        assert len(output) > 0
        # Should stop after first generate since response contains <answer>
        assert agent.server_client.post.call_count == 1
