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
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.langgraph_agent.app import response_output_items
from responses_api_agents.langgraph_agent.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorAgentConfig,
)
from responses_api_agents.langgraph_agent.parallel_thinking_agent import (
    ParallelThinkingAgent,
    ParallelThinkingAgentConfig,
)
from responses_api_agents.langgraph_agent.reflection_agent import (
    ReflectionAgent,
    ReflectionAgentConfig,
)
from responses_api_agents.langgraph_agent.rewoo_agent import ReWOOAgent, ReWOOAgentConfig


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


def _mock_model_response(text: str = "The answer is <answer>42</answer>."):
    payload = deepcopy(MOCK_RESPONSE)
    payload["output"][0]["content"][0]["text"] = text
    mock = AsyncMock()
    mock.json.return_value = payload
    mock.read.return_value = json.dumps(payload)
    mock.cookies = MagicMock()
    mock.cookies.items.return_value = []
    mock.ok = True
    return mock


class TestReflectionAgent:
    def test_response_output_items_drop_internal_input_messages(self) -> None:
        user_prompt = NeMoGymEasyInputMessage(role="user", content="internal prompt")
        assistant_message = NeMoGymResponseOutputMessage(
            id="msg_test",
            content=[NeMoGymResponseOutputText(text="answer", annotations=[])],
        )

        assert response_output_items([user_prompt, assistant_message]) == [assistant_message]

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

    async def test_reflection_model_inputs_are_chronological_without_duplicates(self) -> None:
        agent = ReflectionAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        client = TestClient(agent.setup_webserver())
        agent.server_client.post.side_effect = [
            _mock_model_response("initial draft"),
            _mock_model_response("specific critique"),
            _mock_model_response("revised <answer>42</answer>"),
        ]

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "What is 6 * 7?"}]})
        assert res.status_code == 200

        model_inputs = [call.kwargs["json"].input for call in agent.server_client.post.call_args_list]
        assert [[message.role for message in items] for items in model_inputs] == [
            ["user"],
            ["user", "assistant", "user"],
            ["user", "assistant", "user"],
        ]
        assert [message.content for message in model_inputs[2]] == [
            "What is 6 * 7?",
            "initial draft",
            "specific critique",
        ]


@pytest.mark.parametrize(
    ("agent_class", "config_class"),
    [
        (OrchestratorAgent, OrchestratorAgentConfig),
        (ParallelThinkingAgent, ParallelThinkingAgentConfig),
        (ReWOOAgent, ReWOOAgentConfig),
    ],
)
async def test_staged_model_calls_use_only_the_current_prompt(agent_class, config_class) -> None:
    config = config_class(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
    )
    agent = agent_class(config=config, server_client=MagicMock(spec=ServerClient))
    agent.server_client.post.return_value = _mock_model_response()
    old_output = NeMoGymResponseOutputMessage(
        id="msg_old",
        content=[NeMoGymResponseOutputText(text="old stage", annotations=[])],
    )
    state = {
        "request_body": NeMoGymResponseCreateParamsNonStreaming(input=[]),
        "policy_outputs": [
            NeMoGymEasyInputMessage(role="user", content="old prompt"),
            old_output,
        ],
        "cookies": {},
    }

    await agent._call_model(state, "current stage")

    model_input = agent.server_client.post.call_args.kwargs["json"].input
    assert [(message.role, message.content) for message in model_input] == [("user", "current stage")]
