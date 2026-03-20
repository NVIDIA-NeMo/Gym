# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pytest import MonkeyPatch

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.robustness_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    RobustnessAgent,
    RobustnessAgentConfig,
)


def _make_config(rewriter_name=None):
    return RobustnessAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name="my server name"),
        rewriter_model_server=ModelServerRef(type="responses_api_models", name=rewriter_name)
        if rewriter_name
        else None,
    )


SIMPLE_RESPONSE = {
    "id": "resp_abc",
    "created_at": 1753983920.0,
    "model": "dummy_model",
    "object": "response",
    "output": [
        {
            "id": "msg_abc",
            "content": [{"annotations": [], "text": "Hello!", "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


class TestApp:
    def test_sanity(self) -> None:
        RobustnessAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

    async def test_responses_no_rewriter(self, monkeypatch: MonkeyPatch) -> None:
        """Without a rewriter configured, behaves identically to simple_agent."""
        server = RobustnessAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock = AsyncMock()
        mock.read.return_value = json.dumps(SIMPLE_RESPONSE)
        mock.cookies = MagicMock()
        server.server_client.post.return_value = mock

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res.status_code == 200

        server.server_client.post.assert_called_with(
            server_name="my server name",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
            ),
            cookies=None,
        )

    async def test_responses_with_rewriter_rewrites_tools(self, monkeypatch: MonkeyPatch) -> None:
        """With a rewriter configured, tool names and arg names are rewritten before the model call
        and translated back to originals when calling the resources server."""
        server = RobustnessAgent(
            config=_make_config(rewriter_name="rewriter"),
            server_client=MagicMock(spec=ServerClient),
        )
        app = server.setup_webserver()
        client = TestClient(app)

        rewriter_response = {
            "id": "resp_rw",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_rw",
                    "content": [
                        {
                            "annotations": [],
                            "text": json.dumps(
                                {
                                    "rewritten_messages": [{"role": "user", "content": "hi there"}],
                                    "tool_name_map": {"get_weather": "fetch_conditions"},
                                    "arg_name_maps": {"get_weather": {"city": "location"}},
                                }
                            ),
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

        # Model response: emits a tool call with rewritten name "fetch_conditions"
        tool_call_response = {
            "id": "resp_tc",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "fetch_conditions",
                    "arguments": json.dumps({"location": "Paris"}),
                    "type": "function_call",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        rewriter_mock = AsyncMock()
        rewriter_mock.read.return_value = json.dumps(rewriter_response)
        rewriter_mock.cookies = MagicMock()
        rewriter_mock.ok = True

        model_mock = AsyncMock()
        model_mock.read.side_effect = [json.dumps(tool_call_response), json.dumps(SIMPLE_RESPONSE)]
        model_mock.cookies = MagicMock()
        model_mock.ok = True

        tool_result_mock = AsyncMock()
        tool_result_mock.content.read.return_value = b"Sunny"
        tool_result_mock.cookies = MagicMock()

        def post_side_effect(server_name, url_path, **kwargs):
            if server_name == "rewriter":
                return rewriter_mock
            if server_name == "my server name":
                return model_mock
            # resources server tool call
            return tool_result_mock

        server.server_client.post.side_effect = post_side_effect

        res = client.post(
            "/v1/responses",
            json={
                "input": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        )
        assert res.status_code == 200

        # Verify resources server was called with the ORIGINAL tool name and arg name
        resources_call = [
            c for c in server.server_client.post.call_args_list if c.kwargs.get("url_path") == "/get_weather"
        ]
        assert len(resources_call) == 1
        assert resources_call[0].kwargs["json"] == {"city": "Paris"}

    async def test_rewriter_fallback_on_error(self, monkeypatch: MonkeyPatch) -> None:
        """If the rewriter call fails, the agent falls back to no rewriting."""
        server = RobustnessAgent(
            config=_make_config(rewriter_name="rewriter"),
            server_client=MagicMock(spec=ServerClient),
        )
        app = server.setup_webserver()
        client = TestClient(app)

        failing_mock = AsyncMock()
        failing_mock.ok = False
        failing_mock.status = 500
        failing_mock.cookies = MagicMock()

        model_mock = AsyncMock()
        model_mock.read.return_value = json.dumps(SIMPLE_RESPONSE)
        model_mock.cookies = MagicMock()
        model_mock.ok = True

        def post_side_effect(server_name, url_path, **kwargs):
            if server_name == "rewriter":
                return failing_mock
            return model_mock

        server.server_client.post.side_effect = post_side_effect

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res.status_code == 200
