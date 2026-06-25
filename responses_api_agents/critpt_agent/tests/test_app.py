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

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.critpt_agent.app import (
    CritPtAgent,
    CritPtAgentConfig,
    CritPtAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _extract_output_text,
    _strip_thinking_blocks,
)


TURN2_PROMPT_FPATH = "benchmarks/critpt/prompts/turn2.yaml"


_MODEL_RESPONSE = {
    "id": "resp-id",
    "created_at": 1234.5,
    "model": "test-model",
    "object": "response",
    "output": [
        {
            "id": "msg-id",
            "content": [{"annotations": [], "text": "The answer is 1.23", "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "none",
    "tools": [],
}

_VERIFY_RESPONSE = {
    "responses_create_params": {"input": []},
    "response": _MODEL_RESPONSE,
    "reward": 1.0,
    "problem_id": "5",
    "accuracy": 1.0,
    "timeout_rate": 0.0,
}


def _make_config() -> CritPtAgentConfig:
    return CritPtAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="critpt_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="critpt_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        turn2_prompt_fpath=TURN2_PROMPT_FPATH,
    )


def _make_server() -> CritPtAgent:
    return CritPtAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))


def _mock_post_response(data: dict) -> AsyncMock:
    mock = AsyncMock()
    mock.read.return_value = json.dumps(data).encode()
    mock.cookies = {}
    return mock


def _make_run_body(
    problem_id: str = "5", code_template: str = "def solve():\n    return ???"
) -> CritPtAgentRunRequest:
    return CritPtAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "You are a physics assistant."},
                {"role": "user", "content": "Solve this."},
            ]
        ),
        problem_id=problem_id,
        code_template=code_template,
    )


class TestHelpers:
    def test_extract_output_text_single_message(self):
        response_json = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                }
            ]
        }
        assert _extract_output_text(response_json) == "Hello world"

    def test_extract_output_text_skips_non_message(self):
        response_json = {
            "output": [
                {"type": "function_call", "content": []},
                {"type": "message", "content": [{"type": "output_text", "text": "Answer"}]},
            ]
        }
        assert _extract_output_text(response_json) == "Answer"

    def test_extract_output_text_empty(self):
        assert _extract_output_text({"output": []}) == ""

    def test_strip_think_block(self):
        assert _strip_thinking_blocks("<think>reasoning here</think>Final Answer: 42") == "Final Answer: 42"

    def test_strip_thinking_variant(self):
        assert _strip_thinking_blocks("<thinking>blah</thinking>answer") == "answer"

    def test_strip_multiline_block(self):
        text = "<think>line 1\nline 2\nline 3</think>Final Answer: 42"
        assert _strip_thinking_blocks(text) == "Final Answer: 42"

    def test_strip_multiple_blocks(self):
        text = "<think>step 1</think>middle\n<think>step 2</think>conclusion"
        assert _strip_thinking_blocks(text) == "middle\nconclusion"

    def test_strip_no_tags_unchanged(self):
        assert _strip_thinking_blocks("Final Answer: 42") == "Final Answer: 42"

    def test_strip_unclosed_left_intact(self):
        text = "<think>partial reasoning"
        assert _strip_thinking_blocks(text) == text

    def test_turn2_template_loaded_from_yaml(self):
        server = _make_server()
        template = "def solve():\n    return ???"
        rendered = server._turn2_user_template.format(code_template=template)
        assert "```python" in rendered
        assert template in rendered
        assert "Populate your final answer" in rendered
        assert "No additional reasoning" in rendered


class TestApp:
    def test_sanity(self):
        _make_server()

    @pytest.mark.asyncio
    async def test_run_correct(self):
        server = _make_server()
        server.server_client.post.side_effect = [
            _mock_post_response({}),  # seed_session
            _mock_post_response(_MODEL_RESPONSE),  # Turn 1
            _mock_post_response(_MODEL_RESPONSE),  # Turn 2
            _mock_post_response(_VERIFY_RESPONSE),  # verify
        ]

        request_mock = MagicMock()
        request_mock.cookies = {}

        result = await server.run(request_mock, _make_run_body(problem_id="5"))

        assert result.reward == 1.0
        assert server.server_client.post.call_count == 4

    @pytest.mark.asyncio
    async def test_run_turn2_messages_include_turn1_response(self):
        server = _make_server()
        server.server_client.post.side_effect = [
            _mock_post_response({}),
            _mock_post_response(_MODEL_RESPONSE),
            _mock_post_response(_MODEL_RESPONSE),
            _mock_post_response(_VERIFY_RESPONSE),
        ]

        request_mock = MagicMock()
        request_mock.cookies = {}
        body = _make_run_body(code_template="def solve():\n    return ???")
        await server.run(request_mock, body)

        # The Turn 2 call is the 3rd post (index 2)
        turn2_call = server.server_client.post.call_args_list[2]
        turn2_params = turn2_call.kwargs["json"]
        messages = list(turn2_params.input)

        # Should be: [system, user, assistant (Turn1), user (Turn2)]
        assert len(messages) == 4
        assert messages[2].role == "assistant"
        assert messages[2].content == "The answer is 1.23"
        assert messages[3].role == "user"
        assert "def solve():" in messages[3].content

    @pytest.mark.asyncio
    async def test_run_strips_thinking_from_turn1_before_turn2(self):
        server = _make_server()
        turn1_with_think = {
            **_MODEL_RESPONSE,
            "output": [
                {
                    "id": "msg-id",
                    "content": [
                        {
                            "annotations": [],
                            "text": "<think>I need to compute this...</think>Final Answer: 1.23",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
        }
        server.server_client.post.side_effect = [
            _mock_post_response({}),
            _mock_post_response(turn1_with_think),
            _mock_post_response(_MODEL_RESPONSE),
            _mock_post_response(_VERIFY_RESPONSE),
        ]

        request_mock = MagicMock()
        request_mock.cookies = {}
        await server.run(request_mock, _make_run_body())

        turn2_call = server.server_client.post.call_args_list[2]
        messages = list(turn2_call.kwargs["json"].input)
        assert messages[2].content == "Final Answer: 1.23"  # think block stripped

    @pytest.mark.asyncio
    async def test_run_verify_called_with_problem_id(self):
        server = _make_server()
        server.server_client.post.side_effect = [
            _mock_post_response({}),
            _mock_post_response(_MODEL_RESPONSE),
            _mock_post_response(_MODEL_RESPONSE),
            _mock_post_response(_VERIFY_RESPONSE),
        ]

        request_mock = MagicMock()
        request_mock.cookies = {}
        await server.run(request_mock, _make_run_body(problem_id="42"))

        verify_call = server.server_client.post.call_args_list[3]
        assert verify_call.kwargs["url_path"] == "/verify"
        verify_data = verify_call.kwargs["json"]
        assert verify_data["problem_id"] == "42"
