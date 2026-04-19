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

from pytest import fixture

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import (
    BrowsecompAgent,
    BrowsecompAgentConfig,
)


def _make_config(**kwargs) -> BrowsecompAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
    )
    return BrowsecompAgentConfig(**(defaults | kwargs))


def _make_msg(text: str, msg_id: str = "msg_001") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=msg_id,
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_fn_call(name: str, call_id: str = "call_001", args: dict | None = None) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id="fc_001",
        call_id=call_id,
        name=name,
        arguments=json.dumps(args or {}),
        type="function_call",
    )


def _make_tool_output(call_id: str = "call_001", output: str = "tool result") -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output=output)


def _make_model_response(outputs: list, response_id: str = "resp_001") -> dict:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="test_model",
        object="response",
        output=outputs,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump()


class TestApp:
    @fixture
    def agent(self) -> BrowsecompAgent:
        return BrowsecompAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

    # ---- Sanity ----

    def test_sanity(self) -> None:
        BrowsecompAgent(config=_make_config(), server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        config = _make_config()
        assert config.max_steps == 400
        assert config.keep_rounds == 9999
        assert config.nudge_steps is True
        assert config.max_context_tokens == 196608
        assert config.context_reset_pct == 0.3
        assert config.context_reset_keep_rounds == 3
        assert config.max_run_retries == 1

    # ---- _compact_old_tool_messages ----

    def test_compact_old_tool_messages_no_compaction_needed(self, agent: BrowsecompAgent) -> None:
        """With keep_rounds=2 and only 2 tool outputs, nothing should be replaced."""
        agent.config = _make_config(keep_rounds=2)
        messages = [
            _make_tool_output("c1", "result 1"),
            _make_tool_output("c2", "result 2"),
        ]
        result = agent._compact_old_tool_messages(messages)
        assert result[0].output == "result 1"
        assert result[1].output == "result 2"

    def test_compact_old_tool_messages_replaces_old(self, agent: BrowsecompAgent) -> None:
        """With keep_rounds=1 and 3 tool outputs, the first two should be replaced."""
        agent.config = _make_config(keep_rounds=1)
        messages = [
            _make_tool_output("c1", "result 1"),
            _make_tool_output("c2", "result 2"),
            _make_tool_output("c3", "result 3"),
        ]
        result = agent._compact_old_tool_messages(messages)
        assert result[0].output == "[Previous tool result hidden for context management]"
        assert result[1].output == "[Previous tool result hidden for context management]"
        assert result[2].output == "result 3"

    def test_compact_old_tool_messages_mixed_types(self, agent: BrowsecompAgent) -> None:
        """Non-tool messages should not be affected."""
        agent.config = _make_config(keep_rounds=1)
        msg = _make_msg("some text")
        tool1 = _make_tool_output("c1", "old result")
        tool2 = _make_tool_output("c2", "new result")
        messages = [msg, tool1, tool2]
        result = agent._compact_old_tool_messages(messages)
        assert result[0].content[0].text == "some text"
        assert result[1].output == "[Previous tool result hidden for context management]"
        assert result[2].output == "new result"

    # ---- _extract_last_rounds ----

    def test_extract_last_rounds_empty(self, agent: BrowsecompAgent) -> None:
        agent.config = _make_config(context_reset_keep_rounds=2)
        assert agent._extract_last_rounds([]) == []

    def test_extract_last_rounds_zero(self, agent: BrowsecompAgent) -> None:
        agent.config = _make_config(context_reset_keep_rounds=0)
        messages = [_make_fn_call("search"), _make_tool_output("c1")]
        assert agent._extract_last_rounds(messages) == []

    def test_extract_last_rounds_keeps_one(self, agent: BrowsecompAgent) -> None:
        """keep_rounds=1 should only return the last fn_call + tool_output pair."""
        agent.config = _make_config(context_reset_keep_rounds=1)
        fn1 = _make_fn_call("search", call_id="c1")
        out1 = _make_tool_output("c1", "old")
        fn2 = _make_fn_call("browse", call_id="c2")
        out2 = _make_tool_output("c2", "new")
        messages = [fn1, out1, fn2, out2]
        result = agent._extract_last_rounds(messages)
        assert len(result) == 2
        assert result[0].name == "browse"
        assert result[1].output == "new"

    def test_extract_last_rounds_keeps_two(self, agent: BrowsecompAgent) -> None:
        agent.config = _make_config(context_reset_keep_rounds=2)
        fn1 = _make_fn_call("search", call_id="c1")
        out1 = _make_tool_output("c1", "r1")
        fn2 = _make_fn_call("browse", call_id="c2")
        out2 = _make_tool_output("c2", "r2")
        fn3 = _make_fn_call("search", call_id="c3")
        out3 = _make_tool_output("c3", "r3")
        messages = [fn1, out1, fn2, out2, fn3, out3]
        result = agent._extract_last_rounds(messages)
        assert len(result) == 4
        assert result[0].name == "browse"
        assert result[2].name == "search"

    # ---- responses (multi-turn loop) ----

    async def test_responses_single_turn_no_tools(self, agent: BrowsecompAgent) -> None:
        """Model responds immediately without tool calls — loop exits after one step."""
        final_response = _make_model_response([_make_msg("Final Answer: Paris")])

        mock_http = MagicMock()
        mock_http.ok = True
        mock_http.read = AsyncMock(return_value=json.dumps(final_response).encode())
        mock_http.cookies = {}
        agent.server_client.post = AsyncMock(return_value=mock_http)

        request_mock = MagicMock()
        request_mock.cookies = {}
        response_mock = MagicMock()
        response_mock.set_cookie = MagicMock()

        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is the capital of France?"}]
        )
        result = await agent.responses(request_mock, response_mock, body)

        assert agent.server_client.post.call_count == 1
        assert result.output[-1].content[0].text == "Final Answer: Paris"

    async def test_responses_one_tool_call_then_answer(self, agent: BrowsecompAgent) -> None:
        """Model makes one tool call, then answers — loop runs exactly two model steps."""
        fn_call = _make_fn_call("search", call_id="c1", args={"queries": ["capital France"]})
        tool_response_data = _make_model_response([fn_call])
        final_response_data = _make_model_response([_make_msg("Final Answer: Paris")])

        tool_http = MagicMock()
        tool_http.ok = True
        tool_http.read = AsyncMock(
            side_effect=[
                json.dumps(tool_response_data).encode(),  # model call 1
                json.dumps(final_response_data).encode(),  # model call 2
            ]
        )
        tool_http.content.read = AsyncMock(return_value=b'{"results_string": "Paris is the capital"}')  # tool call
        tool_http.cookies = {}
        agent.server_client.post = AsyncMock(return_value=tool_http)

        request_mock = MagicMock()
        request_mock.cookies = {}
        response_mock = MagicMock()
        response_mock.set_cookie = MagicMock()

        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is the capital of France?"}]
        )
        result = await agent.responses(request_mock, response_mock, body)

        assert agent.server_client.post.call_count == 3  # model + tool + model
        assert result.output[-1].content[0].text == "Final Answer: Paris"

    async def test_responses_respects_max_steps(self) -> None:
        """Agent should stop after max_steps even if no final answer is given."""
        agent = BrowsecompAgent(
            config=_make_config(max_steps=2, nudge_steps=False),
            server_client=MagicMock(spec=ServerClient),
        )
        fn_call = _make_fn_call("search", call_id="c1", args={"queries": ["q"]})
        tool_response_data = _make_model_response([fn_call])

        mock_http = MagicMock()
        mock_http.ok = True
        mock_http.read = AsyncMock(return_value=json.dumps(tool_response_data).encode())
        mock_http.content.read = AsyncMock(return_value=b"{}")
        mock_http.cookies = {}
        agent.server_client.post = AsyncMock(return_value=mock_http)

        request_mock = MagicMock()
        request_mock.cookies = {}
        response_mock = MagicMock()
        response_mock.set_cookie = MagicMock()

        body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hard question"}])
        await agent.responses(request_mock, response_mock, body)

        # max_steps=2: 2 model calls + 2 tool calls = 4 total posts
        assert agent.server_client.post.call_count == 4
