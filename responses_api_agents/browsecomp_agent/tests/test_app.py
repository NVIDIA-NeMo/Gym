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
import re
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import (
    BrowsecompAgent,
    BrowsecompAgentConfig,
    BrowsecompAgentRunRequest,
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


def _make_model_response_with_usage(outputs: list, input_tokens: int, response_id: str = "resp_001") -> dict:
    """Like _make_model_response but carries usage.input_tokens (drives the post-call context reset)."""
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="test_model",
        object="response",
        output=outputs,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=input_tokens,
        ),
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
        assert config.context_reset_tokens == 0
        assert config.context_reset_keep_rounds == 3
        assert config.max_run_retries == 1

    # ---- _reset_threshold ----

    def test_reset_threshold_pct_fallback(self) -> None:
        # context_reset_tokens == 0 (default) -> max_context_tokens * context_reset_pct
        config = _make_config()
        assert BrowsecompAgent._reset_threshold(config) == int(196608 * 0.3)  # 58982

    def test_reset_threshold_absolute_overrides_pct(self) -> None:
        # context_reset_tokens > 0 takes precedence over the pct calc (50k standard)
        config = _make_config(context_reset_tokens=50000)
        assert BrowsecompAgent._reset_threshold(config) == 50000

    def test_reset_threshold_disabled(self) -> None:
        config = _make_config(context_reset_tokens=0, max_context_tokens=0)
        assert BrowsecompAgent._reset_threshold(config) == 0

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
        tool_http.status = 200
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
        mock_http.status = 200
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

    # ---- full trajectory (Part B) ----

    def test_save_trajectory_writes_header_and_all_items(self, tmp_path) -> None:
        agent = BrowsecompAgent(
            config=_make_config(snap_dir=str(tmp_path)),
            server_client=MagicMock(spec=ServerClient),
        )
        input_messages = [NeMoGymResponseOutputMessage.model_construct(type="message", role="user", content=[])]
        full_trajectory = [_make_fn_call("search"), _make_tool_output(), _make_msg("Exact Answer: RIGHT")]
        agent._save_trajectory(
            input_messages=input_messages,
            full_trajectory=full_trajectory,
            task_index="9",
            attempt="0",
            reset_steps=[2, 5],
            reset_count=2,
            num_tool_calls=1,
        )
        path = tmp_path / "sample_9" / "attempt_0_trajectory.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        header = json.loads(lines[0])
        assert header["type"] == "metadata"
        assert header["reset_count"] == 2 and header["reset_steps"] == [2, 5] and header["num_tool_calls"] == 1
        # line 1 header + input prefix + every trajectory item
        assert len(lines) == 1 + len(input_messages) + len(full_trajectory)

    async def test_full_trajectory_survives_reset(self, tmp_path) -> None:
        """A context reset trims new_outputs, but response.output (= full_trajectory) keeps the
        pre-reset tool round, and the last item is still the final answer (grading invariant)."""
        agent = BrowsecompAgent(
            config=_make_config(
                snap_dir=str(tmp_path),
                save_model_call_using_vllm_tokenize_endpoint=False,
                context_reset_tokens=1,  # threshold 1 -> any usage>1 resets
                context_reset_keep_rounds=0,  # reset drops everything from new_outputs
                nudge_steps=False,
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        model1 = _make_model_response_with_usage([_make_fn_call("search", call_id="c1")], input_tokens=0)  # turn 1
        model2 = _make_model_response_with_usage([_make_msg("partial")], input_tokens=999)  # turn 2 -> reset
        model3 = _make_model_response_with_usage([_make_msg("Exact Answer: RIGHT")], input_tokens=0)  # turn 3

        http = MagicMock()
        http.ok = True
        http.status = 200
        http.cookies = {}
        http.read = AsyncMock(
            side_effect=[json.dumps(model1).encode(), json.dumps(model2).encode(), json.dumps(model3).encode()]
        )
        http.content.read = AsyncMock(return_value=b'{"results_string": "tool result"}')
        agent.server_client.post = AsyncMock(return_value=http)

        request_mock = MagicMock()
        request_mock.cookies = {}
        response_mock = MagicMock()
        response_mock.set_cookie = MagicMock()

        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "q"}],
            metadata={"task_index": "7", "attempt": "0"},
        )
        result = await agent.responses(request_mock, response_mock, body)

        types = [getattr(o, "type", None) for o in result.output]
        # full trajectory retains the pre-reset round (would be gone from the trimmed new_outputs window)
        assert "function_call" in types and "function_call_output" in types
        assert len(result.output) == 3
        # grading invariant: last item is the final assistant answer (what Part A grades)
        assert result.output[-1].type == "message" and result.output[-1].content[0].text == "Exact Answer: RIGHT"
        # trajectory.jsonl written with the reset recorded
        traj = tmp_path / "sample_7" / "attempt_0_trajectory.jsonl"
        assert traj.exists()
        lines = traj.read_text().strip().split("\n")
        assert json.loads(lines[0])["reset_steps"] == [2]
        assert len(lines) == 1 + len(body.input) + len(result.output)

    # ---- _last_message_text (bc_frankie last-message retry parity) ----

    def test_last_message_text_returns_final_answer(self) -> None:
        resp = NeMoGymResponse.model_validate(_make_model_response([_make_msg("Exact Answer: Paris")]))
        assert BrowsecompAgent._last_message_text(resp) == "Exact Answer: Paris"

    def test_last_message_text_last_content_bearing_wins_over_earlier(self) -> None:
        """The empty-answer retry keys on the LAST content-bearing assistant message, not the
        concatenation of every assistant turn. A trailing think-only turn -> empty after
        <think>-strip -> retry, even though an earlier turn emitted a real answer (where the old
        aggregated output_text check would NOT have retried)."""
        resp = NeMoGymResponse.model_validate(
            _make_model_response(
                [
                    _make_msg("Exact Answer: Paris", msg_id="m1"),
                    _make_fn_call("search", call_id="c1"),
                    _make_msg("<think>still reasoning</think>", msg_id="m2"),
                ]
            )
        )
        last = BrowsecompAgent._last_message_text(resp)
        assert last == "<think>still reasoning</think>"
        # last-message semantics -> empty after strip -> WOULD retry
        assert re.sub(r"<think>.*?</think>", "", last, flags=re.DOTALL).strip() == ""
        # contrast: the OLD aggregated output_text is non-empty -> would NOT have retried
        assert re.sub(r"<think>.*?</think>", "", resp.output_text, flags=re.DOTALL).strip() == "Exact Answer: Paris"

    def test_last_message_text_walks_past_trailing_tool_call(self) -> None:
        resp = NeMoGymResponse.model_validate(
            _make_model_response(
                [_make_msg("Exact Answer: Paris", msg_id="m1"), _make_fn_call("search", call_id="c1")]
            )
        )
        assert BrowsecompAgent._last_message_text(resp) == "Exact Answer: Paris"

    def test_last_message_text_skips_empty_message(self) -> None:
        """An empty-content trailing message is skipped (mirrors bc_frankie's truthy-content check)."""
        resp = NeMoGymResponse.model_validate(
            _make_model_response([_make_msg("real answer", msg_id="m1"), _make_msg("", msg_id="m2")])
        )
        assert BrowsecompAgent._last_message_text(resp) == "real answer"

    def test_last_message_text_empty_when_no_message(self) -> None:
        resp = NeMoGymResponse.model_validate(_make_model_response([_make_fn_call("search", call_id="c1")]))
        assert BrowsecompAgent._last_message_text(resp) == ""

    def test_last_message_text_empty_output(self) -> None:
        assert BrowsecompAgent._last_message_text(NeMoGymResponse.model_validate(_make_model_response([]))) == ""

    # ---- run() retry wiring ----

    async def test_run_retries_on_think_only_last_message(self) -> None:
        """run() retries the whole trajectory when the last content-bearing turn is think-only,
        even though an earlier turn emitted a real answer. Under the old aggregated output_text
        check this would NOT retry (attempt 0 would be verified), so post.call_count would differ."""
        agent = BrowsecompAgent(config=_make_config(max_run_retries=2), server_client=MagicMock(spec=ServerClient))

        attempt0 = _make_model_response(
            [_make_msg("Exact Answer: EARLY", msg_id="m1"), _make_msg("<think>no answer</think>", msg_id="m2")]
        )
        attempt1 = _make_model_response([_make_msg("Exact Answer: FINAL", msg_id="m3")])
        verify_json = {
            "reward": 1.0,
            "response": attempt1,
            "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
        }

        def _http(read_bytes: bytes | None = None) -> MagicMock:
            m = MagicMock()
            m.ok = True
            m.cookies = {}
            if read_bytes is not None:
                m.read = AsyncMock(return_value=read_bytes)
            return m

        # seed_session, /v1/responses (attempt 0), /v1/responses (attempt 1 after retry), /verify
        agent.server_client.post = AsyncMock(
            side_effect=[
                _http(),
                _http(json.dumps(attempt0).encode()),
                _http(json.dumps(attempt1).encode()),
                _http(json.dumps(verify_json).encode()),
            ]
        )

        request_mock = MagicMock()
        request_mock.cookies = {}
        body = BrowsecompAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}])
        )
        result = await agent.run(request_mock, body)

        assert agent.server_client.post.call_count == 4  # retry fired -> attempt 1 + verify
        assert result.reward == 1.0
