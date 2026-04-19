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
"""Unit tests for FinanceAgent (responses_api_agents/finance_agent/app.py).

Covers:
- _is_context_overflow_error detection
- _is_model_output classification
- _truncate_oldest_exchange (context overflow recovery)
- Agent loop: max_steps enforcement at top of loop
- Agent loop: text-only response terminates loop
- Agent loop: done_tool termination
- Agent loop: context-overflow exception triggers truncation (eval mode)
- Agent loop: non-overflow errors break gracefully (log + break, no raise)
- Agent loop: timeout breaks gracefully
- Agent loop: truncate_on_overflow=False breaks gracefully (training mode)
"""

from unittest.mock import AsyncMock, MagicMock, patch

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.finance_agent.app import FinanceAgent, FinanceAgentConfig


APP_MODULE = "responses_api_agents.finance_agent.app"


# ── Test helpers ──────────────────────────────────────────────────


def _item(type_val, role=None, **kwargs):
    """Lightweight mock output item with a given .type (and optional .role)."""
    m = MagicMock()
    m.type = type_val
    if role is not None:
        m.role = role
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def _fn_call(name="web_search", call_id="c1"):
    return _item("function_call", name=name, call_id=call_id, arguments="{}")


def _fn_output(call_id="c1"):
    return NeMoGymFunctionCallOutput(type="function_call_output", call_id=call_id, output="ok")


def _assistant_msg(text="some text"):
    m = _item("message", role="assistant")
    content_item = MagicMock()
    content_item.text = text
    m.content = [content_item]
    return m


def _reasoning():
    return _item("reasoning")


def _user_continue():
    return NeMoGymEasyInputMessage(role="user", content="Continue.")


def _mock_model_response(output_items):
    resp = MagicMock()
    resp.output = list(output_items)
    resp.incomplete_details = None
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.total_tokens = 15
    usage.input_tokens_details = MagicMock(cached_tokens=0)
    usage.output_tokens_details = MagicMock(reasoning_tokens=0)
    resp.usage = usage
    return resp


def _make_agent(
    max_steps=50,
    model_call_timeout=None,
    truncate_on_overflow=False,
):
    config = FinanceAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        model_server=ModelServerRef(type="responses_api_models", name="ms"),
        max_steps=max_steps,
        model_call_timeout=model_call_timeout,
        truncate_on_overflow=truncate_on_overflow,
    )
    return FinanceAgent(config=config, server_client=MagicMock(spec=ServerClient))


async def _run_loop(agent, model_responses):
    """Run agent.responses() with fully mocked server interactions.

    model_responses: list of mock NeMoGymResponse objects consumed in order.
    Returns the final NeMoGymResponse-like object whose .output holds the full
    conversation history built by the loop.
    """
    body = NeMoGymResponseCreateParamsNonStreaming(input="What is 2+2?")
    request = MagicMock()
    request.cookies = {}
    response_obj = MagicMock()

    async def fake_post(server_name, url_path, json=None, cookies=None):
        mock_resp = MagicMock()
        mock_resp.cookies = {}
        if url_path != "/v1/responses":
            mock_resp.content = MagicMock()
            mock_resp.content.read = AsyncMock(return_value=b"tool_output")
        return mock_resp

    agent.server_client.post = AsyncMock(side_effect=fake_post)
    responses_iter = iter(model_responses)

    with patch(f"{APP_MODULE}.raise_for_status", new_callable=AsyncMock):
        with patch(f"{APP_MODULE}.get_response_json", new_callable=AsyncMock, return_value={}):
            with patch(f"{APP_MODULE}.NeMoGymResponse") as mock_cls:
                mock_cls.model_validate = MagicMock(side_effect=lambda _: next(responses_iter))
                return await agent.responses(request, response_obj, body)


# ═══════════════════════════════════════════════════════════════════
#  _is_model_output
# ═══════════════════════════════════════════════════════════════════


class TestIsModelOutput:
    def test_reasoning_is_model(self):
        assert FinanceAgent._is_model_output(_reasoning()) is True

    def test_function_call_is_model(self):
        assert FinanceAgent._is_model_output(_fn_call()) is True

    def test_assistant_message_is_model(self):
        assert FinanceAgent._is_model_output(_assistant_msg()) is True

    def test_function_call_output_is_not_model(self):
        assert FinanceAgent._is_model_output(_fn_output()) is False

    def test_user_message_is_not_model(self):
        assert FinanceAgent._is_model_output(_user_continue()) is False

    def test_unknown_type_is_not_model(self):
        assert FinanceAgent._is_model_output(_item("other")) is False


# ═══════════════════════════════════════════════════════════════════
#  _is_context_overflow_error
# ═══════════════════════════════════════════════════════════════════


class TestIsContextOverflowError:
    def test_vllm_context_length_message(self):
        e = RuntimeError(
            "This model's maximum context length is 32768 tokens. However, you requested 32818 tokens in the messages."
        )
        assert FinanceAgent._is_context_overflow_error(e) is True

    def test_prompt_too_long(self):
        assert FinanceAgent._is_context_overflow_error(RuntimeError("prompt is too long")) is True

    def test_input_exceeded_context_window(self):
        assert FinanceAgent._is_context_overflow_error(RuntimeError("input exceeded the context window")) is True

    def test_timeout_is_not_overflow(self):
        assert FinanceAgent._is_context_overflow_error(TimeoutError("timed out")) is False

    def test_generic_error_is_not_overflow(self):
        assert FinanceAgent._is_context_overflow_error(RuntimeError("connection reset")) is False

    def test_empty_message_is_not_overflow(self):
        assert FinanceAgent._is_context_overflow_error(RuntimeError("")) is False


# ═══════════════════════════════════════════════════════════════════
#  _truncate_oldest_exchange
# ═══════════════════════════════════════════════════════════════════


class TestTruncateOldestExchange:
    def test_empty_returns_empty(self):
        assert FinanceAgent._truncate_oldest_exchange([]) == []

    def test_single_item_returns_same(self):
        items = [_reasoning()]
        assert FinanceAgent._truncate_oldest_exchange(items) is items

    def test_single_exchange_cannot_truncate(self):
        items = [_reasoning(), _fn_call(), _fn_output()]
        result = FinanceAgent._truncate_oldest_exchange(items)
        assert result is items

    def test_two_exchanges_removes_first(self):
        ex1 = [_reasoning(), _fn_call(), _fn_output()]
        ex2 = [_reasoning(), _fn_call(), _fn_output()]
        result = FinanceAgent._truncate_oldest_exchange(ex1 + ex2)
        assert len(result) == 3
        assert result == ex2

    def test_text_only_plus_continue_removed(self):
        ex1 = [_reasoning(), _assistant_msg()]
        cont = [_user_continue()]
        ex2 = [_reasoning(), _fn_call(), _fn_output()]
        result = FinanceAgent._truncate_oldest_exchange(ex1 + cont + ex2)
        assert result == ex2

    def test_preserves_all_later_exchanges(self):
        ex1 = [_reasoning(), _fn_call(), _fn_output(), _fn_output()]
        ex2 = [_reasoning(), _assistant_msg()]
        cont = [_user_continue()]
        ex3 = [_reasoning(), _fn_call(), _fn_output()]
        result = FinanceAgent._truncate_oldest_exchange(ex1 + ex2 + cont + ex3)
        assert result == ex2 + cont + ex3

    def test_multiple_tool_outputs_in_exchange(self):
        ex1 = [_reasoning(), _fn_call("a"), _fn_call("b"), _fn_output("a"), _fn_output("b")]
        ex2 = [_reasoning(), _fn_call("c"), _fn_output("c")]
        result = FinanceAgent._truncate_oldest_exchange(ex1 + ex2)
        assert result == ex2


# ═══════════════════════════════════════════════════════════════════
#  Agent loop integration tests
# ═══════════════════════════════════════════════════════════════════


class TestFinanceAgentLoop:
    async def test_max_steps_limits_turns(self):
        """Loop should run exactly max_steps turns, then stop."""
        agent = _make_agent(max_steps=3, model_call_timeout=None)
        model_responses = [_mock_model_response([_fn_call("web_search", f"c{i}")]) for i in range(10)]
        result = await _run_loop(agent, model_responses)
        fn_outputs = [o for o in result.output if getattr(o, "type", None) == "function_call_output"]
        assert len(fn_outputs) == 3

    async def test_done_tool_stops_loop(self):
        """submit_final_result should break the loop immediately."""
        agent = _make_agent(max_steps=50, model_call_timeout=None)
        model_responses = [
            _mock_model_response([_fn_call("web_search", "c1")]),
            _mock_model_response([_fn_call("submit_final_result", "c2")]),
        ]
        result = await _run_loop(agent, model_responses)
        fn_names = [getattr(o, "name", None) for o in result.output if getattr(o, "type", None) == "function_call"]
        assert fn_names == ["web_search", "submit_final_result"]

    async def test_text_only_response_stops_loop(self):
        """Text-only response (no function calls) terminates the loop immediately."""
        agent = _make_agent(max_steps=50, model_call_timeout=None)
        model_responses = [
            _mock_model_response([_assistant_msg("Here is my analysis...")]),
        ]
        result = await _run_loop(agent, model_responses)

        assistant_msgs = [
            o
            for o in result.output
            if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant"
        ]
        assert len(assistant_msgs) == 1

        continue_msgs = [
            o for o in result.output if isinstance(o, NeMoGymEasyInputMessage) and o.content == "Continue."
        ]
        assert len(continue_msgs) == 0

    async def test_non_overflow_error_breaks_gracefully(self):
        """Non-context-overflow errors are logged and break the loop (no raise)."""
        agent = _make_agent(max_steps=50, model_call_timeout=None)

        async def always_fail(server_name, url_path, json=None, cookies=None):
            if url_path == "/v1/responses":
                raise RuntimeError("vLLM internal error")
            mock_resp = MagicMock()
            mock_resp.cookies = {}
            return mock_resp

        agent.server_client.post = AsyncMock(side_effect=always_fail)
        body = NeMoGymResponseCreateParamsNonStreaming(input="Question?")
        request = MagicMock()
        request.cookies = {}
        response_obj = MagicMock()

        with patch(f"{APP_MODULE}.raise_for_status", new_callable=AsyncMock):
            with patch(f"{APP_MODULE}.get_response_json", new_callable=AsyncMock, return_value={}):
                result = await agent.responses(request, response_obj, body)

        assert result.id == "error"
        assert result.output == []

    async def test_context_overflow_exception_triggers_truncation(self):
        """With truncate_on_overflow=True, a context-overflow exception removes the oldest exchange."""
        agent = _make_agent(max_steps=50, model_call_timeout=None, truncate_on_overflow=True)

        resp1 = _mock_model_response([_fn_call("web_search", "c1")])
        resp2 = _mock_model_response([_fn_call("edgar_search", "c2")])
        resp3 = _mock_model_response([_fn_call("submit_final_result", "c3")])
        model_call_count = [0]

        async def fake_post(server_name, url_path, json=None, cookies=None):
            mock_resp = MagicMock()
            mock_resp.cookies = {}
            if url_path == "/v1/responses":
                model_call_count[0] += 1
                if model_call_count[0] == 3:
                    raise RuntimeError(
                        "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens."
                    )
            else:
                mock_resp.content = MagicMock()
                mock_resp.content.read = AsyncMock(return_value=b"ok")
            return mock_resp

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        body = NeMoGymResponseCreateParamsNonStreaming(input="Question?")
        request = MagicMock()
        request.cookies = {}
        response_obj = MagicMock()

        validate_responses = iter([resp1, resp2, resp3])

        with patch(f"{APP_MODULE}.raise_for_status", new_callable=AsyncMock):
            with patch(f"{APP_MODULE}.get_response_json", new_callable=AsyncMock, return_value={}):
                with patch(f"{APP_MODULE}.NeMoGymResponse") as mock_cls:
                    mock_cls.model_validate = MagicMock(side_effect=lambda _: next(validate_responses))
                    result = await agent.responses(request, response_obj, body)

        fn_names = [getattr(o, "name", None) for o in result.output if getattr(o, "type", None) == "function_call"]
        assert fn_names == ["edgar_search", "submit_final_result"]

    async def test_timeout_error_breaks_gracefully(self):
        """TimeoutError on model call breaks the loop (no raise), preserving prior output."""
        agent = _make_agent(max_steps=50, model_call_timeout=None, truncate_on_overflow=True)

        resp1 = _mock_model_response([_fn_call("web_search", "c1")])
        model_call_count = [0]

        async def fake_post(server_name, url_path, json=None, cookies=None):
            mock_resp = MagicMock()
            mock_resp.cookies = {}
            if url_path == "/v1/responses":
                model_call_count[0] += 1
                if model_call_count[0] == 2:
                    raise TimeoutError("timed out waiting for model")
            else:
                mock_resp.content = MagicMock()
                mock_resp.content.read = AsyncMock(return_value=b"ok")
            return mock_resp

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        body = NeMoGymResponseCreateParamsNonStreaming(input="Question?")
        request = MagicMock()
        request.cookies = {}
        response_obj = MagicMock()

        with patch(f"{APP_MODULE}.raise_for_status", new_callable=AsyncMock):
            with patch(f"{APP_MODULE}.get_response_json", new_callable=AsyncMock, return_value={}):
                with patch(f"{APP_MODULE}.NeMoGymResponse") as mock_cls:
                    mock_cls.model_validate = MagicMock(return_value=resp1)
                    result = await agent.responses(request, response_obj, body)

        fn_calls = [o for o in result.output if getattr(o, "type", None) == "function_call"]
        assert len(fn_calls) == 1
        fn_outputs = [o for o in result.output if getattr(o, "type", None) == "function_call_output"]
        assert len(fn_outputs) == 1

    async def test_truncate_on_overflow_disabled_breaks_gracefully(self):
        """With truncate_on_overflow=False (training), overflow errors break (no truncation, no raise)."""
        agent = _make_agent(max_steps=50, model_call_timeout=None, truncate_on_overflow=False)

        resp1 = _mock_model_response([_fn_call("web_search", "c1")])
        model_call_count = [0]

        async def fake_post(server_name, url_path, json=None, cookies=None):
            mock_resp = MagicMock()
            mock_resp.cookies = {}
            if url_path == "/v1/responses":
                model_call_count[0] += 1
                if model_call_count[0] == 2:
                    raise RuntimeError("This model's maximum context length is 32768 tokens.")
            else:
                mock_resp.content = MagicMock()
                mock_resp.content.read = AsyncMock(return_value=b"ok")
            return mock_resp

        agent.server_client.post = AsyncMock(side_effect=fake_post)
        body = NeMoGymResponseCreateParamsNonStreaming(input="Question?")
        request = MagicMock()
        request.cookies = {}
        response_obj = MagicMock()

        with patch(f"{APP_MODULE}.raise_for_status", new_callable=AsyncMock):
            with patch(f"{APP_MODULE}.get_response_json", new_callable=AsyncMock, return_value={}):
                with patch(f"{APP_MODULE}.NeMoGymResponse") as mock_cls:
                    mock_cls.model_validate = MagicMock(return_value=resp1)
                    result = await agent.responses(request, response_obj, body)

        fn_calls = [o for o in result.output if getattr(o, "type", None) == "function_call"]
        assert len(fn_calls) == 1
        fn_outputs = [o for o in result.output if getattr(o, "type", None) == "function_call_output"]
        assert len(fn_outputs) == 1
