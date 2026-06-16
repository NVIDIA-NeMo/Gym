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
"""Tests for Finance Agent (responses_api_agents/finance_agent)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.finance_agent.app import (
    FinanceAgent,
    FinanceAgentConfig,
    FinanceAgentRunRequest,
)


_MODEL_SERVER = "model_server"
_RS_SERVER = "resources_server"
_INPUT = {"input": [{"role": "user", "content": "What was revenue?"}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> FinanceAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name=_RS_SERVER),
        model_server=ModelServerRef(type="responses_api_models", name=_MODEL_SERVER),
    )
    defaults.update(overrides)
    return FinanceAgentConfig(**defaults)


def _make_agent_and_client(config: FinanceAgentConfig | None = None):
    """Create agent + TestClient pair (the canonical test pattern for responses_api_agents)."""
    config = config or _make_config()
    agent = FinanceAgent(config=config, server_client=MagicMock(spec=ServerClient))
    app = agent.setup_webserver()
    client = TestClient(app)
    return agent, client


def _text_response(text: str, resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _tool_call_response(tool_name: str, arguments: str, call_id: str = "call_1", resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "fc_1",
                "call_id": call_id,
                "name": tool_name,
                "arguments": arguments,
                "type": "function_call",
                "status": "completed",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _reasoning_response(text: str = "thinking", resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "r1",
                "summary": [{"text": text, "type": "summary_text"}],
                "status": "completed",
                "type": "reasoning",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _dotjson_mock(*json_responses: dict) -> AsyncMock:
    """Mock for server_client.post return value, matching the pattern used by simple_agent tests.

    Sets up both paths used by the finance agent:
    - .read() → JSON string  (used by get_response_json for model calls)
    - .content.read() → bytes (used by tool-call output path)
    """
    mock = AsyncMock()
    mock.ok = True
    mock.content = MagicMock()
    if len(json_responses) == 1:
        data = json.dumps(json_responses[0])
        mock.read.return_value = data
        mock.content.read = AsyncMock(return_value=data.encode())
    else:
        strings = [json.dumps(r) for r in json_responses]
        mock.read.side_effect = strings
        mock.content.read = AsyncMock(side_effect=[s.encode() for s in strings])
    mock.cookies = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Tests: Config and Construction
# ---------------------------------------------------------------------------


class TestFinanceAgentConfig:
    def test_default_config(self) -> None:
        config = _make_config()
        assert config.max_steps is None
        assert config.done_tools == ["submit_final_result"]
        assert config.model_call_timeout is None
        assert config.tool_call_timeout is None
        assert config.truncate_on_overflow is False

    def test_custom_config(self) -> None:
        config = _make_config(
            max_steps=10,
            done_tools=["submit_final_result", "abort"],
            model_call_timeout=30.0,
            tool_call_timeout=60.0,
            truncate_on_overflow=True,
        )
        assert config.max_steps == 10
        assert config.done_tools == ["submit_final_result", "abort"]
        assert config.model_call_timeout == 30.0
        assert config.tool_call_timeout == 60.0
        assert config.truncate_on_overflow is True

    def test_sanity_construction(self) -> None:
        agent, _ = _make_agent_and_client()
        assert agent is not None


# ---------------------------------------------------------------------------
# Tests: Context Overflow Detection
# ---------------------------------------------------------------------------


class TestContextOverflowDetection:
    @pytest.mark.parametrize(
        "msg",
        [
            "maximum context length is 8192 tokens",
            "context length is only 4096 tokens",
            "maximum input length of 32768 tokens exceeded",
            "Please reduce the length of the input",
            "prompt is too long",
            "input exceeded the context window",
            "too large for model with 8192 maximum context length",
            "longer than the model's context length",
            "payload too large",
        ],
    )
    def test_detects_overflow(self, msg: str) -> None:
        assert FinanceAgent._is_context_overflow_error(Exception(msg))

    @pytest.mark.parametrize(
        "msg",
        [
            "connection refused",
            "timeout error",
            "Internal server error",
            "rate limit exceeded",
        ],
    )
    def test_ignores_non_overflow(self, msg: str) -> None:
        assert not FinanceAgent._is_context_overflow_error(Exception(msg))


# ---------------------------------------------------------------------------
# Tests: _is_model_output
# ---------------------------------------------------------------------------


class TestIsModelOutput:
    def test_function_call_is_model_output(self) -> None:
        item = MagicMock(type="function_call")
        assert FinanceAgent._is_model_output(item)

    def test_reasoning_is_model_output(self) -> None:
        item = MagicMock(type="reasoning")
        assert FinanceAgent._is_model_output(item)

    def test_assistant_message_is_model_output(self) -> None:
        item = MagicMock(type="message", role="assistant")
        assert FinanceAgent._is_model_output(item)

    def test_user_message_is_not_model_output(self) -> None:
        item = MagicMock(type="message", role="user")
        assert not FinanceAgent._is_model_output(item)

    def test_function_call_output_is_not_model_output(self) -> None:
        item = MagicMock(type="function_call_output")
        assert not FinanceAgent._is_model_output(item)


# ---------------------------------------------------------------------------
# Tests: _truncate_oldest_exchange
# ---------------------------------------------------------------------------


class TestTruncateOldestExchange:
    @staticmethod
    def _item(type_: str, role: str | None = None) -> MagicMock:
        return MagicMock(type=type_, role=role)

    def test_empty_list(self) -> None:
        assert FinanceAgent._truncate_oldest_exchange([]) == []

    def test_single_item(self) -> None:
        items = [self._item("message", "assistant")]
        assert len(FinanceAgent._truncate_oldest_exchange(items)) == 1

    def test_removes_first_exchange(self) -> None:
        items = [
            self._item("reasoning"),
            self._item("function_call"),
            self._item("function_call_output"),
            self._item("function_call_output"),
            self._item("message", "assistant"),
            self._item("function_call"),
            self._item("function_call_output"),
        ]
        result = FinanceAgent._truncate_oldest_exchange(items)
        assert result[0] is items[4]
        assert len(result) == 3

    def test_no_truncation_when_single_exchange(self) -> None:
        items = [self._item("reasoning"), self._item("function_call")]
        result = FinanceAgent._truncate_oldest_exchange(items)
        assert result == items


# ---------------------------------------------------------------------------
# Tests: responses() via TestClient
# ---------------------------------------------------------------------------


class TestResponses:
    def test_text_only_response_terminates_loop(self) -> None:
        """Model returns a text message → loop exits after one step."""
        agent, client = _make_agent_and_client()
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        data = res.json()
        assert len(data["output"]) == 1
        assert data["output"][0]["type"] == "message"

    def test_tool_call_then_text_terminates(self) -> None:
        """Model calls a tool, gets result, then responds with text."""
        agent, client = _make_agent_and_client()

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        text_data = _text_response("Here is the data.")
        tool_result = {"result": "filing data"}

        model_mock = _dotjson_mock(tool_call_data, text_data)
        rs_mock = _dotjson_mock(tool_result)

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            return rs_mock

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        types = [o["type"] for o in output]
        assert "function_call" in types
        assert "function_call_output" in types
        assert output[-1]["type"] == "message"

    def test_done_tool_terminates_loop(self) -> None:
        """submit_final_result tool call exits the loop immediately."""
        agent, client = _make_agent_and_client()

        tool_call_data = _tool_call_response("submit_final_result", json.dumps({"final_result": "Revenue was $100B"}))
        tool_result = {"status": "ok"}

        model_mock = _dotjson_mock(tool_call_data)
        rs_mock = _dotjson_mock(tool_result)

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            return rs_mock

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        fn_names = [o.get("name") for o in output if o["type"] == "function_call"]
        assert "submit_final_result" in fn_names

    def test_max_steps_terminates_loop(self) -> None:
        """Loop exits after max_steps even if model keeps producing tool calls."""
        config = _make_config(max_steps=2)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        tool_result = {"result": "data"}

        model_mock = _dotjson_mock(tool_call_data, tool_call_data)
        rs_mock = _dotjson_mock(tool_result)

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            return rs_mock

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        fn_calls = [o for o in output if o["type"] == "function_call"]
        assert len(fn_calls) == 2

    def test_model_call_timeout(self) -> None:
        """Model call timeout → loop terminates gracefully."""
        config = _make_config(model_call_timeout=0.01)
        agent, client = _make_agent_and_client(config)

        async def slow_post(**kwargs):
            await asyncio.sleep(10)

        agent.server_client.post = AsyncMock(side_effect=slow_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_tool_call_timeout_returns_error(self) -> None:
        """Tool call timeout → error JSON fed back to model, loop continues."""
        config = _make_config(tool_call_timeout=0.01, max_steps=2)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        text_data = _text_response("I'll try something else.")

        model_mock = _dotjson_mock(tool_call_data, text_data)

        async def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            await asyncio.sleep(10)

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        tool_outputs = [o for o in output if o["type"] == "function_call_output"]
        assert len(tool_outputs) >= 1
        error_payload = json.loads(tool_outputs[0]["output"])
        assert "timed out" in error_payload["error"]

    def test_tool_call_exception_returns_error(self) -> None:
        """Tool call exception → error JSON fed back to model, loop continues."""
        config = _make_config(max_steps=2)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        text_data = _text_response("Something went wrong.")

        model_mock = _dotjson_mock(tool_call_data, text_data)

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            raise ConnectionError("server unavailable")

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        tool_outputs = [o for o in output if o["type"] == "function_call_output"]
        assert len(tool_outputs) >= 1
        error_payload = json.loads(tool_outputs[0]["output"])
        assert "ConnectionError" in error_payload["error"]

    def test_model_error_terminates_loop(self) -> None:
        """Non-overflow model error → loop breaks."""
        agent, client = _make_agent_and_client()

        def fail_post(**kwargs):
            raise RuntimeError("internal server error")

        agent.server_client.post = AsyncMock(side_effect=fail_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_context_overflow_with_truncation(self) -> None:
        """Context overflow with truncate_on_overflow=True → retries after truncation.

        Need >=2 tool-call rounds so _truncate_oldest_exchange can drop one
        and keep another. Overflow on model call 3, success on model call 4.
        """
        config = _make_config(truncate_on_overflow=True, max_steps=10)
        agent, client = _make_agent_and_client(config)

        tc1 = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}), call_id="c1")
        tc2 = _tool_call_response("sec_filing_search", json.dumps({"ticker": "MSFT"}), call_id="c2")
        text = _text_response("Got it.")

        model_calls = {"n": 0}

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                model_calls["n"] += 1
                if model_calls["n"] <= 2:
                    data = tc1 if model_calls["n"] == 1 else tc2
                    return _dotjson_mock(data)
                if model_calls["n"] == 3:
                    raise Exception("maximum context length is 8192 tokens")
                return _dotjson_mock(text)
            return _dotjson_mock({"result": "filing data"})

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        assert output[-1]["type"] == "message"
        assert model_calls["n"] == 4

    def test_context_overflow_without_truncation_breaks(self) -> None:
        """Context overflow with truncate_on_overflow=False → loop breaks."""
        agent, client = _make_agent_and_client(_make_config(truncate_on_overflow=False))

        def fail_post(**kwargs):
            raise Exception("maximum context length is 8192 tokens")

        agent.server_client.post = AsyncMock(side_effect=fail_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_usage_accumulation(self) -> None:
        """Usage tokens accumulate across multiple model calls."""
        config = _make_config(max_steps=3)
        agent, client = _make_agent_and_client(config)

        usage1 = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 30,
        }
        usage2 = {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 200,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 300,
        }

        resp1 = _reasoning_response() | {"usage": usage1}
        resp2 = _text_response("Done.") | {"usage": usage2}

        agent.server_client.post.return_value = _dotjson_mock(resp1, resp2)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        u = res.json()["usage"]
        assert u["input_tokens"] == 110
        assert u["output_tokens"] == 220
        assert u["total_tokens"] == 330

    def test_string_input_converted_to_message(self) -> None:
        """String input is automatically wrapped in a user message."""
        agent, client = _make_agent_and_client()
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hi!"))

        res = client.post("/v1/responses", json={"input": "hello"})
        assert res.status_code == 200

        post_kwargs = agent.server_client.post.call_args_list[0].kwargs
        body = post_kwargs["json"]
        assert isinstance(body.input[0], NeMoGymEasyInputMessage)
        assert body.input[0].content == "hello"

    def test_incomplete_details_max_tokens_breaks(self) -> None:
        """incomplete_details with reason=max_output_tokens → loop exits."""
        agent, client = _make_agent_and_client()

        resp = _text_response("partial") | {"incomplete_details": {"reason": "max_output_tokens"}}
        agent.server_client.post.return_value = _dotjson_mock(resp)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert len(res.json()["output"]) == 1


# ---------------------------------------------------------------------------
# Tests: run() — top-level error handling
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_catches_exceptions(self) -> None:
        """run() wraps exceptions and returns reward=0."""
        agent, _ = _make_agent_and_client()

        agent.server_client.post = AsyncMock(side_effect=RuntimeError("catastrophic failure"))

        body = FinanceAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "test"}]},
            }
        )
        req = MagicMock()
        req.cookies = {}
        result = await agent.run(req, body)
        assert result.reward == 0.0
        assert result.response.id == "error"
