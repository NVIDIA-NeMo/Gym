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
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.tool_call_scaled.app import (
    ToolCallScaledConfig,
    ToolCallScaledServer,
    ToolCallScaledVerifyRequest,
    _compare_params,
    _extract_tool_call,
    _normalize_value,
)


def _make_config():
    return ToolCallScaledConfig(host="0.0.0.0", port=8080, entrypoint="", name="")


def _make_server():
    return ToolCallScaledServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))


def _make_response_with_tool_call(func_name, arguments):
    return MagicMock(
        output=[
            MagicMock(
                model_dump=lambda: {
                    "type": "function_call",
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                    "call_id": "call_1",
                }
            )
        ],
        output_text="",
    )


def _make_response_with_text(text):
    return MagicMock(output=[], output_text=text)


def _make_empty_response():
    return MagicMock(output=[], output_text="")


def _make_responses_create_params():
    return MagicMock(input=[{"role": "user", "content": "test"}], tools=[])


class TestSanity:
    def test_server_instantiation(self):
        server = _make_server()
        assert server is not None


class TestNormalizeValue:
    def test_none(self):
        assert _normalize_value(None) is None

    def test_boolean_true(self):
        assert _normalize_value("true") is True
        assert _normalize_value("yes") is True
        assert _normalize_value("1") is True

    def test_boolean_false(self):
        assert _normalize_value("false") is False
        assert _normalize_value("no") is False
        assert _normalize_value("0") is False

    def test_integer(self):
        assert _normalize_value("42") == 42

    def test_float(self):
        assert _normalize_value("3.14") == 3.14

    def test_string(self):
        assert _normalize_value("hello") == "hello"


class TestCompareParams:
    def test_empty_expected(self):
        result = _compare_params({}, {})
        assert result["total"] == 0

    def test_all_correct(self):
        result = _compare_params({"user_id": 123}, {"user_id": 123})
        assert result["total"] == 1
        assert result["correct"] == 1

    def test_type_flexible_match(self):
        result = _compare_params({"user_id": "123"}, {"user_id": 123})
        assert result["correct"] == 1

    def test_mismatch(self):
        result = _compare_params({"status": "wrong"}, {"status": "active"})
        assert result["correct"] == 0


class TestExtractToolCall:
    def test_function_call_item(self):
        items = [{"type": "function_call", "name": "get_user_orders", "arguments": '{"user_id": 123}'}]
        name, args = _extract_tool_call(items)
        assert name == "get_user_orders"
        assert args == {"user_id": 123}

    def test_no_function_call(self):
        items = [{"type": "message"}]
        name, args = _extract_tool_call(items)
        assert name is None


class TestVerify:
    @pytest.fixture
    def server(self):
        return _make_server()

    @pytest.mark.asyncio
    async def test_exact_match_reward_1(self, server):
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_user_orders", {"user_id": 12345, "limit": 100}),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {"user_id": 12345, "limit": 100},
                "confusion_candidates": ["get_order_history"],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.function_correct is True
        assert result.exact_match is True

    @pytest.mark.asyncio
    async def test_wrong_function_reward_0(self, server):
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_order_history", {}),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {"user_id": 12345},
                "confusion_candidates": ["get_order_history"],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_correct is False
        assert result.confused_with == "get_order_history"

    @pytest.mark.asyncio
    async def test_correct_function_wrong_params(self, server):
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_user_orders", {"user_id": 99999}),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {"user_id": 12345},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_correct is True
        assert result.exact_match is False

    @pytest.mark.asyncio
    async def test_params_not_evaluated_on_wrong_function(self, server):
        """Scaled eval only evaluates params when function is correct."""
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("wrong_func", {"user_id": 12345}),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {"user_id": 12345},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.params_total == 0
        assert result.params_correct == 0

    @pytest.mark.asyncio
    async def test_empty_response(self, server):
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_empty_response(),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_text_fallback_parsing(self, server):
        text_response = json.dumps({"function_name": "get_user_orders", "parameters": {"user_id": 12345}})
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(text_response),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {"user_id": 12345},
                "confusion_candidates": [],
                "function_count": 100,
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_function_correct_no_params(self, server):
        body = ToolCallScaledVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_user_orders", {}),
            verifier_metadata={
                "query_id": 1,
                "expected_function": "get_user_orders",
                "expected_params": {},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.exact_match is True
