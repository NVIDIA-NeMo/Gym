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
from resources_servers.tool_call_enterprise.app import (
    ToolCallEnterpriseConfig,
    ToolCallEnterpriseServer,
    ToolCallEnterpriseVerifyRequest,
    _evaluate_params,
    _extract_tool_call,
    _values_match,
)


def _make_config():
    return ToolCallEnterpriseConfig(host="0.0.0.0", port=8080, entrypoint="", name="")


def _make_server():
    return ToolCallEnterpriseServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))


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


class TestValuesMatch:
    def test_exact_match(self):
        assert _values_match("hello", "hello") is True

    def test_case_insensitive_match(self):
        assert _values_match("Hello", "hello") is True

    def test_type_coercion_match(self):
        assert _values_match(True, "true") is True

    def test_mismatch(self):
        assert _values_match("hello", "world") is False

    def test_none_values(self):
        assert _values_match(None, "hello") is False


class TestEvaluateParams:
    def test_empty_expected(self):
        result = _evaluate_params({}, {})
        assert result["total"] == 0
        assert result["correct"] == 0

    def test_all_correct(self):
        result = _evaluate_params({"key": "value"}, {"key": "value"})
        assert result["total"] == 1
        assert result["correct"] == 1

    def test_partial_correct(self):
        result = _evaluate_params(
            {"a": "1", "b": "wrong"},
            {"a": "1", "b": "right"},
        )
        assert result["total"] == 2
        assert result["correct"] == 1

    def test_missing_predicted_param(self):
        result = _evaluate_params({}, {"key": "value"})
        assert result["total"] == 1
        assert result["correct"] == 0


class TestExtractToolCall:
    def test_function_call_item(self):
        items = [{"type": "function_call", "name": "test_func", "arguments": '{"a": 1}'}]
        name, args = _extract_tool_call(items)
        assert name == "test_func"
        assert args == {"a": 1}

    def test_no_function_call(self):
        items = [{"type": "message", "content": "hello"}]
        name, args = _extract_tool_call(items)
        assert name is None
        assert args == {}

    def test_invalid_json_arguments(self):
        items = [{"type": "function_call", "name": "func", "arguments": "not json"}]
        name, args = _extract_tool_call(items)
        assert name == "func"
        assert args == {}


class TestVerify:
    @pytest.fixture
    def server(self):
        return _make_server()

    @pytest.mark.asyncio
    async def test_exact_match_reward_1(self, server):
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_requirement_details", {"major_version": "R2025x"}),
            verifier_metadata={
                "expected_function": "get_requirement_details",
                "expected_params": {"major_version": "R2025x"},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.function_correct is True
        assert result.exact_match is True
        assert result.params_correct == 1

    @pytest.mark.asyncio
    async def test_wrong_function_reward_0(self, server):
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("wrong_function", {}),
            verifier_metadata={
                "expected_function": "get_requirement_details",
                "expected_params": {"major_version": "R2025x"},
                "confusion_candidates": ["wrong_function"],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_correct is False
        assert result.confused_with == "wrong_function"

    @pytest.mark.asyncio
    async def test_correct_function_wrong_params(self, server):
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_requirement_details", {"major_version": "R2024x"}),
            verifier_metadata={
                "expected_function": "get_requirement_details",
                "expected_params": {"major_version": "R2025x"},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_correct is True
        assert result.exact_match is False
        assert result.params_correct == 0

    @pytest.mark.asyncio
    async def test_empty_response(self, server):
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_empty_response(),
            verifier_metadata={
                "expected_function": "get_requirement_details",
                "expected_params": {},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_correct is False

    @pytest.mark.asyncio
    async def test_text_fallback_parsing(self, server):
        text_response = json.dumps({"function": "get_requirement_details", "parameters": {"major_version": "R2025x"}})
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(text_response),
            verifier_metadata={
                "expected_function": "get_requirement_details",
                "expected_params": {"major_version": "R2025x"},
                "confusion_candidates": [],
                "function_count": 100,
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.function_correct is True

    @pytest.mark.asyncio
    async def test_no_verifier_metadata(self, server):
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_empty_response(),
            verifier_metadata=None,
        )
        result = await server.verify(body)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_function_correct_no_params(self, server):
        body = ToolCallEnterpriseVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_tool_call("get_requirement_details", {}),
            verifier_metadata={
                "expected_function": "get_requirement_details",
                "expected_params": {},
                "confusion_candidates": [],
                "function_count": 20,
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.exact_match is True
