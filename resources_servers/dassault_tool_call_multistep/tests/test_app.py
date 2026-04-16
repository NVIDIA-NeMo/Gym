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
from resources_servers.tool_call_multistep.app import (
    ToolCallMultistepConfig,
    ToolCallMultistepServer,
    ToolCallMultistepVerifyRequest,
    _check_order_correct,
    _extract_json_array,
)


def _make_config():
    return ToolCallMultistepConfig(host="0.0.0.0", port=8080, entrypoint="", name="")


def _make_server():
    return ToolCallMultistepServer(config=_make_config(), server_client=MagicMock(spec=ServerClient))


def _make_response_with_text(text):
    return MagicMock(output=[], output_text=text)


def _make_empty_response():
    return MagicMock(output=[], output_text="")


def _make_responses_create_params():
    return MagicMock(input=[{"role": "user", "content": "test"}])


class TestSanity:
    def test_server_instantiation(self):
        server = _make_server()
        assert server is not None


class TestExtractJsonArray:
    def test_plain_json(self):
        text = '[{"function": "a", "step": 1}, {"function": "b", "step": 2}]'
        result = _extract_json_array(text)
        assert len(result) == 2
        assert result[0]["function"] == "a"

    def test_json_in_code_block(self):
        text = '```json\n[{"function": "a"}]\n```'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_json_with_surrounding_text(self):
        text = 'Here is my plan:\n[{"function": "a"}]\nDone.'
        result = _extract_json_array(text)
        assert len(result) == 1

    def test_invalid_json(self):
        result = _extract_json_array("not json at all")
        assert result is None

    def test_empty_text(self):
        result = _extract_json_array("")
        assert result is None

    def test_thinking_tags_stripped(self):
        text = '<think>reasoning here</think>[{"function": "a"}]'
        result = _extract_json_array(text)
        assert len(result) == 1


class TestCheckOrderCorrect:
    def test_correct_order(self):
        assert _check_order_correct(["a", "b", "c"], ["a", "b", "c"]) is True

    def test_wrong_order(self):
        assert _check_order_correct(["b", "a"], ["a", "b"]) is False

    def test_single_function(self):
        assert _check_order_correct(["a"], ["a"]) is True

    def test_empty_lists(self):
        assert _check_order_correct([], []) is True

    def test_partial_overlap_correct_order(self):
        assert _check_order_correct(["a", "c"], ["a", "b", "c"]) is True

    def test_partial_overlap_wrong_order(self):
        assert _check_order_correct(["c", "a"], ["a", "b", "c"]) is False


class TestVerify:
    @pytest.fixture
    def server(self):
        return _make_server()

    @pytest.mark.asyncio
    async def test_exact_match_reward_1(self, server):
        prediction = json.dumps(
            [
                {"function": "get_customer_orders", "params": {"customer_id": 12345}, "step": 1},
                {"function": "get_product_stock", "params": {"product_id": "<from_step_1>"}, "step": 2},
            ]
        )
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(prediction),
            verifier_metadata={
                "query_id": 1,
                "expected_sequence": [
                    {"function": "get_customer_orders", "params": {"customer_id": 12345}},
                    {"function": "get_product_stock", "params": {"product_id": "<from_step_1>"}},
                ],
                "difficulty": "easy",
            },
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.sequence_exact_match is True
        assert result.function_recall == 1.0
        assert result.function_precision == 1.0
        assert result.order_correct is True

    @pytest.mark.asyncio
    async def test_wrong_sequence_reward_0(self, server):
        prediction = json.dumps(
            [
                {"function": "get_product_stock", "params": {}, "step": 1},
            ]
        )
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(prediction),
            verifier_metadata={
                "query_id": 1,
                "expected_sequence": [
                    {"function": "get_customer_orders", "params": {}},
                    {"function": "get_product_stock", "params": {}},
                ],
                "difficulty": "easy",
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.sequence_exact_match is False
        assert result.function_recall == 0.5
        assert "get_customer_orders" in result.missing_functions

    @pytest.mark.asyncio
    async def test_correct_functions_wrong_order(self, server):
        prediction = json.dumps(
            [
                {"function": "get_product_stock", "params": {}, "step": 1},
                {"function": "get_customer_orders", "params": {}, "step": 2},
            ]
        )
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(prediction),
            verifier_metadata={
                "query_id": 1,
                "expected_sequence": [
                    {"function": "get_customer_orders", "params": {}},
                    {"function": "get_product_stock", "params": {}},
                ],
                "difficulty": "easy",
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_recall == 1.0
        assert result.order_correct is False

    @pytest.mark.asyncio
    async def test_empty_response(self, server):
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_empty_response(),
            verifier_metadata={
                "query_id": 1,
                "expected_sequence": [{"function": "a", "params": {}}],
                "difficulty": "easy",
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_recall == 0.0

    @pytest.mark.asyncio
    async def test_extra_functions(self, server):
        prediction = json.dumps(
            [
                {"function": "a", "params": {}, "step": 1},
                {"function": "b", "params": {}, "step": 2},
                {"function": "c", "params": {}, "step": 3},
            ]
        )
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(prediction),
            verifier_metadata={
                "query_id": 1,
                "expected_sequence": [
                    {"function": "a", "params": {}},
                    {"function": "b", "params": {}},
                ],
                "difficulty": "medium",
            },
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.function_recall == 1.0
        assert result.function_precision == pytest.approx(2.0 / 3.0)
        assert "c" in result.extra_functions

    @pytest.mark.asyncio
    async def test_no_verifier_metadata(self, server):
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_empty_response(),
            verifier_metadata=None,
        )
        result = await server.verify(body)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_difficulty_preserved(self, server):
        prediction = json.dumps([{"function": "a", "params": {}, "step": 1}])
        body = ToolCallMultistepVerifyRequest(
            responses_create_params=_make_responses_create_params(),
            response=_make_response_with_text(prediction),
            verifier_metadata={
                "query_id": 5,
                "expected_sequence": [{"function": "a", "params": {}}],
                "difficulty": "hard",
            },
        )
        result = await server.verify(body)
        assert result.difficulty == "hard"
        assert result.query_id == 5
