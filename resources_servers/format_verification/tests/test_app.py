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
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.format_verification.app import (
    FormatVerificationResourcesServer,
    FormatVerificationResourcesServerConfig,
    FormatVerificationVerifyRequest,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
}


def _make_server() -> FormatVerificationResourcesServer:
    config = FormatVerificationResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return FormatVerificationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(text: str, verifier: dict) -> FormatVerificationVerifyRequest:
    return FormatVerificationVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(text),
        verifier=verifier,
    )


class TestVerifyRegex:
    async def test_lines_meeting_threshold(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "1. First\n2. Second\n3. Third",
                {"type": "regex", "verify_regex": [r"^\d+\."], "verify_min_matches": 2},
            )
        )
        assert result.reward == 1.0
        assert result.match_details["matching_lines"] == 3
        assert result.match_details["passed"] is True

    async def test_lines_below_threshold(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "1. First\n2. Second",
                {"type": "regex", "verify_regex": [r"^\d+\."], "verify_min_matches": 5},
            )
        )
        assert result.reward == 0.0
        assert result.match_details["matching_lines"] == 2
        assert result.match_details["passed"] is False

    async def test_no_patterns(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "anything here",
                {"type": "regex", "verify_regex": [], "verify_min_matches": 1},
            )
        )
        assert result.reward == 0.0
        assert result.match_details["matching_lines"] == 0

    async def test_multiple_patterns_line_counts_once(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "1. First item",
                {"type": "regex", "verify_regex": [r"^\d", r"First"], "verify_min_matches": 1},
            )
        )
        assert result.reward == 1.0
        assert result.match_details["matching_lines"] == 1

    async def test_empty_text(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "",
                {"type": "regex", "verify_regex": [r"hello"], "verify_min_matches": 1},
            )
        )
        assert result.reward == 0.0
        assert result.match_details["matching_lines"] == 0

    async def test_min_matches_zero(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "no match here",
                {"type": "regex", "verify_regex": [r"xyz"], "verify_min_matches": 0},
            )
        )
        assert result.reward == 1.0
        assert result.match_details["passed"] is True

    async def test_inline_prose_type(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "1. First line\n2. Second line",
                {"type": "inline_prose", "verify_regex": [r"^\d+\."], "verify_min_matches": 1},
            )
        )
        assert result.reward == 1.0


class TestVerifyStringMatch:
    async def test_all_present_no_spurious(self) -> None:
        server = _make_server()
        text = "According to [source:1] and [source:2], the answer is clear."
        result = await server.verify(
            _make_verify_request(
                text,
                {
                    "type": "string_match",
                    "expected_markers": ["[source:1]", "[source:2]"],
                    "patterns": [r"\[source:\d+\]"],
                },
            )
        )
        assert result.reward == 1.0
        assert result.match_details["missing"] == []
        assert result.match_details["spurious"] == []
        assert result.match_details["passed"] is True

    async def test_missing_marker(self) -> None:
        server = _make_server()
        text = "According to [source:1], the answer is clear."
        result = await server.verify(
            _make_verify_request(
                text,
                {
                    "type": "string_match",
                    "expected_markers": ["[source:1]", "[source:2]"],
                    "patterns": [],
                },
            )
        )
        assert result.reward == 0.0
        assert result.match_details["missing"] == ["[source:2]"]
        assert result.match_details["passed"] is False

    async def test_spurious_marker(self) -> None:
        server = _make_server()
        text = "According to [source:1] and [source:2], the answer is clear."
        result = await server.verify(
            _make_verify_request(
                text,
                {
                    "type": "string_match",
                    "expected_markers": ["[source:1]"],
                    "patterns": [r"\[source:\d+\]"],
                },
            )
        )
        assert result.reward == 0.0
        assert "[source:2]" in result.match_details["spurious"]
        assert result.match_details["passed"] is False

    async def test_both_missing_and_spurious(self) -> None:
        server = _make_server()
        text = "According to [source:1] and [source:2], the answer is clear."
        result = await server.verify(
            _make_verify_request(
                text,
                {
                    "type": "string_match",
                    "expected_markers": ["[source:1]", "[source:3]"],
                    "patterns": [r"\[source:\d+\]"],
                },
            )
        )
        assert result.reward == 0.0
        assert "[source:3]" in result.match_details["missing"]
        assert "[source:2]" in result.match_details["spurious"]

    async def test_no_patterns_skips_spurious(self) -> None:
        server = _make_server()
        text = "According to [source:1] and [source:2], the answer is clear."
        result = await server.verify(
            _make_verify_request(
                text,
                {
                    "type": "string_match",
                    "expected_markers": ["[source:1]"],
                    "patterns": [],
                },
            )
        )
        assert result.reward == 1.0
        assert result.match_details["spurious"] == []

    async def test_empty_expected(self) -> None:
        server = _make_server()
        result = await server.verify(
            _make_verify_request(
                "anything here",
                {"type": "string_match", "expected_markers": [], "patterns": []},
            )
        )
        assert result.reward == 1.0
        assert result.match_details["passed"] is True


class TestVerifyRouting:
    async def test_unknown_type_raises(self) -> None:
        server = _make_server()
        with pytest.raises(NotImplementedError, match="unknown"):
            await server.verify(_make_verify_request("text", {"type": "unknown"}))

    async def test_extract_multiple_messages(self) -> None:
        server = _make_server()
        response = NeMoGymResponse(
            id="resp_multi",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "1. First\n", "annotations": []}],
                },
                {
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "2. Second\n", "annotations": []}],
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        request = FormatVerificationVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=response,
            verifier={"type": "regex", "verify_regex": [r"^\d+\."], "verify_min_matches": 2},
        )
        result = await server.verify(request)
        assert result.reward == 1.0
        assert result.match_details["matching_lines"] == 2

    async def test_extract_skips_non_message(self) -> None:
        server = _make_server()
        response = NeMoGymResponse(
            id="resp_mixed",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "some_fn",
                    "arguments": "{}",
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        request = FormatVerificationVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=response,
            verifier={"type": "string_match", "expected_markers": ["Hello"], "patterns": []},
        )
        result = await server.verify(request)
        assert result.reward == 1.0
