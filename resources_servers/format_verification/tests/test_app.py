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
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from resources_servers.format_verification.app import (
    FormatVerificationResourcesServer,
    FormatVerificationResourcesServerConfig,
    FormatVerificationVerifyRequest,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
}

DS3_EXAMPLE_FPATH = Path(__file__).parent.parent / "data" / "ds3_citation_format_example.jsonl"


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

    @pytest.mark.parametrize(
        "expected,other",
        [
            ("[ref:2]", "[ref:3]"),
            ("<ref:2>", "<ref:3>"),
            ("{ref:2}", "{ref:3}"),
            ("[source:2]", "[source:10]"),
            ("[2]", "[13]"),
            ("[web:5]", "[web:1]"),
            ("<<2>>", "<<4>>"),
            ("(Part 2)", "(Part 1)"),
            ("(ref 2)", "(ref 4)"),
        ],
    )
    async def test_spurious_marker_with_literal_patterns(self, expected: str, other: str) -> None:
        # The ds3 data sets patterns to the escaped expected markers, which alone never match another marker.
        server = _make_server()
        verifier = {"type": "string_match", "expected_markers": [expected], "patterns": [re.escape(expected)]}

        result = await server.verify(_make_verify_request(f"The answer {expected} {other}.", verifier))
        assert result.reward == 0.0
        assert result.match_details["spurious"] == [other]

        result = await server.verify(_make_verify_request(f"The answer {expected}.", verifier))
        assert result.reward == 1.0
        assert result.match_details["spurious"] == []

    async def test_spurious_marker_reported_once_when_patterns_overlap(self) -> None:
        server = _make_server()
        text = "According to [source:1] and [source:2], the answer is clear."
        result = await server.verify(
            _make_verify_request(
                text,
                {
                    "type": "string_match",
                    "expected_markers": ["[source:1]"],
                    "patterns": [r"\[source:[0-9]+\]"],
                },
            )
        )
        assert result.reward == 0.0
        assert result.match_details["spurious"] == ["[source:2]"]

    async def test_ds3_example_rejects_citing_every_part(self) -> None:
        server = _make_server()
        rows = [json.loads(line) for line in DS3_EXAMPLE_FPATH.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            verifier = row["verifier"]
            expected = verifier["expected_markers"]
            prompt = row["responses_create_params"]["input"][0]["content"]
            num_parts = len(set(re.findall(r"\[Part (\d+)\]", prompt)))
            assert num_parts > len(expected)
            style = re.sub(r"\d+", "{}", expected[0])

            result = await server.verify(_make_verify_request("Answer. " + " ".join(expected), verifier))
            assert result.reward == 1.0

            cite_all = "Answer. " + " ".join(style.format(n) for n in range(1, num_parts + 1))
            result = await server.verify(_make_verify_request(cite_all, verifier))
            assert result.reward == 0.0
            assert len(result.match_details["spurious"]) == num_parts - len(expected)


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
