# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
import asyncio
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.puzzles.app import (
    PuzzlesResourcesServer,
    PuzzlesResourcesServerConfig,
    PuzzlesVerifyRequest,
)


def _make_server() -> PuzzlesResourcesServer:
    config = PuzzlesResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return PuzzlesResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(
    response_content: str, ground_truth, verification_type: str, request_id: int = 1
) -> PuzzlesVerifyRequest:
    response = NeMoGymResponse(
        id=f"resp_test_{request_id}",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": f"msg_test_{request_id}",
                "content": [
                    {
                        "annotations": [],
                        "text": response_content,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return PuzzlesVerifyRequest(
        responses_create_params={"input": []},
        response=response,
        ground_truth=ground_truth,
        verification_type=verification_type,
    )


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    def test_boxed_positive(self) -> None:
        req = _make_request("The answer is \\boxed{42}.", "42", "boxed")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.verification_failed is False
        assert result.verification_type == "boxed"

    def test_boxed_negative(self) -> None:
        req = _make_request("The answer is \\boxed{17}.", "42", "boxed")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_boxed_int_alias(self) -> None:
        req = _make_request("Final: \\boxed{7}", 7, "boxed_int")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_json_block_positive(self) -> None:
        req = _make_request(
            'My reasoning... ```json\n{"answer": "blue"}\n```',
            {"answer": "blue"},
            "json_block",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_json_block_with_solution_key_extracts_inner(self) -> None:
        req = _make_request(
            '```json\n{"solution": {"answer": "yes"}}\n```',
            {"answer": "yes"},
            "json_block",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_json_block_negative(self) -> None:
        req = _make_request(
            '```json\n{"answer": "red"}\n```',
            {"answer": "blue"},
            "json_block",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_dict_match_positive(self) -> None:
        gt = {"alice": {"color": "blue", "pet": "dog"}}
        req = _make_request(
            '```json\n{"solution": {"alice": {"color": "Blue", "pet": "Dog"}}}\n```',
            gt,
            "dict_match",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_dict_match_negative_wrong_value(self) -> None:
        gt = {"alice": {"color": "blue"}}
        req = _make_request(
            '```json\n{"solution": {"alice": {"color": "red"}}}\n```',
            gt,
            "dict_match",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_dict_match_no_solution_key_returns_zero(self) -> None:
        gt = {"alice": {"color": "blue"}}
        req = _make_request(
            '```json\n{"alice": {"color": "blue"}}\n```',
            gt,
            "dict_match",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_double_brackets_positive(self) -> None:
        req = _make_request("Result: [[yes]]", "yes", "double_brackets")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_python_block_literal_list(self) -> None:
        req = _make_request(
            "```python\n[[1, 2], [3, 4]]\n```", [[1, 2], [3, 4]], "python_block"
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_python_block_arithmetic_expression(self) -> None:
        req = _make_request("```python\n2 + 3\n```", 5, "python_block")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_bracket_list_positive(self) -> None:
        req = _make_request("Schedule: [45, 3]", [45, 3], "bracket_list")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_python_2d_list_positive(self) -> None:
        req = _make_request(
            "Solution: [[1, 2], [3, 4]]", [[1, 2], [3, 4]], "python_2d_list"
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_plain_number_positive(self) -> None:
        req = _make_request("After computing, the answer is 42.", 42, "plain_number")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_plain_number_no_number(self) -> None:
        req = _make_request("I cannot compute this.", 42, "plain_number")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0

    def test_tuple_list_order_insensitive(self) -> None:
        req = _make_request(
            "Pairs: [(2, 3), (0, 1)]", [(0, 1), (2, 3)], "tuple_list"
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_bold_yes_no_positive(self) -> None:
        req = _make_request("The answer is **yes**", "yes", "bold_yes_no")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_plain_sequence_positive(self) -> None:
        req = _make_request("Result:\n([{}])", "([{}])", "plain_sequence")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_plain_string_substring_match(self) -> None:
        req = _make_request("The capital is PARIS.", "paris", "plain_string")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_think_tag_stripped_before_verification(self) -> None:
        req = _make_request(
            "<think>let me reason about \\boxed{17}</think>\nFinal: \\boxed{42}",
            "42",
            "boxed",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_tag_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>incomplete reasoning with \\boxed{42}",
            "42",
            "boxed",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_unknown_verification_type_falls_back_to_plain_string(self) -> None:
        req = _make_request(
            "The answer is paris.", "paris", "definitely_not_a_real_type"
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_legacy_json_response_alias(self) -> None:
        req = _make_request(
            '```json\n{"result": [{"answer": "blue"}]}\n```',
            {"result": [{"answer": "blue"}]},
            "json_response",
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_legacy_exact_match_alias(self) -> None:
        req = _make_request(
            "The capital is Paris.", "paris", "exact_match"
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
