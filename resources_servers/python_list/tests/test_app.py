# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from app import PythonListResourcesServer, PythonListResourcesServerConfig, PythonListVerifyRequest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [
                    {
                        "annotations": [],
                        "text": text,
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


def _make_server() -> PythonListResourcesServer:
    return PythonListResourcesServer(
        config=PythonListResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


def _make_request(
    text: str,
    expected_answer: str,
    extraction_mode: str = "final_answer",
) -> PythonListVerifyRequest:
    return PythonListVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "What is the answer?"}],
        },
        response=_make_response(text),
        expected_answer=expected_answer,
        extraction_mode=extraction_mode,
    )


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    async def test_tuple_final_answer_correct(self) -> None:
        server = _make_server()
        req = _make_request("Reasoning...\nFinal answer: (4, 3)", "(4, 3)")
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "(4, 3)"
        assert result.parsed_prediction == [4, 3]

    async def test_json_list_correct(self) -> None:
        server = _make_server()
        req = _make_request('Final answer: ["red", "blue"]', "['red', 'blue']")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_partial_reward_for_wrong_item(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: [1, 9, 3]", "[1, 2, 3]")
        result = await server.verify(req)
        assert result.reward == 2 / 3

    async def test_extra_items_penalized(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: [1, 2, 3]", "[1, 2]")
        result = await server.verify(req)
        assert result.reward == 2 / 3

    async def test_string_items_use_word_f1(self) -> None:
        server = _make_server()
        req = _make_request(
            'Final answer: ["how many omelet are in canada"]',
            '["HOW MANY OMELET ARE IN CANADA?"]',
        )
        result = await server.verify(req)
        assert result.reward > 0.9

    async def test_boxed_extraction(self) -> None:
        server = _make_server()
        req = _make_request("The answer is \\boxed{(1, 5)}.", "(1, 5)", extraction_mode="boxed")
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "(1, 5)"

    async def test_auto_extracts_list_from_sentence(self) -> None:
        server = _make_server()
        req = _make_request("After checking, the answer is [30, 45, 60, 75, 90].", "[30,45,60,75,90]", "auto")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_unparseable_prediction_returns_zero(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: row four column three", "(4, 3)")
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.parsed_prediction is None

    async def test_nested_lists(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: [[1, 2], [3, 4]]", "[[1, 2], [3, 4]]")
        result = await server.verify(req)
        assert result.reward == 1.0
