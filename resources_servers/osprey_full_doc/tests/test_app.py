# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.osprey_full_doc.app import (
    OspreyFullDocResourcesServer,
    OspreyFullDocResourcesServerConfig,
    OspreyFullDocVerifyRequest,
)


def _make_response(*, output=None, error=None):
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output or [],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        error=error,
    )


def _make_server() -> OspreyFullDocResourcesServer:
    return OspreyFullDocResourcesServer(
        config=OspreyFullDocResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


def _make_request(response, ground_truth):
    return OspreyFullDocVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "Extract the line item."}],
            "tools": [
                {
                    "type": "function",
                    "name": "extract",
                    "description": "Extract the line item.",
                    "parameters": {"type": "object", "properties": {"extraction": {"type": "object"}}},
                    "strict": False,
                }
            ],
        },
        response=response,
        doc_name="doc",
        line_item_name="line_item",
        ground_truth=ground_truth,
    )


class TestOspreyFullDocApp:
    def test_sanity(self) -> None:
        _make_server()

    async def test_verify_exact_match(self) -> None:
        ground_truth = {"type": "LIMIT", "limit": {"amount": {"currencyCode": "USD", "units": 650000, "nanos": 0}}}
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": ground_truth}),
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, ground_truth))
        assert result.reward == 1.0
        assert result.is_correct is True
        assert result.wrong_prediction_type is None

    async def test_verify_optional_empty_field_is_cleaned(self) -> None:
        ground_truth = {"type": "LIMIT"}
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": {"type": "LIMIT", "extra": ""}}),
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, ground_truth))
        assert result.reward == 1.0
        assert result.clean_prediction == {"type": "LIMIT"}

    async def test_verify_scalar_ground_truth_exact_match(self) -> None:
        ground_truth = "CLAIMS_MADE_AND_REPORTED"
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": ground_truth}),
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, ground_truth))
        assert result.reward == 1.0
        assert result.is_correct is True
        assert result.wrong_prediction_type is None

    async def test_verify_wrapper_message_before_tool_call_is_ignored(self) -> None:
        ground_truth = {"type": "LIMIT"}
        response = _make_response(
            output=[
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Working on it.", "annotations": []}],
                    "type": "message",
                    "status": "completed",
                },
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": ground_truth}),
                    "type": "function_call",
                },
            ]
        )

        result = await _make_server().verify(_make_request(response, ground_truth))
        assert result.reward == 1.0
        assert result.is_correct is True
        assert result.wrong_prediction_type is None

    async def test_verify_false_positive(self) -> None:
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": {"type": "LIMIT"}}),
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, None))
        assert result.reward == 0.0
        assert result.wrong_prediction_type == "False positive"

    async def test_verify_false_negative(self) -> None:
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": None}),
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, {"type": "LIMIT"}))
        assert result.reward == 0.0
        assert result.wrong_prediction_type == "False negative"

    async def test_verify_incorrect_value(self) -> None:
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": json.dumps({"extraction": {"type": "OTHER"}}),
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, {"type": "LIMIT"}))
        assert result.reward == 0.0
        assert result.wrong_prediction_type == "Incorrect value"

    async def test_verify_malformed_function_call_arguments_is_extraction_error(self) -> None:
        response = _make_response(
            output=[
                {
                    "call_id": "call_1",
                    "name": "extract",
                    "arguments": '{"extraction": ',
                    "type": "function_call",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, {"type": "LIMIT"}))
        assert result.reward == 0.0
        assert result.wrong_prediction_type == "Extraction error"

    async def test_verify_extraction_error(self) -> None:
        response = _make_response(
            output=[
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "No tool call", "annotations": []}],
                    "type": "message",
                    "status": "completed",
                }
            ]
        )

        result = await _make_server().verify(_make_request(response, {"type": "LIMIT"}))
        assert result.reward == 0.0
        assert result.wrong_prediction_type == "Extraction error"

    async def test_verify_api_error(self) -> None:
        result = await _make_server().verify(_make_request(None, {"type": "LIMIT"}))
        assert result.reward == 0.0
        assert result.wrong_prediction_type == "API error"
