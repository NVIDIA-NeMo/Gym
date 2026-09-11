# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for NOOA v0.0.9 ExactMatchScorer semantics."""

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.nooa_capability.app import (
    NOOACapabilityResourcesServer,
    NOOACapabilityResourcesServerConfig,
    NOOACapabilityVerifyRequest,
    _parse_value,
    _values_equal,
)


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="capability-response",
        created_at=0,
        model="nooa",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="capability-message",
                content=[NeMoGymResponseOutputText(annotations=[], text=text)],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _server() -> NOOACapabilityResourcesServer:
    return NOOACapabilityResourcesServer(
        config=NOOACapabilityResourcesServerConfig(
            host="127.0.0.1",
            port=9001,
            entrypoint="app.py",
            name="nooa_capability",
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize(
    ("expected", "actual", "matches"),
    [
        (7, "7", True),
        (7, "7.0", True),
        (7, "7.", True),
        (7, "7.01", False),
        ("Positive", " positive ", True),
        ([1, 2.0], "[1.0, 2]", True),
        ({"answer": 7}, '{"answer": 7, "extra": true}', True),
        ({"answer": 7}, '{"answer": 8}', False),
    ],
)
def test_exact_match_parity_cases(expected: object, actual: object, matches: bool) -> None:
    assert _values_equal(_parse_value(expected), _parse_value(actual)) is matches


@pytest.mark.asyncio
async def test_verify_returns_binary_exact_match_reward() -> None:
    body = NOOACapabilityVerifyRequest(
        responses_create_params={"input": "calculate"},
        expected_result=7,
        response=_response("7.01"),
    )

    result = await _server().verify(body)

    assert result.reward == 0.0
    assert result.expected_result == 7
    assert result.actual_result == 7.01
    assert result.output_correct is False


@pytest.mark.asyncio
async def test_verify_extracts_common_answer_wrapper() -> None:
    body = NOOACapabilityVerifyRequest(
        responses_create_params={"input": "calculate"},
        expected_result=7,
        response=_response('{"result": 7, "explanation": "computed"}'),
    )

    result = await _server().verify(body)

    assert result.reward == 1.0
    assert result.actual_result == 7
    assert result.output_correct is True
