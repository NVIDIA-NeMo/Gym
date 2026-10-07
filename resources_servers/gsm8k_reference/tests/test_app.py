# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.gsm8k_reference.app import (
    INVALID,
    GSM8KReferenceConfig,
    GSM8KReferenceResourcesServer,
    GSM8KReferenceVerifyRequest,
    exact_match,
    flexible_extract,
    strict_extract,
)


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test",
        object="response",
        output=[
            {
                "id": "message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _server() -> GSM8KReferenceResourcesServer:
    config = GSM8KReferenceConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return GSM8KReferenceResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(generation: str, expected_answer: str = "reasoning #### 1234") -> GSM8KReferenceVerifyRequest:
    return GSM8KReferenceVerifyRequest(
        responses_create_params={"input": "question"},
        response=_response(generation),
        expected_answer=expected_answer,
        language_code="hi",
    )


@pytest.mark.parametrize(
    ("generation", "strict", "flexible"),
    [
        ("work #### 1,234.", "1,234.", "1,234."),
        ("work gives 12, then 1,234", INVALID, "1,234"),
        ("no number", INVALID, INVALID),
        ("#### 12 then #### 34", "12", "34"),
    ],
)
def test_reference_extractors(generation: str, strict: str, flexible: str) -> None:
    assert strict_extract(generation) == strict
    assert flexible_extract(generation) == flexible


def test_exact_match_applies_reference_ignores() -> None:
    assert exact_match("$1,234.", "worked solution #### 1,234") == 1.0
    assert exact_match("1235", "worked solution #### 1,234") == 0.0


@pytest.mark.asyncio
async def test_verify_reports_both_reference_metrics() -> None:
    result = await _server().verify(_request("first 12, then 1,234"))
    assert result.reward == 0.0
    assert result.strict_match == 0.0
    assert result.flexible_extract == 1.0
    assert result.strict_prediction == INVALID
    assert result.flexible_prediction == "1,234"


@pytest.mark.asyncio
async def test_strict_match_is_reward() -> None:
    result = await _server().verify(_request("calculation #### 1,234."))
    assert result.reward == 1.0
    assert result.strict_match == 1.0
    assert result.flexible_extract == 1.0


def test_compute_metrics_includes_language_breakdown() -> None:
    tasks = [
        [{"strict_match": 1.0, "flexible_extract": 1.0, "strict_prediction": "1", "language_code": "hi"}],
        [{"strict_match": 0.0, "flexible_extract": 1.0, "strict_prediction": INVALID, "language_code": "ta"}],
    ]
    metrics = _server().compute_metrics(tasks)
    assert metrics["pass@1/strict_match"] == 50.0
    assert metrics["pass@1/flexible_extract"] == 100.0
    assert metrics["hi/pass@1/strict_match"] == 100.0
    assert metrics["ta/pass@1/strict_match"] == 0.0
