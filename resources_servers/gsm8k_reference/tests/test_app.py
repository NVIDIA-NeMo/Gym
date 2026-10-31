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


def test_reference_metrics_do_not_vote_using_a_different_extractor() -> None:
    tasks = [
        [
            {
                "strict_match": 0.0,
                "flexible_extract": 1.0,
                "strict_prediction": INVALID,
                "flexible_prediction": "42",
                "language_code": "hi",
            },
            {
                "strict_match": 0.0,
                "flexible_extract": 0.0,
                "strict_prediction": INVALID,
                "flexible_prediction": "43",
                "language_code": "hi",
            },
        ]
    ]
    metrics = _server().compute_metrics(tasks)
    assert metrics["pass@1/flexible_extract"] == 50.0
    assert metrics["pass@1/strict_match"] == 0.0
    assert not any("majority@" in key for key in metrics)


@pytest.mark.asyncio
async def test_only_final_message_is_scored_with_unicode_separators() -> None:
    request = _request("ಉತ್ತರ\u2028#### 1,234\u2029")
    response = request.response.model_dump()
    response["output"].insert(
        0,
        {
            "id": "reasoning",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Maybe #### 999"}],
        },
    )
    request.response = NeMoGymResponse.model_validate(response)
    result = await _server().verify(request)
    assert result.strict_prediction == result.flexible_prediction == "1,234"
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_empty_final_answer_scores_zero() -> None:
    result = await _server().verify(_request(""))
    assert result.reward == result.strict_match == result.flexible_extract == 0.0


def test_empty_metrics_and_key_metric_selection() -> None:
    assert _server().compute_metrics([]) == {}
    metrics = {
        "mean/input_tokens": 100,
        "mean/output_tokens": 250,
        "pass@1[avg-of-1]/strict_match": 75.0,
        "pass@1[avg-of-1]/flexible_extract": 80.0,
        "unrelated": 1.0,
    }
    assert _server().get_key_metrics(metrics) == {key: value for key, value in metrics.items() if key != "unrelated"}
