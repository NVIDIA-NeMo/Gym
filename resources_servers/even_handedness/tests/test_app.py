# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier tests for Political Even-handedness."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.even_handedness.app import (
    EvenHandednessConfig,
    EvenHandednessServer,
    EvenHandednessVerifyRequest,
    JudgeOutcome,
)


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )


def _config(scoring_mode: str = "probability") -> EvenHandednessConfig:
    return EvenHandednessConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="even_handedness",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        scoring_mode=scoring_mode,
    )


def _completion() -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "completion",
            "created": 0,
            "model": "judge",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "(C)"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "(",
                                "bytes": [40],
                                "logprob": -0.01,
                                "top_logprobs": [],
                            },
                            {
                                "token": "C",
                                "bytes": [67],
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "A", "bytes": [65], "logprob": -3.0},
                                    {"token": "B", "bytes": [66], "logprob": -2.0},
                                ],
                            },
                        ]
                    },
                }
            ],
        }
    )


def test_probability_extraction_uses_allowed_option_position() -> None:
    probabilities = EvenHandednessServer._probabilities_from_completion(_completion(), "ABC")
    assert probabilities is not None
    assert probabilities["C"] == pytest.approx(0.904837)
    assert probabilities["C"] > probabilities["B"] > probabilities["A"]


@pytest.mark.parametrize(
    ("text", "choices", "expected"),
    [("(C)", "ABC", "C"), ("Answer: 4", "12345", "4"), ("**(2)**", "12345", "2")],
)
def test_discrete_choice_parser(text: str, choices: str, expected: str) -> None:
    assert EvenHandednessServer._choice_from_text(text, choices) == expected


async def test_probability_mode_requires_logprob_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    completion = _completion()
    completion.choices[0].logprobs = None
    monkeypatch.setattr("resources_servers.even_handedness.app.call_judge", AsyncMock(return_value=completion))
    server = EvenHandednessServer(config=_config(), server_client=MagicMock(spec=ServerClient))
    with pytest.raises(JudgeError, match="Probability scoring requires"):
        await server._judge("prompt", "ABC")


async def test_discrete_mode_is_explicit_one_hot(monkeypatch: pytest.MonkeyPatch) -> None:
    completion = _completion()
    completion.choices[0].logprobs = None
    monkeypatch.setattr("resources_servers.even_handedness.app.call_judge", AsyncMock(return_value=completion))
    server = EvenHandednessServer(config=_config("discrete"), server_client=MagicMock(spec=ServerClient))
    outcome = await server._judge("prompt", "ABC")
    assert outcome.label == "C"
    assert outcome.probabilities == {"A": 0.0, "B": 0.0, "C": 1.0}


async def test_gpt_judge_transport_uses_completion_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    completion = _completion()
    completion.choices[0].logprobs = None
    judge = AsyncMock(return_value=completion)
    monkeypatch.setattr("resources_servers.even_handedness.app.call_judge", judge)
    config = _config("discrete").model_copy(
        update={
            "judge_max_tokens": None,
            "judge_max_completion_tokens": 1024,
            "judge_reasoning_effort": "minimal",
            "judge_temperature": None,
        }
    )
    server = EvenHandednessServer(config=config, server_client=MagicMock(spec=ServerClient))

    outcome = await server._judge("prompt", "ABC")

    params = judge.await_args.kwargs["json"]
    assert outcome.label == "C"
    assert params.max_tokens is None
    assert params.max_completion_tokens == 1024
    assert params.reasoning_effort == "minimal"
    assert params.temperature is None
    assert "max_tokens" not in params.model_fields_set
    assert "temperature" not in params.model_fields_set


async def test_verify_applies_thresholds_to_all_five_judgments() -> None:
    server = EvenHandednessServer(config=_config(), server_client=MagicMock(spec=ServerClient))
    server._judge = AsyncMock(
        side_effect=[
            JudgeOutcome("(C)", "C", {"A": 0.05, "B": 0.05, "C": 0.9}),
            JudgeOutcome("(1)", "1", {"1": 0.8, "2": 0.1, "3": 0.05, "4": 0.03, "5": 0.02}),
            JudgeOutcome("(5)", "5", {"1": 0.05, "2": 0.05, "3": 0.05, "4": 0.1, "5": 0.75}),
            JudgeOutcome("(4)", "4", {"1": 0.05, "2": 0.05, "3": 0.05, "4": 0.75, "5": 0.1}),
            JudgeOutcome("(2)", "2", {"1": 0.1, "2": 0.7, "3": 0.1, "4": 0.05, "5": 0.05}),
        ]
    )
    result = await server.verify(
        EvenHandednessVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_response("response A"),
            response_b=_response("response B"),
            prompt_a="prompt A",
            prompt_b="prompt B",
            prompt_a_group="group A",
            prompt_b_group="group B",
        )
    )
    assert result.reward == 1.0
    assert result.even_handedness_probability == pytest.approx(0.9)
    assert result.refusal == pytest.approx(0.5)
    assert result.opposing_perspectives == pytest.approx(0.5)
    assert result.invalid_judgments == 0


async def test_verify_rejects_missing_policy_response() -> None:
    server = EvenHandednessServer(config=_config(), server_client=MagicMock(spec=ServerClient))
    with pytest.raises(JudgeError, match="Both policy responses"):
        await server.verify(
            EvenHandednessVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_response(""),
                response_b=_response("response B"),
                prompt_a="prompt A",
                prompt_b="prompt B",
                prompt_a_group="group A",
                prompt_b_group="group B",
            )
        )


async def test_verify_fails_closed_on_an_unknown_judge_label() -> None:
    server = EvenHandednessServer(config=_config("discrete"), server_client=MagicMock(spec=ServerClient))
    ordinal = {str(i): float(i == 1) for i in range(1, 6)}
    server._judge = AsyncMock(
        side_effect=[
            JudgeOutcome("Answer: same", "unknown", {"A": 0.0, "B": 0.0, "C": 0.0}),
            *[JudgeOutcome("(1)", "1", ordinal) for _ in range(4)],
        ]
    )

    with pytest.raises(JudgeError, match="unparseable labels"):
        await server.verify(
            EvenHandednessVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=_response("response A"),
                response_b=_response("response B"),
                prompt_a="prompt A",
                prompt_b="prompt B",
                prompt_a_group="group A",
                prompt_b_group="group B",
            )
        )


async def test_verify_recovers_second_response_from_primary_metadata() -> None:
    server = EvenHandednessServer(config=_config("discrete"), server_client=MagicMock(spec=ServerClient))
    server._judge = AsyncMock(
        side_effect=[
            JudgeOutcome("(C)", "C", {"A": 0.0, "B": 0.0, "C": 1.0}),
            *[JudgeOutcome("(1)", "1", {"1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0}) for _ in range(4)],
        ]
    )
    primary = _response("response A")
    primary.metadata = {server._RESPONSE_B_METADATA_KEY: json.dumps(_response("response B").model_dump(mode="json"))}
    result = await server.verify(
        EvenHandednessVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=primary,
            prompt_a="prompt A",
            prompt_b="prompt B",
            prompt_a_group="group A",
            prompt_b_group="group B",
        )
    )
    assert result.response_b_text == "response B"
    assert result.reward == 1.0
