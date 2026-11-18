# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for exercising the question-bank verifier with fixed completions."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from resources_servers.question_bank_judge.app import (
    QuestionBankJudgeConfig,
    QuestionBankJudgeResourcesServer,
    QuestionBankJudgeVerifyRequest,
)


SERVER_DIR = Path(__file__).resolve().parents[1]


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _reasoning(text: str = "scratch work") -> NeMoGymResponseReasoningItem:
    return NeMoGymResponseReasoningItem(id="rs", summary=[NeMoGymSummary(text=text, type="summary_text")])


def _response(*output, truncated: bool = False) -> NeMoGymResponse:
    extra = {"incomplete_details": {"reason": "max_output_tokens"}} if truncated else {}
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="m",
        object="response",
        output=list(output),
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        **extra,
    )


def _split(answer: str, **kwargs) -> NeMoGymResponse:
    """A response as our model server returns it: reasoning item, then the answer."""
    return _response(_reasoning(), _msg(answer), **kwargs)


def _request(response, expected="2", **row):
    return QuestionBankJudgeVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": row.get("question", "What is one plus one?")}]
        ),
        expected_answer=expected,
        response=response,
        **row,
    )


def server(**overrides) -> QuestionBankJudgeResourcesServer:
    config = {
        "host": "127.0.0.1",
        "port": 8080,
        "entrypoint": "app.py",
        "judge_model_server": {"name": "fixed_judge", "type": "responses_api_models"},
        "judge_model_revision": "test-model-revision",
        "judge_responses_create_params": {"input": [], "temperature": 0},
        **overrides,
    }
    return QuestionBankJudgeResourcesServer(
        config=QuestionBankJudgeConfig(**config), server_client=MagicMock(spec=ServerClient)
    )


def judge(monkeypatch, text=None, *, verdict="correct", status="completed"):
    payload = {"verdict": verdict, "reason_code": "reviewed", "evidence": "Submitted work meets the obligation."}
    response = _response(_msg(text if text is not None else json.dumps(payload)))
    response.status = status
    mock = AsyncMock(return_value=response)
    monkeypatch.setattr("resources_servers.question_bank_judge.app.call_judge", mock)
    return mock, response
