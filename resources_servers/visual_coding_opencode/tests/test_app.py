# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.visual_coding_opencode.app import (
    JUDGE_SYSTEM,
    VisualCodingOpenCodeConfig,
    VisualCodingOpenCodeServer,
    VisualCodingOpenCodeVerifyRequest,
    parse_verdict,
)


def message(text: str) -> dict:
    return {
        "id": "m",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def request(*texts: str, expected: str = "(4,7)") -> VisualCodingOpenCodeVerifyRequest:
    response = NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [message(text) for text in texts],
        }
    )
    return VisualCodingOpenCodeVerifyRequest(
        responses_create_params={"input": "q"}, response=response, expected_answer=expected, question="Which cell?"
    )


@pytest.fixture
def server() -> VisualCodingOpenCodeServer:
    config = VisualCodingOpenCodeConfig(
        name="visual_coding_opencode",
        host="localhost",
        port=1,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_model="gpt",
    )
    return VisualCodingOpenCodeServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ('{"verdict": "equivalent", "confidence": 1, "reason": "same"}', "equivalent"),
        ('Sure. {"verdict": "DIFFERENT", "reason": "x"}', "different"),
        ('{"verdict": "maybe"}', "unsure"),
        ("not json", "unsure"),
        ("", "unsure"),
    ],
)
def test_parse_verdict(reply: str, verdict: str) -> None:
    assert parse_verdict(reply)["verdict"] == verdict


@pytest.mark.asyncio
@pytest.mark.parametrize(("verdict", "reward"), [("equivalent", 1.0), ("different", 0.0), ("unsure", 0.0)])
async def test_reward_is_the_judge_verdict(server, verdict: str, reward: float) -> None:
    with patch.object(
        VisualCodingOpenCodeServer, "judge", AsyncMock(return_value={"verdict": verdict, "reason": ""})
    ) as judge:
        result = await server.verify(request("Looking at row 4.", "It is \\boxed{(4,7)}"))
    assert result.reward == reward and result.judge_verdict == verdict
    assert result.extracted_answer == "(4,7)" and result.string_match_reward == 1.0
    question, expected, extracted, text = judge.call_args.args
    assert (question, expected, extracted) == ("Which cell?", "(4,7)", "(4,7)")
    assert text == "Looking at row 4.\nIt is \\boxed{(4,7)}"


@pytest.mark.asyncio
async def test_unextracted_answer_still_judged(server) -> None:
    with patch.object(
        VisualCodingOpenCodeServer, "judge", AsyncMock(return_value={"verdict": "equivalent", "reason": ""})
    ) as judge:
        result = await server.verify(request("row 4, column 7"))
    assert result.reward == 1.0 and result.extracted_answer is None
    assert judge.call_args.args[2] is None


@pytest.mark.asyncio
async def test_empty_response_is_wrong_without_a_judge_call(server) -> None:
    with patch.object(VisualCodingOpenCodeServer, "judge", AsyncMock()) as judge:
        result = await server.verify(request(""))
    assert result.reward == 0.0 and result.failure_reason == "no_response"
    judge.assert_not_called()


@pytest.mark.asyncio
async def test_judge_error_scores_zero(server) -> None:
    with patch.object(VisualCodingOpenCodeServer, "judge", AsyncMock(side_effect=JudgeError("down"))):
        result = await server.verify(request("\\boxed{(4,7)}"))
    assert result.reward == 0.0 and result.failure_reason == "judge_error"


@pytest.mark.asyncio
async def test_judge_request(server) -> None:
    completion = {
        "id": "c",
        "created": 0,
        "model": "gpt",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"verdict": "equivalent", "reason": "ok"}'},
            }
        ],
    }
    with patch("resources_servers.visual_coding_opencode.app.call_judge", AsyncMock()) as call:
        from nemo_gym.openai_utils import NeMoGymChatCompletion

        call.return_value = NeMoGymChatCompletion.model_validate(completion)
        judged = await server.judge("Q" * 3000, "(4,7)", None, "x" * 5000 + "end")
    params = call.call_args.kwargs["json"]
    assert judged["verdict"] == "equivalent"
    assert params["model"] == "gpt" and params["temperature"] == 0.0 and params["max_tokens"] == 1024
    assert params["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM}
    user = params["messages"][1]["content"]
    assert "Q" * 2000 + "\n" in user and "Q" * 2001 not in user
    assert "<nothing extracted>" in user
    assert "x" * 3997 + "end" in user and "x" * 3998 not in user
