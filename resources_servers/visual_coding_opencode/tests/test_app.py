# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymResponse
from nemo_gym.verification import JudgeError
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


def request(*texts: str, expected: str = "(4,7)", status: str = "completed") -> VisualCodingOpenCodeVerifyRequest:
    response = NeMoGymResponse.model_validate(
        {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
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
    )
    return VisualCodingOpenCodeServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ('{"verdict": "equivalent", "confidence": 1, "reason": "same"}', "equivalent"),
        ('Sure. {"verdict": "DIFFERENT", "reason": "x"}', "different"),
        ('{"verdict": "unsure", "reason": "no final answer"}', "unsure"),
        # A brace ahead of the verdict object, as a reasoning judge that quotes the model's \boxed{} writes.
        ('<think>The model wrote \\boxed{7}.</think>{"verdict": "equivalent", "reason": "same"}', "equivalent"),
        ('{"verdict": "equivalent", "reason": "same cell"} Gold {4,7} matches.', "equivalent"),
    ],
)
def test_parse_verdict(reply: str, verdict: str) -> None:
    assert parse_verdict(reply)["verdict"] == verdict


@pytest.mark.parametrize(
    "reply",
    [
        '{"verdict": "maybe"}',
        '{"reason": "no verdict"}',
        "not json",
        "",
        "[1, 2]",
        # Verdict objects that disagree: taking the first would score a quoted "equivalent" as correct.
        '<think>It says {"verdict": "equivalent"}, but 5 is not 7.</think>{"verdict": "different", "reason": "5"}',
    ],
)
def test_a_reply_without_a_valid_verdict_is_a_judge_failure(reply: str) -> None:
    with pytest.raises(JudgeError):
        parse_verdict(reply)


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
async def test_the_last_boxed_answer_wins_over_an_earlier_answer_line(server) -> None:
    with patch.object(
        VisualCodingOpenCodeServer, "judge", AsyncMock(return_value={"verdict": "equivalent", "reason": ""})
    ) as judge:
        result = await server.verify(request("The answer: crop the grid first.", "It is \\boxed{(4,7)}"))
    assert result.extracted_answer == "(4,7)" and judge.call_args.args[2] == "(4,7)"


@pytest.mark.asyncio
async def test_incomplete_run_is_wrong_without_a_judge_call(server) -> None:
    # A run that timed out or filled its context never sent its final answer, even if it boxed an early guess.
    body = request("A first guess: \\boxed{(4,7)}. Let me check by cropping.", status="incomplete")
    with patch.object(VisualCodingOpenCodeServer, "judge", AsyncMock()) as judge:
        result = await server.verify(body)
    judge.assert_not_called()
    assert result.reward == 0.0 and result.failure_reason == "incomplete_run"


@pytest.mark.asyncio
async def test_empty_response_is_wrong_without_a_judge_call(server) -> None:
    with patch.object(VisualCodingOpenCodeServer, "judge", AsyncMock()) as judge:
        result = await server.verify(request(""))
    assert result.reward == 0.0 and result.failure_reason == "no_response"
    judge.assert_not_called()


@pytest.mark.parametrize("judge_reply", ["judge down", "no choices", "not json", '{"verdict": "maybe"}'])
def test_a_judge_failure_masks_the_row_instead_of_scoring_zero(server, judge_reply: str) -> None:
    async def call_judge(*args, **kwargs):
        if judge_reply == "judge down":
            raise JudgeError("judge down")
        choices = [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": judge_reply}}]
        return NeMoGymChatCompletion.model_validate(
            {
                "id": "c",
                "created": 0,
                "model": "gpt",
                "object": "chat.completion",
                "choices": [] if judge_reply == "no choices" else choices,
            }
        )

    client = TestClient(server.setup_webserver())
    with patch("resources_servers.visual_coding_opencode.app.call_judge", call_judge):
        reply = client.post("/verify", json=request("\\boxed{(4,7)}").model_dump(mode="json"))

    assert reply.status_code == 200, reply.text
    row = reply.json()
    # The base class's judge_failsafe masks the row, so RL leaves it out of the loss.
    assert row["mask_sample"] is True and row["reward"] == 0.0
    assert row["failure_kind"] == "judge_failed"


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
        from nemo_gym.core import NeMoGymChatCompletion

        call.return_value = NeMoGymChatCompletion.model_validate(completion)
        judged = await server.judge("Q" * 3000, "(4,7)", None, "x" * 5000 + "end")
    params = call.call_args.kwargs["json"]
    assert judged["verdict"] == "equivalent"
    # The judge's model server picks the model.
    assert "model" not in params and params["temperature"] == 0.0 and params["max_tokens"] == 1024
    assert params["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM}
    user = params["messages"][1]["content"]
    assert "Q" * 2000 + "\n" in user and "Q" * 2001 not in user
    assert "<nothing extracted>" in user
    assert "Reference answer (gold):\n(4,7)\n" in user
    assert "x" * 3997 + "end" in user and "x" * 3998 not in user


@pytest.mark.asyncio
async def test_the_exported_verifier_fixture_passes() -> None:
    from nemo_gym.verification import exercise_verifier_fixture
    from resources_servers.visual_coding_opencode.app import VERIFIER_FIXTURE

    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert {result.name: result.observed_rewards for result in results} == {
        "judge says equivalent": (1.0,),
        "judge says different": (0.0,),
        "judge unsure": (0.0,),
        "empty response skips the judge": (0.0,),
        "missing expected answer": (),
    }
