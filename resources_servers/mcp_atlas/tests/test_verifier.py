# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.mcp_atlas.verifier import (
    CLAIM_PROMPT,
    MAX_RESPONSE_CHARS,
    coverage,
    final_answer,
    judge_request,
    parse_verdict,
)


def message(text: str, index: int = 0) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"msg{index}",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def response(*items) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0,
        model="m",
        object="response",
        output=list(items),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def test_final_answer_is_the_last_non_empty_assistant_message() -> None:
    call = NeMoGymResponseFunctionToolCall(
        arguments="{}", call_id="c1", name="calculator_calculate", type="function_call", id="fc1"
    )
    out = response(message("Let me check.", 0), call, message("The answer is 7 years.", 1), message("  ", 2))
    assert final_answer(out) == "The answer is 7 years."


def test_final_answer_is_empty_without_assistant_text() -> None:
    call = NeMoGymResponseFunctionToolCall(arguments="{}", call_id="c1", name="x", type="function_call", id="fc1")
    assert final_answer(response(call)) == ""


def test_judge_request_shows_only_claim_and_answer() -> None:
    params = judge_request(claim="The repo was created in 2013.", answer="Created in 2013.", max_tokens=None)
    assert params["temperature"] == 0.0
    assert "max_tokens" not in params
    assert params["response_format"]["json_schema"]["name"] == "claim_evaluation"
    content = params["messages"][0]["content"]
    assert content == CLAIM_PROMPT.format(claim="The repo was created in 2013.", response="Created in 2013.")


def test_judge_request_truncates_long_answers() -> None:
    params = judge_request(claim="c", answer="x" * (MAX_RESPONSE_CHARS + 10), max_tokens=4096)
    assert params["max_tokens"] == 4096
    assert "x" * MAX_RESPONSE_CHARS in params["messages"][0]["content"]
    assert "x" * (MAX_RESPONSE_CHARS + 1) not in params["messages"][0]["content"]


def test_parse_verdict_scores_each_outcome() -> None:
    for outcome, score in (("fulfilled", 1.0), ("partially_fulfilled", 0.5), ("not_fulfilled", 0.0)):
        verdict = parse_verdict("c", json.dumps({"coverage_outcome": outcome, "justification": "j"}))
        assert verdict["score"] == score
        assert verdict["coverage_outcome"] == outcome


def test_unreadable_verdict_is_none() -> None:
    for content in (None, "", "not json", json.dumps({"coverage_outcome": "maybe"}), json.dumps([1])):
        assert parse_verdict("c", content) is None


def test_coverage_is_the_rounded_mean() -> None:
    verdicts = [{"score": 1.0}, {"score": 0.5}, {"score": 0.0}]
    assert coverage(verdicts, 3) == 0.5
    assert coverage([{"score": 1.0}, {"score": 0.0}, {"score": 0.0}], 3) == 0.333
    assert coverage([], 0) == 0.0
