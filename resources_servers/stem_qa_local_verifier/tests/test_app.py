# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.stem_qa_local_verifier.app import (
    StemQALocalVerifierConfig,
    StemQALocalVerifierServer,
    StemQAVerifyRequest,
)


def _make_config() -> StemQALocalVerifierConfig:
    template = str(
        Path(__file__).resolve().parents[1] / "prompt_templates/stem_qa_loose_verifier.txt"
    )
    return StemQALocalVerifierConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=template,
    )


def _make_server(server_client=None) -> StemQALocalVerifierServer:
    if server_client is None:
        server_client = MagicMock(spec=ServerClient)
    return StemQALocalVerifierServer(config=_make_config(), server_client=server_client)


def _msg(text: str, *, msg_id: str = "msg") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=msg_id,
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_request(
    response_content: str,
    ground_truth,
    *,
    user_question: str = "What is something?",
    request_id: int = 1,
) -> StemQAVerifyRequest:
    response = NeMoGymResponse(
        id=f"resp_test_{request_id}",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[_msg(response_content, msg_id=f"msg_{request_id}")],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return StemQAVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": user_question}]},
        response=response,
        ground_truth=ground_truth,
    )


def _judge_response_text(text: str, *, response_id: str = "judge_resp") -> str:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="judge",
        object="response",
        output=[_msg(text, msg_id="judge_msg")],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump_json()


def _wire_judge(server_client: MagicMock, judge_text_or_list) -> None:
    """Wire ``server_client.post(...).read()`` to return judge response JSON.

    Mirrors the pattern used in ``equivalence_llm_judge`` tests (they call
    ``aiohttp_response.read()`` to get raw bytes of the JSON, which
    ``get_response_json`` then parses).
    """
    post_mock = MagicMock()
    if isinstance(judge_text_or_list, list):
        post_mock.read = AsyncMock()
        post_mock.read.side_effect = [_judge_response_text(t) for t in judge_text_or_list]
    else:
        post_mock.read = AsyncMock(return_value=_judge_response_text(judge_text_or_list))
    server_client.post = AsyncMock(return_value=post_mock)


# ---------------------------------------------------------------------------
# Multiple-choice path (no judge call).
# ---------------------------------------------------------------------------


class TestMultipleChoice:
    def test_sanity(self) -> None:
        _make_server()

    def test_correct_letter(self) -> None:
        req = _make_request(
            "I think the answer is (B) because plants take in CO2.\nAnswer: (B)",
            {"style": "multiple_choice", "value": "B"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.style == "multiple_choice"
        assert result.extracted_answer == "B"
        assert result.verification_failed is False
        assert result.judge_evaluation is None

    def test_wrong_letter(self) -> None:
        req = _make_request(
            "Answer: (A)",
            {"style": "multiple_choice", "value": "B"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.extracted_answer == "A"

    def test_no_letter_match(self) -> None:
        req = _make_request(
            "I have no idea.",
            {"style": "multiple_choice", "value": "B"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.extracted_answer == ""

    def test_letter_case_insensitive(self) -> None:
        req = _make_request(
            "answer: (b)",
            {"style": "multiple_choice", "value": "B"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_think_block_stripped_before_letter_extract(self) -> None:
        req = _make_request(
            "<think>Maybe Answer: (A)</think>\nAnswer: (B)",
            {"style": "multiple_choice", "value": "B"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unclosed_think_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>still thinking and Answer: (B)",
            {"style": "multiple_choice", "value": "B"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_ground_truth_as_json_string(self) -> None:
        req = _make_request(
            "Answer: (B)",
            json.dumps({"style": "multiple_choice", "value": "B"}),
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_invalid_ground_truth_marks_verification_failed(self) -> None:
        req = _make_request("Answer: (B)", "not-a-json-blob")
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True


# ---------------------------------------------------------------------------
# Natural-text / LLM-judge path.
# ---------------------------------------------------------------------------


class TestNaturalText:
    def test_judge_returns_score_1(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(
            server_client,
            'I checked it.\n```json\n{"reasons": "matches", "score": 1}\n```',
        )
        rs = _make_server(server_client)

        req = _make_request(
            "<answer>F = ma is Newton's second law.</answer>",
            {"style": "natural_text", "value": "Newton's second law: F = ma"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 1.0
        assert result.style == "natural_text"
        assert result.extracted_answer == "F = ma is Newton's second law."
        assert result.verification_failed is False
        assert result.judge_evaluation is not None
        assert server_client.post.call_count == 1

    def test_judge_returns_score_0(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(
            server_client,
            '```json\n{"reasons": "wrong", "score": 0}\n```',
        )
        rs = _make_server(server_client)

        req = _make_request(
            "<answer>The moon is made of cheese.</answer>",
            {"style": "natural_text", "value": "Newton's second law: F = ma"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_judge_unparseable_response_marks_failure(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(
            server_client,
            "no json block here, just prose without code fences",
        )
        rs = _make_server(server_client)

        req = _make_request(
            "<answer>F = ma</answer>",
            {"style": "natural_text", "value": "F = ma"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True

    def test_judge_http_failure_marks_failure(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("boom"))
        rs = _make_server(server_client)

        req = _make_request(
            "<answer>F = ma</answer>",
            {"style": "natural_text", "value": "F = ma"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        assert result.judge_evaluation is None

    def test_unclosed_think_skips_judge_call(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock()
        rs = _make_server(server_client)

        req = _make_request(
            "<think>incomplete reasoning",
            {"style": "natural_text", "value": "F = ma"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False
        server_client.post.assert_not_called()

    def test_judge_score_handles_latex_backslashes(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        # \geq is invalid JSON; parse_individual_response normalizes by doubling backslashes.
        _wire_judge(
            server_client,
            '```json\n{"reasons": "x \\geq y holds", "score": 1}\n```',
        )
        rs = _make_server(server_client)

        req = _make_request(
            "<answer>x >= y</answer>",
            {"style": "natural_text", "value": "x is at least y"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 1.0
