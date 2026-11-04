# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the chartography resources server (judge mocked; no network)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx, fixture, raises

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NgFailureEvent,
)
from nemo_gym.verification import JudgeError, is_judge_failure
from resources_servers.chartography.app import (
    JUDGE_RESPONSE_JSON_SCHEMA,
    ChartographyConfig,
    ChartographyResourcesServer,
    ChartographyVerifyRequest,
    ParseError,
    extract_final_answer,
    parse_judge_output,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
QUESTION = "At a flow of 35 L/s, what head does the 225 mm impeller deliver? Give the value in m."
GOLDEN = "Head: 34.5 m (acceptable range 33.5 to 35.5 m)"


@fixture
def config() -> ChartographyConfig:
    return ChartographyConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_chat_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        judge_prompt_template_fpath=str(SERVER_DIR / "prompt_templates/judge.txt"),
    )


@fixture
def server(config: ChartographyConfig) -> ChartographyResourcesServer:
    return ChartographyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _assistant_msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text)],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(output: list) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _params() -> NeMoGymResponseCreateParamsNonStreaming:
    """Question text, then the chart image."""
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": QUESTION},
                    {"type": "input_image", "image_url": "data:image/png;base64,chart", "detail": "auto"},
                ],
            }
        ]
    )


def _request(answer_output: list, golden: Any = GOLDEN, prompt: Any = QUESTION) -> ChartographyVerifyRequest:
    meta = {"task_id": "t-1", "domain": "curve_family", "prompt": prompt, "golden_answer": golden}
    return ChartographyVerifyRequest(
        responses_create_params=_params(), response=_response(answer_output), verifier_metadata=meta
    )


def _judge_content(*correct: bool) -> str:
    return json.dumps(
        {
            "final_answer": "34.8 m",
            "parts": [
                {"part": f"part {i + 1}", "answer": "34.8", "correct": c, "reason": "r"} for i, c in enumerate(correct)
            ],
        }
    )


def _mock_judge(server: ChartographyResourcesServer, *contents: str) -> AsyncMock:
    """Return the given judge outputs in order, repeating the last one."""
    payloads = [
        NeMoGymChatCompletion(
            id="j",
            created=0,
            model="judge",
            object="chat.completion",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content=c),
                )
            ],
        ).model_dump_json()
        for c in contents
    ]

    async def _post(*args: Any, **kwargs: Any) -> MagicMock:
        payload = payloads[min(_post.calls, len(payloads) - 1)]
        _post.calls += 1
        resp = MagicMock()
        resp.read = AsyncMock(return_value=payload)
        return resp

    _post.calls = 0
    mock = AsyncMock(side_effect=_post)
    server.server_client.post = mock
    return mock


def test_extract_final_answer_skips_reasoning() -> None:
    reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
    assert extract_final_answer(_response([reasoning, _assistant_msg("34.8 m")])) == "34.8 m"
    assert extract_final_answer(_response([reasoning])) == ""


class TestParseJudgeOutput:
    def test_valid_output(self) -> None:
        final, verdicts = parse_judge_output(_judge_content(True, False))
        assert final == "34.8 m" and [v.correct for v in verdicts] == [True, False]

    def test_markdown_fences_are_stripped(self) -> None:
        _, verdicts = parse_judge_output("```json\n" + _judge_content(True) + "\n```")
        assert [v.correct for v in verdicts] == [True]

    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            "[]",
            '{"parts": [{"correct": true}]}',
            '{"final_answer": "x", "parts": []}',
            '{"final_answer": "x", "parts": [{"part": "a", "correct": "true"}]}',
        ],
    )
    def test_invalid_output_raises(self, raw: str) -> None:
        with raises(ParseError):
            parse_judge_output(raw)


class TestVerify:
    @pytest.mark.asyncio
    async def test_all_parts_correct(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, _judge_content(True, True))
        result = await server.verify(_request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == 1.0 and result.all_pass == 1.0 and result.num_criteria == 2
        assert result.final_answer == "34.8 m" and [v.correct for v in result.verdicts] == [True, True]

    @pytest.mark.asyncio
    async def test_strict_pass_fails_on_any_part(self, server: ChartographyResourcesServer) -> None:
        _mock_judge(server, _judge_content(True, False))
        result = await server.verify(_request([_assistant_msg("Head: 31 m")]))
        assert result.reward == 0.0 and result.all_pass == 0.0
        assert result.mean_criterion_pass == approx(0.5)

    @pytest.mark.asyncio
    async def test_mean_criteria_gives_partial_credit(self, config: ChartographyConfig) -> None:
        server = ChartographyResourcesServer(
            config=config.model_copy(update={"reward_mode": "mean_criteria"}),
            server_client=MagicMock(spec=ServerClient),
        )
        _mock_judge(server, _judge_content(True, False))
        result = await server.verify(_request([_assistant_msg("Head: 31 m")]))
        assert result.reward == approx(0.5) and result.all_pass == 0.0

    @pytest.mark.asyncio
    async def test_judge_sees_question_reference_and_final_answer_only(
        self, server: ChartographyResourcesServer
    ) -> None:
        mock = _mock_judge(server, _judge_content(True))
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        await server.verify(_request([reasoning, _assistant_msg("Head: 34.8 m")]))
        sent = mock.await_args.kwargs["json"]
        assert mock.await_args.kwargs["server_name"] == "judge"
        [message] = sent.messages
        assert QUESTION in message["content"] and GOLDEN in message["content"]
        assert "Head: 34.8 m" in message["content"]
        assert sent.response_format["json_schema"]["schema"] == JUDGE_RESPONSE_JSON_SCHEMA

    @pytest.mark.asyncio
    async def test_empty_answer_scores_zero_without_judging(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, _judge_content(True))
        reasoning = NeMoGymResponseReasoningItem(id="r", summary=[], type="reasoning")
        result = await server.verify(_request([reasoning]))
        assert result.reward == 0.0 and mock.await_count == 0 and not result.mask_sample

    @pytest.mark.asyncio
    async def test_parse_failure_is_retried(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, "not json", _judge_content(True))
        result = await server.verify(_request([_assistant_msg("Head: 34.8 m")]))
        assert result.reward == 1.0 and result.judge_attempts == 2 and mock.await_count == 2

    @pytest.mark.asyncio
    async def test_parse_failure_exhausts_attempts(self, server: ChartographyResourcesServer) -> None:
        mock = _mock_judge(server, '{"final_answer": "x", "parts": []}')
        with raises(JudgeError):
            await server.verify(_request([_assistant_msg("Head: 34.8 m")]))
        assert mock.await_count == 5

    @pytest.mark.asyncio
    async def test_judge_transport_failure_is_reported(self, server: ChartographyResourcesServer) -> None:
        server.server_client.post = AsyncMock(side_effect=ConnectionError("judge down"))
        with raises(NgFailureEvent) as failure:
            await server.verify(_request([_assistant_msg("Head: 34.8 m")]))
        assert is_judge_failure(failure.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("golden,prompt", [("", QUESTION), (GOLDEN, ""), (None, QUESTION)])
    async def test_invalid_reference_is_masked(
        self, server: ChartographyResourcesServer, golden: Any, prompt: Any
    ) -> None:
        mock = _mock_judge(server, _judge_content(True))
        result = await server.verify(_request([_assistant_msg("Head: 34.8 m")], golden=golden, prompt=prompt))
        assert result.mask_sample and result.reward == 0.0 and mock.await_count == 0
        assert result.failure_kind == "chartography:invalid_reference"


def test_example_rows_are_well_formed() -> None:
    rows = [json.loads(line) for line in (SERVER_DIR / "data/example.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == 5
    for row in rows:
        content = row["responses_create_params"]["input"][0]["content"]
        assert [c["type"] for c in content] == ["input_text", "input_image"]
        assert content[1]["image_url"].startswith("data:image/png;base64,")
        meta = row["verifier_metadata"]
        assert meta["prompt"].strip() == content[0]["text"].strip() and meta["golden_answer"].strip()
        assert set(meta) == {"task_id", "domain", "prompt", "golden_answer"}
