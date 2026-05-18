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

from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.rubrics.app import (
    RubricsResourcesServer,
    RubricsResourcesServerConfig,
    RubricsVerifyRequest,
)


def _make_config() -> RubricsResourcesServerConfig:
    template = str(
        Path(__file__).resolve().parents[1] / "prompt_templates/rubrics_verifier.txt"
    )
    return RubricsResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=template,
    )


def _make_server(server_client=None) -> RubricsResourcesServer:
    if server_client is None:
        server_client = MagicMock(spec=ServerClient)
    return RubricsResourcesServer(config=_make_config(), server_client=server_client)


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
    user_question: str = "Explain something briefly.",
    request_id: int = 1,
) -> RubricsVerifyRequest:
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
    return RubricsVerifyRequest(
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


def _wire_judge(server_client: MagicMock, judge_text: str) -> None:
    post_mock = MagicMock()
    post_mock.read = AsyncMock(return_value=_judge_response_text(judge_text))
    server_client.post = AsyncMock(return_value=post_mock)


def _rubrics_judge_json(verdicts: list[bool]) -> str:
    body = {
        f"Rubric-{i}": {
            "title": f"Criterion {i}",
            "reasoning": "ok",
            "passed": v,
        }
        for i, v in enumerate(verdicts)
    }
    return f"reasoning here\n```json\n{json.dumps(body)}\n```"


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    def test_all_rubrics_passed_returns_full_reward(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True, True]))
        rs = _make_server(server_client)

        gt = {"rubrics": [{"criterion": "A", "points": 2}, {"criterion": "B", "points": 1}]}
        req = _make_request("<answer>Some answer</answer>", gt)
        result = asyncio.run(rs.verify(req))

        assert result.reward == approx(1.0)
        assert result.passed_count == 2
        assert result.total_count == 2
        assert result.verification_failed is False
        assert result.extracted_answer == "Some answer"
        assert result.judge_evaluation is not None
        assert result.judge_evaluation.parsed is not None

    def test_partial_pass_uses_weighted_ratio(self) -> None:
        # Pass the first rubric (worth 2) and fail the second (worth 1).
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True, False]))
        rs = _make_server(server_client)

        gt = {"rubrics": [{"criterion": "A", "points": 2}, {"criterion": "B", "points": 1}]}
        req = _make_request("<answer>partial</answer>", gt)
        result = asyncio.run(rs.verify(req))

        assert result.reward == approx(2.0 / 3.0)
        assert result.passed_count == 1
        assert result.total_count == 2

    def test_pitfall_pass_counts_as_passed_no_penalty(self) -> None:
        # Two positive rubrics passed + one pitfall (negative weight) avoided.
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True, True, True]))
        rs = _make_server(server_client)

        gt = {
            "rubrics": [
                {"criterion": "A", "points": 1},
                {"criterion": "B", "points": 1},
                {"criterion": "Pitfall avoided", "points": -1},
            ]
        }
        req = _make_request("<answer>good answer</answer>", gt)
        result = asyncio.run(rs.verify(req))

        # Earned 2/2 positive weight; pitfall avoided contributes 0 to numerator.
        assert result.reward == approx(1.0)
        assert result.passed_count == 3
        assert result.total_count == 3

    def test_pitfall_fail_subtracts_from_reward(self) -> None:
        # Both positives passed (weight 2 each) but pitfall is present (weight -1).
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True, True, False]))
        rs = _make_server(server_client)

        gt = {
            "rubrics": [
                {"criterion": "A", "points": 2},
                {"criterion": "B", "points": 2},
                {"criterion": "Pitfall", "points": -1},
            ]
        }
        req = _make_request("<answer>good but with pitfall</answer>", gt)
        result = asyncio.run(rs.verify(req))

        # earned = 2 + 2 - 1 = 3; total_positive_weight = 4 -> reward 0.75.
        assert result.reward == approx(0.75)
        assert result.passed_count == 2
        assert result.total_count == 3

    def test_legacy_rubric_format_with_weight(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True, False]))
        rs = _make_server(server_client)

        gt = {
            "rubrics": [
                {"title": "T1", "description": "D1", "weight": 1},
                {"title": "T2", "description": "D2", "weight": 1},
            ]
        }
        req = _make_request("<answer>halfway</answer>", gt)
        result = asyncio.run(rs.verify(req))
        assert result.reward == approx(0.5)
        assert result.passed_count == 1
        assert result.total_count == 2

    def test_doubly_nested_rubrics_list_unwrapped(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True]))
        rs = _make_server(server_client)

        gt = {"rubrics": [[{"criterion": "A", "points": 1}]]}
        req = _make_request("<answer>ok</answer>", gt)
        result = asyncio.run(rs.verify(req))
        assert result.reward == approx(1.0)
        assert result.total_count == 1

    def test_ground_truth_as_json_string(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, _rubrics_judge_json([True]))
        rs = _make_server(server_client)

        gt = json.dumps({"rubrics": [{"criterion": "A", "points": 1}]})
        req = _make_request("<answer>ok</answer>", gt)
        result = asyncio.run(rs.verify(req))
        assert result.reward == approx(1.0)

    def test_no_rubrics_marks_verification_failed(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock()
        rs = _make_server(server_client)

        req = _make_request("<answer>anything</answer>", {"rubrics": []})
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        assert result.judge_evaluation is None
        server_client.post.assert_not_called()

    def test_invalid_ground_truth_marks_verification_failed(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock()
        rs = _make_server(server_client)

        req = _make_request("<answer>x</answer>", "not json at all")
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        server_client.post.assert_not_called()

    def test_unclosed_think_returns_zero_no_failure(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock()
        rs = _make_server(server_client)

        req = _make_request(
            "<think>still thinking",
            {"rubrics": [{"criterion": "A", "points": 1}]},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False
        server_client.post.assert_not_called()

    def test_judge_unparseable_marks_verification_failed(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, "no json at all here")
        rs = _make_server(server_client)

        gt = {"rubrics": [{"criterion": "A", "points": 1}]}
        req = _make_request("<answer>x</answer>", gt)
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        # judge call happened, evaluation is recorded but parsed is None
        assert result.judge_evaluation is not None
        assert result.judge_evaluation.parsed is None

    def test_judge_http_failure_marks_verification_failed(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("boom"))
        rs = _make_server(server_client)

        gt = {"rubrics": [{"criterion": "A", "points": 1}]}
        req = _make_request("<answer>x</answer>", gt)
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        assert result.judge_evaluation is None

    def test_string_passed_field_is_coerced_to_bool(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        # Judge returns "yes" / "no" strings instead of bools.
        body = {
            "Rubric-0": {"title": "T0", "reasoning": "ok", "passed": "yes"},
            "Rubric-1": {"title": "T1", "reasoning": "ok", "passed": "no"},
        }
        _wire_judge(server_client, f"```json\n{json.dumps(body)}\n```")
        rs = _make_server(server_client)

        gt = {"rubrics": [{"criterion": "A", "points": 1}, {"criterion": "B", "points": 1}]}
        req = _make_request("<answer>x</answer>", gt)
        result = asyncio.run(rs.verify(req))
        assert result.reward == approx(0.5)
        assert result.passed_count == 1
        assert result.total_count == 2
