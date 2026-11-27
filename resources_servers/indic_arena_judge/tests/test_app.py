# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import math
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from pytest import approx

from nemo_gym.config_types import AggregateMetricsRequest, ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.arena_judge.app import ArenaJudgeConfig, ArenaJudgeServer, ArenaJudgeVerifyRequest
from resources_servers.indic_arena_judge.app import IndicArenaJudgeConfig, IndicArenaJudgeServer
from resources_servers.indic_arena_judge.task_data import TaskData


def _make_response(text: str, model: str = "mock_model") -> NeMoGymResponse:
    """Build a NeMoGymResponse mock — used as the POLICY model's output
    (the candidate answer the judge scores). The policy server is
    expected to return a Responses API shape, whatever backend it runs.
    """
    return NeMoGymResponse(
        id="mock_resp",
        created_at=0.0,
        model=model,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="mock_msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _make_chat_completion(text: str, model: str = "mock_judge") -> dict:
    """Build a raw ChatCompletion dict — what server_client.post returns
    for the JUDGE call via /v1/chat/completions."""
    return {
        "id": "mock_chat",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _mock_judge_sequence(server_mock: MagicMock, judge_texts: list[str]) -> None:
    """Configure server_client.post to return chat.completion responses
    in order — matches what /v1/chat/completions actually returns.

    The judge code path calls ``get_response_json(response)`` which
    runs ``orjson.loads(await response.read())`` — so ``read()`` must
    return the JSON-encoded bytes, not a dict.
    """
    import json as _json

    response_mocks = []
    for text in judge_texts:
        resp = AsyncMock()
        resp.read = AsyncMock(return_value=_json.dumps(_make_chat_completion(text)).encode())
        response_mocks.append(resp)
    server_mock.post = AsyncMock(side_effect=response_mocks)


class TestPointwiseJudge:
    def _config(self, mode: str | None = None) -> IndicArenaJudgeConfig:
        return IndicArenaJudgeConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="indic_arena_judge",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
            **({"evaluation_mode": mode} if mode is not None else {}),
        )

    def _body(self, answer: str = "हिंदी उत्तर", baseline: str | None = None) -> ArenaJudgeVerifyRequest:
        return ArenaJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response(answer),
            question="हिंदी में उत्तर दें।",
            baseline_answer=baseline,
            language="hi",
            uid="hi:one",
        )

    @staticmethod
    def _judgement(**overrides: object) -> str:
        return json.dumps(
            {"correctness": 4, "relevance": 3, "completeness": 2, "language_quality": 3, "explanation": "Assessment."}
            | overrides
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("baseline", [None, "ignored baseline"])
    async def test_default_one_call_and_normalized_reward(self, baseline: str | None) -> None:
        client = MagicMock(spec=ServerClient)
        _mock_judge_sequence(client, [self._judgement()])
        server = IndicArenaJudgeServer(config=self._config(), server_client=client)
        result = await server.verify(self._body(baseline=baseline))
        assert client.post.await_count == 1
        prompt = str(client.post.call_args.kwargs["json"].messages)
        assert "हिंदी उत्तर" in prompt
        assert "ignored baseline" not in prompt
        assert result.reward == approx(0.75)
        assert result.pointwise_score == approx(75.0)
        assert result.pointwise_scores == {"correctness": 4, "relevance": 3, "completeness": 2, "language_quality": 3}
        assert result.evaluation_mode == "pointwise"
        assert result.invalid_pointwise is False
        assert result.verdict_gen_base is None and result.verdict_base_gen is None
        assert result.model_dump()["language"] == "hi"
        aggregate = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[result.model_dump()]))
        assert aggregate.key_metrics["pointwise/score"] == approx(75.0)
        assert aggregate.key_metrics["pointwise/language/hi/score"] == approx(75.0)
        assert not any(name.startswith("arena_elo/") for name in aggregate.agent_metrics)

    @pytest.mark.asyncio
    async def test_empty_answer_is_zero_without_judge_call(self) -> None:
        client = MagicMock(spec=ServerClient)
        server = IndicArenaJudgeServer(config=self._config(), server_client=client)
        result = await server.verify(self._body(answer=" \n"))
        assert result.reward == result.pointwise_score == 0.0
        assert set(result.pointwise_scores.values()) == {0}
        assert result.invalid_pointwise is False
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_judge_response_is_flagged(self) -> None:
        client = MagicMock(spec=ServerClient)
        _mock_judge_sequence(client, ["The answer is good."])
        server = IndicArenaJudgeServer(config=self._config(), server_client=client)
        result = await server.verify(self._body())
        assert result.invalid_pointwise is True
        assert result.pointwise_score is None and result.pointwise_scores is None
        assert result.reward == 0.0
        assert result.judgement_pointwise == "The answer is good."
        metrics = server.compute_metrics([[result.model_dump()]])
        assert math.isnan(metrics["pointwise/score"])
        assert metrics["pointwise/invalid_scores"] == 1

    @pytest.mark.asyncio
    async def test_judge_transport_failure_is_not_a_score(self) -> None:
        client = MagicMock(spec=ServerClient)
        client.post = AsyncMock(side_effect=RuntimeError("endpoint unavailable"))
        server = IndicArenaJudgeServer(config=self._config(), server_client=client)
        with pytest.raises(JudgeError, match="endpoint unavailable"):
            await server.verify(self._body())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("baseline", [None, "", "  "])
    async def test_pairwise_rejects_missing_baseline(self, baseline: str | None) -> None:
        client = MagicMock(spec=ServerClient)
        server = IndicArenaJudgeServer(config=self._config(mode="pairwise"), server_client=client)
        with pytest.raises(HTTPException, match="nonempty baseline_answer") as error:
            await server.verify(self._body(baseline=baseline))
        assert error.value.status_code == 422
        client.post.assert_not_called()

    @pytest.mark.parametrize("fenced", [False, True])
    def test_parse_json_and_fenced_json(self, fenced: bool) -> None:
        text = self._judgement()
        if fenced:
            text = f"```json\n{text}\n```"
        result = IndicArenaJudgeServer._parse_pointwise(text)
        assert result.correctness == 4 and result.completeness == 2

    @pytest.mark.parametrize("score", [-1, 5, 1.5, "4", True, None])
    def test_rejects_invalid_dimension_scores(self, score: object) -> None:
        assert IndicArenaJudgeServer._parse_pointwise(self._judgement(correctness=score)) is None

    @pytest.mark.parametrize("text", ["", "{}", '{"correctness": 4', "[]", "null"])
    def test_rejects_missing_or_truncated_scores(self, text: str) -> None:
        assert IndicArenaJudgeServer._parse_pointwise(text) is None

    def test_rejects_unknown_fields_and_missing_explanation(self) -> None:
        assert IndicArenaJudgeServer._parse_pointwise(self._judgement(extra_score=4)) is None
        assert IndicArenaJudgeServer._parse_pointwise(self._judgement(explanation="")) is None

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValidationError, match="evaluation_mode"):
            self._config(mode="unknown")

    def test_metrics_average_repeats_then_tasks_and_split_languages(self) -> None:
        def scored(language: str, score: int) -> dict:
            return {
                "language": language,
                "pointwise_score": score,
                "pointwise_scores": dict.fromkeys(
                    ["correctness", "relevance", "completeness", "language_quality"], score // 25
                ),
            }

        server = IndicArenaJudgeServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        tasks = [
            [scored("hi", 100), scored("hi", 0), {"language": "hi", "invalid_pointwise": True}],
            [scored("ta", 75)],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["pointwise/score"] == approx(62.5)
        assert metrics["pointwise/correctness"] == approx(2.5)
        assert metrics["pointwise/valid_scores"] == 3
        assert metrics["pointwise/invalid_scores"] == 1
        assert metrics["pointwise/scored_tasks"] == 2
        assert metrics["pointwise/language/hi/score"] == approx(50.0)
        assert metrics["pointwise/language/ta/score"] == approx(75.0)
        assert not any(name.startswith(("arena_elo/", "pass@")) for name in metrics)
        assert server.get_key_metrics(metrics) == metrics

    def test_metrics_empty_tasks(self) -> None:
        server = IndicArenaJudgeServer(config=self._config(), server_client=MagicMock(spec=ServerClient))
        metrics = server.compute_metrics([[]])
        assert math.isnan(metrics["pointwise/score"])
        assert metrics["pointwise/valid_scores"] == metrics["pointwise/invalid_scores"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["hard_prompt", "creative_writing"])
async def test_indic_pairwise_matches_english_verdicts_and_category_prompts(category: str) -> None:
    helper = TestPointwiseJudge()
    body = helper._body(baseline="संदर्भ उत्तर").model_copy(update={"category": category})
    clients = [MagicMock(spec=ServerClient), MagicMock(spec=ServerClient)]
    for client in clients:
        _mock_judge_sequence(client, ["[[A>B]]", "[[B>A]]"])
    english_config = ArenaJudgeConfig.model_validate(helper._config("pairwise").model_dump())
    english = ArenaJudgeServer(config=english_config, server_client=clients[0])
    indic = IndicArenaJudgeServer(config=helper._config("pairwise"), server_client=clients[1])
    before = await english.verify(body)
    after = await indic.verify(body)
    assert after.reward == before.reward == 1.0
    assert after.category == before.category == category
    assert after.verdict_gen_base == before.verdict_gen_base == "A>B"
    assert after.verdict_base_gen == before.verdict_base_gen == "B>A"
    assert after.evaluation_mode == "pairwise"
    for english_call, indic_call in zip(clients[0].post.call_args_list, clients[1].post.call_args_list, strict=True):
        assert english_call.kwargs["json"] == indic_call.kwargs["json"]
    metrics = indic.compute_metrics([[after.model_dump()]])
    assert metrics["arena_elo/score"] == 100.0
    assert not any(key.startswith("pointwise/") for key in metrics)
    assert "arena_elo/score" in indic.get_key_metrics(metrics)


@pytest.mark.asyncio
async def test_indic_mode_does_not_change_english_baseline_or_response_contract() -> None:
    helper = TestPointwiseJudge()
    IndicArenaJudgeServer(config=helper._config(), server_client=MagicMock(spec=ServerClient))
    client = MagicMock(spec=ServerClient)
    _mock_judge_sequence(client, ["[[A=B]]", "[[A=B]]"])
    english_config = ArenaJudgeConfig.model_validate(helper._config("pairwise").model_dump())
    english = ArenaJudgeServer(config=english_config, server_client=client)
    result = await english.verify(helper._body())
    assert client.post.await_count == 2
    assert result.reward == 0.0
    assert "evaluation_mode" not in result.model_dump()
    assert "pointwise_score" not in result.model_dump()
    assert "evaluation_mode" not in ArenaJudgeConfig.model_fields


@pytest.mark.asyncio
async def test_indic_sanitization_applies_before_pointwise_judging() -> None:
    helper = TestPointwiseJudge()
    client = MagicMock(spec=ServerClient)
    _mock_judge_sequence(client, [helper._judgement()])
    config = helper._config().model_copy(update={"sanitize_generations": True})
    server = IndicArenaJudgeServer(config=config, server_client=client)
    await server.verify(helper._body(answer="हिंदी\ud800 उत्तर"))
    messages = client.post.call_args.kwargs["json"].messages
    # Surrogate halves cannot be encoded as UTF-8 in a judge HTTP request.
    encoded = json.dumps(messages, ensure_ascii=False).encode("utf-8", errors="strict")
    assert "हिंदी".encode() in encoded


@pytest.mark.asyncio
async def test_blank_pairwise_baseline_with_sanitization_is_rejected() -> None:
    helper = TestPointwiseJudge()
    client = MagicMock(spec=ServerClient)
    config = helper._config("pairwise").model_copy(update={"sanitize_generations": True})
    server = IndicArenaJudgeServer(config=config, server_client=client)
    with pytest.raises(HTTPException, match="nonempty baseline_answer"):
        await server.verify(helper._body(baseline=" \n"))
    client.post.assert_not_called()


def test_task_schema_preserves_language_and_source_metadata_without_baseline() -> None:
    task = TaskData.model_validate({"question": "हिंदी में उत्तर दें।", "language": "hi", "uid": "hi:one", "row_id": "one"})
    assert task.question == "हिंदी में उत्तर दें।"
    assert task.baseline_answer is None
    assert task.language == "hi" and task.model_dump()["row_id"] == "one"
    assert task.model_dump()["uid"] == "hi:one"
