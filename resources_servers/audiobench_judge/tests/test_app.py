# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the audiobench_judge resources server.

Two test surfaces:

* Pure helper tests for ``extract_judge_result`` — the parser that pulls
  the rating out of judge text. Mirrors NeMo Skills'
  ``AudioMetrics._extract_judge_result`` byte-for-byte.
* Server-level tests with a mocked judge model server, asserting reward
  routing, the AudioBench prompt formatting, score aggregation, and the
  empty-generation short-circuit.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.audiobench_judge.app import (
    AudioBenchJudgeConfig,
    AudioBenchJudgeResourcesServer,
    AudioBenchJudgeVerifyRequest,
    extract_judge_result,
)
from resources_servers.audiobench_judge.generate_example_data import SAMPLES, main, make_example
from resources_servers.audiobench_judge.task_data import TaskData


EXAMPLE_DATA_PATH = Path(__file__).parents[1] / "data" / "example.jsonl"

MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
    "parallel_tool_calls": True,
}


def _make_config(use_chat_completions_for_judge: bool = False) -> AudioBenchJudgeConfig:
    return AudioBenchJudgeConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(
            type="responses_api_models",
            name="audiobench_judge_model",
        ),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        use_chat_completions_for_judge=use_chat_completions_for_judge,
    )


def _make_response_with_text(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "role": "assistant",
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_judge_post_mock(judge_text: str):
    """Build a mock ``server_client.post`` that returns a NeMoGymResponse-shaped JSON."""

    async def _read_json(*_args, **_kwargs) -> dict[str, Any]:
        return _make_response_with_text(judge_text).model_dump()

    response = MagicMock(ok=True)
    response.json = AsyncMock(side_effect=_read_json)
    response.read = AsyncMock(return_value=json.dumps(_make_response_with_text(judge_text).model_dump()).encode())
    response.text = AsyncMock(return_value=json.dumps(_make_response_with_text(judge_text).model_dump()))
    return response


def _make_chat_judge_post_mock(content: str | None) -> MagicMock:
    """Build a mock ``server_client.post`` reply shaped like a /v1/chat/completions response."""
    chat_completion = {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 0,
        "model": "dummy",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}},
        ],
    }
    response = MagicMock(ok=True)
    response.read = AsyncMock(return_value=json.dumps(chat_completion).encode())
    return response


def _make_server(judge_text: str = "Rating: 5") -> AudioBenchJudgeResourcesServer:
    config = _make_config()
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=_make_judge_post_mock(judge_text))
    return AudioBenchJudgeResourcesServer(config=config, server_client=server_client)


def _make_chat_server(content: str | None) -> AudioBenchJudgeResourcesServer:
    config = _make_config(use_chat_completions_for_judge=True)
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=_make_chat_judge_post_mock(content))
    return AudioBenchJudgeResourcesServer(config=config, server_client=server_client)


def _make_verify_request(generation: str, expected_answer: str, question: str) -> AudioBenchJudgeVerifyRequest:
    return AudioBenchJudgeVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response_with_text(generation),
        question=question,
        expected_answer=expected_answer,
    )


# ──────────────────────────────────────────────────────────────────────────────
# extract_judge_result — pure helper tests (byte-equivalent with Skills)
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractJudgeResult:
    """Mirrors ``AudioMetrics._extract_judge_result`` precedence — Rating > Judgement > yes/no fallback."""

    def test_rating_5_is_correct(self) -> None:
        assert extract_judge_result("Explanation: ...\nRating: 5") == (True, 5.0)

    def test_rating_4_is_correct(self) -> None:
        assert extract_judge_result("Rating: 4") == (True, 4.0)

    def test_rating_3_is_correct(self) -> None:
        # Threshold: rating >= 3 → correct.
        assert extract_judge_result("Rating: 3") == (True, 3.0)

    def test_rating_2_is_incorrect(self) -> None:
        assert extract_judge_result("Rating: 2") == (False, 2.0)

    def test_rating_0_is_incorrect(self) -> None:
        assert extract_judge_result("Rating: 0") == (False, 0.0)

    def test_rating_clamped_above_5(self) -> None:
        # Out-of-range ratings are clamped to 0..5.
        assert extract_judge_result("Rating: 7") == (True, 5.0)

    def test_rating_clamped_below_0(self) -> None:
        # Pattern ``[0-9]+`` only matches non-negative; "Rating: -1" doesn't
        # match the rating regex and falls through.
        assert extract_judge_result("Rating: -1")[0] is False

    def test_rating_float_supported(self) -> None:
        assert extract_judge_result("Rating: 3.5") == (True, 3.5)

    def test_judgement_yes_maps_to_5(self) -> None:
        # Legacy binary fallback.
        assert extract_judge_result("Reasoning: ...\nJudgement: Yes") == (True, 5.0)

    def test_judgement_no_maps_to_0(self) -> None:
        assert extract_judge_result("Reasoning: ...\nJudgement: No") == (False, 0.0)

    def test_rating_takes_precedence_over_judgement(self) -> None:
        # If both appear, Rating wins (matches Skills' precedence order).
        assert extract_judge_result("Rating: 4\nJudgement: No") == (True, 4.0)

    def test_plain_yes_fallback(self) -> None:
        assert extract_judge_result("yes, this looks good") == (True, 5.0)

    def test_plain_no_fallback(self) -> None:
        assert extract_judge_result("no, this is wrong") == (False, 0.0)

    def test_no_match_returns_zero(self) -> None:
        assert extract_judge_result("¯\\_(ツ)_/¯") == (False, 0.0)

    def test_empty_text_returns_zero(self) -> None:
        assert extract_judge_result("") == (False, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Server-level tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAudioBenchJudgeServer:
    async def test_perfect_rating_gives_reward_one(self) -> None:
        server = _make_server(judge_text="Explanation: matches.\nRating: 5")
        body = _make_verify_request(
            generation="The audio is silent.",
            expected_answer="Silence.",
            question="Describe the audio.",
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.judge_correct is True
        assert result.judge_rating == 5.0
        assert result.generation == "The audio is silent."
        assert result.judge_evaluation is not None

    async def test_low_rating_gives_reward_zero(self) -> None:
        server = _make_server(judge_text="Explanation: misses.\nRating: 1")
        body = _make_verify_request(
            generation="It's loud rock music.",
            expected_answer="Silence.",
            question="Describe the audio.",
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.judge_correct is False
        assert result.judge_rating == 1.0

    async def test_threshold_at_3(self) -> None:
        # Rating exactly at threshold → correct.
        server = _make_server(judge_text="Rating: 3")
        body = _make_verify_request("close enough", "the right answer", "what?")
        result = await server.verify(body)
        assert result.judge_correct is True
        assert result.judge_rating == 3.0

    async def test_empty_generation_short_circuits(self) -> None:
        """Empty generation → rating 0; judge LLM is NOT called (saves tokens)."""
        server = _make_server(judge_text="Rating: 5")  # would be 5 if called
        body = _make_verify_request(generation="", expected_answer="anything", question="q?")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.judge_rating == 0.0
        assert result.judge_evaluation is None
        # Confirm the judge model was never hit.
        server.server_client.post.assert_not_called()

    async def test_judge_call_failure_is_not_a_wrong_answer(self) -> None:
        """A failed judge call raises JudgeError; the failsafe masks the row instead of scoring it 0."""
        server = _make_server(judge_text="Rating: 5")
        server.server_client.post = AsyncMock(side_effect=RuntimeError("judge unreachable"))
        body = _make_verify_request("an answer", "ref", "q?")
        with pytest.raises(JudgeError, match="judge unreachable"):
            await server.verify(body)
        result = json.loads((await judge_failsafe(server.verify)(body)).body)
        assert result["_ng_failure_class"] == "judge_failed"
        assert result["mask_sample"] is True
        assert result["reward"] == 0.0

    async def test_judge_http_error_is_not_a_wrong_answer(self) -> None:
        """A non-2xx judge reply raises JudgeError before the body is parsed."""
        server = _make_server()
        error_reply = MagicMock(ok=False)
        error_reply.raise_for_status = MagicMock(side_effect=RuntimeError("401 Unauthorized"))
        error_reply.content.read = AsyncMock(return_value=b"unauthorized")
        server.server_client.post = AsyncMock(return_value=error_reply)
        with pytest.raises(JudgeError, match="401 Unauthorized"):
            await server.verify(_make_verify_request("an answer", "ref", "q?"))

    async def test_chat_completions_judge_records_evaluation(self) -> None:
        """The chat-completions judge path scores the reply and still populates judge_evaluation."""
        server = _make_chat_server("Rating: 4")
        result = await server.verify(_make_verify_request("My answer", "The reference", "q?"))
        assert result.reward == 1.0
        assert result.judge_rating == 4.0
        assert result.judge_evaluation.response.output_text == "Rating: 4"
        call_kwargs = server.server_client.post.call_args.kwargs
        assert call_kwargs["url_path"] == "/v1/chat/completions"
        assert call_kwargs["json"].max_tokens == 512
        assert "The reference" in call_kwargs["json"].messages[0]["content"]

    async def test_chat_completions_empty_judge_reply_rated_zero(self) -> None:
        """A received-but-empty judge reply is a legitimate rating of 0, not a judge failure."""
        server = _make_chat_server(None)
        result = await server.verify(_make_verify_request("My answer", "ref", "q?"))
        assert result.reward == 0.0
        assert result.judge_rating == 0.0
        assert result.judge_evaluation.response.output_text == ""

    async def test_generation_ignores_non_text_output_items(self) -> None:
        """Only assistant output_text parts are judged; reasoning items and refusals are skipped."""
        server = _make_server(judge_text="Rating: 5")
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {
                    "id": "msg_1",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {"type": "refusal", "refusal": "no"},
                        {"type": "output_text", "text": "final answer", "annotations": []},
                    ],
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        body = AudioBenchJudgeVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=response,
            question="q?",
            expected_answer="ref",
        )
        result = await server.verify(body)
        assert result.generation == "final answer"

    async def test_row_fields_reach_the_rollout(self) -> None:
        """subset_for_metrics (and other row fields) survive verify so compute_metrics() can group by them."""
        server = _make_server(judge_text="Rating: 5")
        body = _make_verify_request("an answer", "ref", "q?").model_copy(
            update={"subset_for_metrics": "alpaca_audio_test"}
        )
        result = (await server.verify(body)).model_dump()
        assert result["subset_for_metrics"] == "alpaca_audio_test"
        assert "alpaca_audio_test/pass@1[avg-of-1]/accuracy" in server.compute_metrics([[result]])

    async def test_judge_prompt_is_audiobench_format(self) -> None:
        """Judge gets the AudioBench rating prompt — byte-equivalent with Skills' judge/audiobench.yaml."""
        server = _make_server(judge_text="Rating: 4")
        body = _make_verify_request(
            generation="My answer",
            expected_answer="The reference",
            question="What's in the audio?",
        )
        await server.verify(body)
        call_kwargs = server.server_client.post.call_args.kwargs
        sent_input = call_kwargs["json"].input
        assert len(sent_input) == 1
        assert sent_input[0].role == "user"
        assert "[Reference Answer]" in sent_input[0].content
        assert "[Model Answer]" in sent_input[0].content
        assert "[Question]" in sent_input[0].content
        assert "Rating: (int)" in sent_input[0].content
        assert "The reference" in sent_input[0].content
        assert "My answer" in sent_input[0].content
        assert "What's in the audio?" in sent_input[0].content


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics + get_key_metrics tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAggregateMetrics:
    def test_compute_metrics_perfect(self) -> None:
        server = _make_server()
        rollout = {
            "judge_correct": True,
            "judge_rating": 5.0,
            "generation": "good answer",
            "expected_answer": "ref",
            "question": "q",
            "reward": 1.0,
        }
        tasks = [[rollout, rollout]]
        metrics = server.compute_metrics(tasks)
        assert metrics["pass@1[avg-of-2]/accuracy"] == approx(100.0)
        # judge_score: 5.0 * 20 = 100.0
        assert metrics["pass@1[avg-of-2]/judge_score"] == approx(100.0)

    def test_compute_metrics_mid_rating(self) -> None:
        server = _make_server()
        rollouts = [
            {"judge_correct": True, "judge_rating": 4.0, "generation": "ok"},
            {"judge_correct": False, "judge_rating": 2.0, "generation": "meh"},
        ]
        tasks = [rollouts]
        metrics = server.compute_metrics(tasks)
        # accuracy: 1/2 = 50.0
        assert metrics["pass@1[avg-of-2]/accuracy"] == approx(50.0)
        # judge_score: avg(4,2)*20 = 60.0
        assert metrics["pass@1[avg-of-2]/judge_score"] == approx(60.0)

    def test_compute_metrics_per_dataset_breakdown(self) -> None:
        """Rows stamped with subset_for_metrics get per-sub-dataset metrics alongside the overall ones."""
        server = _make_server()
        tasks = [
            [
                {
                    "judge_correct": True,
                    "judge_rating": 5.0,
                    "generation": "a",
                    "subset_for_metrics": "alpaca_audio_test",
                }
            ],
            [{"judge_correct": False, "judge_rating": 1.0, "generation": "b", "subset_for_metrics": "mmau_mini"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["pass@1[avg-of-1]/accuracy"] == approx(50.0)
        assert metrics["alpaca_audio_test/pass@1[avg-of-1]/accuracy"] == approx(100.0)
        assert metrics["mmau_mini/pass@1[avg-of-1]/judge_score"] == approx(20.0)

    def test_no_answer_flag(self) -> None:
        empty = AudioBenchJudgeResourcesServer._score_fn(
            {"judge_correct": False, "judge_rating": 0.0, "generation": ""}
        )
        full = AudioBenchJudgeResourcesServer._score_fn(
            {"judge_correct": True, "judge_rating": 5.0, "generation": "yes"}
        )
        assert empty["no_answer"] == 1.0
        assert full["no_answer"] == 0.0

    def test_get_key_metrics_picks_highest_k(self) -> None:
        server = _make_server()
        agent_metrics = {
            "pass@1[avg-of-2]/accuracy": 80.0,
            "pass@1[avg-of-4]/accuracy": 75.0,
            "pass@2/accuracy": 85.0,
            "pass@4/accuracy": 90.0,
            "mean/input_tokens": 7,
            "mean/output_tokens": 42,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["pass@1[avg-of-4]/accuracy"] == 75.0
        assert key["pass@4/accuracy"] == 90.0
        assert key["mean/input_tokens"] == 7
        assert key["mean/output_tokens"] == 42


# ──────────────────────────────────────────────────────────────────────────────
# Example data
# ──────────────────────────────────────────────────────────────────────────────


def test_example_data_matches_generator_and_task_data_schema() -> None:
    rows = [json.loads(line) for line in EXAMPLE_DATA_PATH.read_text().splitlines() if line.strip()]
    generated_rows = [
        make_example(sample_id=f"example-{i:02d}", question=s["question"], expected_answer=s["expected_answer"])
        for i, s in enumerate(SAMPLES)
    ]
    assert rows == generated_rows
    for row in rows:
        assert row["responses_create_params"]["metadata"]["audio_data"].startswith("data:audio/wav;base64,")
        TaskData.model_validate({k: v for k, v in row.items() if k != "responses_create_params"})


def test_generate_example_data_main_writes_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_path = tmp_path / "example.jsonl"
    monkeypatch.setattr("sys.argv", ["generate_example_data.py", "--out", str(out_path)])
    main()
    assert out_path.read_text() == EXAMPLE_DATA_PATH.read_text()


async def test_verifier_fixture() -> None:
    from nemo_gym.verifier_fixture import exercise_verifier_fixture
    from resources_servers.audiobench_judge.app import VERIFIER_FIXTURE

    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed"}
