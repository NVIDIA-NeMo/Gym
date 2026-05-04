# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Unit tests for the asr_with_pc resources server.

Each test fixes the model output and the reference transcript and asserts the
WER values against numeric expectations computed offline (jiwer 3.x).
"""

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.asr_with_pc.app import (
    ASRWithPCConfig,
    ASRWithPCResourcesServer,
    ASRWithPCVerifyRequest,
    calculate_per,
    evaluate_asr,
    evaluate_asr_pc,
    extract_punctuation,
    normalize_whitespace,
    preprocess_asr_text,
    split_tokens,
)


MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
    "parallel_tool_calls": True,
}


def _make_server(task_type: str = "ASR-PC") -> ASRWithPCResourcesServer:
    config = ASRWithPCConfig(host="0.0.0.0", port=8080, entrypoint="", name="", task_type=task_type)
    return ASRWithPCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(assistant_text: str) -> NeMoGymResponse:
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
                "content": [{"type": "output_text", "text": assistant_text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(assistant_text: str, expected_answer: str) -> ASRWithPCVerifyRequest:
    return ASRWithPCVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(assistant_text),
        expected_answer=expected_answer,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pure helper tests
# ──────────────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_normalize_whitespace(self) -> None:
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("\thello\n  world\t") == "hello world"
        assert normalize_whitespace("") == ""

    def test_split_tokens(self) -> None:
        assert split_tokens("Hello, world!") == ["Hello", ",", "world", "!"]
        assert split_tokens("don't stop") == ["don", "'", "t", "stop"]
        assert split_tokens("") == []

    def test_extract_punctuation(self) -> None:
        assert extract_punctuation("Hello, world!") == [",", "!"]
        assert extract_punctuation("plain text") == []
        assert extract_punctuation("It's 3.14 — nice.") == ["'", ".", "—", "."]

    def test_preprocess_asr_text_lowercases_and_strips_punct(self) -> None:
        # whisper-normalizer lowercases, strips most punctuation, expands numerics
        result = preprocess_asr_text("Hello, World!")
        assert result == "hello world"

    def test_calculate_per_identical(self) -> None:
        assert calculate_per("Hello, world!", "Hello, world!") == 0.0

    def test_calculate_per_no_punct(self) -> None:
        assert calculate_per("hello world", "hello world") == 0.0

    def test_calculate_per_missing_punct(self) -> None:
        # Reference has 1 punct, hyp has 0 → 1 deletion / (0+0+1+0) = 1.0
        assert calculate_per("hello, world", "hello world") == 1.0

    def test_calculate_per_extra_punct(self) -> None:
        # Reference has 0 punct, hyp has 1 → 1 insertion / (0+0+0+1) = 1.0
        assert calculate_per("hello world", "hello, world") == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_asr_pc tests (pure deterministic numerics)
# ──────────────────────────────────────────────────────────────────────────────


class TestEvaluateAsrPc:
    def test_perfect_match(self) -> None:
        result = evaluate_asr_pc("Hello, world.", "Hello, world.")
        assert result["wer"] == 0.0
        assert result["wer_c"] == 0.0
        assert result["wer_pc"] == 0.0
        assert result["per"] == 0.0
        assert result["is_correct"] is True

    def test_completely_different(self) -> None:
        result = evaluate_asr_pc("Hello world", "Goodbye universe")
        # Both tokens substituted: WER = 2/2 = 1.0
        assert result["wer"] == pytest.approx(1.0)
        assert result["wer_c"] == pytest.approx(1.0)
        assert result["is_correct"] is False

    def test_punct_only_diff(self) -> None:
        # Same words, different punctuation → wer == wer_c == 0; wer_pc > 0; per > 0
        result = evaluate_asr_pc("Hello, world.", "Hello world")
        assert result["wer"] == pytest.approx(0.0)
        assert result["wer_c"] == pytest.approx(0.0)
        assert result["wer_pc"] > 0.0
        assert result["per"] == pytest.approx(1.0)

    def test_case_only_diff(self) -> None:
        # Capitalization differs → wer == 0 (Whisper-normalized lowercases);
        # wer_c is jiwer over case-preserved text → > 0 (jiwer is case-sensitive).
        result = evaluate_asr_pc("Hello world", "hello world")
        assert result["wer"] == pytest.approx(0.0)
        assert result["wer_c"] > 0.0
        assert result["wer_pc"] > 0.0

    def test_returns_normalized_strings(self) -> None:
        result = evaluate_asr_pc("Hello, World!", "hello world")
        assert "text" in result
        assert "pred_text" in result
        assert result["text"] == "hello world"
        assert result["pred_text"] == "hello world"

    def test_evaluate_asr_perfect(self) -> None:
        """task_type=ASR path: standard WER only, Whisper-normalized."""
        result = evaluate_asr("Hello, world.", "Hello, world.")
        assert result["wer"] == 0.0
        assert result["is_correct"] is True

    def test_evaluate_asr_empty_reference_returns_none(self) -> None:
        """HF Open ASR Leaderboard convention: drop empty-reference rows."""
        result = evaluate_asr("", "anything")
        assert result["wer"] is None
        assert result["is_correct"] is None

    def test_evaluate_asr_empty_hypothesis_substitutes_empty(self) -> None:
        result = evaluate_asr("hello world", "")
        # ``evaluate_asr`` substitutes "empty" for an empty hypothesis
        assert result["pred_text"] == "empty"
        assert result["wer"] > 0.0

    def test_threshold_at_50_percent(self) -> None:
        # wer_pc < 0.5 → is_correct True; >= 0.5 → False
        good = evaluate_asr_pc("the quick brown fox", "the quick brown FOX")
        bad = evaluate_asr_pc("hello there", "totally wrong words")
        assert good["wer_pc"] < 0.5
        assert good["is_correct"] is True
        assert bad["wer_pc"] >= 0.5
        assert bad["is_correct"] is False


# ──────────────────────────────────────────────────────────────────────────────
# Server-level tests
# ──────────────────────────────────────────────────────────────────────────────


class TestASRWithPCServer:
    def test_sanity(self) -> None:
        server = _make_server()
        assert server is not None

    async def test_perfect_transcription_gives_reward_one(self) -> None:
        server = _make_server()
        body = _make_verify_request("Hello, world.", "Hello, world.")
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.is_correct is True
        assert result.wer == 0.0
        assert result.wer_pc == 0.0

    async def test_wrong_transcription_gives_reward_zero(self) -> None:
        server = _make_server()
        body = _make_verify_request("totally unrelated text", "Hello, world.")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.is_correct is False
        assert result.wer > 0.5

    async def test_empty_response_gives_reward_zero(self) -> None:
        server = _make_server()
        body = _make_verify_request("", "Hello, world.")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.is_correct is False

    async def test_normalized_strings_persisted(self) -> None:
        server = _make_server()
        body = _make_verify_request("Hello, World!", "Hello, world!")
        result = await server.verify(body)
        assert result.text == "hello world"
        assert result.pred_text == "hello world"
        assert result.ref_pc_tok != ""
        assert result.hyp_pc_tok != ""

    async def test_no_message_output_treated_as_empty(self) -> None:
        server = _make_server()
        empty_response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=empty_response,
            expected_answer="some text",
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.pred_text == ""

    async def test_asr_task_type_dispatch(self) -> None:
        """task_type=ASR scores standard WER only (no PC)."""
        server = _make_server(task_type="ASR")
        body = _make_verify_request("hello world", "hello world")
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.is_correct is True
        assert result.wer == 0.0
        # PC fields are zeroed under task_type=ASR
        assert result.wer_pc == 0.0
        assert result.wer_c == 0.0

    async def test_per_row_task_type_overrides_server_default(self) -> None:
        """A row with task_type=ASR beats the server's task_type=ASR-PC default."""
        server = _make_server(task_type="ASR-PC")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("hello world"),
            expected_answer="hello world",
            task_type="ASR",
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        # Standard WER computed (PC variants left at zero, since task_type=ASR).
        assert result.wer == 0.0
        assert result.wer_pc == 0.0

    async def test_unsupported_task_type_raises(self) -> None:
        # Pydantic enforces the Literal so this raises at request validation,
        # but we cover the server-side branch as a defense-in-depth check.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ASRWithPCVerifyRequest(
                responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
                response=_make_response("x"),
                expected_answer="y",
                task_type="Translation",  # not in the Literal union
            )


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics + get_key_metrics tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAggregateMetrics:
    def test_compute_metrics_perfect(self) -> None:
        """All rollouts perfect → corpus_wer == 0, pass@1 accuracy == 100."""
        server = _make_server()
        rollout = evaluate_asr_pc("hello world", "hello world")
        rollout["expected_answer"] = "hello world"
        # Two tasks, two rollouts each
        tasks = [[rollout, rollout], [rollout, rollout]]
        metrics = server.compute_metrics(tasks)

        assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(100.0)
        # Standard WER: corpus-level.
        assert metrics["corpus_wer@k=2"] == pytest.approx(0.0)
        # wer_pc / wer_c / per: mean-of-per-sample.
        assert metrics["wer_pc@k=2"] == pytest.approx(0.0)
        assert metrics["wer_c@k=2"] == pytest.approx(0.0)
        assert metrics["per@k=2"] == pytest.approx(0.0)

    def test_compute_metrics_all_wrong(self) -> None:
        server = _make_server()
        rollout = evaluate_asr_pc("hello world", "totally different output")
        tasks = [[rollout, rollout]]
        metrics = server.compute_metrics(tasks)

        assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(0.0)
        assert metrics["corpus_wer@k=2"] > 0.0
        # wer_pc is a sample-mean and should also be > 0 when all samples are wrong.
        assert metrics["wer_pc@k=2"] > 0.0

    def test_compute_metrics_empty_tasks(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_get_key_metrics_picks_highest_k(self) -> None:
        server = _make_server()
        agent_metrics = {
            "pass@1[avg-of-2]/accuracy": 80.0,
            "pass@1[avg-of-4]/accuracy": 75.0,
            "pass@2/accuracy": 85.0,
            "pass@4/accuracy": 90.0,
            "corpus_wer@k=2": 12.0,
            "corpus_wer@k=4": 11.0,
            "wer_c@k=4": 9.0,
            "wer_pc@k=4": 14.0,
            "per@k=4": 22.0,
            "mean/output_tokens": 42,
        }
        key = server.get_key_metrics(agent_metrics)

        # Highest k for pass@1[avg-of-k] is 4
        assert key["pass@1[avg-of-4]/accuracy"] == 75.0
        assert key["pass@4/accuracy"] == 90.0
        # WER aggregates exposed under headline names.
        assert key["wer"] == 11.0  # corpus_wer@k=4
        assert key["wer_c"] == 9.0
        assert key["wer_pc"] == 14.0
        assert key["per"] == 22.0
        assert key["mean/output_tokens"] == 42

    def test_score_fn_no_answer_flag(self) -> None:
        """Empty pred_text → no_answer == 1.0; non-empty → 0.0."""
        empty_score = ASRWithPCResourcesServer._score_fn({"is_correct": False, "per": 0.0, "pred_text": ""})
        full_score = ASRWithPCResourcesServer._score_fn({"is_correct": True, "per": 0.0, "pred_text": "hello"})
        assert empty_score["no_answer"] == 1.0
        assert full_score["no_answer"] == 0.0
