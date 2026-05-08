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
    evaluate_hallucination,
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


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_hallucination tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEvaluateHallucination:
    def test_normal_speech_rate_not_hallucinating(self) -> None:
        # 50 chars over 5 seconds = 600 chars/min — well below 1500 threshold.
        result = evaluate_hallucination("anything", "x" * 50, audio_duration=5.0)
        assert result["hallucination_rate"] == 0.0
        assert result["is_correct"] is True
        assert result["char_rate"] == pytest.approx(600.0)

    def test_excessive_char_rate_flagged_as_hallucination(self) -> None:
        # 200 chars over 5 seconds = 2400 chars/min — above 1500 threshold.
        result = evaluate_hallucination("", "x" * 200, audio_duration=5.0)
        assert result["hallucination_rate"] == 1.0
        assert result["is_correct"] is False
        assert result["char_rate"] == pytest.approx(2400.0)

    def test_missing_duration_returns_safe_default(self) -> None:
        result = evaluate_hallucination("ref", "hyp", audio_duration=None)
        assert result["hallucination_rate"] == 0.0
        assert result["char_rate"] == 0.0
        assert result["is_correct"] is True
        assert result["error"] == "missing_audio_duration"

    def test_zero_duration_returns_safe_default(self) -> None:
        result = evaluate_hallucination("ref", "hyp", audio_duration=0.0)
        assert result["error"] == "missing_audio_duration"
        assert result["is_correct"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination + ASR_LEADERBOARD server-level dispatch
# ──────────────────────────────────────────────────────────────────────────────


class TestHallucinationDispatch:
    async def test_hallucination_short_hyp_is_correct(self) -> None:
        """Empty expected_answer + 50 chars / 5s → is_correct, hallucination_rate=0."""
        server = _make_server(task_type="Hallucination")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("x" * 50),
            expected_answer="",
            audio_duration=5.0,
        )
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.is_correct is True
        assert result.hallucination_rate == 0.0
        assert result.char_rate == pytest.approx(600.0)

    async def test_hallucination_long_hyp_is_flagged(self) -> None:
        """200 chars / 5s = 2400 chars/min → hallucination_rate=1.0, is_correct=False."""
        server = _make_server(task_type="Hallucination")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("x" * 200),
            expected_answer="",
            audio_duration=5.0,
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.is_correct is False
        assert result.hallucination_rate == 1.0

    async def test_hallucination_missing_duration_is_correct(self) -> None:
        server = _make_server(task_type="Hallucination")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("x" * 200),
            expected_answer="",
            audio_duration=None,
        )
        result = await server.verify(body)
        # Missing duration falls back to is_correct=True.
        assert result.is_correct is True
        assert result.hallucination_rate == 0.0


class TestAsrLeaderboardDispatch:
    async def test_primary_wer_matches_evaluate_asr(self) -> None:
        """Primary WER against expected_answer mirrors ``evaluate_asr``."""
        server = _make_server(task_type="ASR_LEADERBOARD")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("one hundred dollars"),
            expected_answer="one hundred dollars",
            reference_fields=["text_tn", "text_itn"],
            text_tn="$100",
            text_itn="one hundred dollars",
        )
        result = await server.verify(body)
        # Primary metric: hyp matches expected_answer → wer == 0.
        primary = evaluate_asr("one hundred dollars", "one hundred dollars")
        assert result.wer == pytest.approx(primary["wer"])
        assert result.is_correct is True
        # Per-reference fields surfaced via ConfigDict(extra="allow").
        dump = result.model_dump()
        assert "wer_tn" in dump
        assert "wer_itn" in dump
        assert "is_correct_tn" in dump
        assert "is_correct_itn" in dump
        # text_itn matches the hypothesis exactly → wer_itn == 0, is_correct_itn == True.
        assert dump["wer_itn"] == pytest.approx(0.0)
        assert dump["is_correct_itn"] is True
        # text_tn ("$100") normalizes equivalently to "one hundred dollars" via
        # whisper-normalizer numeric expansion → also high agreement.
        assert dump["is_correct_tn"] is True

    async def test_high_wer_reference_marked_incorrect(self) -> None:
        server = _make_server(task_type="ASR_LEADERBOARD")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("totally unrelated transcription"),
            expected_answer="hello world",
            reference_fields=["text_alt"],
            text_alt="hello world",
        )
        result = await server.verify(body)
        dump = result.model_dump()
        # ``text_alt`` has the ``text_`` prefix → suffix is ``alt``.
        assert "wer_alt" in dump
        assert dump["wer_alt"] > 0.0
        assert dump["is_correct_alt"] is False

    async def test_non_text_prefixed_field_uses_full_name(self) -> None:
        """Field names without the 'text_' prefix pass through unchanged as the suffix."""
        server = _make_server(task_type="ASR_LEADERBOARD")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("hello world"),
            expected_answer="hello world",
            reference_fields=["alternate"],
            alternate="hello world",
        )
        result = await server.verify(body)
        dump = result.model_dump()
        # No "text_" prefix → suffix is the full field name.
        assert "wer_alternate" in dump
        assert dump["is_correct_alternate"] is True

    async def test_missing_reference_field_raises(self) -> None:
        """A reference_fields entry that isn't on the request must raise."""
        server = _make_server(task_type="ASR_LEADERBOARD")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("hi"),
            expected_answer="hi",
            reference_fields=["text_missing"],
        )
        with pytest.raises(ValueError, match="reference_fields entry 'text_missing'"):
            await server.verify(body)

    async def test_no_reference_fields_runs_primary_only(self) -> None:
        """ASR_LEADERBOARD without reference_fields should still produce primary WER."""
        server = _make_server(task_type="ASR_LEADERBOARD")
        body = ASRWithPCVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=_make_response("hello world"),
            expected_answer="hello world",
        )
        result = await server.verify(body)
        assert result.wer == 0.0
        assert result.is_correct is True


# ──────────────────────────────────────────────────────────────────────────────
# Mixed-task_type aggregation
# ──────────────────────────────────────────────────────────────────────────────


class TestMixedTaskTypeAggregation:
    def test_hallucination_rate_aggregates_as_mean(self) -> None:
        """Hallucination rollouts → ``hallucination_rate`` aggregates as sample mean."""
        server = _make_server(task_type="Hallucination")
        # Three tasks: rates 1.0, 0.0, 0.0 → mean 1/3 ≈ 33.33% (after *100).
        rollouts = [
            [
                {
                    "is_correct": False,
                    "per": 0.0,
                    "pred_text": "long",
                    "hallucination_rate": 1.0,
                    "text": "",
                    "wer_c": None,
                    "wer_pc": None,
                }
            ],
            [
                {
                    "is_correct": True,
                    "per": 0.0,
                    "pred_text": "short",
                    "hallucination_rate": 0.0,
                    "text": "",
                    "wer_c": None,
                    "wer_pc": None,
                }
            ],
            [
                {
                    "is_correct": True,
                    "per": 0.0,
                    "pred_text": "short",
                    "hallucination_rate": 0.0,
                    "text": "",
                    "wer_c": None,
                    "wer_pc": None,
                }
            ],
        ]
        metrics = server.compute_metrics(rollouts)
        # compute_pass_majority_metrics emits hallucination_rate via _score_fn:
        # mean across tasks @ k=1 → (1.0 + 0.0 + 0.0)/3 = 0.3333..., then *100.
        assert metrics["pass@1[avg-of-1]/hallucination_rate"] == pytest.approx(100.0 / 3.0)

    def test_per_reference_wer_corpus_level(self) -> None:
        """ASR_LEADERBOARD per-reference WERs aggregate corpus-level via jiwer."""
        server = _make_server(task_type="ASR_LEADERBOARD")
        # Two tasks, one rollout each. Both predict "hello world" exactly.
        # text_tn matches one row, mismatches the other.
        rollouts = [
            [
                {
                    "is_correct": True,
                    "per": 0.0,
                    "wer_c": None,
                    "wer_pc": None,
                    "text": "hello world",
                    "pred_text": "hello world",
                    "text_tn": "hello world",
                    "wer_tn": 0.0,
                    "is_correct_tn": True,
                }
            ],
            [
                {
                    "is_correct": True,
                    "per": 0.0,
                    "wer_c": None,
                    "wer_pc": None,
                    "text": "hello world",
                    "pred_text": "hello world",
                    "text_tn": "totally different words",
                    "wer_tn": 1.0,
                    "is_correct_tn": False,
                }
            ],
        ]
        metrics = server.compute_metrics(rollouts)
        # Corpus WER for the canonical "text"/"pred_text" — both perfect → 0.
        assert metrics["corpus_wer@k=1"] == pytest.approx(0.0)
        # Per-reference corpus WER built from text_tn → has nontrivial WER.
        assert "wer_tn@k=1" in metrics
        assert metrics["wer_tn@k=1"] > 0.0

    def test_get_key_metrics_surfaces_per_reference_wer(self) -> None:
        """``wer_<suffix>`` from compute_metrics flows up under headline names."""
        server = _make_server()
        agent_metrics = {
            "pass@1[avg-of-2]/accuracy": 80.0,
            "pass@2/accuracy": 85.0,
            "corpus_wer@k=2": 12.0,
            "wer_c@k=2": 9.0,
            "wer_pc@k=2": 14.0,
            "per@k=2": 22.0,
            "wer_tn@k=2": 15.0,
            "wer_itn@k=2": 11.0,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["wer"] == 12.0
        assert key["wer_tn"] == 15.0
        assert key["wer_itn"] == 11.0
        # Reserved suffixes (c, pc) stay under their canonical headline names.
        assert key["wer_c"] == 9.0
        assert key["wer_pc"] == 14.0
