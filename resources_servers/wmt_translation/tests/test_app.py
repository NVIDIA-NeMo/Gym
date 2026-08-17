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
import logging
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from app import (
    BLEU_MIGRATION_WARNING,
    SPBLEU_TOKENIZER,
    WmtTranslationResourcesServer,
    WmtTranslationResourcesServerConfig,
    WmtTranslationVerifyRequest,
    _build_comet_actor_class,
    _normalize_for_scoring,
    _strip_reasoning_preamble,
)

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_server(
    compute_comet: bool = False,
    strip_reasoning: bool = False,
    comet_num_shards: int = 8,
    language_consistency_backend: Optional[str] = "wmt24pp_cld2",
    language_consistency_warning_threshold: float = 50.0,
) -> WmtTranslationResourcesServer:
    # Tests default strip_reasoning=False so plain-text generations score
    # against the reference directly. Production default is True (drops the
    # <think>...</think> preamble) for reasoning-model outputs.
    config = WmtTranslationResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        compute_comet=compute_comet,
        strip_reasoning=strip_reasoning,
        comet_num_shards=comet_num_shards,
        language_consistency_backend=language_consistency_backend,
        language_consistency_warning_threshold=language_consistency_warning_threshold,
    )
    return WmtTranslationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_request(text: str, translation: str, generation: str, target_language: str) -> WmtTranslationVerifyRequest:
    return WmtTranslationVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": f"Translate: {text}"}],
            "parallel_tool_calls": False,
            "temperature": 0,
        },
        response=_make_response(generation),
        text=text,
        translation=translation,
        source_language="en",
        target_language=target_language,
        source_lang_name="English",
        target_lang_name="German",
    )


class TestStripReasoningPreamble:
    def test_takes_text_after_close_tag(self) -> None:
        text = "We need to translate the segment.\n</think>\nProlog"
        assert _strip_reasoning_preamble(text) == "Prolog"

    def test_strips_trailing_newlines_before_answer(self) -> None:
        text = "Reasoning.\n</think>\n\nHallo Welt."
        # Only leading newlines get lstripped; embedded newlines in the answer stay.
        assert _strip_reasoning_preamble(text) == "Hallo Welt."

    def test_uses_last_close_tag(self) -> None:
        # Edge case: model emits </think> inside the reasoning (rare, but defensive).
        text = "Step 1: </think> Step 2 thinking. </think>\nFinal."
        assert _strip_reasoning_preamble(text) == "Final."

    def test_empty_when_truncated_mid_reasoning(self) -> None:
        # <think> opened but never closed → truncated reasoning, count as no-answer.
        assert _strip_reasoning_preamble("<think>unfinished reasoning...") == ""

    def test_returns_text_unchanged_when_no_reasoning_tags(self) -> None:
        # Endpoints that return reasoning as a structured output[i].type="reasoning"
        # block (OpenAI Responses API style) leave output_text clean of <think> /
        # </think>. The text is the answer — don't blank it.
        assert _strip_reasoning_preamble("Hallo Welt.") == "Hallo Welt."

    def test_empty_input(self) -> None:
        assert _strip_reasoning_preamble("") == ""


class TestScoringHelpers:
    def test_normalizes_canonically_equivalent_text_to_nfc(self) -> None:
        assert _normalize_for_scoring("Cafe\u0301") == "Café"

    def test_spbleu_always_uses_flores200_tokenizer(self) -> None:
        assert SPBLEU_TOKENIZER == "flores200"


class TestServerSetup:
    def test_warns_that_spbleu_is_not_comparable_to_prior_bleu(self, caplog) -> None:
        server = _make_server()

        with caplog.at_level(logging.WARNING, logger="app"):
            server.setup_webserver()

        assert BLEU_MIGRATION_WARNING in caplog.messages


class TestVerify:
    @pytest.mark.parametrize("target_language", ["de_DE", "ja_JP", "zho_Hans"])
    async def test_spbleu_uses_flores200_for_every_target_language(self, target_language: str) -> None:
        server = _make_server(language_consistency_backend=None)
        request = _make_request(
            text="Hello world.",
            translation="Hallo Welt.",
            generation="Hallo Welt.",
            target_language=target_language,
        )
        with patch("app.BLEU") as mock_bleu:
            mock_bleu.return_value.sentence_score.return_value.score = 100.0
            result = await server.verify(request)

        assert result.sentence_spbleu == 100.0
        mock_bleu.assert_called_once_with(tokenize="flores200", effective_order=True)
        mock_bleu.return_value.sentence_score.assert_called_once_with("Hallo Welt.", ["Hallo Welt."])

    async def test_spbleu_metric_is_reused_across_verify_calls(self) -> None:
        server = _make_server(language_consistency_backend=None)
        request = _make_request(
            text="Hello world.",
            translation="Hallo Welt.",
            generation="Hallo Welt.",
            target_language="de_DE",
        )
        with patch("app.BLEU") as mock_bleu:
            mock_bleu.return_value.sentence_score.return_value.score = 100.0
            first_result = await server.verify(request)
            second_result = await server.verify(request)

        assert first_result.sentence_spbleu == 100.0
        assert second_result.sentence_spbleu == 100.0
        mock_bleu.assert_called_once_with(tokenize="flores200", effective_order=True)
        assert mock_bleu.return_value.sentence_score.call_count == 2

    async def test_empty_generation_scores_zero(self) -> None:
        server = _make_server()
        request = _make_request(
            text="Hello world.",
            translation="Hallo Welt.",
            generation="",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.reward == 0.0
        assert result.sentence_chrf == 0.0
        assert result.sentence_spbleu == 0.0
        assert result.generation == ""
        # Empty output is 0% target language.
        assert result.language_consistency_score == 0.0

    async def test_language_consistency_high_for_correct_language(self) -> None:
        server = _make_server()
        ref = "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog in the beautiful garden.",
            translation=ref,
            generation=ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.language_consistency_score is not None
        assert result.language_consistency_score > 0.5

    async def test_language_consistency_low_for_wrong_language(self) -> None:
        server = _make_server()
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation="Der schnelle braune Fuchs springt über den faulen Hund.",
            generation="This is clearly an English sentence, not German at all.",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.language_consistency_score is not None
        assert result.language_consistency_score < 0.5

    @patch("app.get_language_consistency_backend")
    async def test_language_consistency_backend_receives_full_flores_code(
        self, mock_get_language_consistency_backend
    ) -> None:
        scorer = MagicMock(return_value=0.75)
        mock_get_language_consistency_backend.return_value = scorer
        server = _make_server(language_consistency_backend="flores_glotlid")
        request = _make_request(
            text="Hello.",
            translation="مرحبًا.",
            generation="مرحبًا.",
            target_language="apc_Arab_sout3123",
        )

        result = await server.verify(request)

        mock_get_language_consistency_backend.assert_called_once_with("flores_glotlid")
        scorer.assert_called_once_with("مرحبًا.", "apc_Arab_sout3123")
        assert result.language_consistency_score == 0.75

    @patch("app.get_language_consistency_backend")
    async def test_language_consistency_backend_is_reused_across_verify_calls(
        self, mock_get_language_consistency_backend
    ) -> None:
        scorer = MagicMock(return_value=0.75)
        mock_get_language_consistency_backend.return_value = scorer
        server = _make_server(language_consistency_backend="flores_glotlid")
        request = _make_request(
            text="Hello.",
            translation="مرحبًا.",
            generation="مرحبًا.",
            target_language="apc_Arab_sout3123",
        )

        with patch.object(WmtTranslationResourcesServer, "_score_sentence_spbleu", return_value=100.0):
            first_result = await server.verify(request)
            second_result = await server.verify(request)

        mock_get_language_consistency_backend.assert_called_once_with("flores_glotlid")
        assert scorer.call_args_list == [
            (("مرحبًا.", "apc_Arab_sout3123"),),
            (("مرحبًا.", "apc_Arab_sout3123"),),
        ]
        assert first_result.language_consistency_score == 0.75
        assert second_result.language_consistency_score == 0.75

    async def test_language_consistency_none_when_disabled(self) -> None:
        server = _make_server(language_consistency_backend=None)
        ref = "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog in the beautiful garden.",
            translation=ref,
            generation=ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.language_consistency_score is None

    @patch("app.get_language_consistency_backend")
    async def test_language_consistency_none_when_no_backend_is_configured(
        self, mock_get_language_consistency_backend
    ) -> None:
        server = _make_server(language_consistency_backend=None)
        ref = "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog in the beautiful garden.",
            translation=ref,
            generation=ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.language_consistency_score is None

        empty_request = _make_request(
            text="The quick brown fox jumps over the lazy dog in the beautiful garden.",
            translation=ref,
            generation="",
            target_language="de_DE",
        )
        empty_result = await server.verify(empty_request)
        assert empty_result.language_consistency_score is None

        mock_get_language_consistency_backend.assert_not_called()

    async def test_perfect_generation_high_reward(self) -> None:
        server = _make_server()
        # Long enough for 4-gram precisions to be non-zero.
        ref = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation=ref,
            generation=ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.sentence_chrf > 50.0
        assert result.sentence_spbleu > 50.0
        assert result.reward == result.sentence_chrf / 100.0
        assert result.generation == ref

    async def test_canonically_equivalent_unicode_scores_perfectly(self) -> None:
        server = _make_server()
        request = _make_request(
            text="Coffee.",
            translation="Café",
            generation="Cafe\u0301",
            target_language="fr_FR",
        )
        result = await server.verify(request)
        assert result.sentence_chrf == pytest.approx(100.0)
        assert result.sentence_spbleu == pytest.approx(100.0)
        assert result.reward == pytest.approx(1.0)
        assert result.generation == "Cafe\u0301"

    async def test_bad_generation_low_reward(self) -> None:
        server = _make_server()
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation="Der schnelle braune Fuchs springt \u00fcber den faulen Hund.",
            generation="Something entirely unrelated in English about cats.",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.reward < 0.2

    async def test_strip_reasoning_recovers_score(self) -> None:
        """With strip_reasoning=True, a reasoning preamble must not
        contaminate the scored generation."""
        server = _make_server(strip_reasoning=True)
        ref = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund."
        reasoning_preamble = (
            "We need to translate to German, without additional explanation. "
            "Output just the translated sentence.\n</think>\n"
        )
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation=ref,
            generation=reasoning_preamble + ref,
            target_language="de_DE",
        )
        result = await server.verify(request)
        # Exactly the reference should produce a near-perfect score.
        assert result.generation == ref
        assert result.sentence_chrf > 50.0
        assert result.sentence_spbleu > 50.0

    async def test_strip_reasoning_empty_when_truncated_mid_reasoning(self) -> None:
        """If <think> opens but never closes, verify() emits no generation."""
        server = _make_server(strip_reasoning=True)
        request = _make_request(
            text="Hello.",
            translation="Hallo.",
            generation="<think>We are still thinking about the answer, no close tag.",
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.generation == ""
        assert result.reward == 0.0

    async def test_strip_reasoning_keeps_text_without_reasoning_tags(self) -> None:
        """Clean output_text (no <think>/</think>) passes through strip_reasoning.

        Endpoints returning reasoning as a structured output[i].type='reasoning'
        block (OpenAI Responses API style) leave output_text with just the
        answer — blanking it on strip_reasoning=True zeros rewards everywhere.
        """
        server = _make_server(strip_reasoning=True)
        ref = "Der schnelle braune Fuchs springt über den faulen Hund."
        request = _make_request(
            text="The quick brown fox jumps over the lazy dog.",
            translation=ref,
            generation=ref,  # clean, no <think> tags
            target_language="de_DE",
        )
        result = await server.verify(request)
        assert result.generation == ref
        assert result.sentence_chrf > 50.0
        assert result.sentence_spbleu > 50.0


class TestComputeMetrics:
    def test_corpus_spbleu_uses_flores200_for_every_language_pair(self) -> None:
        server = _make_server(compute_comet=False)
        tasks = [
            [
                {
                    "translation": "Hallo Welt.",
                    "generation": "Hallo Welt.",
                    "source_language": "en",
                    "target_language": target_language,
                }
            ]
            for target_language in ("de_DE", "ja_JP", "zho_Hans")
        ]
        with patch("app.corpus_spbleu") as mock_spbleu:
            mock_spbleu.return_value.score = 100.0
            server.compute_metrics(tasks)

        assert len(mock_spbleu.call_args_list) == 3
        assert all(call.kwargs["tokenize"] == "flores200" for call in mock_spbleu.call_args_list)

    def test_empty_tasks(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_corpus_chrf_normalizes_canonically_equivalent_unicode(self) -> None:
        server = _make_server(compute_comet=False)
        metrics = server.compute_metrics(
            [
                [
                    {
                        "translation": "Café au lait chaud",
                        "generation": "Cafe\u0301 au lait chaud",
                        "source_language": "en",
                        "target_language": "fr_FR",
                    }
                ]
            ]
        )
        assert metrics["en->fr_FR/chrF"] == pytest.approx(100.0)
        assert metrics["en->fr_FR/spBLEU"] == pytest.approx(100.0)

    def test_spbleu_and_chrf_per_pair_and_aggregations(self) -> None:
        """Feed two language pairs x two rollouts each; expect spBLEU and chrF
        cross-pair aggregates, and std-dev keys without COMET fields."""
        server = _make_server(compute_comet=False)
        # Two tasks with two rollouts each: rollout 0 is perfect and rollout 1
        # is a slight variant, so the standard deviation across runs is
        # non-zero while both runs have positive chrF.
        de_ref = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund in dem sch\u00f6nen Garten."
        de_perfect = de_ref
        de_variant = "Der schnelle braune Fuchs springt \u00fcber den faulen Hund im sch\u00f6nen Garten."
        fr_ref = "Le renard brun rapide saute par dessus le chien paresseux dans le beau jardin."
        fr_perfect = fr_ref
        fr_variant = "Le renard brun rapide saute au dessus du chien paresseux dans le beau jardin."
        tasks = [
            # Task 1: en->de_DE
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": de_ref,
                    "generation": de_perfect,
                    "source_language": "en",
                    "target_language": "de_DE",
                },
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": de_ref,
                    "generation": de_variant,
                    "source_language": "en",
                    "target_language": "de_DE",
                },
            ],
            # Task 2: en->fr_FR
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": fr_ref,
                    "generation": fr_perfect,
                    "source_language": "en",
                    "target_language": "fr_FR",
                },
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": fr_ref,
                    "generation": fr_variant,
                    "source_language": "en",
                    "target_language": "fr_FR",
                },
            ],
        ]
        m = server.compute_metrics(tasks)

        # Per-pair chrF
        assert "en->de_DE/chrF" in m
        assert "en->fr_FR/chrF" in m
        assert m["en->de_DE/chrF"] > 0
        assert m["en->fr_FR/chrF"] > 0
        # Std-dev keys exist for per-pair
        assert "en->de_DE/chrF_std_dev_across_runs" in m
        assert "en->fr_FR/chrF_std_dev_across_runs" in m

        # Per-pair spBLEU
        assert m["en->de_DE/spBLEU"] > 0
        assert m["en->fr_FR/spBLEU"] > 0
        assert "en->de_DE/spBLEU_std_dev_across_runs" in m
        assert "en->fr_FR/spBLEU_std_dev_across_runs" in m

        # Cross-pair aggregations
        assert "xx->xx/chrF" in m
        assert "en->xx/chrF" in m
        assert "xx->de_DE/chrF" in m
        assert "xx->fr_FR/chrF" in m
        assert "xx->xx/spBLEU" in m
        assert "en->xx/spBLEU" in m
        assert "xx->de_DE/spBLEU" in m
        assert "xx->fr_FR/spBLEU" in m

        # No COMET when disabled
        assert not any(k.endswith("/comet") for k in m)

    def test_comet_disabled_does_not_call_ray(self) -> None:
        """With compute_comet=False, compute_metrics must not call Ray's
        COMET path or add /comet keys even when triples would otherwise exist."""
        server = _make_server(compute_comet=False)
        tasks = [
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": "Der schnelle braune Fuchs springt \u00fcber den faulen Hund im sch\u00f6nen Garten.",
                    "generation": "Der schnelle braune Fuchs springt \u00fcber den faulen Hund im sch\u00f6nen Garten.",
                    "source_language": "en",
                    "target_language": "de_DE",
                }
            ]
        ]
        m = server.compute_metrics(tasks)
        # spBLEU and chrF are emitted; /comet keys are not.
        assert "en->de_DE/chrF" in m
        assert "en->de_DE/spBLEU" in m
        for k in m:
            assert "/comet" not in k

    def test_get_key_metrics_filters(self) -> None:
        server = _make_server()
        agent = {
            "xx->xx/chrF": 35.0,
            "xx->xx/spBLEU": 30.0,
            "xx->xx/comet": 78.0,
            "en->xx/chrF": 32.0,
            "en->xx/spBLEU": 27.0,
            "en->xx/comet": 77.0,
            "eng_Latn->xx/chrF": 31.0,
            "eng_Latn->xx/spBLEU": 26.0,
            "eng_Latn->xx/comet": 76.0,
            "eng_Latn->xx/language_consistency": 96.0,
            "en->de_DE/chrF": 30.0,  # not in key metrics
            "mean/reward": 0.45,  # not in key metrics
        }
        key = server.get_key_metrics(agent)
        assert set(key.keys()) == {
            "xx->xx/chrF",
            "xx->xx/spBLEU",
            "xx->xx/comet",
            "en->xx/chrF",
            "en->xx/spBLEU",
            "en->xx/comet",
            "eng_Latn->xx/chrF",
            "eng_Latn->xx/spBLEU",
            "eng_Latn->xx/comet",
            "eng_Latn->xx/language_consistency",
        }

    def test_per_row_comet_scores_emit_aggregate_metrics(self) -> None:
        """When rollouts carry comet_score (from verify()'s actor pool await),
        compute_metrics buckets those values directly into per-pair and
        cross-pair aggregates."""
        server = _make_server(compute_comet=True)
        de_ref = "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten."
        fr_ref = "Le renard brun rapide saute par dessus le chien paresseux dans le beau jardin."
        tasks = [
            [
                {
                    "text": "T1",
                    "translation": de_ref,
                    "generation": de_ref,
                    "source_language": "en",
                    "target_language": "de_DE",
                    "comet_score": 0.85,
                },
            ],
            [
                {
                    "text": "T2",
                    "translation": de_ref,
                    "generation": de_ref,
                    "source_language": "en",
                    "target_language": "de_DE",
                    "comet_score": 0.95,
                },
            ],
            [
                {
                    "text": "T3",
                    "translation": fr_ref,
                    "generation": fr_ref,
                    "source_language": "en",
                    "target_language": "fr_FR",
                    "comet_score": 0.90,
                },
            ],
        ]
        m = server.compute_metrics(tasks)
        # Per-pair: en->de_DE = mean(0.85, 0.95) × 100 = 90.0
        # Per-pair: en->fr_FR = 0.90 × 100 = 90.0
        assert m["en->de_DE/comet"] == pytest.approx(90.0)
        assert m["en->fr_FR/comet"] == pytest.approx(90.0)
        # Cross-pair aggregations.
        assert m["xx->xx/comet"] == pytest.approx(90.0)
        assert m["en->xx/comet"] == pytest.approx(90.0)
        assert m["xx->de_DE/comet"] == pytest.approx(90.0)
        assert m["xx->fr_FR/comet"] == pytest.approx(90.0)

    def test_per_row_language_consistency_scores_emit_aggregate_metrics(self) -> None:
        """When rollouts carry language_consistency_score (computed in verify()), compute_metrics
        buckets those values into per-pair and cross-pair aggregates."""
        server = _make_server(
            compute_comet=False,
            language_consistency_backend="wmt24pp_cld2",
        )
        de_ref = "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten."
        fr_ref = "Le renard brun rapide saute par dessus le chien paresseux dans le beau jardin."
        tasks = [
            [
                {
                    "text": "T1",
                    "translation": de_ref,
                    "generation": de_ref,
                    "source_language": "en",
                    "target_language": "de_DE",
                    "language_consistency_score": 0.80,
                },
            ],
            [
                {
                    "text": "T2",
                    "translation": de_ref,
                    "generation": de_ref,
                    "source_language": "en",
                    "target_language": "de_DE",
                    "language_consistency_score": 1.00,
                },
            ],
            [
                {
                    "text": "T3",
                    "translation": fr_ref,
                    "generation": fr_ref,
                    "source_language": "en",
                    "target_language": "fr_FR",
                    "language_consistency_score": 0.90,
                },
            ],
        ]
        m = server.compute_metrics(tasks)
        # Per-pair: en->de_DE = mean(0.80, 1.00) × 100 = 90.0; en->fr_FR = 90.0
        assert m["en->de_DE/language_consistency"] == pytest.approx(90.0)
        assert m["en->fr_FR/language_consistency"] == pytest.approx(90.0)
        # Cross-pair aggregations.
        assert m["xx->xx/language_consistency"] == pytest.approx(90.0)
        assert m["en->xx/language_consistency"] == pytest.approx(90.0)
        assert m["xx->de_DE/language_consistency"] == pytest.approx(90.0)
        assert m["xx->fr_FR/language_consistency"] == pytest.approx(90.0)
        # spBLEU and chrF are present; COMET is absent (disabled).
        assert "en->de_DE/chrF" in m
        assert "en->de_DE/spBLEU" in m
        assert not any(k.endswith("/comet") for k in m)

    def test_no_language_consistency_rows_emits_no_language_consistency_keys(self) -> None:
        """Rollouts without language_consistency_score yield no /language_consistency keys."""
        server = _make_server(
            compute_comet=False,
            language_consistency_backend="wmt24pp_cld2",
        )
        tasks = [
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten.",
                    "generation": "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten.",
                    "source_language": "en",
                    "target_language": "de_DE",
                    # No language_consistency_score field.
                }
            ]
        ]
        m = server.compute_metrics(tasks)
        assert "en->de_DE/chrF" in m
        assert "en->de_DE/spBLEU" in m
        assert not any("/language_consistency" in k for k in m)

    def test_warns_for_low_language_consistency_pairs_with_scores(self, caplog) -> None:
        server = _make_server(
            compute_comet=False,
            language_consistency_backend="flores_glotlid",
            language_consistency_warning_threshold=50.0,
        )
        tasks = [
            [
                {
                    "translation": "reference text",
                    "generation": "generated text",
                    "source_language": src,
                    "target_language": tgt,
                    "language_consistency_score": score,
                }
            ]
            for src, tgt, score in (
                ("en", "jpn_Jpan", 0.40),
                ("de", "jpn_Jpan", 0.20),
                ("en", "fra_Latn", 0.50),
            )
        ]

        with caplog.at_level(logging.WARNING, logger="app"):
            server.compute_metrics(tasks)

        warning = next(
            message for message in caplog.messages if message.startswith("Warning - the following language pairs")
        )
        assert "scores < 50.0" in warning
        assert "  de -> jpn_Jpan: 20.0\n  en -> jpn_Jpan: 40.0" in warning
        assert "en -> fra_Latn" not in warning

    def test_does_not_warn_when_language_consistency_pairs_meet_threshold(self, caplog) -> None:
        server = _make_server(
            compute_comet=False,
            language_consistency_backend="wmt24pp_cld2",
            language_consistency_warning_threshold=50.0,
        )
        tasks = [
            [
                {
                    "translation": "reference text",
                    "generation": "generated text",
                    "source_language": "en",
                    "target_language": "ja_JP",
                    "language_consistency_score": 0.50,
                }
            ]
        ]

        with caplog.at_level(logging.WARNING, logger="app"):
            server.compute_metrics(tasks)

        assert not any(message.startswith("Warning - the following language pairs") for message in caplog.messages)

    def test_no_comet_rows_emits_local_metrics_only(self) -> None:
        """Rollouts without comet_score (compute_comet disabled mid-run, or
        actor pool unavailable) yield spBLEU/chrF metrics with no /comet keys."""
        server = _make_server(compute_comet=True)
        tasks = [
            [
                {
                    "text": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
                    "translation": "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten.",
                    "generation": "Der schnelle braune Fuchs springt über den faulen Hund im schönen Garten.",
                    "source_language": "en",
                    "target_language": "de_DE",
                    # No comet_score field → simulates pool unavailable.
                }
            ]
        ]
        m = server.compute_metrics(tasks)
        assert "en->de_DE/chrF" in m
        assert "en->de_DE/spBLEU" in m
        assert not any(k.endswith("/comet") for k in m)


class TestBuildCometActorClass:
    """Unit tests for _build_comet_actor_class() — the pre-dispatch setup logic.

    The inner @ray.remote actor class requires a live Ray cluster + GPUs +
    unbabel-comet (a gated ~10B checkpoint), so we can't construct the actor
    itself in unit tests. What we *can* cover is the code that builds it:
    venv Python resolution, cross-node Python mirror, env_vars propagation,
    and ray.remote decoration args.
    """

    def _stub_ray_remote(self, captured: dict):
        """Return a ray.remote replacement that captures the decoration kwargs."""

        def _ray_remote(**decorator_kwargs):
            captured["decorator_kwargs"] = decorator_kwargs

            def _decorate(cls_or_fn):
                class _Decorated:
                    _wrapped = cls_or_fn

                    @staticmethod
                    def remote(*args, **kwargs):
                        raise AssertionError("actor must not instantiate in unit tests")

                return _Decorated

            return _decorate

        return _ray_remote

    def _fake_venv(self, tmp_path: Path) -> Path:
        """Build a fake uv-style Python install and return its venv python path."""
        uv_root = tmp_path / "uv" / "cpython-3.12.12-linux-x86_64-gnu"
        venv_bin = tmp_path / "venv" / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (uv_root / "bin").mkdir(parents=True)
        real_python = uv_root / "bin" / "python3.12"
        real_python.write_text("")
        fake_python = venv_bin / "python3.12"
        fake_python.symlink_to(real_python)
        (venv_bin.parent / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
        return fake_python

    def test_propagates_runtime_env_and_pins_py_executable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """_build_comet_actor_class must request ray.remote with an env_vars
        dict that preserves CUDA_VISIBLE_DEVICES, threads site-packages onto
        PYTHONPATH, propagates HF_HOME so actors find the prepared cache,
        and pins py_executable to the cross-node-mirrored Python. HF
        offline/online flags are intentionally NOT overridden — the
        benchmark prepare step pre-fetches the COMET model + tokenizer
        into HF_HOME so runtime is fully offline.
        """
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setenv("HF_HOME", "/tmp/hf_home")
        monkeypatch.setenv("PYTHONPATH", "/existing/pp")

        captured = {}
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote(captured)))

        _build_comet_actor_class()

        kw = captured["decorator_kwargs"]
        assert kw["num_gpus"] == 0
        assert kw["resources"] == {"extra_gpu": 1}

        env = kw["runtime_env"]["env_vars"]
        assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
        assert "site-packages" in env["PYTHONPATH"]
        assert "/existing/pp" in env["PYTHONPATH"]
        assert env["HF_HOME"] == "/tmp/hf_home"
        # No HF online/offline overrides — actors inherit from parent.
        assert "HF_HUB_OFFLINE" not in env
        assert "TRANSFORMERS_OFFLINE" not in env
        # No token propagation — runtime is offline post-prepare.
        assert "HF_TOKEN" not in env
        assert "HUGGING_FACE_HUB_TOKEN" not in env

        py_exec = kw["runtime_env"]["py_executable"]
        assert py_exec.startswith(str(mirror_root))
        assert py_exec.endswith("bin/python3.12")
        # Mirror was performed on first invocation.
        assert (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").exists()

    def test_reuses_existing_mirror_without_recopy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Second invocation must skip copytree when the mirror already exists."""
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin").mkdir(parents=True)
        (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").write_text("")

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        import shutil as shutil_mod

        with patch.object(shutil_mod, "copytree") as mock_copy:
            _build_comet_actor_class()
            mock_copy.assert_not_called()

    def test_stale_unique_tmp_from_prior_crash_does_not_block_new_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Per-writer staging paths are isolated, so a stale tmp from a crashed builder is ignored."""
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        mirror_root.mkdir()
        stale_tmp = mirror_root / ".cpython-3.12.12-linux-x86_64-gnu.tmp.someoldhost.99999.deadbeef"
        stale_tmp.mkdir()
        (stale_tmp / "leftover.txt").write_text("from prior crashed builder")

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        _build_comet_actor_class()

        assert (mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin" / "python3.12").exists()
        assert stale_tmp.exists()

    def test_concurrent_builders_both_succeed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Two builders racing the same cache root both return cleanly, leave one mirror, no leaks."""
        import shutil as shutil_mod
        import threading

        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        # Barrier at the start of copytree forces both threads into the rename phase together.
        barrier = threading.Barrier(2)
        real_copytree = shutil_mod.copytree

        def _coordinated_copytree(*args, **kwargs):
            barrier.wait(timeout=5)
            return real_copytree(*args, **kwargs)

        monkeypatch.setattr(shutil_mod, "copytree", _coordinated_copytree)

        results: list[BaseException | None] = [None, None]

        def _run(idx: int) -> None:
            try:
                _build_comet_actor_class()
            except BaseException as exc:
                results[idx] = exc

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert results[0] is None, f"builder 0 raised: {results[0]!r}"
        assert results[1] is None, f"builder 1 raised: {results[1]!r}"

        published = mirror_root / "cpython-3.12.12-linux-x86_64-gnu"
        assert (published / "bin" / "python3.12").exists()

        leftover_tmps = list(mirror_root.glob(".cpython-3.12.12-linux-x86_64-gnu.tmp.*"))
        assert leftover_tmps == [], f"staging tmp dirs leaked: {leftover_tmps}"

    def test_rename_loser_adopts_winner_mirror(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A rename collision against a valid published mirror is silently adopted and staging is cleaned."""
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"
        mirror_root.mkdir()
        published_bin = mirror_root / "cpython-3.12.12-linux-x86_64-gnu" / "bin"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        # Simulate "winner published between our copytree and our rename".
        def _failing_rename(self, *_args, **_kwargs):
            published_bin.mkdir(parents=True, exist_ok=True)
            (published_bin / "python3.12").write_text("winner's python")
            raise OSError(39, "Directory not empty")

        with patch.object(Path, "rename", new=_failing_rename):
            _build_comet_actor_class()

        assert (published_bin / "python3.12").read_text() == "winner's python"
        leftover_tmps = list(mirror_root.glob(".cpython-3.12.12-linux-x86_64-gnu.tmp.*"))
        assert leftover_tmps == [], f"staging tmp dirs leaked: {leftover_tmps}"

    def test_rename_failure_without_valid_mirror_reraises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A rename OSError with no published mirror re-raises (genuine permission / disk error)."""
        import app as app_module

        fake_python = self._fake_venv(tmp_path)
        mirror_root = tmp_path / "mirror_cache"

        monkeypatch.setattr(sys, "executable", str(fake_python))
        monkeypatch.setenv("WMT_TRANSLATION_COMET_PY_CACHE", str(mirror_root))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        def _failing_rename(self, *_args, **_kwargs):
            raise OSError(13, "Permission denied")

        with patch.object(Path, "rename", new=_failing_rename), pytest.raises(OSError, match="Permission denied"):
            _build_comet_actor_class()

    def test_raises_if_sys_executable_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Defensive: if sys.executable points at a vanished path, fail loudly."""
        import app as app_module

        monkeypatch.setattr(sys, "executable", str(tmp_path / "does_not_exist"))
        monkeypatch.setattr(app_module, "ray", MagicMock(remote=self._stub_ray_remote({})))

        with pytest.raises(RuntimeError, match="sys.executable doesn't exist"):
            _build_comet_actor_class()
