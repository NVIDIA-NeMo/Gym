# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import in22_metrics
import pytest
from app import WmtTranslationResourcesServer, WmtTranslationResourcesServerConfig, WmtTranslationVerifyRequest
from in22_metrics import clean_in22_response, compute_in22_metrics, corpus_metrics, normalize_in22

from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="model",
        object="response",
        output=[
            {
                "id": "message",
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


def _server() -> WmtTranslationResourcesServer:
    config = WmtTranslationResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        compute_comet=False,
        language_consistency_backend=None,
        metric_profile="in22",
    )
    return WmtTranslationResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" translation ", "translation"),
        ("<think>reasoning</think>\ntranslation", "translation"),
        ("<think>unfinished", ""),
        ("```text\ntranslation\n```", "translation"),
        ("```\nline one\nline two\n```", "line one\nline two"),
    ],
)
def test_clean_response_matches_reference_filter(raw: str, expected: str) -> None:
    assert clean_in22_response(raw) == expected


def test_indic_normalization_uses_reference_normalizer_and_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    normalizer = MagicMock()
    normalizer.normalize.return_value = "normalized"
    tokenizer = MagicMock()
    tokenizer.trivial_tokenize.return_value = ["normalized", "tokens"]
    monkeypatch.setattr(in22_metrics, "_indic_tools", lambda language: (normalizer, tokenizer))

    assert normalize_in22("  input  ", "hi") == "normalized tokens"
    normalizer.normalize.assert_called_once_with("input")
    tokenizer.trivial_tokenize.assert_called_once_with("normalized", "hi")


def test_english_does_not_load_indic_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        in22_metrics,
        "_indic_tools",
        lambda _language: pytest.fail("Indic tools should not load for English"),
    )
    assert normalize_in22("  English text  ", "en") == "English text"


def test_corpus_metrics_are_perfect_for_identical_english_text() -> None:
    metrics = corpus_metrics(
        ["This is a complete reference sentence."],
        ["This is a complete reference sentence."],
        "en",
    )
    assert metrics == pytest.approx({"chrf": 100.0, "chrf++": 100.0, "bleu": 100.0})


# Frozen from the supplied reference utils.py, SHA-256:
# da5ee94c4e74d82dc6d0a689802932f5a6cae0ecd8df0a9fb8ba5b522fc56481
# SacreBLEU 2.6.0, IndicNLP 0.92, no UrduHack; provenance is in the IN22 README.
@pytest.mark.parametrize(
    ("language", "reference", "hypothesis", "expected"),
    [
        (
            "en",
            "Hello, world! This is a test.",
            "Hello world; this is another test.",
            {"chrf": 46.846724488, "chrf++": 43.965680971, "bleu": 12.256200970},
        ),
        (
            "hi",
            "क़लम से ख़त लिखा गया।",
            "क़लम से खत लिखा गया।",
            {"chrf": 75.123844648, "chrf++": 74.270091269, "bleu": 37.991784283},
        ),
        (
            "ur",
            "یہ ایک خوبصورت کتاب ہے۔",
            "یہ ایک اچھی کتاب ہے ۔",
            {"chrf": 42.191340425, "chrf++": 49.623907753, "bleu": 37.991784283},
        ),
    ],
    ids=["english-punctuation", "hindi-nukta", "urdu"],
)
def test_corpus_metrics_match_frozen_reference_scores(
    language: str, reference: str, hypothesis: str, expected: dict[str, float]
) -> None:
    assert corpus_metrics([reference], [hypothesis], language) == pytest.approx(expected, rel=0, abs=1e-6)


def test_compute_metrics_groups_pairs_and_rollout_indices() -> None:
    tasks = [
        [
            {
                "translation": "A complete English reference sentence.",
                "generation": "A complete English reference sentence.",
                "source_language": "hi",
                "target_language": "en",
            }
        ],
        [
            {
                "translation": "Another complete English reference sentence.",
                "generation": "Another complete English reference sentence.",
                "source_language": "hi",
                "target_language": "en",
            }
        ],
    ]

    metrics = compute_in22_metrics(tasks)

    for key in ("hi->en/chrf", "hi->en/chrf++", "hi->en/bleu", "xx->xx/chrf", "xx->en/bleu"):
        assert metrics[key] == pytest.approx(100.0)
    assert metrics["hi->en/chrf_std_dev_across_runs"] == 0.0


def test_corpus_and_macro_metrics_do_not_average_sentence_scores() -> None:
    def rollout(source: str, reference: str, generation: str) -> dict:
        return {
            "translation": reference,
            "generation": generation,
            "source_language": source,
            "target_language": "en",
        }

    references = ["A long reference sentence with several words to translate correctly.", "Yes."]
    hypotheses = [references[0], "No."]
    tasks = [[rollout("hi", ref, hyp), rollout("hi", ref, "")] for ref, hyp in zip(references, hypotheses)]
    tasks.append([rollout("bn", "Another English sentence.", ""), rollout("bn", "Another English sentence.", "")])
    metrics = _server().compute_metrics(tasks)
    expected = corpus_metrics(references, hypotheses, "en")["chrf++"]
    sentence_mean = sum(in22_metrics.sentence_metrics(r, h, "en")[1] for r, h in zip(references, hypotheses)) / 2
    assert expected != pytest.approx(sentence_mean)
    assert metrics["hi->en/chrf++"] == pytest.approx(expected / 2)
    assert metrics["hi->en/chrf++_std_dev_across_runs"] == pytest.approx(expected / 2)
    # Equal weight per language, despite Hindi having twice as many tasks.
    assert metrics["xx->en/chrf++"] == pytest.approx(expected / 4)


def test_real_indic_normalization_and_empty_hypothesis() -> None:
    assert normalize_in22("यह एक वाक्य है।", "hi") == "यह एक वाक्य है ।"
    assert corpus_metrics(["यह एक वाक्य है।"], [""], "hi") == {"chrf": 0.0, "chrf++": 0.0, "bleu": 0.0}


@pytest.mark.parametrize("urduhack_available", [False, True])
def test_urdu_tokenization_is_independent_of_optional_normalizer(
    monkeypatch: pytest.MonkeyPatch, urduhack_available: bool
) -> None:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    normalizer = MagicMock()
    normalizer.normalize.return_value = "changed by optional normalizer"
    factory = MagicMock(return_value=normalizer)
    if not urduhack_available:
        factory.side_effect = ModuleNotFoundError("No module named 'urduhack'")
    monkeypatch.setattr(IndicNormalizerFactory, "get_normalizer", factory)
    in22_metrics._indic_tools.cache_clear()
    try:
        reference = "یہ ایک مکمل جملہ ہے۔"
        hypothesis = "یہ\tایک  مکمل جملہ ہے ۔"
        assert normalize_in22(reference, "ur") == normalize_in22(hypothesis, "ur") == "یہ ایک مکمل جملہ ہے ۔"
        assert corpus_metrics([reference], [hypothesis], "ur") == pytest.approx(
            {"chrf": 100.0, "chrf++": 100.0, "bleu": 100.0}
        )
        factory.assert_not_called()
    finally:
        in22_metrics._indic_tools.cache_clear()


def test_required_normalizer_dependency_failure_is_not_silenced(monkeypatch: pytest.MonkeyPatch) -> None:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    monkeypatch.setattr(
        IndicNormalizerFactory,
        "get_normalizer",
        MagicMock(side_effect=ImportError("required normalization dependency")),
    )
    in22_metrics._indic_tools.cache_clear()
    try:
        with pytest.raises(ImportError, match="required normalization dependency"):
            normalize_in22("यह एक वाक्य है।", "hi")
    finally:
        in22_metrics._indic_tools.cache_clear()


async def test_server_uses_in22_cleanup_and_scores() -> None:
    reference = "This is a complete reference sentence."
    request = WmtTranslationVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Translate"}]},
        response=_response(f"<think>reasoning</think>\n```text\n{reference}\n```"),
        text="source",
        translation=reference,
        source_language="hi",
        target_language="en",
        source_lang_name="Hindi",
        target_lang_name="English",
    )

    result = await _server().verify(request)

    assert result.generation == reference
    assert result.reward == pytest.approx(1.0)
    assert result.sentence_chrf == pytest.approx(100.0)
    assert result.sentence_chrfpp == pytest.approx(100.0)
    assert result.sentence_bleu == pytest.approx(100.0)
    assert result.sentence_spbleu is None


async def test_server_scores_truncated_reasoning_as_empty() -> None:
    request = WmtTranslationVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Translate"}]},
        response=_response("<think>unfinished"),
        text="source",
        translation="reference",
        source_language="en",
        target_language="hi",
        source_lang_name="English",
        target_lang_name="Hindi",
    )

    result = await _server().verify(request)

    assert result.generation == ""
    assert result.reward == 0.0
    assert result.sentence_chrfpp == 0.0
    assert result.sentence_bleu == 0.0
