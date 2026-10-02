# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import in22_metrics
import pytest
from app import WmtTranslationResourcesServer, WmtTranslationResourcesServerConfig, WmtTranslationVerifyRequest
from in22_metrics import clean_in22_response, compute_in22_metrics, corpus_metrics, normalize_in22

from nemo_gym.openai_utils import NeMoGymResponse
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
