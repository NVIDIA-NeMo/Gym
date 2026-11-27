# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the speech scoring primitives."""

import unicodedata
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.asr_with_pc.app import ASRWithPCConfig, ASRWithPCResourcesServer
from resources_servers.asr_with_pc.speech_metrics import (
    evaluate_multilingual_asr,
    evaluate_open_asr,
)


def _make_server() -> ASRWithPCResourcesServer:
    config = ASRWithPCConfig(host="0.0.0.0", port=8080, entrypoint="", name="", task_type="OpenASR")
    return ASRWithPCResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_open_asr_merges_equivalent_compounds() -> None:
    result = evaluate_open_asr("Everything is all right", "everything is alright")
    assert result["wer"] == 0.0
    assert result["wer_errors"] == 0
    assert result["wer_ref_words"] == 3


def test_multilingual_cer_uses_character_denominator() -> None:
    result = evaluate_multilingual_asr("你好世界", "你好世", language="cmn_hans_cn", use_cer=True)
    assert result["wer"] == pytest.approx(0.25)
    assert result["wer_errors"] == 1
    assert result["wer_ref_words"] == 4
    assert result["cer"] == pytest.approx(0.25)


def test_multilingual_defaults_keep_fleurs_normalization() -> None:
    korean_nfd = unicodedata.normalize("NFD", "한국")
    decomposed = evaluate_multilingual_asr(
        korean_nfd,
        korean_nfd,
        language="ko_kr",
        remove_diacritics=True,
    )
    assert decomposed["cer_ref_words"] == len(korean_nfd)

    compound = evaluate_multilingual_asr(
        "ice cream",
        "icecream",
        language="de_de",
        remove_diacritics=True,
        normalize_compound=True,
        merge_compounds=False,
    )
    assert compound["wer"] == 0.0


@pytest.mark.parametrize(
    ("language", "reference", "hypothesis"),
    [
        # Turkish num2words output carries diacritics ("yüz") that the spelled-out hypothesis loses in the fold.
        ("tr_tr", "1900 yılında", "bin dokuz yüz yılında"),
        # Thai number words carry tone/vowel marks.
        ("th_th", "ราคา 5 บาท", "ราคาห้าบาท"),
        # Korean number words are composed Hangul syllables; NFKD splits them into jamo.
        ("ko_kr", "2주 만에", "이 주 만에"),
    ],
)
def test_multilingual_numbers_are_spelled_out_before_diacritic_folding(
    language: str, reference: str, hypothesis: str
) -> None:
    result = evaluate_multilingual_asr(
        reference,
        hypothesis,
        language=language,
        use_cer=language != "tr_tr",
        remove_diacritics=True,
        normalize_compound=True,
    )
    assert result["wer"] == 0.0, result


def test_open_asr_chunks_are_scored_as_one_session() -> None:
    server = _make_server()
    tasks = []
    for chunk_index, hypothesis in enumerate(("hello", "world")):
        score = evaluate_open_asr("hello world", hypothesis)
        score.update(
            expected_answer="hello world",
            raw_pred_text=hypothesis,
            parent_id="call-1",
            chunk_index=chunk_index,
            subset_for_metrics="earnings22_aa",
            benchmark_profile="asr_leaderboard",
        )
        tasks.append([score])

    metrics = server.compute_metrics(tasks)
    assert metrics["corpus_wer@k=1"] == 0.0
    assert metrics["wer::subset::earnings22_aa@k=1"] == 0.0
    assert metrics["wer::profile::asr_leaderboard@k=1"] == 0.0


def test_han_adjacent_cer_whitespace_ignores_segmentation_spaces():
    ref = "执 政 党 swapo 也 在 议 会"
    hyp = "执政党swapo也在议会。"
    keep = evaluate_multilingual_asr(ref, hyp, language="cmn_hans_cn", use_cer=True, normalize_compound=False)
    han = evaluate_multilingual_asr(ref, hyp, language="cmn_hans_cn", use_cer=True, cer_whitespace_mode="han_adjacent")
    assert keep["wer"] > 0.3
    assert han["wer"] == 0.0
    assert han["text"] == "执政党swapo也在议会"


def test_han_adjacent_keeps_latin_word_boundaries():
    from resources_servers.asr_with_pc.speech_metrics import normalize_cer_whitespace

    assert normalize_cer_whitespace("今 天 use AI", "han_adjacent") == "今天use AI"
    with pytest.raises(ValueError):
        normalize_cer_whitespace("x", "bogus")
