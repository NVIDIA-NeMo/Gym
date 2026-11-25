# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MUCS 2021 scorer (normal and transliterated WER), driven with small hand-counted rows.

The official maps are replaced by tiny ones; their download and verification are tested in ``test_mucs_assets.py``.
"""

import asyncio
import importlib

import pytest


mucs = importlib.import_module("resources_servers.audio_eval._scoring.dataset.mucs2021.scorer")


@pytest.fixture
def mapping_requests(monkeypatch):
    """Replace the downloaded official maps with tiny ones and record which map files were requested."""
    maps = {
        "punct.mapping.txt": {"?": "<q>"},
        "hin2engwords.txt": {"कंप्यूटर": "computer", "<q>": "<ignored>"},
        "ben2engwords.txt": {"কম্পিউটার": "computer"},
    }
    requested = []

    def load(name):
        requested.append(name)
        return maps[name]

    monkeypatch.setattr(mucs, "load_official_mapping", load)
    return requested


def mucs_row(reference, generation, language="hindi-english"):
    return {"expected_answer": reference, "generation": generation, "extra_fields": {"mucs_language": language}}


def test_mucs_tokens_and_mapping_follow_the_official_script():
    assert mucs.whitespace_tokens("  Hello   WORLD\n") == ["hello", "world"]
    assert mucs.apply_mapping(["a", "b", "c"], {"b": "x y", "c": ""}) == ["a", "x", "y"]


def test_mucs_transliteration_map_only_affects_twer(mapping_requests):
    result = mucs.evaluate_mucs2021_sample(mucs_row("I have computer", "I have कंप्यूटर"))
    assert (result["wer_errors"], result["twer_errors"], result["ref_words"]) == (1, 0, 3)
    assert result["wer"] == pytest.approx(1 / 3) and result["twer"] == 0.0
    assert result["text"] == "i have computer" and result["pred_text"] == "i have कंप्यूटर"
    assert result["transliterated_pred_text"] == "i have computer"
    assert result["skipped_empty_reference"] is False
    assert mapping_requests == ["punct.mapping.txt", "hin2engwords.txt"]


def test_mucs_punctuation_map_applies_before_the_word_map(mapping_requests):
    result = mucs.evaluate_mucs2021_sample(mucs_row("what <q>", "what ?"))
    assert result["wer_errors"] == 0
    # The punctuated token <q> is then rewritten by the word map, so the transliterated score diverges.
    assert result["transliterated_pred_text"] == "what <ignored>" and result["twer_errors"] == 1


def test_mucs_bengali_uses_its_own_word_map(mapping_requests):
    result = mucs.evaluate_mucs2021_sample(mucs_row("computer", "কম্পিউটার", language="bengali-english"))
    assert (result["wer_errors"], result["twer_errors"]) == (1, 0)
    assert mapping_requests == ["punct.mapping.txt", "ben2engwords.txt"]


def test_mucs_empty_reference_is_skipped_like_the_official_loop(mapping_requests):
    result = mucs.evaluate_mucs2021_sample(mucs_row("", "spurious words"))
    assert result["skipped_empty_reference"] is True and result["wer"] is None and result["twer"] is None
    assert (result["wer_errors"], result["twer_errors"], result["ref_words"]) == (0, 0, 0)


@pytest.mark.parametrize("extra_fields", [{}, {"mucs_language": "tamil-english"}])
def test_mucs_requires_a_known_language(mapping_requests, extra_fields):
    with pytest.raises(ValueError, match="Missing or invalid extra_fields.mucs_language"):
        mucs.evaluate_mucs2021_sample({"expected_answer": "a", "generation": "a", "extra_fields": extra_fields})
    with pytest.raises(ValueError, match="Missing or invalid extra_fields.mucs_language"):
        mucs.evaluate_mucs2021_sample({"expected_answer": "a", "generation": "a"})


def test_mucs_evaluator_scores_asynchronously(mapping_requests):
    result = asyncio.run(mucs.MUCS2021Evaluator({}).eval_single(mucs_row("a b", "a")))
    assert result["wer_errors"] == 1 and result["ref_words"] == 2


def test_mucs_metrics_sum_utterance_counts_and_skip_empty_references(mapping_requests):
    metrics = mucs.MUCS2021Metrics()
    for reference, generation in (("I have computer", "I have कंप्यूटर"), ("", "foo"), ("a b", "")):
        metrics.update([mucs.evaluate_mucs2021_sample(mucs_row(reference, generation))])
    values = metrics.get_metrics()["pass@1"]
    assert (values["wer_errors"], values["twer_errors"], values["ref_words"]) == (3, 2, 5)
    assert values["wer"] == 60.0 and values["twer"] == 40.0
    assert values["scored_entries"] == 2 and values["skipped_empty_references"] == 1
    assert values["no_answer"] == pytest.approx(100.0 / 3)
    assert values["num_entries"] == 3


def test_mucs_metrics_contracts():
    assert mucs.MUCS2021Metrics()._get_score_dict({"wer_errors": 0, "skipped_empty_reference": False}) == {
        "exact": True
    }
    assert mucs.MUCS2021Metrics()._get_score_dict({"wer_errors": 0, "skipped_empty_reference": True}) == {
        "exact": False
    }
    with pytest.raises(ValueError, match="one greedy generation"):
        mucs.MUCS2021Metrics(max_k=2)
    with pytest.raises(ValueError, match="exactly one prediction"):
        mucs.MUCS2021Metrics().update([])
    with pytest.raises(ValueError, match="no non-empty reference words"):
        mucs.MUCS2021Metrics().get_metrics()
