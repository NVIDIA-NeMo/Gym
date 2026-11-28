# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLEURS language-ID evaluator and metrics: locale labels, region groups, the reference label set and the alias."""

import asyncio

import pytest

from resources_servers.audio_eval._scoring.dataset.fleurs.languages import (
    ASR_REPORTING_LOCALES,
    FLEURS_LANGUAGES,
    LID_LOCALES,
    get_lang_group,
)
from resources_servers.audio_eval._scoring.evaluation.evaluator.fleurs_lid import (
    LID_ALIASES,
    FleursLIDEvaluator,
    evaluate_fleurs_lid_sample,
)
from resources_servers.audio_eval._scoring.evaluation.metrics.fleurs_lid_metrics import FleursLIDMetrics
from resources_servers.audio_eval.score import resolve


def test_the_language_id_locales_are_the_asr_locales_without_armenian_and_norwegian():
    assert len(LID_LOCALES) == 36 and len(ASR_REPORTING_LOCALES) == 38
    assert set(ASR_REPORTING_LOCALES) - set(LID_LOCALES) == {"hy_am", "nb_no"}
    assert set(LID_LOCALES) <= set(FLEURS_LANGUAGES) and len(FLEURS_LANGUAGES) == 102


def test_the_profile_names_resolve_to_these_classes():
    assert resolve("evaluators", "fleurs_lid") is FleursLIDEvaluator
    assert resolve("metrics", "fleurs_lid") is FleursLIDMetrics


def test_sample_reports_locale_and_geographic_group():
    result = evaluate_fleurs_lid_sample({"expected_answer": "bg_bg", "generation": "bg_bg"})
    assert result["lid_correct"] and result["locale"] == "bg_bg"
    assert result["lang_group"] == result["label_group"] == get_lang_group("bg_bg") == "eastern_european_ee"
    assert result["valid_label"] is True


def test_prediction_outside_the_language_id_locales_is_an_invalid_label():
    wrong = evaluate_fleurs_lid_sample({"expected_answer": "bg_bg", "generation": "not_a_locale"})
    assert wrong["lid_correct"] is False and wrong["valid_label"] is False
    # A locale that is not one of the 36 is a valid label only when it is the reference itself.
    extra = "af_za"
    assert extra not in LID_LOCALES
    assert evaluate_fleurs_lid_sample({"expected_answer": extra, "generation": extra})["valid_label"] is True
    assert evaluate_fleurs_lid_sample({"expected_answer": "bg_bg", "generation": extra})["valid_label"] is False


def test_a_default_run_accepts_every_fleurs_locale_as_the_reference():
    """A default run prepares all 102 locales, so every one of them is a legal reference and a correct answer."""
    for locale in FLEURS_LANGUAGES:
        result = evaluate_fleurs_lid_sample({"expected_answer": locale, "generation": locale})
        assert result["lid_correct"] and result["valid_label"] and result["lang_group"] == get_lang_group(locale)


def test_reference_must_be_a_fleurs_locale():
    with pytest.raises(ValueError, match="Non-canonical language-ID reference label"):
        evaluate_fleurs_lid_sample({"expected_answer": "xx_yy", "generation": "xx_yy"})


def test_evaluator_scores_asynchronously():
    result = asyncio.run(FleursLIDEvaluator({}).eval_single({"expected_answer": "ar_eg", "generation": "zh_cn"}))
    assert result["lid_correct"] is False and result["locale"] == "ar_eg"


def test_zh_cn_is_graded_as_the_canonical_chinese_label_and_nothing_else_is_aliased():
    assert LID_ALIASES == {"zh_cn": "cmn_hans_cn"}
    result = evaluate_fleurs_lid_sample({"expected_answer": "cmn_hans_cn", "generation": "zh_cn\n"})
    assert result["lid_correct"] and result["predicted_label"] == "cmn_hans_cn" and result["valid_label"]
    assert not evaluate_fleurs_lid_sample({"expected_answer": "cmn_hans_cn", "generation": "ja_jp"})["lid_correct"]
    assert not evaluate_fleurs_lid_sample({"expected_answer": "ja_jp", "generation": "zh_cn"})["lid_correct"]
    assert not evaluate_fleurs_lid_sample({"expected_answer": "yue_hant_hk", "generation": "zh_cn"})["lid_correct"]


def test_metrics_name_the_label_count_num_locales():
    metrics = FleursLIDMetrics()
    for point in (
        {"expected_answer": "ar_eg", "generation": "ar_eg"},
        {"expected_answer": "bg_bg", "generation": "ar_eg"},
    ):
        # The verify path merges the evaluator result into the source row.
        metrics.update([{**point, **evaluate_fleurs_lid_sample(point)}])
    values = metrics.get_metrics()["pass@1"]
    assert values["num_locales"] == 2 and "num_labels" not in values
    assert values["accuracy_micro"] == 50.0 and values["accuracy::bg_bg"] == 0.0
    assert values[f"accuracy_group::{get_lang_group('ar_eg')}"] == 100.0
