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
from unittest.mock import patch

import pycld2 as cld2
import pytest
from wmt24pp_cld2_language_consistency import (
    _sanitize_text,
    wmt24pp_cld2_language_consistency_score,
)


def _detection_result(*details):
    return True, 100, details


def test_sanitize_text_removes_unsupported_control_characters() -> None:
    assert _sanitize_text("a\x00b\x07c\x1bd") == "abcd"


def test_sanitize_text_preserves_printable_unicode_and_whitespace() -> None:
    text = "Zażółć 中文\n\t\r\u2003"
    assert _sanitize_text(text) == text


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_empty_input_returns_zero_without_calling_cld2(mock_detect) -> None:
    assert wmt24pp_cld2_language_consistency_score("", "de_DE") == 0.0
    mock_detect.assert_not_called()


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_control_characters_only_returns_zero_without_calling_cld2(mock_detect) -> None:
    assert wmt24pp_cld2_language_consistency_score("\x00\x07\x1b", "de_DE") == 0.0
    mock_detect.assert_not_called()


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_expected_language_can_be_found_when_not_top_prediction(mock_detect) -> None:
    mock_detect.return_value = _detection_result(
        ("ENGLISH", "en", 70, 1000),
        ("GERMAN", "de", 30, 500),
    )
    assert wmt24pp_cld2_language_consistency_score("mixed text", "de_DE") == pytest.approx(0.30)


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_missing_expected_language_returns_zero(mock_detect) -> None:
    mock_detect.return_value = _detection_result(("ENGLISH", "en", 100, 1000))

    assert wmt24pp_cld2_language_consistency_score("English text", "de_DE") == 0.0


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_empty_detection_details_returns_zero(mock_detect) -> None:
    mock_detect.return_value = _detection_result()

    assert wmt24pp_cld2_language_consistency_score("Hallo Welt", "de_DE") == 0.0


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_unknown_wmt_language_code_raises_before_calling_cld2(mock_detect) -> None:
    with pytest.raises(ValueError, match=r"Unknown WMT24\+\+ language code"):
        wmt24pp_cld2_language_consistency_score("text", "unknown_CODE")

    mock_detect.assert_not_called()


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_cld2_receives_sanitized_text_and_detection_options(mock_detect) -> None:
    mock_detect.return_value = _detection_result(("GERMAN", "de", 100, 1000))

    wmt24pp_cld2_language_consistency_score("Hallo\x00\nWelt", "de_DE")

    mock_detect.assert_called_once_with(
        "Hallo\nWelt",
        isPlainText=True,
        bestEffort=True,
    )


@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_cld2_error_is_logged_and_returns_zero(mock_detect, caplog) -> None:
    mock_detect.side_effect = cld2.error("detection failed")

    with caplog.at_level(logging.ERROR, logger="wmt24pp_cld2_language_consistency"):
        score = wmt24pp_cld2_language_consistency_score("Hallo Welt", "de_DE")

    assert score == 0.0
    error_records = [record for record in caplog.records if record.name == "wmt24pp_cld2_language_consistency"]
    assert len(error_records) == 1
    assert "CLD2 detection failed for target language" in error_records[0].message
    assert error_records[0].exc_info is not None


@pytest.mark.parametrize(
    ("wmt_code", "cld2_code"),
    [
        ("fil_PH", "tl"),
        ("he_IL", "iw"),
        ("zh_TW", "zh-Hant"),
        ("ar_EG", "ar"),
        ("ar_SA", "ar"),
        ("sw_KE", "sw"),
        ("sw_TZ", "sw"),
    ],
)
@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_exceptional_language_code_mappings(mock_detect, wmt_code: str, cld2_code: str) -> None:
    mock_detect.return_value = _detection_result(("EXPECTED", cld2_code, 80, 1000))

    assert wmt24pp_cld2_language_consistency_score("target-language text", wmt_code) == pytest.approx(0.80)


@pytest.mark.parametrize(
    ("percent", "expected_fraction"),
    [
        (0, 0.0),
        (100, 1.0),
    ],
)
@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_boundary_percentages(mock_detect, percent: int, expected_fraction: float) -> None:
    mock_detect.return_value = _detection_result(("GERMAN", "de", percent, 1000))

    assert wmt24pp_cld2_language_consistency_score("Hallo Welt", "de_DE") == expected_fraction


@pytest.mark.parametrize("reliability_score", [-100, 0, 1_000_000])
@patch("wmt24pp_cld2_language_consistency.cld2.detect")
def test_cld2_reliability_score_is_ignored(mock_detect, reliability_score: int) -> None:
    mock_detect.return_value = _detection_result(("GERMAN", "de", 42, reliability_score))

    assert wmt24pp_cld2_language_consistency_score("Hallo Welt", "de_DE") == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("wmt_code", "text"),
    [
        (
            "de_DE",
            "Der schnelle braune Fuchs springt über den faulen Hund. "
            "Heute scheint die Sonne und die Kinder spielen im Garten.",
        ),
        (
            "fr_FR",
            "Le renard brun rapide saute par-dessus le chien paresseux. "
            "Aujourd'hui, le soleil brille et les enfants jouent dans le jardin.",
        ),
        (
            "es_MX",
            "El rápido zorro marrón salta sobre el perro perezoso. Hoy brilla el sol y los niños juegan en el jardín.",
        ),
        (
            "ja_JP",
            "素早い茶色のキツネが怠け者の犬を飛び越えます。今日は天気がよく、子どもたちは庭で遊んでいます。",
        ),
        (
            "zh_CN",
            "快速的棕色狐狸跳过了那只懒狗。今天天气很好，孩子们正在花园里快乐地玩耍。",
        ),
        (
            "zh_TW",
            "快速的棕色狐狸跳過了那隻懶狗。今天天氣很好，孩子們正在花園裡快樂地玩耍。",
        ),
    ],
)
def test_real_cld2_gives_expected_language_a_high_consistency_score(wmt_code: str, text: str) -> None:
    assert wmt24pp_cld2_language_consistency_score(text, wmt_code) > 0.5


@pytest.mark.parametrize(
    ("wmt_code", "text"),
    [
        (
            "fr_FR",  # Text actually in German
            "Der schnelle braune Fuchs springt über den faulen Hund. "
            "Heute scheint die Sonne und die Kinder spielen im Garten.",
        ),
        (
            "de_DE",  # Text actually in French
            "Le renard brun rapide saute par-dessus le chien paresseux. "
            "Aujourd'hui, le soleil brille et les enfants jouent dans le jardin.",
        ),
        (
            "ja_JP",  # Text actually in Spanish
            "El rápido zorro marrón salta sobre el perro perezoso. Hoy brilla el sol y los niños juegan en el jardín.",
        ),
        (
            "zh_TW",  # Text actually in Simplified Chinese
            "快速的棕色狐狸跳过了那只懒狗。今天天气很好，孩子们正在花园里快乐地玩耍。",
        ),
    ],
)
def test_real_cld2_gives_wrong_language_a_low_consistency_score(wmt_code: str, text: str) -> None:
    assert wmt24pp_cld2_language_consistency_score(text, wmt_code) < 0.5
