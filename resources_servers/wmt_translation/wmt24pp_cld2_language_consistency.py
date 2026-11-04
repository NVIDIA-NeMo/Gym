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

import pycld2 as cld2


LOG = logging.getLogger(__name__)


# WMT24++ distinguishes regional varieties that CLD2 cannot. Arabic, French,
# Portuguese, and Swahili each have two WMT24++ locales that map to one CLD2
# language code.
# This means that _WMT24PP_TO_CLD2_CODES[wmt24pp_language_code] == cld2_code
# is a **lenient** language-level comparison for those locales, not a check of
# the requested regional variety.
_WMT24PP_TO_CLD2_CODES = {
    "ar_EG": "ar",  # Egyptian Arabic -> Arabic
    "ar_SA": "ar",  # Saudi Arabic -> Arabic
    "bg_BG": "bg",
    "bn_IN": "bn",
    "ca_ES": "ca",
    "cs_CZ": "cs",
    "da_DK": "da",
    "de_DE": "de",
    "el_GR": "el",
    "es_MX": "es",
    "et_EE": "et",
    "fa_IR": "fa",
    "fi_FI": "fi",
    "fil_PH": "tl",
    "fr_CA": "fr",
    "fr_FR": "fr",
    "gu_IN": "gu",
    "he_IL": "iw",
    "hi_IN": "hi",
    "hr_HR": "hr",
    "hu_HU": "hu",
    "id_ID": "id",
    "is_IS": "is",
    "it_IT": "it",
    "ja_JP": "ja",
    "kn_IN": "kn",
    "ko_KR": "ko",
    "lt_LT": "lt",
    "lv_LV": "lv",
    "ml_IN": "ml",
    "mr_IN": "mr",
    "nl_NL": "nl",
    "no_NO": "no",
    "pa_IN": "pa",
    "pl_PL": "pl",
    "pt_BR": "pt",
    "pt_PT": "pt",
    "ro_RO": "ro",
    "ru_RU": "ru",
    "sk_SK": "sk",
    "sl_SI": "sl",
    "sr_RS": "sr",
    "sv_SE": "sv",
    "sw_KE": "sw",  # Kenyan Swahili -> Swahili
    "sw_TZ": "sw",  # Tanzanian Swahili -> Swahili
    "ta_IN": "ta",
    "te_IN": "te",
    "th_TH": "th",
    "tr_TR": "tr",
    "uk_UA": "uk",
    "ur_PK": "ur",
    "vi_VN": "vi",
    "zh_CN": "zh",
    "zh_TW": "zh-Hant",
    "zu_ZA": "zu",
}


def _sanitize_text(text: str) -> str:
    return "".join(c for c in text if c.isprintable() or c.isspace())


def wmt24pp_cld2_language_consistency_score(text: str, expected_language_code: str) -> float:
    """
    Return the fraction of `text` (0.0-1.0) that CLD2 attributes to the
    expected language for `expected_language_code`.

    CLD2 has no "score language X" function: it always reports its own top guesses,
    so we detect and then read off the percentage for the language we wanted.
    Returns 0.0 when that language is not among the detections (e.g. the model
    translated into the wrong language, or produced empty output).

    WMT24++ has multiple regional varieties of Arabic, French, Portuguese, and
    Swahili that CLD2 cannot distinguish. Checks for those locales validate the
    major language only and cannot detect a wrong regional variety.
    """
    text = _sanitize_text(text)  # pycld2 rejects some control chars
    if not text:
        return 0.0

    try:
        expected_code = _WMT24PP_TO_CLD2_CODES[expected_language_code]
    except KeyError as exc:
        raise ValueError(f"Unknown WMT24++ language code: {expected_language_code!r}") from exc

    # details is up to three (name, code, percent, score) tuples; `percent` is
    # the share of the text detected as that language. A code appears at most
    # once, so find the matching entry (if any) and return its percentage.
    try:
        _, _, details = cld2.detect(text, isPlainText=True, bestEffort=True)
    except cld2.error:
        LOG.exception(
            "CLD2 detection failed for target language %r; returning 0.0",
            expected_language_code,
        )
        return 0.0
    for _name, code, percent, _score in details:
        if code == expected_code:
            return percent / 100.0
    return 0.0
