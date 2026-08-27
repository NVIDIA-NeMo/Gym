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

"""
This script:
1) downloads the wmt24pp data and makes sure each wmt24pp
    language is present in _WMT24PP_TO_CLD2_CODES.

2) For each language:
    2.a) Run CLD2 on the entire wmt24pp human reference concatenated
           together and raise an error if it is not classified
           as the expected language
    2.b) Run CLD2 on each sentence in the wmt24pp human reference
           and raise an error if the average per-sentence score is
           below an allowed minimum.

This confirms that the mappings in _WMT24PP_TO_CLD2_CODES,
in combination with CLD2, are correct.

These checks are too expensive to include in unit tests,
so they are provided instead as a standalone script.

"""

import sys
from pathlib import Path

import pycld2 as cld2
from datasets import get_dataset_config_names, load_dataset


_WMT_TRANSLATION_DIR = Path(__file__).resolve().parents[1]
if str(_WMT_TRANSLATION_DIR) not in sys.path:
    sys.path.insert(0, str(_WMT_TRANSLATION_DIR))

from wmt24pp_cld2_language_consistency import (
    _WMT24PP_TO_CLD2_CODES,
    _sanitize_text,
    wmt24pp_cld2_language_consistency_score,
)


def _cld2_predict(text: str) -> str:
    """
    Return most likely CLD2 language code for given text
    """
    text = _sanitize_text(text)  # pycld2 rejects some control chars
    _, _, details = cld2.detect(text, isPlainText=True)
    cld2_lang_code = details[0][1]  # top (name, code, percent, score) -> code
    return cld2_lang_code


def _validate_language_mappings() -> None:
    # worst observed average per-sentence language-consistency score was 0.64012 for Croatian (hr_HR)
    minimum_average_per_sentence_language_consistency_score = 0.60
    minimum_fraction_sents_in_expected_language = 0.60

    # Verify _WMT24PP_TO_CLD2_CODES has full coverage of the wmt24pp languages
    available_wmt_targets = {
        config_name.removeprefix("en-")
        for config_name in get_dataset_config_names("google/wmt24pp")
        if config_name.startswith("en-")
    }
    mapped_wmt_targets = set(_WMT24PP_TO_CLD2_CODES)
    if available_wmt_targets != mapped_wmt_targets:
        missing_mappings = sorted(available_wmt_targets - mapped_wmt_targets)
        unknown_mappings = sorted(mapped_wmt_targets - available_wmt_targets)
        raise AssertionError(
            f"WMT24++/CLD2 mapping mismatch: missing mappings={missing_mappings}, unknown mappings={unknown_mappings}"
        )
    else:
        print("_WMT24PP_TO_CLD2_CODES mapping coverage: pass")

    for wmt_code, expected_cld2_code in _WMT24PP_TO_CLD2_CODES.items():
        per_sent_scores = []

        rows = load_dataset("google/wmt24pp", f"en-{wmt_code}", split="train")
        text = " ".join(r["target"] for r in rows if not r["is_bad_source"])
        for r in rows:
            if not r["is_bad_source"]:
                sent = r["target"]
                score = wmt24pp_cld2_language_consistency_score(sent, wmt_code)
                per_sent_scores.append(score)

        if not per_sent_scores:
            raise AssertionError(f"{wmt_code}: no valid target sentences found")

        # 2) Verify all valid rows in the train split (concatenated together) return the expected language
        actual_cld2_code = _cld2_predict(text)
        if actual_cld2_code != expected_cld2_code:
            raise AssertionError(
                f"{wmt_code}: expected CLD2 {expected_cld2_code!r}, CLD2 detected {actual_cld2_code!r}"
            )

        # 3) Verify the average per-sentence language-consistency score is >= allowed minimum
        average_score = sum(per_sent_scores) / len(per_sent_scores)
        if average_score < minimum_average_per_sentence_language_consistency_score:
            raise AssertionError(
                f"{wmt_code}: average per-sentence language-consistency score {average_score:.5f} "
                f"is below minimum {minimum_average_per_sentence_language_consistency_score:.2f}"
            )

        fraction_sents_in_expected_language = sum(score > 0.5 for score in per_sent_scores) / len(per_sent_scores)
        if fraction_sents_in_expected_language <= minimum_fraction_sents_in_expected_language:
            raise AssertionError(
                f"{wmt_code}: fraction of sentences in the expected language "
                f"{fraction_sents_in_expected_language:.5f} is not greater than minimum "
                f"{minimum_fraction_sents_in_expected_language:.2f}"
            )

        print(f"{wmt_code} -> {expected_cld2_code} mapping: pass")


if __name__ == "__main__":
    _validate_language_mappings()
