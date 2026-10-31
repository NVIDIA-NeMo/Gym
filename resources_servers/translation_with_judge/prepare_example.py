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
"""Builds data/example.jsonl from FLORES-200 devtest sentence pairs.

The canonical FLORES-200 dataset (``facebook/flores`` on Hugging Face) is
gated, so this script embeds the five devtest sentence pairs directly
instead of downloading them -- they're the same five rows already committed
in data/example.jsonl, reproduced here deterministically so the row-wrapping
logic (PROMPT_TEMPLATES, below) is documented in code rather than only in
the data file. Rerun with `python prepare_example.py` after changing a
template or adding a row; it overwrites data/example.jsonl.

Source: FLORES-200 devtest split (CC-BY-SA 4.0, Meta AI), see README.md.
"""

import json

from app import _lang_name


# Each entry: (src_lang, tgt_lang, source_sentence, solution, direction, prompt_style).
# `direction` records which side of the pair is English: "src2tgt" when English is
# src_lang, "tgt2src" when English is tgt_lang -- provenance only, not read by verify().
EXAMPLES = [
    (
        "eng_Latn",
        "hin_Deva",
        "In late 2017, Siminoff appeared on shopping television channel QVC.",
        "2017 के अंत में, सिमिनॉफ़ शॉपिंग टेलीविज़न चैनल QVC में दिखाई दिए.",
        "src2tgt",
        "task_provided",
    ),
    (
        "eng_Latn",
        "ben_Beng",
        "The find also grants insight into the evolution of feathers in birds.",
        "অনুসন্ধানটি পাখির পালকের বিবর্তনের  বিষয়েও পরিজ্ঞান দেয়",
        "src2tgt",
        "minimal_bracket",
    ),
    (
        "eng_Latn",
        "tel_Telu",
        "Over four million people went to Rome to attend the funeral.",
        "అంత్యక్రియలకు హాజరు కావడానికి, సుమారు నాలుగు మిలియన్ల మంది ప్రజలు రోమ్‌కు వెళ్లారు.",
        "src2tgt",
        "professional_accurate",
    ),
    (
        "tam_Taml",
        "eng_Latn",
        "802.11n தரம் 2.4Ghz மற்றும் 5.0Ghz அலைவரிசை ஆகிய இரண்டிலும் செயல்படும்.",
        "The 802.11n standard operates on both the 2.4Ghz and 5.0Ghz frequencies.",
        "tgt2src",
        "task_provided",
    ),
    (
        "kan_Knda",
        "eng_Latn",
        "ಈ ಸಮಯದಲ್ಲಿ ಜೆರ್ರಿ ಚಂಡಮಾರುತದಿಂದ ಭೂಮಿಗೆ ಯಾವುದೇ ಆತಂಕವಿಲ್ಲ ಎಂದು ರಾಷ್ಟ್ರೀಯ ಚಂಡಮಾರುತ ಕೇಂದ್ರ (NHC) ಹೇಳುತ್ತದೆ.",
        "The National Hurricane Center (NHC) says that at this point Jerry poses no threat to land.",
        "tgt2src",
        "minimal_bracket",
    ),
]

# Mirrors the instruction-wrapping styles a real training curriculum draws from --
# `prompt` is provenance only (see TranslationWithJudgeRunRequest in app.py); verify()
# always scores against the unwrapped `source_sentence`, not this wrapped text.
PROMPT_TEMPLATES = {
    "task_provided": "Translate the following sentence from {src_lang_name} to {tgt_lang_name}.\n\n{sentence}",
    "minimal_bracket": "[{src_lang_name} -> {tgt_lang_name}]\n{sentence}",
    "professional_accurate": "Provide an accurate {tgt_lang_name} translation of the following:\n\n{sentence}",
}


def main() -> None:
    rows = []
    for src_lang, tgt_lang, source_sentence, solution, direction, prompt_style in EXAMPLES:
        content = PROMPT_TEMPLATES[prompt_style].format(
            src_lang_name=_lang_name(src_lang),
            tgt_lang_name=_lang_name(tgt_lang),
            sentence=source_sentence,
        )
        prompt = [{"role": "user", "content": content}]
        rows.append(
            {
                "prompt": prompt,
                "source_sentence": source_sentence,
                "responses_create_params": {"input": prompt},
                "solution": solution,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "direction": direction,
                "prompt_style": prompt_style,
                "dataset_type": "translation",
                "agent_ref": {"type": "responses_api_agents", "name": "translation_with_judge_simple_agent"},
            }
        )

    with open("data/example.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
