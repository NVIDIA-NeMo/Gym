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
"""Pure-Python tests for benchmarks/polymath/prepare.py.

Hits ``format_entry`` with hand-rolled instruction dicts; does NOT
network out to HuggingFace or the upstream PolyMath repo (those are
covered by the on-cluster ``ng_prepare_benchmark`` step).
"""

from benchmarks.polymath.prepare import DIFFICULTY_LEVEL_DIC, QUESTION_TEMPLATE, format_entry


QUERY_DIC = {
    "en": "Please reason step by step, and put your final answer within \\boxed{}.",
    "zh": "请逐步推理，并将最终答案置于 \\boxed{} 内。",
}

LANGUAGE_CONTROL_DIC = {
    "forcing_en": {
        "en": "",
        "zh": "Please answer in English.",
    },
}


def test_format_entry_default() -> None:
    entry = {"question": "What is 2 + 2?", "answer": "4"}
    out = format_entry(
        entry,
        language="en",
        difficulty="medium",
        query_dic=QUERY_DIC,
        language_control_dic=LANGUAGE_CONTROL_DIC,
        language_control_mode=None,
    )

    expected_question = QUESTION_TEMPLATE.format(
        question="What is 2 + 2?",
        instruction=QUERY_DIC["en"],
        lang_control="",
    ).strip()

    assert out["question"] == expected_question
    assert out["problem"] == "What is 2 + 2?"
    assert out["expected_answer"] == "4"
    assert out["language"] == "en"
    assert out["subset_for_metrics"] == "en"
    assert out["difficulty"] == "medium"
    assert out["weight"] == 2
    assert out["language_control_mode"] is None


def test_format_entry_with_language_control() -> None:
    entry = {"question": "计算 5 + 3。", "answer": "8"}
    out = format_entry(
        entry,
        language="zh",
        difficulty="top",
        query_dic=QUERY_DIC,
        language_control_dic=LANGUAGE_CONTROL_DIC,
        language_control_mode="forcing_en",
    )
    assert out["language"] == "zh"
    assert out["weight"] == 8
    assert out["language_control_mode"] == "forcing_en"
    # The Chinese question should pick up the English forcing suffix.
    assert "Please answer in English." in out["question"]


def test_difficulty_weights_match_skills() -> None:
    # WeightedMathMetrics in Skills hardcodes these mappings — pin them
    # so a future change is loud.
    assert DIFFICULTY_LEVEL_DIC == {"low": 1, "medium": 2, "high": 4, "top": 8}
