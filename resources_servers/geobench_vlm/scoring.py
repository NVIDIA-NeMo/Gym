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
"""Score GEOBench-VLM answers the way the released evaluator reads them.

Multiple-choice items ask for a single letter, A through E. The temporal
script treats the first character of the response as that letter. This scorer
also accepts a letter written as ``Answer: C`` so a complete sentence is not
marked wrong when the choice is explicit.
"""

from __future__ import annotations

import re


LETTERS = "ABCDE"
_EXPLICIT = re.compile(r"(?i)(?:answer|choice|option)\s*(?:is|:)?\s*([A-E])\b")
_LEADING = re.compile(r"(?i)^\s*([A-E])(?:\s*[\.\)\:]|$)")
_GOLD = re.compile(r"(?i)^\s*([A-E])\b")


def gold_letter(option: str | None) -> str | None:
    match = _GOLD.match(option or "")
    return match.group(1).upper() if match else None


def first_letter(text: str | None) -> str | None:
    raw = (text or "").lstrip()
    if raw and raw[0].upper() in LETTERS:
        return raw[0].upper()
    return None


def predicted_letter(text: str | None) -> str | None:
    raw = text or ""
    explicit = _EXPLICIT.search(raw)
    if explicit:
        return explicit.group(1).upper()
    leading = _LEADING.match(raw)
    if leading:
        return leading.group(1).upper()
    return first_letter(raw)


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def score_answer(
    prediction: str | None,
    ground_truth_option: str | None,
    ground_truth: str | None = None,
) -> dict:
    gold = gold_letter(ground_truth_option)
    parsed = predicted_letter(prediction)
    official = first_letter(prediction)
    if gold is not None:
        correct = parsed == gold
        official_correct = official == gold
    else:
        expected = _normalize(ground_truth)
        correct = bool(expected) and _normalize(prediction) == expected
        official_correct = correct
    return {
        "reward": 1.0 if correct else 0.0,
        "geobench_vlm_correct": correct,
        "geobench_vlm_first_letter_correct": official_correct,
        "predicted_letter": parsed or "",
        "gold_letter": gold or "",
    }
