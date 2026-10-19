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
script treats the first character of the response as that letter; that rule
is reported separately as ``geobench_vlm_first_letter_correct``.

The reward parser is stricter about what counts as a letter and looser about
formatting. It reads, in order: the last ``Answer: C`` statement, the last
``option C`` or ``choice C`` mention, a leading letter set off by punctuation or markup (``C.``,
``(C)``, ``**C**``), and a lone letter at the end. A capital letter that
starts a word ("Based on...", "A harbor") is not an answer, and a lowercase
letter only counts when it is the whole reply. Reasoning inside
``<think>...</think>`` is ignored.
"""

from __future__ import annotations

import re


LETTERS = "ABCDE"
_THINK = re.compile(r"(?s)<think>.*?</think>")
_ANSWER = re.compile(r"(?i:answer)\s*(?i:is)?\s*:?\s*[*_(\[]*([A-E])(?![A-Za-z])")
_CHOICE = re.compile(r"(?i:choice|option)\s*(?i:is)?\s*:?\s*[*_(\[]*([A-E])(?![A-Za-z])")
_LEADING = re.compile(r"^[\s*_(\[]*([A-E])(?:[*_)\]]+|\s*[.:)]|\s*$)")
_TRAILING = re.compile(r"(?:^|\s)[*_(\[]*([A-E])[*_)\]]*\.?\s*$")
_LONE = re.compile(r"^[\s*_(\[]*([a-eA-E])[*_)\].]*\s*$")
_GOLD = re.compile(r"(?i)^\s*([A-E])\b")


def _answer_text(text: str | None) -> str:
    raw = _THINK.sub("", text or "")
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[1]
    return raw


def gold_letter(option: str | None) -> str | None:
    match = _GOLD.match(option or "")
    return match.group(1).upper() if match else None


def first_letter(text: str | None) -> str | None:
    raw = (text or "").lstrip()
    if raw and raw[0].upper() in LETTERS:
        return raw[0].upper()
    return None


def predicted_letter(text: str | None) -> str | None:
    raw = _answer_text(text)
    for pattern in (_ANSWER, _CHOICE):
        explicit = pattern.findall(raw)
        if explicit:
            return explicit[-1]
    for pattern in (_LEADING, _TRAILING, _LONE):
        match = pattern.search(raw)
        if match:
            return match.group(1).upper()
    return None


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
