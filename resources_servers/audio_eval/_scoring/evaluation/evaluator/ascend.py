# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The punctuation list and the MER/CER tokenization follow the official ASCEND scorer, utils.py at commit
# 8c50c2e2c65f8555eb1d7655310dd585dc1f6710 of https://github.com/HLTCHKUST/ASCEND (MIT License):
#
# Copyright (c) 2022 HKUST
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# Modifications: the punctuation list and the two tokenizers are unchanged apart from formatting and a pinned-jieba
# check; the edit distance, the per-sample scoring, the symmetric normalization of the hypothesis and the evaluator
# class are new.

"""ASCEND MER/CER evaluator with two scoring protocols.

The official ASCEND code normalizes references before constructing the CTC label vocabulary, then tokenizes decoded
predictions without another normalization pass. CTC predictions are therefore already constrained to the normalized
alphabet; free-form audio-language-model generations are not. The unsuffixed ``mer``/``cer`` keys apply the same
normalization to the reference and the hypothesis. The ``*_upstream_literal`` keys tokenize the raw hypothesis, as the
official code does.
"""

from __future__ import annotations

import re
from typing import Any

import jieba

from resources_servers.audio_eval._scoring.edit_distance import edit_distance
from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator


JIEBA_VERSION = "0.42.1"


# Copied from the official ASCEND utils.py scorer. Upstream uses a character class, so the multi-character-looking
# entry '""' acts as the individual ignored character '"'.
CHARS_TO_IGNORE = [
    ",",
    "?",
    "¿",
    ".",
    "!",
    "¡",
    ";",
    "；",
    ":",
    '""',
    "%",
    '"',
    "�",
    "ʿ",
    "·",
    "჻",
    "~",
    "՞",
    "؟",
    "،",
    "।",
    "॥",
    "«",
    "»",
    "„",
    "“",
    "”",
    "「",
    "」",
    "‘",
    "’",
    "《",
    "》",
    "(",
    ")",
    "{",
    "}",
    "=",
    "\x60",
    "_",
    "+",
    "<",
    ">",
    "…",
    "–",
    "°",
    "´",
    "ʾ",
    "‹",
    "›",
    "©",
    "®",
    "—",
    "→",
    "。",
    "、",
    "﹂",
    "﹁",
    "‧",
    "～",
    "﹏",
    "，",
    "｛",
    "｝",
    "（",
    "）",
    "［",
    "］",
    "【",
    "】",
    "‥",
    "〽",
    "『",
    "』",
    "〝",
    "〟",
    "⟨",
    "⟩",
    "〜",
    "：",
    "！",
    "？",
    "♪",
    "؛",
    "/",
    "\\",
    "º",
    "−",
    "^",
    "ʻ",
    "ˆ",
]
_IGNORE_RE = re.compile("[" + re.escape("".join(CHARS_TO_IGNORE)) + "]")


def normalize_ascend(text: str) -> str:
    """Apply ASCEND's official lowercasing and punctuation removal."""
    return _IGNORE_RE.sub("", text).lower() + " "


def mer_tokens(text: str) -> list[str]:
    """Jieba-segment, preserving ASCII spans and splitting non-ASCII spans."""
    if jieba.__version__ != JIEBA_VERSION:
        raise RuntimeError(f"ASCEND scoring requires jieba=={JIEBA_VERSION}; found {jieba.__version__!r}")
    output = []
    for token in jieba.lcut(text):
        if not token.strip():
            continue
        output.extend([token] if token.isascii() else list(token))
    return output


def cer_tokens(text: str) -> list[str]:
    """Every non-space character is one unit."""
    return [character for character in text if character.strip()]


def evaluate_ascend_sample(data_point: dict[str, Any]) -> dict[str, Any]:
    """Score one sample under both protocols (see the module docstring)."""
    reference = normalize_ascend(str(data_point["expected_answer"]))
    raw_generation = str(data_point.get("generation") or "")
    generation = raw_generation.strip()
    hypothesis = normalize_ascend(generation)
    reference_mer = mer_tokens(reference)
    reference_cer = cer_tokens(reference)

    mer_errors = edit_distance(reference_mer, mer_tokens(hypothesis))
    cer_errors = edit_distance(reference_cer, cer_tokens(hypothesis))
    upstream_mer_errors = edit_distance(reference_mer, mer_tokens(raw_generation))
    upstream_cer_errors = edit_distance(reference_cer, cer_tokens(raw_generation))
    mer_ref_units = len(reference_mer)
    cer_ref_chars = len(reference_cer)
    return {
        "generation": generation,
        "text": reference,
        "pred_text": hypothesis,
        "mer": mer_errors / mer_ref_units if mer_ref_units else None,
        "mer_errors": mer_errors,
        "mer_ref_units": mer_ref_units,
        "cer": cer_errors / cer_ref_chars if cer_ref_chars else None,
        "cer_errors": cer_errors,
        "cer_ref_chars": cer_ref_chars,
        "is_correct": mer_errors == 0,
        "mer_upstream_literal": upstream_mer_errors / mer_ref_units if mer_ref_units else None,
        "mer_upstream_literal_errors": upstream_mer_errors,
        "mer_upstream_literal_ref_units": mer_ref_units,
        "cer_upstream_literal": upstream_cer_errors / cer_ref_chars if cer_ref_chars else None,
        "cer_upstream_literal_errors": upstream_cer_errors,
        "cer_upstream_literal_ref_chars": cer_ref_chars,
        "is_correct_upstream_literal": upstream_mer_errors == 0,
    }


class ASCENDEvaluator(BaseEvaluator):
    """Evaluate ASCEND transcripts with mixed and character error rates."""

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        return evaluate_ascend_sample(data_point)
