# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text helpers shared by the audio scorers: whitespace and compound-word handling, CER whitespace modes and the
generation-loop guardrail.

The ``asr_with_pc`` server scores with the same definitions; each server is its own package with its own
environment, so this server keeps its own copy instead of importing the other one.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    """Collapse runaway repetition in a decoded transcript.

    A character repeated 21 or more times, and then an exact block of 1-4096
    characters repeated 20 or more times, are each replaced by a single copy. The
    thresholds follow the repetition fix of Qwen3-ASR
    (https://github.com/QwenLM/Qwen3-ASR, ``detect_and_fix_repetitions``, Apache-2.0),
    but this is an independent regular-expression implementation: no code is copied
    and it accepts longer blocks (Qwen3-ASR limits them to 20 characters).
    """
    text = _CHAR_RUN.sub(lambda m: m.group(1), text)
    return _BLOCK_RUN.sub(lambda m: m.group(1), text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_compound_pairs(reference: str, hypothesis: str) -> tuple[str, str]:
    """Join equivalent mismatch spans that differ only in word boundaries."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    new_ref: list[str] = []
    new_hyp: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref_words, hyp_words).get_opcodes():
        if tag == "equal":
            new_ref.extend(ref_words[i1:i2])
            new_hyp.extend(hyp_words[j1:j2])
            continue
        ref_compound = "".join(ref_words[i1:i2])
        hyp_compound = "".join(hyp_words[j1:j2])
        if ref_compound == hyp_compound:
            new_ref.append(ref_compound)
            new_hyp.append(hyp_compound)
        else:
            new_ref.extend(ref_words[i1:i2])
            new_hyp.extend(hyp_words[j1:j2])
    return " ".join(new_ref), " ".join(new_hyp)


# Han ideographs used by the han_adjacent CER whitespace predicate and by the Han component of the
# scriptio-continua segmentation set of the audio evaluator.
HAN_RANGES = (
    "\u3007"  # Ideographic number zero
    "\u3400-\u4dbf"  # CJK Unified Ideographs Extension A
    "\u4e00-\u9fff"  # CJK Unified Ideographs
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\U00020000-\U0002a6df"  # Extension B
    "\U0002a700-\U0002b73f"  # Extension C
    "\U0002b740-\U0002b81f"  # Extension D
    "\U0002b820-\U0002ceaf"  # Extension E
    "\U0002ceb0-\U0002ebef"  # Extension F
    "\U0002ebf0-\U0002ee5f"  # Extension I
    "\U0002f800-\U0002fa1f"  # Compatibility Ideographs Supplement
    "\U00030000-\U0003134f"  # Extension G
    "\U00031350-\U000323af"  # Extension H
)
HAN_RE = re.compile(f"[{HAN_RANGES}]")
VALID_CER_WHITESPACE_MODES = frozenset({"keep", "all", "han_adjacent"})


def normalize_cer_whitespace(text: str, mode: str = "keep") -> str:
    """Normalize whitespace for CER.

    ``han_adjacent`` removes a whitespace run only when the nearest non-space
    character on either side is a Han ideograph, so FLEURS Mandarin/Cantonese
    character-segmentation spaces are ignored while Latin word boundaries stay.
    """
    if mode not in VALID_CER_WHITESPACE_MODES:
        raise ValueError(f"Invalid CER whitespace mode {mode!r}; expected one of {sorted(VALID_CER_WHITESPACE_MODES)}")
    if mode == "keep":
        return text
    if mode == "all":
        return re.sub(r"\s+", "", text)
    output: list[str] = []
    index = 0
    while index < len(text):
        if not text[index].isspace():
            output.append(text[index])
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end].isspace():
            end += 1
        left = text[index - 1] if index > 0 else ""
        right = text[end] if end < len(text) else ""
        if not (HAN_RE.fullmatch(left) or HAN_RE.fullmatch(right)):
            output.append(text[index:end])
        index = end
    return "".join(output)
