# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# `normalize_alimeeting_text` is a Python translation of the AliMeeting text cleanup of
# https://github.com/yufan-aslp/AliMeeting at commit 692a034fd510f1720f547c2c91e0fbadc9c24bf2: the rules of
# `asr/local/text_normalize.pl` (marker and punctuation deletion, full-width letter folding, upper-casing; the perl
# script carries "Copyright Chao Weng") and the `sed` deletions of `！`, `？`, `(~)`, `(SIL)` and `(/OVERLAP)` in
# `asr/local/alimeeting_data_prep.sh`. License: Apache-2.0, as the repository README declares ("Code license"); the
# commit has no LICENSE file.
# Modifications: rewritten in Python; applied to one string instead of a whitespace-split Kaldi text file; all
# whitespace is removed so the text can be split into characters; the `SRC` -> `$` substitution of the far-field
# data preparation is not ported.
#
# `normalize_aishell4_text` is not copied: it implements the one-line rule that the AISHELL-4 repository documents in
# `data_preparation/Readme.md` (https://github.com/felixfuyihui/AISHELL-4 at commit
# bad82b77c3753df1b232c5c6491cd3e2f2e32d24, Apache-2.0): "delete the non-lexical information in `text` and do
# tokenizer", with the vim substitutions `[^\u4e00-\u9fa5]` (delete) and `([\u4e00-\u9fa5])` (split into characters).

"""Reference text normalization for the Mandarin meeting cpCER evaluation (AISHELL-4 and AliMeeting).

The evaluator applies the corpus text normalization and character tokenization before the minimum-error speaker
permutation. The AliMeeting cleanup is adapted from the released AliMeeting data-preparation scripts (see the
header); the AISHELL-4 rule is the Han-only filter its repository documents.
"""

from __future__ import annotations

import re


_MARKERS = ("<sil>", "<%>", "<->", "<$>", "<#>", "<_>", "<space>")
_DELETE_CHARS = "`&,，丶。、？"
_FULLWIDTH_ASCII = str.maketrans({"Ａ": "A", "ａ": "A", "ｂ": "B", "ｃ": "C", "ｋ": "K", "ｔ": "T"})


def normalize_alimeeting_text(text: str) -> str:
    """Apply the AliMeeting cleanup: drop markers and punctuation, fold full-width letters, upper-case Latin text."""
    normalized = str(text)
    for marker in _MARKERS:
        normalized = normalized.replace(marker, "")
    normalized = normalized.translate(_FULLWIDTH_ASCII)
    normalized = normalized.translate(str.maketrans("", "", _DELETE_CHARS))
    for marker in ("！", "？", "(~)", "(SIL)", "(/OVERLAP)"):
        normalized = normalized.replace(marker, "")
    normalized = re.sub(r"\s+", "", normalized)
    if re.search(r"[A-Za-z]", normalized):
        normalized = normalized.upper()
    return normalized


def normalize_aishell4_text(text: str) -> str:
    """Keep only Han characters (U+4E00 to U+9FA5), the rule the AISHELL-4 data preparation readme documents."""
    return re.sub(r"[^一-龥]", "", str(text))


def normalize_mandarin_meeting_text(text: str, corpus: str) -> str:
    """Select the released corpus-specific reference normalization."""
    if corpus in {"alimeeting", "alimeeting-sot"}:
        return normalize_alimeeting_text(text)
    if corpus in {"aishell4", "aishell4-sot"}:
        return normalize_aishell4_text(text)
    raise ValueError(f"Unsupported Mandarin meeting corpus: {corpus!r}")


def cpcer_token_string(text: str, corpus: str) -> str:
    """Return whitespace-delimited Unicode characters for cpCER edit distance."""
    normalized = normalize_mandarin_meeting_text(text, corpus)
    return " ".join(character for character in normalized if not character.isspace())
