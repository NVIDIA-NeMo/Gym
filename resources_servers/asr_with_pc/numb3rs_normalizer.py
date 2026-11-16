# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative Numb3rs spelling normalization; no number-value conversion.

Whisper's English normalizer rewrites number words as digits in both the reference and the hypothesis, which hides
exactly the errors Numb3rs measures. This normalizer is a local evaluation protocol, not an official Numb3rs one:
the same category-dependent rules apply independently to the reference and the hypothesis.
"""

import re


# Expand only conventional initialisms in categories where they are unambiguous.
# In particular, never join arbitrary single letters: repeated "o" denotes zeros.
_INITIALISMS = {
    "ADDRESS": "sr us cr",
    "TIME": "am pm utc gmt bst est edt cst cdt mst mdt pst pdt cet cest ist et ct ast eet",
    "TELEPHONE": "isbn oclc",
}
_ZERO_CATEGORIES = {
    "ADDRESS",
    "CARDINAL",
    "DATE",
    "DECIMAL",
    "DIGIT",
    "FRACTION",
    "MEASURE",
    "MONEY",
    "ORDINAL",
    "TELEPHONE",
    "TIME",
}


def normalize_numb3rs_text(text: str, category: str) -> str:
    """Ignore case/prose punctuation, zero spelling and conventional initialisms.

    Preserve currency, percentages, signs and separators inside numbers so that
    "$1.5", "15", "1/5" and "-15" cannot become equal. Keep number words,
    repeated zeros, optional "and", and chemical names unchanged.
    """
    text = text.lower().replace("−", "-")
    chars = []
    for i, char in enumerate(text):
        between_digits = i > 0 and i + 1 < len(text) and text[i - 1].isdigit() and text[i + 1].isdigit()
        sign = char in "+-" and i + 1 < len(text) and text[i + 1].isdigit() and (i == 0 or not text[i - 1].isalnum())
        if char.isalnum() or char.isspace() or char == "_" or char in "$€£¥%+=" or sign:
            chars.append(char)
        elif char in ".,:/-" and between_digits:
            chars.append(char)
    text = " ".join("".join(chars).split())
    for initialism in _INITIALISMS.get(category, "").split():
        text = re.sub(rf"\b{initialism}\b", " ".join(initialism), text)
    if category in _ZERO_CATEGORIES:
        text = re.sub(r"\boh\b", "o", text)
    return " ".join(text.split())
