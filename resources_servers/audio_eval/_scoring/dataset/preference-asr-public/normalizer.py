# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the normalizer package of the Preference-ASR release:
#   source:   https://github.com/nithinraok/preference-asr-bench (directory normalizer/)
#   revision: efa5cae9aecb56ce28aeea6105425ded654acbdd
#   license:  Apache-2.0 (the package itself is vendored there from the Open ASR Leaderboard, Copyright 2022 The
#             OpenAI team and The HuggingFace Team, Apache-2.0, which copied it from OpenAI Whisper, MIT)
# Modifications: the vendored English normalizer is not copied; it is rebuilt from the ``whisper-normalizer`` package
# (0.1.12) plus the release's small replacer and spelling changes, and the released preference-aware skip logic is
# reimplemented on top of it. ``target_case`` is recovered from ``sub_category`` (the released manifest omits it),
# and unsupported values raise instead of being ignored.

"""Compact parity implementation of the Preference-ASR release normalizers.

The public release vendors a full copy of Whisper's English normalizer. This module rebuilds it from
``whisper-normalizer==0.1.12`` plus the release's small changes, then layers its preference skip logic.
The released manifest omits ``target_case`` even though its helper reads it, so
we deterministically recover that value from the case ``sub_category``.
"""

from __future__ import annotations

import re
from functools import cache

from whisper_normalizer.english import EnglishTextNormalizer, remove_symbols_and_diacritics


_REMOVED_REPLACERS = (r"\bkinda\b", r"\bsorta\b", r"\bdunno\b", r"\bcause\b")
_SPELLING_REMOVALS = {"archeological", "cancelation", "cancelations", "travelog", "travelogs"}
_SPELLING_UPDATES = {
    "archaeological": "archeological",
    "cancellation": "cancelation",
    "cancellations": "cancelations",
    "mm": "hmm",
    "travelogue": "travelog",
    "travelogues": "travelogs",
    "archaeology": "archeology",
    "pummelled": "pummeled",
    "pummelling": "pummeling",
    "snowploughs": "snowplows",
}
_CASE_TARGETS = {"lower_case", "upper_case", "proper_punctuated_capitalized_format"}


def _release_english_normalizer() -> EnglishTextNormalizer:
    normalizer = EnglishTextNormalizer()
    for pattern in _REMOVED_REPLACERS:
        normalizer.replacers.pop(pattern)
    for word in _SPELLING_REMOVALS:
        normalizer.standardize_spellings.mapping.pop(word)
    normalizer.standardize_spellings.mapping.update(_SPELLING_UPDATES)
    return normalizer


@cache
def standard_normalizer() -> EnglishTextNormalizer:
    """Return the exact full normalizer vendored by the pinned public release."""
    return _release_english_normalizer()


class PreferenceAwareNormalizer:
    """Released selective normalizer with the manifest's case-field repair."""

    def __init__(self):
        base = _release_english_normalizer()
        self.ignore_patterns = r"\b(ah|eh|uh|um|umm|uhh|hmm|mm|mhm|mmm)\b"
        self.replacers = base.replacers.copy()
        self.standardize_numbers = base.standardize_numbers
        self.standardize_spellings = base.standardize_spellings

    @staticmethod
    def _compute_skip_flags(preference_type, sub_category, direction, target_case):
        sub_category = set(sub_category or [])
        flags = {
            "skip_lowercase": False,
            "skip_filler_removal": False,
            "skip_period_removal": False,
            "skip_number_standardization": False,
            "skip_orphan_symbol_removal": False,
            "preserve_punctuation": False,
        }
        if preference_type == "normalization":
            flags["skip_number_standardization"] = True
            if "symbols" in sub_category:
                flags["skip_orphan_symbol_removal"] = True
            if "website_links" in sub_category:
                flags["skip_period_removal"] = True
                flags["skip_orphan_symbol_removal"] = True
        elif preference_type == "disfluencies" and direction == "keep":
            flags["skip_filler_removal"] = True
        elif preference_type == "case":
            flags["skip_lowercase"] = True
            if target_case in {"proper_punctuated_capitalized_format", "upper_case"}:
                flags["preserve_punctuation"] = True
                flags["skip_period_removal"] = True
                flags["skip_orphan_symbol_removal"] = True
        return flags

    @staticmethod
    def _collapse_dotted_acronyms(text: str) -> str:
        return re.sub(
            r"(?<!\w)((?:[a-zA-Z]\.){2,})",
            lambda match: match.group().replace(".", ""),
            text,
        )

    @staticmethod
    def _collapse_spaced_acronyms(text: str) -> str:
        tokens = text.split()
        result: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if len(token) == 1 and token.isalpha() and token not in ("a", "i"):
                next_index = index + 1
                while next_index < len(tokens) and len(tokens[next_index]) == 1 and tokens[next_index].isalpha():
                    next_index += 1
                if next_index - index >= 2:
                    result.append("".join(tokens[index:next_index]))
                    index = next_index
                    continue
            result.append(token)
            index += 1
        return " ".join(result)

    def __call__(
        self,
        text: str,
        preference_type: str = "",
        sub_category: list[str] | None = None,
        direction: str = "",
        target_case: str = "",
    ) -> str:
        flags = self._compute_skip_flags(preference_type, sub_category, direction, target_case)
        if not flags["skip_lowercase"]:
            text = text.lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        if flags["skip_filler_removal"]:
            text = re.sub(self.ignore_patterns, "um", text)
        else:
            text = re.sub(self.ignore_patterns, "", text)
        text = re.sub(r"\s+'", "'", text)
        for pattern, replacement in self.replacers.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        text = self._collapse_dotted_acronyms(text)
        if not flags["skip_period_removal"]:
            text = re.sub(r"\.([^0-9]|$)", r" \1", text)
        keep_chars = ".%$¢€£"
        if flags["preserve_punctuation"]:
            keep_chars += "?!',\":;-"
        text = remove_symbols_and_diacritics(text, keep=keep_chars)
        if not flags["skip_number_standardization"]:
            text = self.standardize_numbers(text)
        text = self.standardize_spellings(text)
        if not flags["skip_orphan_symbol_removal"]:
            text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
            text = re.sub(r"([^0-9])%", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)
        return self._collapse_spaced_acronyms(text).strip()

    @staticmethod
    def target_case_from_entry(entry: dict) -> str:
        explicit = entry.get("target_case")
        if explicit:
            if explicit not in _CASE_TARGETS:
                raise ValueError(f"Unsupported Preference-ASR target_case: {explicit!r}")
            return explicit
        if entry.get("preference_type") != "case":
            return ""
        matches = [value for value in entry.get("sub_category", []) if value in _CASE_TARGETS]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot recover Preference-ASR target_case from sub_category={entry.get('sub_category')!r}"
            )
        return matches[0]

    def normalize_entry(self, text: str, entry: dict) -> str:
        return self(
            text,
            preference_type=entry.get("preference_type", ""),
            sub_category=entry.get("sub_category"),
            direction=entry.get("direction", ""),
            target_case=self.target_case_from_entry(entry),
        )


@cache
def preference_normalizer() -> PreferenceAwareNormalizer:
    return PreferenceAwareNormalizer()
