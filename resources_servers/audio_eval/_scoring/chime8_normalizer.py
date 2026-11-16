# SPDX-FileCopyrightText: Copyright (c) 2024 Samuele Cornell
# SPDX-FileCopyrightText: Copyright (c) 2022 OpenAI
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Adapted from https://github.com/chimechallenge/chime-utils/blob/152882404f572d40769ef02bf91c5a9a9cfc9c78/chime_utils/text_norm/whisper_like/english.py
# (with ``remove_symbols_and_diacritics`` from ``basic.py`` in the same directory). chime-utils is MIT
# licensed and derives this normalizer from the OpenAI Whisper English normalizer, also MIT licensed.
# Changes: the ``more_itertools`` dependency is replaced by ``_windowed``; the basic helper is inlined; the
# classes are renamed and made private; the JSON tables are loaded from ``chime8_normalizer_data/`` (copied
# unchanged from the same directory upstream); ``normalize`` is added here and applies the normalizer until its
# output stops changing; comments and docstrings are condensed and the code is reformatted.
# See ATTRIBUTIONS.md and CHIME_UTILS_LICENSE.

"""CHIME-8 text normalization without a chime-utils or MeetEval dependency.

This is the ``chime8`` normalizer that ``meeteval-wer --normalizer chime8`` obtains from
``chime_utils.text_norm.get_txt_norm("chime8")``. It is idempotent once ``normalize`` converges.
"""

from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Match, Optional, Union


__all__ = [
    "Chime8TextNormalizer",
    "normalize",
]

_DATA_DIR = Path(__file__).resolve().parent / "chime8_normalizer_data"

# non-ASCII letters that are not separated by "NFKD" normalization
_ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def _remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """Drop diacritics / punctuation; keep characters listed in *keep*."""
    return "".join(
        (
            c
            if c in keep
            else (
                _ADDITIONAL_DIACRITICS[c]
                if c in _ADDITIONAL_DIACRITICS
                else ("" if unicodedata.category(c) == "Mn" else " " if unicodedata.category(c)[0] in "MSP" else c)
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def _windowed(items: List, size: int):
    """Simple sliding window helper (replaces ``more_itertools.windowed``)."""
    if size <= 0:
        raise ValueError("window size must be positive")
    n = len(items)
    for i in range(n - size + 1):
        yield tuple(items[i : i + size])


class _EnglishNumberNormalizer:
    """
    Convert spelled-out numbers into arabic numbers.

    Copied from Whisper / chime-utils with no behavioural changes.
    """

    def __init__(self):
        self.zeros = {"o", "zero"}
        self.ones = {
            name: i
            for i, name in enumerate(
                [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ],
                start=1,
            )
        }
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s") for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()}
        self.tens_ordinal = {name.replace("y", "ieth"): (value, "th") for name, value in self.tens.items()}
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {name + "s": (value, "s") for name, value in self.multipliers.items()}
        self.multipliers_ordinal = {name + "th": (value, "th") for name, value in self.multipliers.items()}
        self.multipliers_suffixed = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = set(
            [
                key
                for mapping in [
                    self.zeros,
                    self.ones,
                    self.ones_suffixed,
                    self.tens,
                    self.tens_suffixed,
                    self.multipliers,
                    self.multipliers_suffixed,
                    self.preceding_prefixers,
                    self.following_prefixers,
                    self.suffixers,
                    self.specials,
                ]
                for key in mapping
            ]
        )
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for prev, current, next in _windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                f = to_fraction(current_without_prefix)
                assert f is not None
                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        value = str(value) + str(current)
                        continue
                    yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator
                else:
                    value = current_without_prefix
            elif current not in self.words:
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        assert value[-1] == "0"
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    raise ValueError(f"Unexpected token: {current}")
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)
        return s


class _EnglishReverseNumberNormalizer(_EnglishNumberNormalizer):
    """Approximate inverse of ``_EnglishNumberNormalizer`` for ASR compatibility."""

    def __init__(self):
        super().__init__()
        self.int_to_ones = {v: k for k, v in self.ones.items()}
        self.int_to_tens = {v: k for k, v in self.tens.items()}
        self.str_to_ones_suffixed = {str(n) + s: k for k, (n, s) in self.ones_suffixed.items()}
        self.str_to_tens_suffixed = {str(n) + s: k for k, (n, s) in self.tens_suffixed.items()}

    def __call__(self, s: str):
        s = re.sub(r"\$(\d+(\.\d+)?)", r"\1 dollars", s)
        s = re.sub(r"(\d+(\.\d+)?)\$", r"\1 dollars", s)
        s = re.sub(r"(\d+(\.\d+)?)%", r"\1 percent", s)

        def number_to_words(w: str):
            if w.isdigit():
                # Avoid pathological generated digit strings in num2words.
                if len(w) > 64:
                    return w
                num = int(w)
                if w == "000":
                    return "thousand"
                if num == 0:
                    return "zero"
                if num == 100:
                    return "hundred"
                if 0 < num < 1000:
                    hundreds, remainder = divmod(num, 100)
                    tens, ones = divmod(remainder, 10)
                    h = [f"{self.int_to_ones[hundreds]} hundred"] if hundreds > 0 else []
                    if 0 < remainder <= 19:
                        t = [self.int_to_ones[remainder]]
                        o = []
                    else:
                        t = [self.int_to_tens[tens * 10]] if tens > 0 else []
                        o = [self.int_to_ones[ones]] if ones > 0 else []
                    return " ".join(h + t + o)
                if num == 1000:
                    return "thousand"
                return w
            w = self.str_to_ones_suffixed.get(w, w)
            w = self.str_to_tens_suffixed.get(w, w)
            return w

        return " ".join(number_to_words(w) for w in s.split())


class _EnglishSpellingNormalizer:
    """British-American spelling mappings from Whisper / chime-utils."""

    def __init__(self, mapping_name: str = "english.json"):
        mapping_path = _DATA_DIR / mapping_name
        with open(mapping_path, encoding="utf-8") as f:
            self.mapping = json.load(f)

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class Chime8TextNormalizer:
    """
    CHIME-8 scoring text normalizer (Whisper-derived, idempotent, less aggressive).

    Key behaviour:
    - lowercases, strips bracketed / parenthetical annotations
    - expands contractions and title abbreviations (Mr. -> mister, etc.)
    - removes filler words (hmm, uh, ah, eh) by default
    - converts numerals to spelled-out form (reverse-number path) for fair ASR scoring
    - normalises UK/US spellings
    """

    def __init__(
        self,
        standardize_numbers: bool = False,
        standardize_numbers_rev: bool = True,
        remove_fillers: bool = True,
    ):
        self.replacers = {
            r"\u2019": ("'"),
            r"\b(hm+)\b|\b(mhm)\b|\b(mm+)\b|\b(m+h)\b|\b(hm+)\b|\b(um+)\b|\b(uhm+)\b": ("hmm"),
            r"\b(a+h+)\b|\b(ha+)\b": "ah",
            r"[!?.]+(?=$|\s)": "",
            r"\b(o+h+)\b|\b(h+o+)\b": "oh",
            r"\b(u+h+)\b|\b(h+u+)\b|\b(h+u+h+)\b": "uh",
            r"\b(wi\sfi)\b": "wifi",
            r"\b(goin)\b": "going",
            r"\wi-fi\b": "wifi",
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\bokay\b": "ok",
            r"\bsetup\b": "set up",
            r"\beveryday\b": "every day",
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        if standardize_numbers:
            self.standardize_numbers = _EnglishNumberNormalizer()
            assert not standardize_numbers_rev
        else:
            self.standardize_numbers = None

        if standardize_numbers_rev:
            self.standardize_numbers_rev = _EnglishReverseNumberNormalizer()
        else:
            self.standardize_numbers_rev = None

        self.standardize_spellings = _EnglishSpellingNormalizer()
        self.pre_standardize_spellings = _EnglishSpellingNormalizer("pre_english.json")

        if remove_fillers:
            self.fillers = ["hmm", "uh", "ah", "eh"]
        else:
            self.fillers = None

    def __call__(self, s: str) -> str:
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = self.pre_standardize_spellings(s)
        s = re.sub(r"\s+'", "'", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        s = _remove_symbols_and_diacritics(s, keep=".%$¢€£")

        if self.standardize_numbers is not None:
            s = self.standardize_numbers(s)

        if self.standardize_numbers_rev is not None:
            s = self.standardize_numbers_rev(s)

        s = self.standardize_spellings(s)
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        if self.fillers:
            s = re.sub(r"\b(" + "|".join(self.fillers) + r")\b", "", s)

        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"^\s+|\s+$", "", s)

        return s


_DEFAULT_NORMALIZER = Chime8TextNormalizer()


def normalize(text: str, *, max_passes: int = 5) -> str:
    """
    Apply CHIME-8 text normalization until idempotent (at most *max_passes*).

    This mirrors the loop recommended in chime-utils / MeetEval scoring pipelines.
    """
    words = text
    for _ in range(max_passes):
        words_ = _DEFAULT_NORMALIZER(words)
        if words == words_:
            return words
        words = words_
    raise RuntimeError(f"CHIME-8 normalizer did not converge within {max_passes} passes for input: {text!r}")
