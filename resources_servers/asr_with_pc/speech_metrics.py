# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech scoring primitives shared by the ASR resource server.

Per-sample scorers return additive edit counts so Gym can compute corpus
metrics and per-subset breakdowns without averaging utterance error rates.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import jiwer
from kaldialign import batch_error_rate
from num2words import num2words
from whisper_normalizer.english import EnglishTextNormalizer

from resources_servers.asr_with_pc.open_asr_normalizer import OpenASREnglishTextNormalizer


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


def _wer_counts(reference: str, hypothesis: str, *, merge_compounds: bool = False) -> dict[str, Any]:
    if merge_compounds:
        measures = batch_error_rate(
            [tuple(reference.split())],
            [tuple(hypothesis.split())],
            merge_compounds=True,
        )
        return {
            "wer": float(measures["err_rate"]),
            "wer_errors": int(measures["total"]),
            "wer_ref_words": int(measures["ref_len"]),
            "wer_substitutions": int(measures["sub"]),
            "wer_insertions": int(measures["ins"]),
            "wer_deletions": int(measures["del"]),
        }

    measures = jiwer.process_words(reference, hypothesis)
    errors = measures.substitutions + measures.deletions + measures.insertions
    ref_words = measures.substitutions + measures.deletions + measures.hits
    return {
        "wer": errors / ref_words if ref_words else 0.0,
        "wer_errors": errors,
        "wer_ref_words": ref_words,
        "wer_substitutions": measures.substitutions,
        "wer_insertions": measures.insertions,
        "wer_deletions": measures.deletions,
    }


def _cer_counts(reference: str, hypothesis: str, *, prefix: str = "cer") -> dict[str, Any]:
    measures = jiwer.process_characters(reference, hypothesis)
    errors = measures.substitutions + measures.deletions + measures.insertions
    ref_chars = measures.substitutions + measures.deletions + measures.hits
    return {
        prefix: errors / ref_chars if ref_chars else 0.0,
        f"{prefix}_errors": errors,
        f"{prefix}_ref_words": ref_chars,
        f"{prefix}_substitutions": measures.substitutions,
        f"{prefix}_insertions": measures.insertions,
        f"{prefix}_deletions": measures.deletions,
    }


_OPEN_ASR_NORMALIZER = OpenASREnglishTextNormalizer()
_ENGLISH_NORMALIZER = EnglishTextNormalizer()


def normalize_open_asr_text(text: str) -> str:
    return normalize_whitespace(_OPEN_ASR_NORMALIZER(text or ""))


def evaluate_open_asr(reference: str, hypothesis: str) -> dict[str, Any]:
    """Score the current Hugging Face Open ASR English protocol."""
    ref = normalize_open_asr_text(reference)
    hyp = normalize_open_asr_text(hypothesis)
    if not ref:
        return {"wer": None, "is_correct": None, "text": "", "pred_text": hyp}
    result = _wer_counts(ref, hyp, merge_compounds=True)
    result.update(is_correct=result["wer"] < 0.5, text=ref, pred_text=hyp)
    return result


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


def _remove_symbols(text: str, *, remove_diacritics: bool) -> str:
    def replace(character: str) -> str:
        if remove_diacritics and character in _ADDITIONAL_DIACRITICS:
            return _ADDITIONAL_DIACRITICS[character]
        category = unicodedata.category(character)
        if remove_diacritics and category == "Mn":
            return ""
        if category[0] in "MSP":
            return " "
        return character

    form = "NFKD" if remove_diacritics else "NFKC"
    return "".join(replace(character) for character in unicodedata.normalize(form, text))


class MultilingualTextNormalizer:
    def __init__(self, *, remove_diacritics: bool = False):
        self.remove_diacritics = remove_diacritics

    @staticmethod
    def _normalize_numbers(text: str, language: str) -> str:
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        def replace(match: re.Match[str]) -> str:
            try:
                return num2words(int(match.group()), lang=language)
            except Exception:  # noqa: BLE001 - unsupported locales keep digits.
                return match.group()

        return re.sub(r"\d+", replace, text)

    def __call__(self, text: str, *, language: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        # Spell digits out BEFORE folding diacritics. num2words output carries
        # diacritics/marks (Turkish "yüz", "beş"; Thai tone marks; Hangul
        # syllables that NFKD splits into jamo). Spelling out after the fold
        # leaves those marks in the reference while the hypothesis, which spells
        # numbers in words, was already folded, producing spurious errors.
        # Only punctuation/symbols are replaced here so digit groups stay intact.
        text = "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFKC", text))
        text = self._normalize_numbers(text, language)
        text = _remove_symbols(text, remove_diacritics=self.remove_diacritics)
        return normalize_whitespace(re.sub(r"[^\w\s]", "", text))


def normalize_multilingual_text(text: str, *, language: str, remove_diacritics: bool = False) -> str:
    language = (language or "").split("_")[0]
    if language in {"", "en"}:
        return normalize_whitespace(_ENGLISH_NORMALIZER((text or "").lower()))
    return MultilingualTextNormalizer(remove_diacritics=remove_diacritics)(text, language=language)


_HAN_RANGES = (
    "\u3007"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uf900-\ufaff"
    "\U00020000-\U0002a6df"
    "\U0002a700-\U0002ebef"
    "\U0002ebf0-\U0002ee5f"
    "\U0002f800-\U0002fa1f"
    "\U00030000-\U000323af"
)
_HAN_RE = re.compile(f"[{_HAN_RANGES}]")
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
        if not (_HAN_RE.fullmatch(left) or _HAN_RE.fullmatch(right)):
            output.append(text[index:end])
        index = end
    return "".join(output)


def evaluate_multilingual_asr(
    reference: str,
    hypothesis: str,
    *,
    language: str,
    use_cer: bool = False,
    remove_diacritics: bool = False,
    normalize_compound: bool = True,
    merge_compounds: bool = False,
    cer_whitespace_mode: str = "keep",
) -> dict[str, Any]:
    normalize_cer_whitespace("", mode=cer_whitespace_mode)  # validate
    if cer_whitespace_mode != "keep":
        # Non-keep whitespace modes are never combined with compound merging.
        normalize_compound = False
    ref = normalize_multilingual_text(reference, language=language, remove_diacritics=remove_diacritics)
    hyp = normalize_multilingual_text(hypothesis, language=language, remove_diacritics=remove_diacritics)
    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)
    if cer_whitespace_mode != "keep":
        ref = normalize_cer_whitespace(ref, mode=cer_whitespace_mode)
        hyp = normalize_cer_whitespace(hyp, mode=cer_whitespace_mode)
    if not ref:
        return {"wer": None, "cer": None, "is_correct": None, "text": "", "pred_text": hyp}
    if use_cer:
        result = _cer_counts(ref, hyp, prefix="wer")
    else:
        result = _wer_counts(ref, hyp, merge_compounds=merge_compounds)
    result.update(_cer_counts(ref, hyp, prefix="cer"))
    result.update(is_correct=result["wer"] < 0.5, text=ref, pred_text=hyp)
    return result
