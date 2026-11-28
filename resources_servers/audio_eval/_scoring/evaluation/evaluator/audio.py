# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA-NeMo/Skills (Apache-2.0), nemo_skills/evaluation/evaluator/audio.py
# at commit da85a881d972e6fec847b90cf553a0bf9bf10638: https://github.com/NVIDIA-NeMo/Skills
# Modifications (NVIDIA-authored; no addition is taken from another source): the file-level `eval_audio` entry point
# and the helpers `calculate_per`, `extract_punctuation`, `split_tokens`, `_expand_contractions`,
# `_normalize_digits_to_words` and `_remove_non_speech_elements` are removed, and whitespace and compound-word
# helpers moved to `_scoring/text.py`. Added: mixed error rate (`evaluate_mer`, `mixed_segment`), multi-speaker
# cpWER (`evaluate_msasr`, speaker-tag parsing and normalization; the counts come from `metrics/cpwer_utils.py` and
# the `chime8` normalization from `_scoring/chime8_normalizer.py`), punctuation error counts, CER whitespace modes,
# `register_normalization_mode` and `register_task_type` (a benchmark's scorer module adds its own), `coerce_text_field`,
# the `lower_nopunct` normalization and the per-row edit counts that corpus aggregation sums. `evaluate_asr`,
# `evaluate_asr_pc`, `evaluate_cer`, `evaluate_sample`, the multilingual normalizer and `AudioEvaluator` (which now
# validates its settings when built) are changed accordingly.

"""Audio evaluation framework supporting ASR, ASR-PC, Translation, CER, and more."""

import logging
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from resources_servers.audio_eval._scoring.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from resources_servers.audio_eval._scoring.text import (
    HAN_RANGES,
    HAN_RE,
    VALID_CER_WHITESPACE_MODES,
    normalize_cer_whitespace,
    normalize_compound_pairs,
    normalize_whitespace,
)


LOG = logging.getLogger(__name__)


@dataclass(kw_only=True)
class AudioEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for audio evaluation."""

    normalize_asr_pc_standard_wer: bool = True
    strip_helpful_prefixes: bool = False
    apply_normalization: bool = True
    # See VALID_NORMALIZATION_MODES for accepted values.
    normalization_mode: str = "standard"

    # Optional list of reference fields to calculate WER against (e.g., ["text_tn", "text_itn"])
    # For each field, WER will be computed and stored with corresponding metric name
    reference_fields: list[str] | None = None

    # Multispeaker ASR (cpWER) settings
    speaker_token: str = "<spk:*>"
    cpwer_normalization: str = "chime8"


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
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


def remove_symbols_and_diacritics(s: str, keep: str = ""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


def remove_symbols_keep_marks(s: str):
    """
    Replace symbols and punctuation (categories S*, P*) with a space, but KEEP combining
    marks (M*) and letters/numbers. For abugida scripts (Thai/Lao/Myanmar/Khmer/Indic),
    the vowel and tone marks are meaning-bearing, unlike Latin diacritics, so they must
    not be stripped. Uses NFC to keep precomposed forms intact.
    """
    return "".join(" " if unicodedata.category(c)[0] in "SP" else c for c in unicodedata.normalize("NFC", s))


class MultilingualTextNormalizer:
    """Multilingual text normalizer with optional number normalization.

    Call with just text for standard normalization.
    Pass lang= to also convert digits to words via num2words.
    """

    def __init__(self, remove_diacritics: bool = True, preserve_marks: bool = False):
        # preserve_marks keeps combining marks (M*) — essential vowels/tones in abugida
        # scripts — while still removing symbols/punctuation. Overrides remove_diacritics.
        self.preserve_marks = preserve_marks
        if preserve_marks:
            self.clean = remove_symbols_keep_marks
        else:
            self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

    def _normalize_numbers(self, text, lang):
        import num2words

        # Join space-separated thousand groups (e.g. "10 000" -> "10000")
        text = re.sub(r"(\d)\s+(\d{3})\b", r"\1\2", text)

        # Convert remaining digit sequences to words
        def _replace(m):
            try:
                return num2words.num2words(int(m.group()), lang=lang)
            except Exception:
                return m.group()

        return re.sub(r"\d+", _replace, text)

    def __call__(self, s: str, lang=None):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        # Remove punctuations and extra spaces. Skip the [^\w\s] strip when preserving
        # marks (it would delete combining marks, which \w does not match); self.clean
        # has already removed symbols/punctuation in that case.
        if not self.preserve_marks:
            s = re.sub(r"[^\w\s]", "", s)
        s = normalize_whitespace(s)

        if lang is not None:
            s = self._normalize_numbers(s, lang)
        return s


# Known model failure responses that should be treated as empty transcriptions
_FAILURE_RESPONSES = [
    r"the speech is in audio format and needs to be transcribed",
    r"i do not have access to audio",
    r"i cannot access audio",
    r"i'm sorry.*i do not have access",
    r"as an ai language model.*i do not have access",
]


def extract_asr_text(generation: str) -> str:
    """Extract ASR text from generation."""

    def parse_qwen_asr_output(generation: str) -> str:
        _ASR_TEXT_TAG = "<asr_text>"
        s = str(generation).strip()
        has_tag = _ASR_TEXT_TAG in s
        if has_tag:
            s = s.split(_ASR_TEXT_TAG, 1)[1]
        return s.strip()

    result = parse_qwen_asr_output(generation)
    return result.strip()


def strip_helpful_prefixes(text: str) -> str:
    """Strip ASR response prefixes like 'The audio says: ...' for accurate WER.

    Also removes SRT subtitle timestamps that can appear in vLLM chunked audio generation.
    """
    result = text.strip()

    # Check for model failure responses
    for failure_pattern in _FAILURE_RESPONSES:
        if re.search(failure_pattern, result, flags=re.IGNORECASE):
            return ""

    # Remove SRT subtitle timestamps (vLLM chunked audio artifact)
    result = re.sub(r"\d+\s+\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s+", "", result)
    result = re.sub(r"\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s*", "", result)
    result = re.sub(r"\\n\d+\s+(?=\d{2}:\d{2})", " ", result)
    result = re.sub(r"\n\d+\s+(?=\d{2}:\d{2})", " ", result)

    # Extract from double quotes
    match = re.search(r'"((?:\\.|[^"\\])*)"', result)
    if match:
        result = match.group(1)

    # Handle colon-quote patterns (e.g., "The audio says: 'hello'")
    if ": '" in result:
        result = result.split(": '", 1)[1]
        # Remove trailing quote (with punctuation either inside or outside)
        result = re.sub(r"[.!?]?'[.!?]?\s*$", "", result)
    elif ":'" in result:
        result = result.split(":'", 1)[1]
        result = re.sub(r"[.!?]?'[.!?]?\s*$", "", result)
    else:
        # Single quote extraction - only match quotes at end of string (not contractions like o'clock)
        match = re.search(r"'(.+?)'\s*\.?\s*$", result)
        if match:
            result = match.group(1)

    return result.strip()


_HAN_RANGES = HAN_RANGES
_HAN_RE = HAN_RE

# Scriptio-continua scripts (no inter-word spaces) whose units are tokenized
# individually for Mixed Error Rate, while space-delimited scripts (e.g. Latin)
# keep their whitespace-delimited words. Covers every CER-style script that
# occurs in CS-FLEURS plus the obvious siblings (Lao/Khmer).
_CONTINUA_RANGES = (
    _HAN_RANGES + "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "ㇰ-ㇿ"  # Katakana Phonetic Extensions
    "ᄀ-ᇿ"  # Hangul Jamo
    "㄰-㆏"  # Hangul Compatibility Jamo
    "가-힣"  # Hangul Syllables
    "฀-๿"  # Thai
    "຀-໿"  # Lao
    "က-႟"  # Myanmar
    "ꩠ-ꩿ"  # Myanmar Extended-A
    "ក-៿"  # Khmer
)
_CONTINUA_RE = re.compile(f"[{_CONTINUA_RANGES}]")


def _grapheme_clusters(text: str) -> list[str]:
    """Group each base character with its trailing combining marks.

    A lightweight, dependency-free grapheme segmentation: combining marks
    (Unicode categories Mn/Mc/Me) attach to the preceding base. This keeps a
    Thai/Myanmar consonant together with its vowel signs and tone marks, which
    naive per-codepoint splitting would tear apart.
    """
    clusters: list[str] = []
    for ch in text:
        if clusters and unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def mixed_segment(text: str, grapheme: bool = False) -> list[str]:
    """Tokenize code-switched text for Mixed Error Rate (MER).

    Each scriptio-continua unit (Han/kana/Hangul/Thai/Lao/Myanmar/Khmer) becomes
    its own token; runs of space-delimited script (e.g. Latin) stay as their
    whitespace-delimited words. The result, joined with spaces, can be scored
    with the standard word-level edit distance so Mandarin is counted by
    character and English by word in the same utterance.

    Args:
        text: Normalized transcription (normalize first, then segment).
        grapheme: When True, scriptio-continua units are grapheme clusters
            (base + combining marks) rather than single codepoints. Needed for
            Thai/Myanmar to avoid splitting tone/vowel marks off their base;
            unnecessary for Han/kana/Hangul (no combining marks). Default False
            keeps per-codepoint counting, consistent with the evaluator's CER.
    """
    units = _grapheme_clusters(text) if grapheme else list(text)
    tokens: list[str] = []
    buffer: list[str] = []

    def _flush():
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for unit in units:
        if unit[0].isspace():
            _flush()
        elif _CONTINUA_RE.match(unit[0]):
            _flush()
            tokens.append(unit)
        else:
            buffer.append(unit)
    _flush()
    return tokens


ASR_PC_PUNCTUATION = (".", ",", "?")
_ASR_PC_TOKEN_RE = re.compile(r"[\w']+|[.,?]")


def _asr_pc_tokens(text: str, *, punctuation: bool, lowercase: bool) -> list[str]:
    """Tokenize the frozen LibriSpeech-PC protocol without text/number normalization."""
    tokens = _ASR_PC_TOKEN_RE.findall(normalize_whitespace(text))
    if not punctuation:
        tokens = [token for token in tokens if token not in ASR_PC_PUNCTUATION]
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def punctuation_error_counts(reference: str, hypothesis: str) -> dict[str, int | float]:
    """Return NeMo's official masked-position PER counts for ``[.,?]``."""
    ref_tokens = _asr_pc_tokens(reference, punctuation=True, lowercase=False)
    hyp_tokens = _asr_pc_tokens(hypothesis, punctuation=True, lowercase=False)
    punctuation = set(ASR_PC_PUNCTUATION)
    mask = "[PUNCT]"
    ref_masked = [mask if token in punctuation else token for token in ref_tokens]
    hyp_masked = [mask if token in punctuation else token for token in hyp_tokens]

    ref_len = len(ref_masked)
    hyp_len = len(hyp_masked)
    costs = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    backtrace = [[""] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for ref_index in range(1, ref_len + 1):
        costs[ref_index][0] = ref_index
        backtrace[ref_index][0] = "D"
    for hyp_index in range(1, hyp_len + 1):
        costs[0][hyp_index] = hyp_index
        backtrace[0][hyp_index] = "I"

    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            if ref_masked[ref_index - 1] == hyp_masked[hyp_index - 1]:
                costs[ref_index][hyp_index] = costs[ref_index - 1][hyp_index - 1]
                backtrace[ref_index][hyp_index] = "C"
                continue
            substitution = costs[ref_index - 1][hyp_index - 1] + 1
            insertion = costs[ref_index][hyp_index - 1] + 1
            deletion = costs[ref_index - 1][hyp_index] + 1
            costs[ref_index][hyp_index] = min(substitution, insertion, deletion)
            if costs[ref_index][hyp_index] == substitution:
                backtrace[ref_index][hyp_index] = "S"
            elif costs[ref_index][hyp_index] == insertion:
                backtrace[ref_index][hyp_index] = "I"
            else:
                backtrace[ref_index][hyp_index] = "D"

    correct = 0
    substitutions = 0
    ref_index = ref_len
    hyp_index = hyp_len
    while ref_index > 0 or hyp_index > 0:
        operation = backtrace[ref_index][hyp_index]
        if operation == "C":
            ref_token = ref_tokens[ref_index - 1]
            hyp_token = hyp_tokens[hyp_index - 1]
            if ref_token in punctuation or hyp_token in punctuation:
                if ref_token == hyp_token:
                    correct += 1
                else:
                    substitutions += 1
            ref_index -= 1
            hyp_index -= 1
        elif operation == "S":
            ref_index -= 1
            hyp_index -= 1
        elif operation == "I":
            hyp_index -= 1
        elif operation == "D":
            ref_index -= 1
        else:
            raise AssertionError(f"Missing PER backtrace at ({ref_index}, {hyp_index})")

    ref_punctuation = sum(token in punctuation for token in ref_tokens)
    hyp_punctuation = sum(token in punctuation for token in hyp_tokens)
    deletions = ref_punctuation - correct - substitutions
    insertions = hyp_punctuation - correct - substitutions
    operations = correct + substitutions + deletions + insertions
    errors = substitutions + deletions + insertions
    return {
        "per": errors / operations if operations else 0.0,
        "per_correct": correct,
        "per_substitutions": substitutions,
        "per_deletions": deletions,
        "per_insertions": insertions,
        "per_errors": errors,
        "per_operations": operations,
    }


def _prefixed_wer_counts(reference: str, hypothesis: str, prefix: str) -> dict[str, int | float]:
    counts = _wer_with_counts(reference, hypothesis)
    return {
        prefix: counts["wer"],
        f"{prefix}_errors": counts["wer_errors"],
        f"{prefix}_ref_words": counts["wer_ref_words"],
        f"{prefix}_substitutions": counts["wer_substitutions"],
        f"{prefix}_insertions": counts["wer_insertions"],
        f"{prefix}_deletions": counts["wer_deletions"],
    }


def evaluate_asr_pc(
    reference: str, hypothesis: str, normalize_standard_wer: bool = True, normalization_mode: str = "standard"
) -> dict[str, Any]:
    """Evaluate the frozen LibriSpeech-PC WER/WER-C/WER-PC/PER protocol.

    The compatibility arguments remain accepted, but ASR-PC intentionally
    never uses Whisper TN/ITN.  It applies only whitespace normalization,
    case folding for WER, and the target punctuation set ``[.,?]``.
    """
    del normalize_standard_wer, normalization_mode
    ref_std = " ".join(_asr_pc_tokens(reference, punctuation=False, lowercase=True))
    hyp_std = " ".join(_asr_pc_tokens(hypothesis, punctuation=False, lowercase=True))
    ref_c = " ".join(_asr_pc_tokens(reference, punctuation=False, lowercase=False))
    hyp_c = " ".join(_asr_pc_tokens(hypothesis, punctuation=False, lowercase=False))
    ref_pc = " ".join(_asr_pc_tokens(reference, punctuation=True, lowercase=False))
    hyp_pc = " ".join(_asr_pc_tokens(hypothesis, punctuation=True, lowercase=False))

    result = _wer_with_counts(ref_std, hyp_std)
    result.update(_prefixed_wer_counts(ref_c, hyp_c, "wer_c"))
    result.update(_prefixed_wer_counts(ref_pc, hyp_pc, "wer_pc"))
    result.update(punctuation_error_counts(reference, hypothesis))
    result["is_correct"] = result["wer_pc_errors"] == 0
    result["text"] = ref_std
    result["pred_text"] = hyp_std
    return result


VALID_NORMALIZATION_MODES = (
    "standard",
    "none",
    "no_tn_itn",
    "multilingual",
    "lower_nopunct",
)
# Normalization modes and task types of a single benchmark are not part of this module: the benchmark's evaluator
# module registers them with ``register_normalization_mode`` / ``register_task_type`` when it is imported.
_EXTRA_NORMALIZATION_MODES: dict[str, Callable[..., str]] = {}
_EXTRA_TASK_TYPES: dict[str, Callable[[dict[str, Any], str, "AudioEvaluatorConfig"], dict[str, Any]]] = {}


def register_normalization_mode(name: str, normalizer: Callable[..., str]) -> None:
    """Add a ``normalization_mode``. ``normalizer(text, **kwargs)`` returns the text before whitespace collapsing."""
    if name in VALID_NORMALIZATION_MODES or name in _EXTRA_NORMALIZATION_MODES:
        raise ValueError(f"normalization_mode {name!r} is already defined")
    _EXTRA_NORMALIZATION_MODES[name] = normalizer


def register_task_type(
    name: str, handler: Callable[[dict[str, Any], str, "AudioEvaluatorConfig"], dict[str, Any]]
) -> None:
    """Add a ``task_type``. ``handler(sample, generation, config)`` returns the updates to merge into the row."""
    if name in _EXTRA_TASK_TYPES:
        raise ValueError(f"task_type {name!r} is already defined")
    _EXTRA_TASK_TYPES[name] = handler


def valid_normalization_modes() -> tuple[str, ...]:
    return VALID_NORMALIZATION_MODES + tuple(_EXTRA_NORMALIZATION_MODES)


VALID_CPWER_NORMALIZATION_MODES = ("chime8", "none")


def resolve_cer_whitespace_mode(override: bool | str | None = None) -> str:
    """Resolve a validated CER whitespace mode.

    A string override accepts ``keep``, ``all``, or ``han_adjacent``.
    A Boolean override is the ``cer_strip_whitespace`` metadata flag: ``True`` means ``all`` and ``False``
    means ``keep``. Values such as the string ``"false"`` are rejected rather
    than silently evaluating as truthy.
    """
    if override is None:
        return "keep"
    if isinstance(override, bool):
        return "all" if override else "keep"
    if not isinstance(override, str) or override not in VALID_CER_WHITESPACE_MODES:
        raise ValueError(
            f"Invalid CER whitespace override {override!r}; expected a boolean or one of "
            f"{sorted(VALID_CER_WHITESPACE_MODES)}"
        )
    return override


def resolve_sample_cer_whitespace_mode(extra_fields: dict[str, Any]) -> str:
    """Resolve CER whitespace behavior from one dataset record.

    Datasets should set ``cer_whitespace_mode``. The Boolean
    ``cer_strip_whitespace`` is also accepted, but specifying both is an error.
    CS-FLEURS keeps its paper-comparable all-whitespace CER by default; an
    explicit record override always wins.
    """
    has_mode = "cer_whitespace_mode" in extra_fields
    has_legacy = "cer_strip_whitespace" in extra_fields
    if has_mode and has_legacy:
        raise ValueError("Set only one of cer_whitespace_mode and cer_strip_whitespace")
    if has_mode:
        mode = extra_fields["cer_whitespace_mode"]
        if not isinstance(mode, str):
            raise ValueError(f"cer_whitespace_mode must be one of {sorted(VALID_CER_WHITESPACE_MODES)}")
        return resolve_cer_whitespace_mode(mode)
    if has_legacy:
        strip_whitespace = extra_fields["cer_strip_whitespace"]
        if not isinstance(strip_whitespace, bool):
            raise ValueError("cer_strip_whitespace must be a boolean")
        return resolve_cer_whitespace_mode(strip_whitespace)
    # Code-switched MER records use paper-comparable all-whitespace CER in
    # parallel with their boundary-preserving headline MER.
    if extra_fields.get("use_mer", False):
        return "all"
    return "keep"


TASKS_NEED_PARSING = {
    "ASR",
    "ASR-PC",
    "ASR_LEADERBOARD",
    "Multilingual-ASR",
    "MS-ASR",
    "CER",
    "Hallucination",
    "PC-Rate",
}

SOT_STRICT_TAG_RES = {
    "spk": re.compile(r"<spk:(?P<spk>\d+)>", re.IGNORECASE),
    "s_bracket": re.compile(r"\[(?P<bracket>s\d+)\]", re.IGNORECASE),
    "both": re.compile(r"\[(?P<bracket>s\d+)\]|<spk:(?P<spk>\d+)>", re.IGNORECASE),
}
SOT_CANONICAL_TAG_RE = re.compile(
    r"\[(?P<bracket>s\d+)\]|<spk:(?P<spk>\d+)>|(?:^|\s)speaker[_\s-]*(?P<speaker_line>\d+)\s*[:\-]",
    re.IGNORECASE,
)


def _append_msasr_segment(segments: list[dict[str, str]], speaker: str, text: str) -> None:
    text = normalize_whitespace(text)
    if text:
        segments.append({"speaker": speaker, "text": text})


def _speaker_from_sot_match(match: re.Match) -> str:
    groups = match.groupdict()
    if groups.get("bracket") is not None:
        return groups["bracket"].lower()
    if groups.get("spk") is not None:
        return f"s{int(groups['spk'])}"
    return f"s{int(groups['speaker_line'])}"


def _parse_sot_segments(
    text: str,
    *,
    default_speaker: str = "s0",
    mode: str = "canonical",
    strict_tag_style: str = "spk",
) -> list[dict[str, str]]:
    if mode not in {"strict", "canonical"}:
        raise ValueError(f"Unsupported SOT parse mode: {mode}")
    if strict_tag_style not in SOT_STRICT_TAG_RES:
        raise ValueError(
            f"Unsupported SOT strict tag style: {strict_tag_style}. Expected one of {sorted(SOT_STRICT_TAG_RES)}"
        )

    cleaned = normalize_whitespace(text or "")
    tag_re = SOT_STRICT_TAG_RES[strict_tag_style] if mode == "strict" else SOT_CANONICAL_TAG_RE
    matches = list(tag_re.finditer(cleaned))
    if not matches:
        if mode == "canonical" and cleaned:
            return [{"speaker": default_speaker, "text": cleaned}]
        return []

    segments: list[dict[str, str]] = []
    current_speaker: str | None = None
    cursor = 0
    for match in matches:
        chunk = cleaned[cursor : match.start()]
        if chunk:
            if current_speaker is None:
                if mode == "canonical":
                    _append_msasr_segment(segments, default_speaker, chunk)
            else:
                _append_msasr_segment(segments, current_speaker, chunk)

        current_speaker = _speaker_from_sot_match(match)
        cursor = match.end()

    tail = cleaned[cursor:]
    if tail:
        _append_msasr_segment(segments, current_speaker or default_speaker, tail)
    return segments


def parse_sot_speaker_streams(
    text: str,
    *,
    default_speaker: str = "s0",
    mode: str = "canonical",
    strict_tag_style: str = "spk",
) -> OrderedDict[str, str]:
    """Parse a serialized-overlap transcript into speaker-wise text streams."""
    streams: OrderedDict[str, list[str]] = OrderedDict()
    for segment in _parse_sot_segments(
        text,
        default_speaker=default_speaker,
        mode=mode,
        strict_tag_style=strict_tag_style,
    ):
        streams.setdefault(segment["speaker"], []).append(segment["text"])
    return OrderedDict((speaker, normalize_whitespace(" ".join(chunks))) for speaker, chunks in streams.items())


def resolve_speaker_tag_style(speaker_token: str) -> str:
    """Map configured speaker token pattern to SOT strict tag style."""
    token = (speaker_token or "<spk:*>").strip().lower()
    if token in {"<spk:*>", "<spk:n>", "spk", "spk_tag"}:
        return "spk"
    if token in {"[s*]", "[sn]", "s_bracket", "[s0]"}:
        return "s_bracket"
    if token in {"both", "all"}:
        return "both"
    raise ValueError(f"Unsupported speaker_token '{speaker_token}'. Expected '<spk:*>' (default), '[s*]', or 'both'.")


def normalize_cpwer_stream(text: str, mode: str = "chime8") -> str:
    """Normalize one speaker stream for cpWER scoring."""
    if mode not in VALID_CPWER_NORMALIZATION_MODES:
        raise ValueError(
            f"Invalid cpwer_normalization '{mode}'. Available options: {', '.join(VALID_CPWER_NORMALIZATION_MODES)}"
        )
    if mode == "none":
        return normalize_whitespace(text or "")
    from resources_servers.audio_eval._scoring.chime8_normalizer import normalize as normalize_chime8

    return normalize_chime8(text or "")


def evaluate_msasr(
    reference: str,
    hypothesis: str,
    *,
    speaker_token: str = "<spk:*>",
    cpwer_normalization: str = "chime8",
    hypothesis_mode: str = "canonical",
) -> dict[str, Any]:
    """Evaluate multispeaker ASR with corpus-level cpWER building blocks."""
    from resources_servers.audio_eval._scoring.evaluation.metrics.cpwer_utils import session_cpwer_counts

    tag_style = resolve_speaker_tag_style(speaker_token)
    ref_streams = parse_sot_speaker_streams(reference, mode="strict", strict_tag_style=tag_style)
    hyp_streams = parse_sot_speaker_streams(
        hypothesis,
        mode=hypothesis_mode,
        strict_tag_style=tag_style,
    )

    normalized_ref = {
        speaker: normalize_cpwer_stream(text, mode=cpwer_normalization) for speaker, text in ref_streams.items()
    }
    normalized_hyp = {
        speaker: normalize_cpwer_stream(text, mode=cpwer_normalization) for speaker, text in hyp_streams.items()
    }

    if not normalized_ref:
        return {
            "cpwer": None,
            "is_correct": None,
            "text": "",
            "pred_text": " ".join(normalized_hyp.values()).strip(),
        }

    counts = session_cpwer_counts(normalized_ref, normalized_hyp)
    ref_text = " ".join(normalized_ref.values()).strip()
    hyp_text = " ".join(normalized_hyp.values()).strip()
    return {
        "cpwer": counts["cpwer"],
        "cpwer_errors": counts["errors"],
        "cpwer_substitutions": counts["substitutions"],
        "cpwer_insertions": counts["insertions"],
        "cpwer_deletions": counts["deletions"],
        "cpwer_ref_words": counts["ref_words"],
        "speaker_assignment": counts["assignment"],
        "reference_streams": dict(normalized_ref),
        "hypothesis_streams": dict(normalized_hyp),
        "is_correct": counts["errors"] == 0,
        "text": ref_text,
        "pred_text": hyp_text,
    }


def resolve_asr_normalization_mode(config: AudioEvaluatorConfig) -> str:
    """Resolve effective normalization mode for ASR-family tasks.

    - no_tn_itn is explicit and does not use whisper normalization.
    - Other modes respect apply_normalization toggle.
    """
    if config.normalization_mode == "no_tn_itn":
        return "no_tn_itn"
    return config.normalization_mode if config.apply_normalization else "none"


def preprocess_asr_text(text: str, mode: str = "standard", **kwargs) -> str:
    """Normalize ASR text for WER calculation.

    Args:
        text: Raw text.
        mode: Normalization mode:
            - "standard": Whisper normalization (default) - converts number words to digits
            - "none": No normalization (whitespace only)
            - "no_tn_itn": Lowercase + remove punctuation, no number word conversion (for TN/ITN eval)
            - "multilingual": Multilingual normalization
            - "lower_nopunct": Lowercase + remove punctuation while preserving combining marks
            Modes registered with ``register_normalization_mode`` are accepted too.
        **kwargs: Additional keyword arguments.
    """
    if mode not in VALID_NORMALIZATION_MODES and mode not in _EXTRA_NORMALIZATION_MODES:
        raise ValueError(
            f"Invalid normalization_mode '{mode}'. Available options: {', '.join(valid_normalization_modes())}"
        )

    if mode == "none":
        return normalize_whitespace(text)

    if mode == "no_tn_itn":
        # Lowercase + remove punctuation + whitespace normalization
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return normalize_whitespace(text)

    if mode == "lower_nopunct":
        # Lowercase + drop punctuation only, PRESERVING combining marks (Mn/Mc).
        # Unlike no_tn_itn (whose [^\w\s] also strips abugida vowel/tone marks,
        # since \w does not match them), this keeps Thai/Myanmar/Indic vowels and
        # tones. Matches the CS-FLEURS paper's "case insensitive and unpunctuated"
        # character error rate, so per-language CER is comparable to the paper.
        text = text.lower()
        text = "".join("" if unicodedata.category(ch).startswith("P") else ch for ch in text)
        return normalize_whitespace(text)

    from whisper_normalizer.english import EnglishTextNormalizer

    if mode == "standard":
        text = text.lower()
        text = EnglishTextNormalizer()(text)
        return normalize_whitespace(text)

    if mode in _EXTRA_NORMALIZATION_MODES:
        return normalize_whitespace(_EXTRA_NORMALIZATION_MODES[mode](text, **kwargs))

    # "multilingual" uses multilingual normalization for non-English languages
    # and whisper normalization for English
    if mode == "multilingual":
        text = text.lower()
        lang = kwargs["lang"]
        if lang in [None, "en"]:
            text = EnglishTextNormalizer()(text)
        else:
            text = MultilingualTextNormalizer(
                remove_diacritics=kwargs.get("remove_diacritics", False),
                preserve_marks=kwargs.get("preserve_marks", False),
            )(text, lang=lang)
        return normalize_whitespace(text)


def _wer_with_counts(ref: str, hyp: str, merge_compounds: bool = False) -> dict[str, Any]:
    """Compute WER and raw counts, optionally using Open ASR compound merging."""
    if merge_compounds:
        try:
            from kaldialign import batch_error_rate

            measures = batch_error_rate(
                [tuple(ref.split())],
                [tuple(hyp.split())],
                merge_compounds=True,
            )
        except ImportError:
            from kaldialign import edit_distance

            ref_words = ref.split()
            ref, hyp = normalize_compound_pairs(ref, hyp)
            measures = edit_distance(tuple(ref.split()), tuple(hyp.split()))
            measures["ref_len"] = len(ref_words)
            measures["err_rate"] = measures["total"] / measures["ref_len"] if measures["ref_len"] else 0.0
        return {
            "wer": measures["err_rate"],
            "wer_errors": measures["total"],
            "wer_ref_words": measures["ref_len"],
            "wer_substitutions": measures["sub"],
            "wer_insertions": measures["ins"],
            "wer_deletions": measures["del"],
        }

    import jiwer

    wer_score = jiwer.wer(ref, hyp)
    measures = jiwer.process_words(ref, hyp)
    wer_errors = measures.substitutions + measures.deletions + measures.insertions
    wer_ref_words = measures.substitutions + measures.deletions + measures.hits

    return {
        "wer": wer_score,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "wer_substitutions": measures.substitutions,
        "wer_insertions": measures.insertions,
        "wer_deletions": measures.deletions,
    }


def _cer_with_counts(ref: str, hyp: str, key_prefix: str = "cer") -> dict[str, Any]:
    """Compute CER and return both the score and raw error/reference counts for corpus-level aggregation."""
    import jiwer

    assert key_prefix in ["cer", "wer"], "key_prefix must be 'cer' or 'wer'"

    cer_score = jiwer.cer(ref, hyp)
    measures = jiwer.process_characters(ref, hyp)
    cer_errors = measures.substitutions + measures.deletions + measures.insertions
    cer_ref_chars = measures.substitutions + measures.deletions + measures.hits

    return {
        f"{key_prefix}": cer_score,
        f"{key_prefix}_errors": cer_errors,
        f"{key_prefix}_ref_words": cer_ref_chars,
        f"{key_prefix}_substitutions": measures.substitutions,
        f"{key_prefix}_insertions": measures.insertions,
        f"{key_prefix}_deletions": measures.deletions,
    }


def evaluate_asr(
    reference: str, hypothesis: str, normalization_mode: str = "standard", normalize_compound: bool = False, **kwargs
) -> dict[str, Any]:
    """Evaluate ASR: computes WER with normalization.

    Args:
        reference: Ground truth transcription.
        hypothesis: Model output transcription.
        normalization_mode: "standard", "multilingual", "lower_nopunct", "none", or "no_tn_itn".
        normalize_compound: Whether to normalize compound pairs.
        **kwargs: Additional keyword arguments.
    """
    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    # Match the HF Open ASR Leaderboard: drop samples whose normalized
    # reference is empty rather than scoring them against a placeholder.
    if not ref:
        result = {"wer": None, "is_correct": None, "text": "", "pred_text": hyp or ""}
        return result

    merge_compounds = kwargs.get("merge_compounds", False)
    if not hyp and not merge_compounds:
        hyp = "empty"
    result = _wer_with_counts(ref, hyp, merge_compounds=merge_compounds)
    result["is_correct"] = result["wer"] < 0.5
    result["text"] = ref
    result["pred_text"] = hyp
    return result


_BLEU_TOKENIZE_BY_LANG = {
    "ja": "ja-mecab",
    "zh": "zh",
    "cmn": "zh",
    "yue": "zh",
    "ko": "ko-mecab",
}


def resolve_bleu_tokenize(tgt_lang: str | None) -> str:
    """Resolve sacrebleu tokenize from a target language code."""
    if not isinstance(tgt_lang, str):
        return "13a"
    lang_code = re.split(r"[_-]", tgt_lang)[0]  # "zh-CN" (CoVoST2) and "cmn_hans_cn" (FLEURS) both resolve
    return _BLEU_TOKENIZE_BY_LANG.get(lang_code, "13a")


def evaluate_translation(
    reference: str,
    hypothesis: str,
    tgt_lang: str | None = None,
) -> dict[str, Any]:
    """Evaluate translation: computes sentence-level BLEU score."""
    tokenize = resolve_bleu_tokenize(tgt_lang)
    try:
        import sacrebleu

        text = reference.strip()
        pred_text = hypothesis.strip()
        bleu = sacrebleu.sentence_bleu(pred_text, [text], tokenize=tokenize)
        bleu_score = bleu.score / 100.0

        return {
            "bleu": bleu_score,
            "is_correct": bleu_score > 0.3,
            "text": text,
            "pred_text": pred_text,
            "bleu_tokenize": tokenize,
        }
    except Exception as e:
        return {
            "bleu": 0.0,
            "is_correct": False,
            "error": str(e),
            "text": reference.strip(),
            "pred_text": hypothesis.strip(),
            "bleu_tokenize": tokenize,
        }


def evaluate_cer(
    reference: str,
    hypothesis: str,
    normalization_mode: str = "none",
    key_prefix: str = "cer",
    normalize_compound: bool = False,
    strip_whitespace: bool | None = None,
    whitespace_mode: str = "keep",
    **kwargs,
) -> dict[str, Any]:
    """Evaluate CER: character-level edit distance.

    ``whitespace_mode`` controls how spaces are scored: ``keep`` preserves them,
    ``all`` removes them globally, and ``han_adjacent`` removes only spaces
    adjacent to Han characters. ``strip_whitespace`` is the
    Boolean spelling for ``all``/``keep``.
    """

    # Validate the named mode before interpreting the ``strip_whitespace`` Boolean so an
    # invalid mode is never hidden behind a misleading conflict error.
    normalize_cer_whitespace("", mode=whitespace_mode)
    if strip_whitespace is not None:
        if not isinstance(strip_whitespace, bool):
            raise ValueError("strip_whitespace must be a boolean when provided")
        if whitespace_mode != "keep":
            raise ValueError("Set only one of whitespace_mode and strip_whitespace")
        whitespace_mode = "all" if strip_whitespace else "keep"
    if normalize_compound and whitespace_mode == "han_adjacent":
        raise ValueError("normalize_compound cannot be combined with han_adjacent CER whitespace mode")

    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    ref = normalize_cer_whitespace(ref, mode=whitespace_mode)
    hyp = normalize_cer_whitespace(hyp, mode=whitespace_mode)

    # Mirror evaluate_asr: drop samples whose normalized reference is empty rather
    # than dividing by zero in the character edit distance.
    if not ref:
        return {key_prefix: None, "is_correct": None, "text": "", "pred_text": hyp or ""}

    result = _cer_with_counts(ref, hyp, key_prefix=key_prefix)
    result["is_correct"] = result[key_prefix] < 0.5
    result["text"] = ref
    result["pred_text"] = hyp
    return result


def evaluate_mer(
    reference: str,
    hypothesis: str,
    normalization_mode: str = "none",
    normalize_compound: bool = False,
    grapheme: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Evaluate Mixed Error Rate for code-switched ASR.

    Scriptio-continua scripts are counted by character and space-delimited
    scripts by word within the same utterance (see ``mixed_segment``), then
    scored with the standard word-level edit distance. Results are reported
    under the ``wer*`` keys so the mixed figure occupies the headline WER column
    (matching how ``Multilingual-ASR`` folds CER into ``wer*``).
    """
    ref = preprocess_asr_text(reference, mode=normalization_mode, **kwargs)
    hyp = preprocess_asr_text(hypothesis, mode=normalization_mode, **kwargs)

    if normalize_compound:
        ref, hyp = normalize_compound_pairs(ref, hyp)

    # Mirror evaluate_asr/evaluate_cer: drop samples whose normalized reference
    # is empty rather than scoring against a placeholder.
    if not ref:
        return {"wer": None, "is_correct": None, "text": "", "pred_text": hyp or ""}

    if not hyp:
        hyp = "empty"

    ref_seg = " ".join(mixed_segment(ref, grapheme=grapheme))
    hyp_seg = " ".join(mixed_segment(hyp, grapheme=grapheme))

    result = _wer_with_counts(ref_seg, hyp_seg)
    result["is_correct"] = result["wer"] < 0.5
    # Store the human-readable (unsegmented) normalized text, not the spaced tokens.
    result["text"] = ref
    result["pred_text"] = hyp
    return result


def evaluate_hallucination(reference: str, hypothesis: str, audio_context: dict = None) -> dict[str, Any]:
    """Detect potential hallucinations via speaking rate anomaly.

    Normal speech: ~600-900 chars/minute. Higher rates suggest repetition/hallucination.
    Requires audio_duration in audio_context.
    """
    audio_duration = audio_context.get("audio_duration") if audio_context else None

    if not audio_duration or audio_duration <= 0:
        return {
            "hallucination_rate": 0.0,
            "char_rate": 0.0,
            "is_correct": True,
            "error": "missing_audio_duration",
            "text": reference,
            "pred_text": hypothesis,
        }

    char_count = len(hypothesis)
    # Convert to chars/minute
    char_rate = (char_count / audio_duration) * 60.0

    # Hallucination threshold: >1500 chars/min (25 chars/second * 60)
    is_hallucinating = char_rate > 1500.0

    return {
        "hallucination_rate": 1.0 if is_hallucinating else 0.0,
        "char_rate": round(char_rate, 2),
        "is_correct": not is_hallucinating,
        "text": reference,
        "pred_text": hypothesis,
    }


def evaluate_pc_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Evaluate detailed Punctuation and Capitalization metrics."""
    # Extract punctuation with positions
    ref_puncts = [(m.group(), m.start()) for m in re.finditer(r"[.,!?;:\-]", reference)]
    hyp_puncts = [(m.group(), m.start()) for m in re.finditer(r"[.,!?;:\-]", hypothesis)]

    # Punctuation matching (within 2 char tolerance)
    matched = 0
    for ref_p, ref_pos in ref_puncts:
        for hyp_p, hyp_pos in hyp_puncts:
            if ref_p == hyp_p and abs(ref_pos - hyp_pos) <= 2:
                matched += 1
                break

    punct_precision = matched / len(hyp_puncts) if hyp_puncts else 0.0
    punct_recall = matched / len(ref_puncts) if ref_puncts else 0.0
    punct_f1 = (
        2 * punct_precision * punct_recall / (punct_precision + punct_recall)
        if (punct_precision + punct_recall) > 0
        else 0.0
    )

    # Capitalization: check sentence starts and word capitals
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) != len(hyp_words):
        cap_accuracy = 0.0
    else:
        cap_matches = sum(
            1 for r, h in zip(ref_words, hyp_words, strict=True) if r and h and r[0].isupper() == h[0].isupper()
        )
        cap_accuracy = cap_matches / len(ref_words) if ref_words else 0.0

    # Overall PC rate (average of punct F1 and cap accuracy)
    pc_rate = (punct_f1 + cap_accuracy) / 2.0

    return {
        "pc_rate": round(pc_rate, 3),
        "punct_precision": round(punct_precision, 3),
        "punct_recall": round(punct_recall, 3),
        "punct_f1": round(punct_f1, 3),
        "cap_accuracy": round(cap_accuracy, 3),
        "is_correct": pc_rate > 0.5,
        "text": reference,
        "pred_text": hypothesis,
    }


class AudioEvaluator(BaseEvaluator):
    """Audio evaluator supporting ASR, ASR-PC, Translation, CER, etc."""

    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = AudioEvaluatorConfig(**self.config)
        # Fail when the evaluator is built (at server startup), not per row at scoring time.
        if self.eval_config.normalization_mode not in valid_normalization_modes():
            raise ValueError(
                f"Invalid normalization_mode '{self.eval_config.normalization_mode}'. "
                f"Available options: {', '.join(valid_normalization_modes())}"
            )
        if self.eval_config.cpwer_normalization not in VALID_CPWER_NORMALIZATION_MODES:
            raise ValueError(
                f"Invalid cpwer_normalization '{self.eval_config.cpwer_normalization}'. "
                f"Available options: {', '.join(VALID_CPWER_NORMALIZATION_MODES)}"
            )

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        """Evaluate single audio sample - can be called during generation.

        Returns dict of updates to be merged into data_point by BaseEvaluator.
        """
        return evaluate_sample(data_point, self.eval_config)


def coerce_text_field(value: Any) -> str:
    """Convert loose text or reference fields (a string, mapping or list) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "expected_answer", "answer", "transcript", "reference"):
            if key in value and value[key] is not None:
                return coerce_text_field(value[key])
        return " ".join(coerce_text_field(item) for item in value.values() if item is not None).strip()
    if isinstance(value, (list, tuple)):
        return " ".join(coerce_text_field(item) for item in value if item is not None).strip()
    return str(value).strip()


def evaluate_sample(sample: dict[str, Any], config: AudioEvaluatorConfig) -> dict[str, Any]:
    """Evaluate single sample based on task_type. Returns dict of updates to merge."""
    updates = {}
    task_type = sample.get("task_type", "unknown")
    generation = coerce_text_field(sample.get("generation", ""))
    expected_answer = coerce_text_field(sample.get("expected_answer", ""))

    # Extract ASR text from generation
    # E.g Qwen ASR uses <asr_text> tags to indicate the ASR text
    if task_type in TASKS_NEED_PARSING:
        generation = extract_asr_text(generation)

    # Strip helpful prefixes for ASR tasks (e.g., "The audio says: ...")
    if config.strip_helpful_prefixes:
        generation = strip_helpful_prefixes(generation)

    if task_type == "ASR-PC":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr_pc(
            expected_answer,
            generation,
            normalize_standard_wer=config.normalize_asr_pc_standard_wer,
            normalization_mode=mode,
        )
        updates.update(metrics)

    elif task_type == "ASR":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates["predicted_answer"] = generation
        updates.update(metrics)

    elif task_type == "MS-ASR":
        metrics = evaluate_msasr(
            expected_answer,
            generation,
            speaker_token=config.speaker_token,
            cpwer_normalization=config.cpwer_normalization,
        )
        updates.update(metrics)
        updates["predicted_answer"] = generation

    elif task_type == "ASR_LEADERBOARD":
        mode = resolve_asr_normalization_mode(config)
        metrics = evaluate_asr(expected_answer, generation, normalization_mode=mode)
        updates.update(metrics)

        # Additional WER calculation for specified reference fields
        if config.reference_fields:
            for ref_field in config.reference_fields:
                ref_value = sample[ref_field]  # fail if field is missing - user-specified fields must exist
                # Compute WER against this reference field
                ref_metrics = evaluate_asr(ref_value, generation, normalization_mode=mode)
                # Derive metric name from field name (e.g., "text_tn" -> "wer_tn")
                metric_suffix = ref_field.replace("text_", "") if ref_field.startswith("text_") else ref_field
                updates[f"wer_{metric_suffix}"] = ref_metrics["wer"]
                updates[f"is_correct_{metric_suffix}"] = ref_metrics["is_correct"]

    elif task_type in ["AST", "Translation", "Multilingual-AST"]:
        extra_fields = sample.get("extra_fields", {})
        tgt_lang = extra_fields.get("tgt_lang", None)
        metrics = evaluate_translation(expected_answer, generation, tgt_lang)
        updates.update(metrics)

    elif task_type == "Multilingual-ASR":
        mode = resolve_asr_normalization_mode(config)
        extra_fields = sample.get("extra_fields", {})
        use_cer = extra_fields.get("use_cer", False)
        use_mer = extra_fields.get("use_mer", False)
        src_lang = extra_fields.get("src_lang", None)
        cer_whitespace_mode = resolve_sample_cer_whitespace_mode(extra_fields)
        if src_lang is not None and "_" in src_lang:
            src_lang = src_lang.split("_")[0]
        preprocess_kwargs = {
            "lang": src_lang,
            "remove_diacritics": extra_fields.get("remove_diacritics", True),
        }
        # For code-switched data (use_mer, i.e. cs-fleurs), preserve combining marks in
        # the multilingual normalization so MER counts Thai/Myanmar vowels/tones (and so
        # grapheme-cluster segmentation is meaningful). num2words and other steps still
        # apply. Monolingual benchmarks strip combining marks.
        if use_mer or extra_fields.get("preserve_marks", False):
            preprocess_kwargs["preserve_marks"] = True
        # Only normalize compound pairs for non-English languages by default.
        normalize_compound = extra_fields.get("normalize_compound", src_lang not in [None, "en"])
        merge_compounds = extra_fields.get("merge_compounds", False)
        # Non-keep whitespace semantics must not be undermined by the
        # mismatch-region compound normalizer, which can join Latin words.
        # Explicit keep and a False ``cer_strip_whitespace`` keep the
        # compound normalization that applies when whitespace is scored.
        cer_normalize_compound = normalize_compound and not use_mer and cer_whitespace_mode == "keep"

        # Mixed WER column: CJK-style languages (use_cer) fold their CER into the WER
        # column (stored under wer* keys), all other languages contribute standard WER.
        # Code-switched samples (use_mer) instead get a true Mixed Error Rate, counting
        # scriptio-continua scripts by character and space-delimited scripts by word in
        # the same utterance; also stored under the wer* column. Monolingual benchmarks
        # (e.g. fleurs) never set use_mer, so their use_cer/WER behavior is unchanged.
        if use_mer:
            metrics = evaluate_mer(
                expected_answer,
                generation,
                normalization_mode=mode,
                # mixed_segment already handles compact and spaced continua
                # scripts. Compound normalization would erase English word
                # boundaries before MER can score them.
                normalize_compound=False,
                grapheme=extra_fields.get("mer_grapheme", False),
                **preprocess_kwargs,
            )
        elif use_cer:
            metrics = evaluate_cer(
                expected_answer,
                generation,
                normalization_mode=mode,
                key_prefix="wer",  # fold CER into the mixed WER column
                normalize_compound=cer_normalize_compound,
                whitespace_mode=cer_whitespace_mode,
                **preprocess_kwargs,
            )
        else:
            metrics = evaluate_asr(
                expected_answer,
                generation,
                normalization_mode=mode,
                normalize_compound=normalize_compound,
                merge_compounds=merge_compounds,
                **preprocess_kwargs,
            )
        updates.update(metrics)

        # Parallel only-CER column: character error rate computed for every sample,
        # reported separately (cer* keys) alongside the mixed WER column.
        # Code-switched benchmarks (use_mer, i.e. cs-fleurs) score this CER with
        # mark-preserving normalization (lowercase + unpunctuated, combining marks
        # kept) so it matches the CS-FLEURS paper's CER. Configured Mandarin and
        # Cantonese benchmarks ignore Han-adjacent formatting spaces; Japanese,
        # Korean, Vietnamese, and other CER languages retain whitespace by default.
        cer_mode = "lower_nopunct" if use_mer else mode
        cer_metrics = evaluate_cer(
            expected_answer,
            generation,
            normalization_mode=cer_mode,
            key_prefix="cer",
            normalize_compound=cer_normalize_compound,
            whitespace_mode=cer_whitespace_mode,
            **preprocess_kwargs,
        )
        # Merge only cer* keys so the mixed metric's is_correct/text/pred_text are preserved.
        for cer_key in (
            "cer",
            "cer_errors",
            "cer_ref_words",
            "cer_substitutions",
            "cer_insertions",
            "cer_deletions",
        ):
            if cer_key in cer_metrics:
                updates[cer_key] = cer_metrics[cer_key]

    elif task_type == "CER":
        extra_fields = sample.get("extra_fields", {})
        metrics = evaluate_cer(
            expected_answer,
            generation,
            normalization_mode="none",
            key_prefix="cer",
            whitespace_mode=resolve_sample_cer_whitespace_mode(extra_fields),
        )
        updates.update(metrics)

    elif task_type == "Hallucination":
        audio_context = {"audio_duration": sample.get("audio_duration")}
        metrics = evaluate_hallucination(expected_answer, generation, audio_context)
        updates.update(metrics)

    elif task_type == "PC-Rate":
        metrics = evaluate_pc_rate(expected_answer, generation)
        updates.update(metrics)

    elif task_type in _EXTRA_TASK_TYPES:
        updates.update(_EXTRA_TASK_TYPES[task_type](sample, generation, config))

    else:
        if "requires_judge" not in sample:
            updates["requires_judge"] = True
        if "is_correct" not in sample:
            updates["is_correct"] = False

    audio_duration = sample.get("audio_duration", None)
    if audio_duration and audio_duration > 0 and expected_answer and generation:
        # chars/minute (chars/second * 60)
        updates["ref_char_rate"] = (len(expected_answer) / audio_duration) * 60.0
        updates["hyp_char_rate"] = (len(generation) / audio_duration) * 60.0
        updates["char_rate_diff"] = abs(updates["hyp_char_rate"] - updates["ref_char_rate"])

    return updates
