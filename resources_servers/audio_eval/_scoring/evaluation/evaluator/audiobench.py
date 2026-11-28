# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The normalization modes and the Spoken-MQA protocol follow the metric definitions published with AudioBench
# (https://github.com/AudioLLMs/AudioBench at commit 85dcdb88c41e6ebb2bcb3ce19b15d061dfef9dba). The AudioBench
# source code is under a Creative Commons NonCommercial license and none of it is copied here: the normalization
# tables, answer extraction and comparison in this file are independent implementations of the same definitions.

"""AudioBench evaluator: the generic audio evaluator plus AudioBench's normalization modes and Spoken-MQA task.

Importing this module registers the ``audiobench`` and ``audiobench_earnings`` normalization modes and the ``MathQA``
task type with the generic audio evaluator; a profile selects it with ``++eval_type=audiobench``.
"""

import re
from functools import lru_cache
from typing import Any

from resources_servers.audio_eval._scoring.evaluation.evaluator.audio import (
    AudioEvaluator,
    AudioEvaluatorConfig,
    coerce_text_field,
    register_normalization_mode,
    register_task_type,
)


_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS_WORDS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
# Standalone numerals 0-19 and the multiples of ten up to 90 are spelled out.
_DIGIT_WORDS = {str(n): word for n, word in enumerate(_NUMBER_WORDS)}
_DIGIT_WORDS.update({str(10 * (n + 2)): word for n, word in enumerate(_TENS_WORDS)})
_DIGIT_WORD_RE = re.compile(r"\b(" + "|".join(_DIGIT_WORDS) + r")\b")


def _normalize_digits_to_words(text: str) -> str:
    """Spell out standalone numerals (e.g., '1' -> 'one')."""
    return _DIGIT_WORD_RE.sub(lambda match: _DIGIT_WORDS[match.group(1)], text)


_CONTRACTION_EXPANSIONS = {
    **{f"{pronoun}'s": f"{pronoun} is" for pronoun in ("he", "she", "it", "that")},
    **{f"{pronoun}'re": f"{pronoun} are" for pronoun in ("you", "we", "they")},
    **{f"{pronoun}'ve": f"{pronoun} have" for pronoun in ("i", "you", "we", "they")},
    "i'm": "i am",
    **{f"{verb}n't": f"{verb} not" for verb in ("is", "are", "was", "were", "has", "have", "had", "does", "did")},
    "don't": "do not",
}
_CONTRACTION_RE = re.compile(r"\b(" + "|".join(map(re.escape, _CONTRACTION_EXPANSIONS)) + r")\b")


def _expand_contractions(text: str) -> str:
    """Expand common English contractions (e.g., "I'm" -> "I am")."""
    return _CONTRACTION_RE.sub(lambda match: _CONTRACTION_EXPANSIONS[match.group(1)], text)


_FILLER_RE = re.compile(r"\b(uh|umm|um|er|ah)\b")


def _remove_non_speech_elements(text: str) -> str:
    """Remove filler words (uh, um, er, ah)."""
    return _FILLER_RE.sub("", text)


@lru_cache(maxsize=1)
def _get_audiobench_earnings_english_normalizer():
    """Reproduce the Whisper normalizer snapshot used by AudioBench 85dcdb8 (as installed from whisper-normalizer).

    whisper-normalizer 0.1.12 added four slang expansions after the Whisper
    snapshot used by AudioBench. Its remaining executable normalizer code
    and spelling map are identical to that snapshot, so remove only those
    additions. the audio_eval pyproject pins whisper-normalizer==0.1.12.

    This compatibility normalizer is intentionally used only by the standalone
    Earnings-21/22 leaves. AudioBench judge/nonjudge tasks use the
    ``audiobench`` mode.
    """
    from whisper_normalizer.english import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    post_snapshot_replacers = {
        r"\bkinda\b": "kind of",
        r"\bsorta\b": "sort of",
        r"\bdunno\b": "do not know",
        r"\bcause\b": "because",
    }
    for pattern, expected_replacement in post_snapshot_replacers.items():
        actual_replacement = normalizer.replacers.get(pattern)
        if actual_replacement != expected_replacement:
            raise RuntimeError(
                "whisper-normalizer no longer matches the pinned 0.1.12 compatibility contract: "
                f"expected replacer {pattern!r} -> {expected_replacement!r}, found {actual_replacement!r}"
            )
        del normalizer.replacers[pattern]
    return normalizer


def _normalize_audiobench(text: str, *, earnings: bool = False, **kwargs: Any) -> str:
    """Whisper normalization, then numerals spelled out, contractions expanded, bracketed and filler text dropped.

    The Earnings-only mode removes four slang rules added to the Whisper normalizer after the snapshot AudioBench
    pins; the broad ``audiobench`` mode keeps them.
    """
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    text = text.lower()
    text = _get_audiobench_earnings_english_normalizer()(text) if earnings else EnglishTextNormalizer()(text)
    text = _normalize_digits_to_words(text)
    text = _expand_contractions(text)
    text = re.sub(r"(\[|\(|\{|\<)[^\(\)\\n\[\]]*(\]|\)|\}|\>)", "", text)
    jiwer_process = jiwer.Compose(
        [
            jiwer.RemoveMultipleSpaces(),
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveKaldiNonWords(),
            jiwer.RemovePunctuation(),
        ]
    )
    text = jiwer_process(text)
    return _remove_non_speech_elements(text)


register_normalization_mode("audiobench", _normalize_audiobench)
register_normalization_mode("audiobench_earnings", lambda text, **kwargs: _normalize_audiobench(text, earnings=True))


SPOKEN_MQA_ANSWER_TRIGGERS = (
    "The answer is:",
    "The answer is",
    "the answer is",
    "####",
)
_SPOKEN_MQA_NUMBER_RE = re.compile(r"-?\d+/?\.?\d*")
_SPOKEN_MQA_FRACTION_RE = re.compile(r"(-?\d+)/(\d+)")
_SPOKEN_MQA_OPTION_LABELS = frozenset({"(a)", "(b)", "(c)", "(d)", "(e)", "(f)"})


def _spoken_mqa_canonical_number(value: str) -> str:
    """Canonical text of a numeral: integral values lose the decimal part, other floats use ``str``.

    A ``a/b`` fraction is evaluated; anything else that is not a number is returned unchanged.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        fraction = _SPOKEN_MQA_FRACTION_RE.fullmatch(str(value))
        if fraction is None or int(fraction.group(2)) == 0:
            return str(value)
        number = int(fraction.group(1)) / int(fraction.group(2))
    return str(int(number)) if number.is_integer() else str(number)


def clean_spoken_mqa_prediction(prediction: Any) -> str:
    """Extract the final numeric answer from a Spoken-MQA generation.

    The text after the last answer trigger is searched for numerals and the first one is taken; without a trigger
    the last numeral of the whole text is taken. If a trigger occurs more than once (a repeated worked example),
    only the first paragraph is considered.
    """
    text = coerce_text_field(prediction).strip("\n")
    if any(text.count(trigger) > 1 for trigger in SPOKEN_MQA_ANSWER_TRIGGERS):
        text = text.split("\n\n")[0]

    segments = re.split("|".join(SPOKEN_MQA_ANSWER_TRIGGERS), text)
    has_trigger = len(segments) > 1
    tail = segments[-1] if has_trigger else text
    tail = tail.strip("\n").rstrip(".").rstrip("/").strip(" ").replace(",", "")
    numerals = [_spoken_mqa_canonical_number(token) for token in _SPOKEN_MQA_NUMBER_RE.findall(tail)]
    if not numerals:
        return ""
    chosen = numerals[0] if has_trigger else numerals[-1]
    return chosen.rstrip(".").rstrip("/")


def _spoken_mqa_reference_list(reference: Any, protocol: str) -> list[str]:
    if isinstance(reference, dict):
        reference = reference.get("text", reference)
    if isinstance(reference, (list, tuple)):
        references = [coerce_text_field(value) for value in reference]
    else:
        references = [coerce_text_field(reference)]

    if protocol == "reasoning":
        references = [
            _spoken_mqa_canonical_number(answer.split("#### ")[-1].replace(",", "") if "####" in answer else answer)
            for answer in references
        ]
    elif protocol != "arithmetic":
        raise ValueError(f"Unknown Spoken-MQA protocol: {protocol}")

    return references or ["empty"]


def _spoken_mqa_answers_match(prediction: str, reference: str) -> bool:
    prediction = prediction.replace("\\text{", "").rstrip("}") if "\\text{" in prediction else prediction
    reference = reference.replace("\\text{", "").rstrip("}") if "\\text{" in reference else reference
    prediction, reference = prediction.lower(), reference.lower()
    if reference in _SPOKEN_MQA_OPTION_LABELS:
        return reference in prediction
    return prediction == reference


def evaluate_spoken_mqa(reference: Any, hypothesis: Any, protocol: str) -> dict[str, Any]:
    """Score one Spoken-MQA row with the official AudioBench ``acc`` protocol."""
    parsed_prediction = clean_spoken_mqa_prediction(hypothesis)
    parse_failure = not parsed_prediction
    prediction = parsed_prediction or "empty"
    prediction = prediction[:100]
    references = _spoken_mqa_reference_list(reference, protocol)
    if not references or not any(references):
        references = ["empty"]
    is_correct = any(_spoken_mqa_answers_match(prediction, answer) for answer in references)
    return {
        "is_correct": is_correct,
        "predicted_answer": prediction,
        "spoken_mqa_prediction": prediction,
        "spoken_mqa_references": references,
        "spoken_mqa_parse_failure": parse_failure,
        "requires_judge": False,
    }


def _math_qa(sample: dict[str, Any], generation: str, config: AudioEvaluatorConfig) -> dict[str, Any]:
    return evaluate_spoken_mqa(
        sample.get("expected_answer", ""), generation, sample.get("spoken_mqa_protocol", "arithmetic")
    )


register_task_type("MathQA", _math_qa)


class AudioBenchEvaluator(AudioEvaluator):
    """The audio evaluator with AudioBench's normalization modes and Spoken-MQA task registered."""
