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

# Faithful port of the deterministic verification used by NeMo Skills'
# `hotpotqa_closedbook` benchmark:
#
#   * SQuAD-style answer normalization (`normalize_answer`).
#   * Token-overlap F1/EM with yes/no/noanswer special cases (`answer_f1_score`,
#     `answer_exact_match`).
#   * JSON-extractor for the model's predicted answer (`parse_generation`).
#   * Alternative-aware substring matching (`is_correct`, `is_correct_strict`)
#     with a small surface-form alternatives generator (`normalize_gt`).
#
# The Skills sources for this code:
#   nemo_skills/evaluation/metrics/hotpotqa_metrics.py
#   nemo_skills/evaluation/metrics/hotpotqa_filtering.py
#
# Skills attributes the original answer scoring to the official HotpotQA
# evaluation script (https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
# and the alternatives / filtering logic to
# hmaron/nvidia-research-tlv-nemotron-hallucination-detection.

import json
import re
import string
from collections import Counter
from typing import Optional, Tuple


__all__ = [
    "answer_exact_match",
    "answer_f1_score",
    "is_correct",
    "is_correct_strict",
    "normalize_answer",
    "normalize_gt",
    "parse_generation",
]


# ──────────────────────────────────────────────────────────
# SQuAD-style normalization & answer scoring
# (port of Skills' hotpotqa_metrics.py)
# ──────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    """Normalize answer string (official HotpotQA / SQuAD normalization)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def answer_f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Compute token-overlap F1, precision, and recall.

    Returns (f1, precision, recall). Special-cases yes/no/noanswer tokens.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0.0, 0.0, 0.0)

    if normalized_prediction in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def answer_exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized prediction matches normalized ground truth, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _try_parse_answer_json(text: str) -> Optional[Tuple[str, list]]:
    """Try to parse a JSON string as a HotpotQA answer object.

    Returns (answer, supporting_facts_list) or None if not parseable.
    """
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or "answer" not in parsed:
            return None
        answer = str(parsed["answer"])
        sp = parsed.get("supporting_facts", [])
        if isinstance(sp, list):
            valid_sp = []
            for item in sp:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        valid_sp.append([str(item[0]), int(item[1])])
                    except (ValueError, TypeError):
                        continue
            return answer, valid_sp
        return answer, []
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _extract_json_candidates(text: str) -> list:
    """Extract all brace-delimited JSON candidate strings, ordered by position."""
    candidates = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    candidates.append(text[i : j + 1])
                    i = j + 1
                    break
            else:
                break
        else:
            i += 1
    return candidates


def parse_generation(generation: str) -> Tuple[str, list]:
    """Parse the model generation to extract the predicted answer and supporting facts.

    Searches for JSON objects containing an "answer" key. When reasoning precedes
    the JSON output, the *last* valid JSON object is used (the model is prompted
    to end its response with the JSON).

    Returns (answer_string, supporting_facts_list). Falls back to the raw
    generation text when no JSON is found.
    """
    if not generation:
        return "", []

    text = generation.strip()

    md_matches = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    for md_match in reversed(md_matches):
        result = _try_parse_answer_json(md_match.group(1))
        if result is not None:
            return result

    candidates = _extract_json_candidates(text)
    for candidate in reversed(candidates):
        result = _try_parse_answer_json(candidate)
        if result is not None:
            return result

    return text, []


# ──────────────────────────────────────────────────────────
# Ground-truth filtering & alternatives
# (port of Skills' hotpotqa_filtering.py)
# ──────────────────────────────────────────────────────────


_NUM_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "nineteenth": "19th",
    "twentieth": "20th",
    "twenty-first": "21st",
}
_NUM_DIGITS = {v: k for k, v in _NUM_WORDS.items()}

_MAX_GT_LENGTH = 40
_MIN_ALT_LENGTH = 3

_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "to",
        "by",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "with",
        "from",
        "that",
        "this",
        "it",
        "its",
        "his",
        "her",
        "my",
        "our",
        "their",
        "no",
        "not",
        "but",
        "if",
        "as",
        "into",
        "about",
        "than",
        "then",
    ]
)


def _normalize_unicode(s: str) -> str:
    """Normalize unicode whitespace, hyphens, and quotes for substring matching."""
    for c in "      ":
        s = s.replace(c, " ")
    for c in "‐‑‒–—―":
        s = s.replace(c, "-")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def _gt_alternatives(gt: str) -> Tuple[list, list]:
    """Generate valid surface-form alternatives for a ground-truth answer.

    Returns (sorted_alternatives, list_of_rule_tags_that_fired).
    """
    alts = {gt}
    rules = []

    for prefix in ("the ", "a ", "an "):
        if gt.lower().startswith(prefix):
            alts.add(gt[len(prefix) :])
            rules.append("strip_article")
            break

    stripped = gt.replace('"', "").replace("“", "").replace("”", "").strip()
    if stripped and stripped != gt:
        alts.add(stripped)
        rules.append("strip_quotes")

    if "(" in gt:
        no_parens = re.sub(r"\s*\([^)]*\)\s*", " ", gt).strip()
        no_parens = re.sub(r"\s+", " ", no_parens)
        if no_parens and no_parens != gt:
            alts.add(no_parens)
        for inner in re.findall(r"\(([^)]+)\)", gt):
            inner = inner.strip()
            if len(inner) > 1:
                alts.add(inner)
        rules.append("normalize_parens")

    gt_low = gt.lower().strip()
    if gt_low in _NUM_WORDS:
        alts.add(_NUM_WORDS[gt_low])
        rules.append("number_word_to_digit")
    if gt_low in _NUM_DIGITS:
        alts.add(_NUM_DIGITS[gt_low])
        rules.append("number_digit_to_word")

    no_commas = re.sub(r"(\d),(\d{3})", r"\1\2", gt)
    while no_commas != gt and re.search(r"(\d),(\d{3})", no_commas):
        no_commas = re.sub(r"(\d),(\d{3})", r"\1\2", no_commas)
    if no_commas != gt:
        alts.add(no_commas)
        rules.append("strip_number_commas")

    if gt and gt[-1] in ".,;:!?":
        alts.add(gt[:-1].rstrip())
        rules.append("strip_trailing_punct")

    if "." in gt:
        no_dots = re.sub(r"(?<!\d)\.(?!\d)", "", gt)
        if no_dots and len(no_dots) > 1 and no_dots != gt:
            alts.add(no_dots)
            rules.append("strip_abbrev_dots")

    if "-" in gt and not gt.startswith("-"):
        no_hyphen = re.sub(r"\s+", " ", gt.replace("-", " ")).strip()
        if no_hyphen != gt:
            alts.add(no_hyphen)
            rules.append("hyphen_to_space")

    if " & " in gt:
        alts.add(gt.replace(" & ", " and "))
        rules.append("ampersand_to_and")
    if " and " in gt.lower():
        idx = gt.lower().index(" and ")
        alts.add(gt[:idx] + " & " + gt[idx + 5 :])
        rules.append("and_to_ampersand")

    extra = set()
    for alt in list(alts):
        for prefix in ("the ", "a ", "an "):
            if alt.lower().startswith(prefix):
                extra.add(alt[len(prefix) :])
    alts |= extra

    normed = set()
    for a in alts:
        a = re.sub(r"\s+", " ", a.strip())
        if a and (len(a) >= _MIN_ALT_LENGTH or a == gt.strip() or a.isdigit()):
            normed.add(a)

    return sorted(normed), rules


def _is_multi_word_name(gt: str) -> bool:
    """True if GT looks like a multi-word proper name unreliable for substring matching."""
    parts = gt.strip().rstrip(".").split()
    n = len(parts)
    if n in (3, 4):
        return all(p[0].isupper() for p in parts) and all(p.lower() not in _STOPWORDS for p in parts)
    if n in (5, 6):
        caps = [p for p in parts if p[0].isupper() and p.lower() not in _STOPWORDS]
        return len(caps) >= 3
    return False


def _should_remove(gt: str) -> Tuple[bool, str]:
    """Return (flag, reason). Reason is '' if not removed."""
    if len(gt) > _MAX_GT_LENGTH:
        return True, "gt_too_long"
    if _is_multi_word_name(gt):
        return True, "multi_word_name"
    return False, ""


def normalize_gt(gt_answer: str) -> dict:
    """Normalize a single ground-truth answer on-the-fly.

    Returns dict with keys:
        alternatives (list[str]): Valid surface forms (always includes original).
        should_remove (bool): True if unreliable for substring eval.
        remove_reason (str): '' | 'gt_too_long' | 'multi_word_name'.
        edited (bool): True if any rule fired.
        edit_reasons (list[str]): Tags of rules that fired.
    """
    alts, alt_rules = _gt_alternatives(gt_answer)
    remove, remove_reason = _should_remove(gt_answer)
    edit_reasons = list(alt_rules)
    if remove_reason:
        edit_reasons.append(remove_reason)
    return {
        "alternatives": alts,
        "should_remove": remove,
        "remove_reason": remove_reason,
        "edited": bool(edit_reasons),
        "edit_reasons": edit_reasons,
    }


def is_correct(alternatives, model_answer: str) -> bool:
    """Lenient: any alternative is a substring of the (normalized) model answer."""
    ans = _normalize_unicode(model_answer.lower())
    return any(_normalize_unicode(alt.lower()) in ans for alt in alternatives)


def is_correct_strict(alternatives, model_answer: str) -> bool:
    """Stricter matching that reduces false positives.

    Additional gates over is_correct():
      - Short alternatives (<=4 chars): require word-boundary match.
      - Long model answers (>80 chars): reject if match starts after position 40.
    """
    ans = _normalize_unicode(model_answer.lower())
    ans_len = len(ans)

    for alt in alternatives:
        alt_norm = _normalize_unicode(alt.lower())
        if not alt_norm:
            continue
        if alt_norm not in ans:
            continue
        if len(alt_norm) <= 4:
            if not re.search(r"(?<!\w)" + re.escape(alt_norm) + r"(?!\w)", ans):
                continue
        if ans_len > 80:
            pos = ans.find(alt_norm)
            if pos > 40:
                continue
        return True
    return False
