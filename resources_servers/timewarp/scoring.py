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
#
# The verifier logic and the LLM-judge prompt are adapted from TimeWarp,
# src/browsergym/timewarp/evaluators.py at commit 4978e69
# (https://github.com/sparklabutah/timewarp), which declares the MIT License (reproduced
# below). Upstream reads each task from a JSON file and wraps every verifier in a class; here
# each verifier is a function of (answer, reference_answers), and the judge prompt is sent
# through a Gym model server instead of the OpenAI SDK. Matching rules, error messages and the
# judge wording are unchanged, except for one trailing space dropped from the prompt.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES and contributors, licensed
# under the Apache License 2.0 (SPDX header above).
#
# MIT License
#
# Copyright (c) 2026 The TimeWarp Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""TimeWarp answer scoring: the deterministic verifiers and the LLM-judge prompt.

A task lists its verifiers in ``eval_types``; they are AND-ed. Each deterministic verifier
reads its spec from ``reference_answers`` and raises ``ValueError`` on a malformed spec rather
than returning 0.0, because a silent zero is indistinguishable from a failed episode.
"""

import logging
import re
from typing import Any, Callable, Mapping

from resources_servers.timewarp.normalization import (
    as_entry_list,
    contains_entry,
    equals_entry,
    extract_numbers,
    find_entry,
    numbers_match,
    scope_text,
    to_decimal,
)


logger = logging.getLogger(__name__)

#: Verifier identifiers accepted in a task's ``eval_types``. Everything except ``llm_judge``
#: is deterministic.
SUPPORTED_EVAL_TYPES = ("string_match", "number_match", "list_match", "exact_match", "llm_judge")
DETERMINISTIC_EVAL_TYPES = tuple(t for t in SUPPORTED_EVAL_TYPES if t != "llm_judge")

_THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL)
_UNPAIRED_THINK_CLOSE = re.compile(r"^.*</(think|thinking)>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop ``<think>``/``<thinking>`` blocks so reasoning never counts as the answer.

    A chat template may open the block in the prompt, leaving only an unpaired closing tag in
    the generation; everything up to the last such tag is reasoning as well.
    """
    text = _THINK_BLOCK.sub("", text)
    return _UNPAIRED_THINK_CLOSE.sub("", text).strip()


def validate_eval_types(eval_types: list[str]) -> None:
    """Reject the task specs upstream's ``evaluator_router`` rejects."""
    if not eval_types:
        raise ValueError("eval_types must list at least one evaluator")
    for eval_type in eval_types:
        if eval_type not in SUPPORTED_EVAL_TYPES:
            raise ValueError(
                f"eval_type {eval_type} is not supported. Supported types: {', '.join(SUPPORTED_EVAL_TYPES)}"
            )


def clean_answer(answer: str) -> str:
    """Strip quotes, collapse whitespace and lowercase (upstream ``Evaluator.clean_answer``)."""
    answer = answer.strip()
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    answer = re.sub(r"\s+", " ", answer)
    return answer.strip().lower()


def exact_match(answer: str, references: Mapping[str, Any]) -> float:
    """The legacy ``exact_match`` eval type: the cleaned answer equals the cleaned reference."""
    if "exact_match" in references:
        return float(clean_answer(answer) == clean_answer(references["exact_match"]))
    return 0.0


def string_match(answer: str, references: Mapping[str, Any]) -> float:
    """Match a free-text answer against ``exact_match``/``must_include``/``must_exclude`` entries.

    ``exact_match`` is any-of over the normalized answer; every ``must_include`` entry must
    appear on word boundaries and no ``must_exclude`` entry may. Entries may carry
    ``" |OR| "`` alternatives or be ``^regex$`` leaves. ``scope`` is ``"full"`` (default) or
    ``"first_sentence"``.
    """
    answer = scope_text(answer, references.get("scope", "full"))
    exact = as_entry_list(references.get("exact_match"), "exact_match")
    includes = as_entry_list(references.get("must_include"), "must_include")
    excludes = as_entry_list(references.get("must_exclude"), "must_exclude")

    if not (exact or includes or excludes):
        raise ValueError(
            "string_match requires at least one of 'exact_match', 'must_include' or 'must_exclude' in reference_answers"
        )

    if exact and not any(equals_entry(answer, entry) for entry in exact):
        logger.debug("string_match: no exact_match alternative matched %r", answer)
        return 0.0
    for entry in includes:
        if not contains_entry(answer, entry):
            logger.debug("string_match: missing required %r in %r", entry, answer)
            return 0.0
    for entry in excludes:
        if contains_entry(answer, entry):
            logger.debug("string_match: forbidden %r present in %r", entry, answer)
            return 0.0
    return 1.0


def number_match(answer: str, references: Mapping[str, Any]) -> float:
    """Require every number in ``reference_answers.number_match`` among the answer's numbers.

    The spec is ``{"value": 7000000}`` or ``{"values": [5, 1.8]}``; comparison is exact unless
    ``rel_tolerance``/``abs_tolerance`` is given, and a ``values`` entry may carry its own.
    """
    spec = references.get("number_match")
    if not isinstance(spec, dict):
        raise ValueError("number_match requires a 'number_match' object in reference_answers")

    answer = scope_text(answer, spec.get("scope", "full"))
    default_rel = spec.get("rel_tolerance")
    default_abs = spec.get("abs_tolerance")

    if "values" in spec:
        requirements = spec["values"]
        if not isinstance(requirements, list) or not requirements:
            raise ValueError("number_match 'values' must be a non-empty list")
    elif "value" in spec:
        requirements = [spec["value"]]
    else:
        raise ValueError("number_match requires a 'value' or 'values' key")

    candidates = extract_numbers(answer)
    if not candidates:
        logger.debug("number_match: no numbers found in %r", answer)
        return 0.0

    for requirement in requirements:
        if isinstance(requirement, dict):
            expected = to_decimal(requirement["value"])
            rel = requirement.get("rel_tolerance", default_rel)
            abs_ = requirement.get("abs_tolerance", default_abs)
        else:
            expected = to_decimal(requirement)
            rel, abs_ = default_rel, default_abs
        if not any(numbers_match(candidate, expected, rel, abs_) for candidate in candidates):
            logger.debug("number_match: %s not found among %s", expected, candidates)
            return 0.0
    return 1.0


def list_match(answer: str, references: Mapping[str, Any]) -> float:
    """Require every item of ``reference_answers.list_match.items`` in the answer.

    Each item is a list of interchangeable spellings (a bare string is a one-element list).
    With ``ordered: true`` the items' first occurrences must follow the listed order;
    ``forbidden`` entries must not appear at all.
    """
    spec = references.get("list_match")
    if not isinstance(spec, dict):
        raise ValueError("list_match requires a 'list_match' object in reference_answers")

    items = spec.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("list_match requires a non-empty 'items' list")

    answer = scope_text(answer, spec.get("scope", "full"))
    offsets = []
    for item in items:
        alternatives = as_entry_list(item, "list_match.items entry")
        if not alternatives:
            raise ValueError("list_match items must not be empty")
        matches = [offset for offset in (find_entry(answer, alt) for alt in alternatives) if offset is not None]
        if not matches:
            logger.debug("list_match: missing item %r in %r", alternatives, answer)
            return 0.0
        offsets.append(min(matches))

    if spec.get("ordered", False) and any(a >= b for a, b in zip(offsets, offsets[1:])):
        logger.debug("list_match: items out of order (offsets=%s)", offsets)
        return 0.0

    for entry in as_entry_list(spec.get("forbidden"), "list_match.forbidden"):
        if contains_entry(answer, entry):
            logger.debug("list_match: forbidden %r present in %r", entry, answer)
            return 0.0
    return 1.0


DETERMINISTIC_SCORERS: dict[str, Callable[[str, Mapping[str, Any]], float]] = {
    "exact_match": exact_match,
    "string_match": string_match,
    "number_match": number_match,
    "list_match": list_match,
}


# --- LLM judge ---------------------------------------------------------------------------
#
# Only tasks with no objectively checkable answer still use the judge (one per split). The
# prompt and the verdict parsing are TimeWarp's; which model answers is a Gym config choice.

JUDGE_PROMPT_TEMPLATE = """Help a teacher grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.
Input:
- question: {question}
- reference answer: {reference}
- student answer: {pred}

Special Sequence: The string 'N/A' that you see is a special sequence that means 'not achievable'

Output: You must respond with EXACTLY one of the following words (nothing else):
1) 'correct': if the answer is semantically equivalent to the reference.
   - Numeric values must match exactly (including units, signs, and scale) unless the question/reference clearly allows an approximation or rounding.
   - If an estimate is allowed, the student must still be reasonably close and not contradict the reference.
   - Ordered lists/steps/rankings must match exactly in both the element values and order.
   - Unordered lists/sets must contain the same element values; the order of the elements does not matter.
   - Extra information is allowed only if it does not introduce contradictions or change the meaning.
2) 'partially correct': if the answer is somewhat related but incomplete or inaccurate
3) 'incorrect': if the answer is wrong or unrelated
Do not include any additional text, explanation, or formatting. Only respond with one of the three words above."""

JUDGE_SYSTEM_MESSAGE = "You are a helpful assistant"


def judge_references(references: Mapping[str, Any]) -> list[str]:
    """The ``fuzzy_match`` golds an ``llm_judge`` task accepts (any one suffices)."""
    golds = references.get("fuzzy_match")
    if golds is None:
        return []
    return [golds] if isinstance(golds, str) else list(golds)


def build_judge_messages(*, question: str, reference: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
        {"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(question=question, reference=reference, pred=answer)},
    ]


def parse_judge_verdict(text: str) -> float:
    """1.0 for 'correct', 0.0 for 'partially correct', 'incorrect' or anything unparseable.

    Negative verdicts are checked first because both contain 'correct' as a substring.
    """
    verdict = text.lower().strip()
    if verdict == "correct":
        return 1.0
    if verdict in ("partially correct", "incorrect"):
        return 0.0
    if "incorrect" in verdict or "partially correct" in verdict:
        return 0.0
    if "correct" in verdict:
        logger.warning("LLM judge returned non-exact response %r; using fallback matching", verdict)
        return 1.0
    logger.warning("LLM judge returned unexpected response %r; scoring 0.0", verdict)
    return 0.0
