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
"""Chartography golden-answer judge: response schema, verdict parsing and pass rule.

Mirrors Surge AI's own scorer (https://github.com/surge-ai/chartography, ``src/chartography/scorer.py`` at
commit 3f1bf837232d3918c2cc35c6e2418c9e70d2a57b). The judge prompt is ``prompt_templates/judge.txt`` (the
upstream ``JUDGE_PROMPT_TEMPLATE``, unchanged), and the response schema and the pass rule are upstream's. The
Inspect AI wrappers are replaced, so the judge can be any OpenAI-compatible endpoint, and a few details differ:

- the three placeholders are filled in one pass, so placeholder text inside the response is never expanded;
- the question, golden answer and response are stripped of surrounding whitespace before rendering;
- parsing also rejects a non-finite score and a score too large to round, as unparseable output that is retried.
"""

from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, List, TypedDict


# Upstream MAX_RETRIES = 4: one call plus four retries on unparseable output.
MAX_ATTEMPTS = 5

# Same as upstream ResponseSchema(name="golden_answer_score", json_schema=...).
RESPONSE_SCHEMA_NAME = "golden_answer_score"
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "required": ["scores"],
    "properties": {
        "scores": {
            "type": "array",
            "description": "One entry per criterion in the golden answer",
            "items": {
                "type": "object",
                "required": ["criterion_id", "extracted_answer", "score", "rationale"],
                "properties": {
                    "criterion_id": {
                        "type": "string",
                        "description": "'criterion_n' where n is the criterion's 1-based order of appearance in the golden answer",
                    },
                    "extracted_answer": {
                        "type": "string",
                        "description": "The model's final answer, verbatim or lightly normalized",
                    },
                    "score": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "1 if the response matches the golden answer, 0 if it does not",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One or two sentences explaining the comparison and which rule applied",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


class ParseError(ValueError):
    """Judge output failed strict parsing or validation."""


class ParsedCriterion(TypedDict):
    criterion_id: str
    extracted_answer: str
    score: Decimal
    rationale: str


_PLACEHOLDER = re.compile(r"\{\{ rubric\.(response_generation_prompt|response_to_grade|all_criteria) \}\}")


def render_judge_prompt(template: str, question: str, golden_answer: str, response_text: str) -> str:
    """Fill the upstream template's three placeholders in one pass.

    Chained ``str.replace`` calls would expand a placeholder that appears inside the response (for example
    ``{{ rubric.all_criteria }}`` written by the policy) into the golden answer; one pass never rescans what it
    substituted.
    """
    values = {
        "response_generation_prompt": question,
        "response_to_grade": response_text,
        "all_criteria": golden_answer,
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def parse_verdict(raw_text: str) -> List[ParsedCriterion]:
    """Strict parse of the judge output: markdown fences stripped, a non-empty ``scores`` array required."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"\A```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\Z", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ParseError("Judge output must be a JSON object")
    entries = payload.get("scores")
    if not isinstance(entries, list) or not entries:
        raise ParseError("Judge output must contain a non-empty 'scores' array")
    return [_parse_criterion(entry) for entry in entries]


def _parse_criterion(entry: Any) -> ParsedCriterion:
    if not isinstance(entry, dict):
        raise ParseError("Each 'scores' entry must be a JSON object")
    missing = {"criterion_id", "extracted_answer", "score", "rationale"} - entry.keys()
    if missing:
        raise ParseError(f"'scores' entry missing {sorted(missing)}")
    raw_score: Any = entry["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
        raise ParseError("'score' must be a number or string")
    if not isinstance(entry["rationale"], str):
        raise ParseError("'rationale' must be a string")
    try:
        score = Decimal(str(raw_score).strip())
    except InvalidOperation as e:
        raise ParseError(f"'score' is not a valid decimal: {raw_score!r}") from e
    if not score.is_finite():
        raise ParseError(f"'score' must be finite: {raw_score!r}")
    try:
        verdict_passes(score)
    except InvalidOperation as e:
        # Too large to round to four decimals (for example 1e30): unparseable, so it is retried.
        raise ParseError(f"'score' is out of range: {raw_score!r}") from e
    return ParsedCriterion(
        criterion_id=_scalar_str(entry["criterion_id"], "criterion_id"),
        extracted_answer=_scalar_str(entry["extracted_answer"], "extracted_answer"),
        score=score,
        rationale=entry["rationale"],
    )


def _scalar_str(value: Any, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ParseError(f"'{field}' must be a string or number")
    return str(value)


def verdict_passes(score: Decimal) -> bool:
    """A criterion passes when its score, rounded half-up to four decimals, is at least 1."""
    return score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1
