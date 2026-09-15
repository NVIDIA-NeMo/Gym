# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Score parser for generative reward model outputs.

Parses JSON-formatted scores from GenRM model output text.

Expected output format (unified for single-rubric and multi-rubric):
```json
{
    "rubric_evaluations": [
        {
            "rubric_id": 1,
            "response_1_analysis": "...",
            "response_2_analysis": "...",
            "score_1": <1-5>,
            "score_2": <1-5>,
            "ranking": <1-6>
        }
    ],
    "overall": {
        "response_1_analysis": "...",
        "response_2_analysis": "...",
        "score_1": <1-5>,
        "score_2": <1-5>,
        "ranking": <1-6>
    }
}
```
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RubricScores:
    rubric_id: int
    score_1: float
    score_2: float
    ranking: float


@dataclass
class ParsedScores:
    score_1: float  # overall score_1
    score_2: float  # overall score_2
    ranking: float  # overall ranking
    rubric_scores: List[RubricScores] = field(default_factory=list)


def extract_scores(output: str) -> Optional[ParsedScores]:
    """Extract scores and ranking from GenRM model output.

    Searches for JSON in the output text, trying:
    1. Fenced JSON blocks (```json {...} ```)
    2. Any {...} JSON objects, taking the last valid one

    Expects the unified format with "rubric_evaluations" and "overall" keys.

    Args:
        output: Text output from the GenRM model (message content only).

    Returns:
        ParsedScores if overall scores were found and are valid floats.
        None otherwise.
    """
    parsed = _try_parse_json(output)

    if parsed is None:
        logger.warning("No parseable JSON found in GenRM output: %s", output)
        return None

    try:
        # Extract overall scores
        overall = parsed.get("overall")
        if not isinstance(overall, dict):
            logger.warning("'overall' is not a dict (got %s) in GenRM output: %s", type(overall).__name__, output)
            return None
        if not all(k in overall for k in ("score_1", "score_2", "ranking")):
            logger.warning("'overall' missing required keys (got %s) in GenRM output: %s", list(overall.keys()), output)
            return None

        # Extract per-rubric scores — must be present and all parseable
        raw_rubrics = parsed.get("rubric_evaluations")
        if not isinstance(raw_rubrics, list) or len(raw_rubrics) == 0:
            logger.warning("'rubric_evaluations' missing or empty (got %s) in GenRM output: %s", type(raw_rubrics).__name__, output)
            return None

        rubric_scores = []
        for rs in raw_rubrics:
            rubric_scores.append(RubricScores(
                rubric_id=int(rs["rubric_id"]),
                score_1=float(rs["score_1"]),
                score_2=float(rs["score_2"]),
                ranking=float(rs["ranking"]),
            ))

        return ParsedScores(
            score_1=float(overall["score_1"]),
            score_2=float(overall["score_2"]),
            ranking=float(overall["ranking"]),
            rubric_scores=rubric_scores,
        )
    except (TypeError, ValueError, KeyError) as e:
        logger.warning("Failed to extract scores (%s: %s) from GenRM output: %s", type(e).__name__, e, output)
        return None


def _try_parse_json(output: str) -> Optional[dict]:
    """Try to extract a valid JSON dict from model output text.

    Tries fenced JSON blocks first, then balanced brace extraction.
    Handles nested JSON (arrays/objects inside the top-level object).
    """
    # Strategy 1: fenced JSON blocks — extract content between ```json and ```
    for match in re.finditer(r"```json\s*([\s\S]*?)\s*```", output, flags=re.IGNORECASE):
        # Find the outermost balanced {} within the fenced block
        for candidate in _find_balanced_braces(match.group(1)):
            result = _safe_json_loads(candidate)
            if result is not None:
                return result

    # Strategy 2: find balanced { ... } blocks in the full text
    last_valid = None
    for candidate in _find_balanced_braces(output):
        result = _safe_json_loads(candidate)
        if result is not None:
            last_valid = result

    return last_valid


def _find_balanced_braces(text: str) -> list:
    """Find all top-level balanced { ... } substrings in text."""
    results = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                results.append(text[start:i + 1])
                start = -1

    return results


def _fix_invalid_escapes(text: str) -> str:
    r"""Replace invalid JSON escape sequences with escaped backslashes.

    JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
    Model outputs often contain LaTeX (e.g. \Delta, \ln) which are invalid.
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def _safe_json_loads(text: str) -> Optional[dict]:
    """Parse JSON string, returning None on failure or if not a dict."""
    for candidate in [text, _fix_invalid_escapes(text)]:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None
