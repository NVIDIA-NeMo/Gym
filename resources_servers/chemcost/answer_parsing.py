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
# Adapted from ChemCost, https://github.com/HeinzVonHank/ChemCost at d7de381d782835e51840d7d371586d3648836cfd:
# ReActAgent._try_parse_final_answer in src/chemcost/baselines/react_agent.py. NVIDIA modifications: one
# brace-matching pass instead of a scan per brace, possessive quantifiers in the fallback pattern, a bound on the
# text handed to json, and integer-digit and nesting errors treated as invalid output.
# The original code carries this notice:
#
# Copyright (c) 2026 The ChemCost Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The upstream final-answer parser, with the same selection and bounded work.

Upstream scans forward from every ``{`` to its depth-matched ``}`` (quadratic in opening braces), and its
fallback pattern backtracks over long whitespace runs. Here one stack pass finds the same candidate spans, and
possessive quantifiers on every whitespace run match the same strings without backtracking (each run is followed
by a non-whitespace token). Two departures, both only on pathological output: json sees at most
``MAX_JSON_CHARS`` in total (balanced but invalid nested candidates would otherwise be decoded again and again),
and a candidate json refuses (too deep, an integer over Python's digit limit) gives no answer, as upstream's
evaluator records the error its parser raises. Both end as an abstention.
"""

import bisect
import json
import re
from typing import Optional


KEY = "predicted_cost_per_gram"
# A real final answer is under 1 kB; a run of failed candidates past this many characters is not an answer.
MAX_JSON_CHARS = 1_000_000
_COST_PATTERN = re.compile(
    r"(?:cost\s+per\s+gram|per[-\s]gram\s+cost|predicted_cost_per_gram)"
    r"[\s\"*]*+(?:\([^)]{0,20}\))?[\s\"*]*+[:=][\s*]*+(?:USD\s*+)?\$?\s*+"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def _brace_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of every ``{`` with a depth-matched ``}``, as upstream's per-start scan finds them, by start."""
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            spans.append((stack.pop(), i))
    spans.sort()
    return spans


def parse_final_answer(text: Optional[str]) -> Optional[dict]:
    """Upstream ``ReActAgent._try_parse_final_answer``: the last JSON object holding the key, else a labelled figure."""
    if not isinstance(text, str) or not text:
        return None
    key_starts = [m.start() for m in re.finditer(re.escape(KEY), text)]
    budget = MAX_JSON_CHARS
    found = None
    # Upstream tries every key-bearing object by start and keeps the last one that parses.
    for start, end in _brace_spans(text):
        # The span holds the key when some occurrence starts at or after `start` and ends by `end`.
        first = bisect.bisect_left(key_starts, start)
        if first == len(key_starts) or key_starts[first] + len(KEY) - 1 > end:
            continue
        budget -= end + 1 - start
        if budget < 0:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
        except (ValueError, RecursionError):
            # Upstream raises here and its evaluator records a null prediction; an earlier draft never wins.
            return None
        if isinstance(parsed, dict):
            parsed.setdefault("predicted_components", [])
            parsed.pop("_harness_error", None)
            found = parsed
    if found is not None:
        return found
    cost_matches = _COST_PATTERN.findall(text)
    if cost_matches:
        return {"predicted_cost_per_gram": float(cost_matches[-1].replace(",", "")), "predicted_components": []}
    return None
