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
"""Secondary structure as a set of base pairs.

Pair sets, not strings, so the same pair written in a different bracket family still matches.
"""

from __future__ import annotations


BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


def to_pairs(structure: str) -> set[tuple[int, int]]:
    """Base pairs as 0-indexed (i, j) with i < j, across every bracket family."""
    pairs: set[tuple[int, int]] = set()
    for opener, closer in BRACKETS:
        stack: list[int] = []
        for index, char in enumerate(structure):
            if char == opener:
                stack.append(index)
            elif char == closer:
                if not stack:
                    raise ValueError(f"unbalanced {closer!r} at position {index}")
                pairs.add((stack.pop(), index))
        if stack:
            raise ValueError(f"unbalanced {opener!r} at position {stack[0]}")
    return pairs


def is_crossing(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when two pairs cross (a pseudoknot)."""
    (i, j), (m, n) = a, b
    return i < m < j < n or m < i < n < j


def compare(predicted: str, target: str) -> dict:
    """Agreement between two structures, overall and over crossing pairs only.

    Crossing pairs are reported separately because a nested-only folding engine cannot express them.
    """
    if len(predicted) != len(target):
        raise ValueError(f"length mismatch: predicted {len(predicted)}, target {len(target)}")
    got, want = to_pairs(predicted), to_pairs(target)
    want_crossing = {p for p in want if any(is_crossing(p, q) for q in want)}

    def f1(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        shared = len(a & b)
        if not shared:
            return 0.0
        precision, recall = shared / len(a), shared / len(b)
        return 2 * precision * recall / (precision + recall)

    overall = round(f1(got, want), 4)
    # None, not 0.0, when the target has no crossing pair, or a perfect nested solve looks half wrong.
    # Precision counts only the prediction's own crossing pairs.
    got_crossing = {p for p in got if any(is_crossing(p, q) for q in got)}
    crossed = round(f1(got_crossing, want_crossing), 4) if want_crossing else None
    return {
        "f1": overall,
        "crossed_pair_f1": crossed,
        "openknot_proxy": openknot_proxy(overall, crossed),
        "n_pairs_predicted": len(got),
        "n_pairs_target": len(want),
        "n_shared": len(got & want),
        "crossing_recall": round(len(got & want_crossing) / len(want_crossing), 4) if want_crossing else None,
        "n_crossing_target": len(want_crossing),
        "exact": got == want,
    }


def to_dot_bracket(pairs: list[tuple[int, int]], length: int) -> str:
    """Encode a pair set as extended dot-bracket, keeping the crossings.

    A plain single-family dot-bracket drops pseudoknot pairs, so encode from the pair list instead.
    """
    families = "()", "[]", "{}", "<>"
    assigned: list[list[tuple[int, int]]] = [[] for _ in families]
    out = ["."] * length
    for i, j in sorted((min(a, b), max(a, b)) for a, b in pairs):
        for layer, taken in enumerate(assigned):
            if not any(is_crossing((i, j), other) for other in taken):
                taken.append((i, j))
                out[i], out[j] = families[layer]
                break
        else:
            raise ValueError(f"pair ({i}, {j}) crosses every available bracket family")
    return "".join(out)


def openknot_proxy(overall: float, crossed: float | None) -> float:
    """Mean of overall and crossing-pair F1, mirroring the shape of the OpenKnot score.

    The real score needs SHAPE reactivity data, so only its half-weight on crossing pairs is reproduced.
    With no crossing pairs in the target, it is the overall F1.
    """
    if crossed is None:
        return overall
    return round((overall + crossed) / 2, 4)
