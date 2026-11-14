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

"""Grid parsing shared by the verifier and the ARC code agent.

A final answer is a block ``<answer> ... </answer>`` holding one grid row per line, digits separated by single
spaces. The last block of a reply is read. A block that is empty, has blank lines, has anything other than single
digits, or has rows of different lengths is not an answer.
"""

import re
from typing import List, Optional, Sequence


Grid = List[List[int]]

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalized_grid(grid: Sequence[Sequence[int]]) -> str:
    """Canonical text of a grid: one row per line, digits packed without separators."""
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def is_grid(value: object) -> bool:
    """True for a non-empty rectangular list of lists of ints in 0-9."""
    if not isinstance(value, list) or not value or not all(isinstance(r, list) and r for r in value):
        return False
    return len({len(r) for r in value}) == 1 and all(
        isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in value for v in r
    )


def unreadable_reason(text: str) -> Optional[str]:
    """Why the last <answer> block of ``text`` cannot be read as a grid; None when there is no block or it reads."""
    blocks = _ANSWER.findall(text or "")
    if not blocks:
        return None
    rows = [line.split() for line in blocks[-1].strip().splitlines()]
    if not rows or not all(rows):
        return "it is empty or has blank lines"
    if any(not (len(c) == 1 and c.isdigit()) for r in rows for c in r):
        return "it contains something other than single digits separated by spaces"
    widths = sorted({len(r) for r in rows})
    if len(widths) > 1:
        return f"the rows have different lengths ({', '.join(map(str, widths))} cells)"
    return None


def parse_answer(text: str) -> Optional[Grid]:
    """The grid in the last <answer> block of ``text``, or None if there is no readable block."""
    blocks = _ANSWER.findall(text or "")
    if not blocks:
        return None
    rows: Grid = []
    for line in blocks[-1].strip().splitlines():
        cells = line.split()
        if not cells or any(not (len(c) == 1 and c.isdigit()) for c in cells):
            return None
        rows.append([int(c) for c in cells])
    if not rows or len({len(r) for r in rows}) != 1:
        return None
    return rows
