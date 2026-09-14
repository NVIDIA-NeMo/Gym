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
"""Split an IOI statement into one subtask-targeted variant per scoring row.

- `| Subtask | Score | Additional Constraints |` -- five of the six tasks. One
  variant per row; rows are positionally aligned with `subtasks/*.json`.
- `| Subtasks | $S$ | $P$ |` -- `magiccity` only, whose 50 subtasks differ solely
  in `K` and are published as eight banded rows (`6 - 10`, `13 - 50`, ...). We
  follow the statement and emit eight variants, not fifty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SCORING_HEADING = re.compile(r"^##\s+.*(?:subtask|scoring).*$", re.IGNORECASE)
NEXT_H2 = re.compile(r"^##\s")
CONSTRAINTS_HEADING = re.compile(r"^##\s+Constraints\s*$", re.IGNORECASE)

TARGETING_NOTE = (
    "Do note that you DO NOT necessarily have to solve for the general case, "
    "but only for the subproblem defined by the following constraints:"
)

NO_EXTRA = re.compile(r"no\s+additional\s+constraints", re.IGNORECASE)
#: `6 - 10`, `13 – 50`, or a bare `1`.
BAND = re.compile(r"^\s*(\d+)\s*(?:[-–—]\s*(\d+))?\s*$")


@dataclass
class Variant:
    """One subtask-targeted statement."""

    label: str  # the row's own label, e.g. "1" or "6 - 10"
    score: float  # points as printed in the table
    constraint: str  # additional-constraints text ("" when there are none)
    statement: str  # the rewritten markdown
    band: tuple[int, int] | None = None  # inclusive subtask range, magiccity only


def _section(lines: list[str], predicate) -> tuple[int, int] | None:
    """[start, end) of the first `## ` section whose heading matches."""
    start = next((i for i, line in enumerate(lines) if predicate(line)), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines)) if NEXT_H2.match(lines[j])), len(lines))
    return start, end


def _first_table(block: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """(header lines, data rows, everything else) for the FIRST table in `block`.

    Only the first table defines subtasks. A scoring section often carries a
    second one -- classroom's partial-credit `| Condition | $X$ |`, ballmachine's
    per-subtask limits under `### Subtask 3` -- which applies to every subtask and
    must survive verbatim into each variant.
    """
    start = next((i for i, line in enumerate(block) if line.strip().startswith("|")), None)
    if start is None:
        return [], [], list(block)
    end = start
    while end < len(block) and (block[end].strip().startswith("|") or not block[end].strip()):
        if not block[end].strip() and end > start:
            break
        end += 1

    header, rows = [], []
    for line in block[start:end]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if set(cells[0]) <= set(":- ") or cells[0].lower().startswith("subtask"):
            header.append(stripped)
            continue
        rows.append(cells)
    return header, rows, block[:start] + block[end:]


def _rewrite_constraints(lines: list[str], constraint: str) -> list[str]:
    """Insert the targeting note and append the subtask's own constraint."""
    span = _section(lines, CONSTRAINTS_HEADING.match)
    if span is None:
        return lines
    start, end = span
    body = lines[start + 1 : end]

    # Append after the last bullet so it reads as one list.
    last_bullet = max((i for i, line in enumerate(body) if line.lstrip().startswith("*")), default=-1)
    added = [f"* {constraint}"] if constraint else []
    if last_bullet >= 0:
        body = body[: last_bullet + 1] + added + body[last_bullet + 1 :]
    else:
        body = added + body

    return lines[: start + 1] + ["", TARGETING_NOTE] + body + lines[end:]


def _reduced_scoring(header: str, table_header: list[str], row: list[str], extra: list[str]) -> list[str]:
    """The scoring section cut down to the targeted row, with any prose kept."""
    out = ["", header, ""]
    out.extend(extra)
    if table_header:
        out.extend(table_header)
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return out


def split(statement: str) -> list[Variant]:
    """One Variant per scoring row. Returns [] if no scoring table is found."""
    lines = statement.splitlines()
    span = _section(lines, lambda line: bool(SCORING_HEADING.match(line)))
    if span is None:
        return []
    start, end = span
    heading = lines[start]
    block = lines[start + 1 : end]
    table_header, rows, prose = _first_table(block)
    if not rows:
        return []

    banded = (
        "additional constraints" not in " ".join(rows[0]).lower()
        and len(rows[0]) >= 3
        and any(BAND.match(r[0]) and BAND.match(r[0]).group(2) for r in rows)
    )

    variants: list[Variant] = []
    for row in rows:
        label = row[0]
        number = re.search(r"\d+(?:\.\d+)?", row[1])
        score_text = number.group(0) if number else "0"

        if banded:
            m = BAND.match(label)
            if not m:
                continue
            low = int(m.group(1))
            high = int(m.group(2) or m.group(1))
            band = (low, high)
            # Derived, not lifted: the section states subtask i uses K = i.
            constraint = f"$K = {low}$" if low == high else f"$K$ is an integer with ${low} \\leq K \\leq {high}$"
            score = float(score_text) * (high - low + 1)
        else:
            band = None
            constraint = "" if NO_EXTRA.search(row[-1]) else row[-1]
            score = float(score_text)

        rewritten = _rewrite_constraints(list(lines), constraint)
        # Recompute the scoring span: rewriting constraints shifted the lines.
        new_span = _section(rewritten, lambda line: bool(SCORING_HEADING.match(line)))
        if new_span is not None:
            s, e = new_span
            rewritten = rewritten[:s] + _reduced_scoring(heading, table_header, row, prose) + rewritten[e:]

        variants.append(
            Variant(label=label, score=score, constraint=constraint, statement="\n".join(rewritten), band=band)
        )
    return variants
