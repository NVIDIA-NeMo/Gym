# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row contract of prepare.py against synthetic workbooks: ids, decoding fields, and the
two ways an upstream change would silently produce a different benchmark."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest

from benchmarks.safe_child_llm.prepare import EXPECTED_ROWS_PER_SPLIT, UPSTREAM_REVISION, _read_first_sheet, _render


def _workbook(rows: list[dict[str, str]]) -> bytes:
    """A minimal XLSX with inline strings, the shape prepare.py reads from the released files."""
    headers = list(rows[0])

    def cell(reference: str, value: str) -> str:
        return f'<c r="{reference}" t="inlineStr"><is><t>{value}</t></is></c>'

    def column(index: int) -> str:
        return chr(ord("A") + index)

    lines = ['<row r="1">' + "".join(cell(f"{column(i)}1", h) for i, h in enumerate(headers)) + "</row>"]
    for number, row in enumerate(rows, start=2):
        lines.append(
            f'<row r="{number}">'
            + "".join(cell(f"{column(i)}{number}", row[h]) for i, h in enumerate(headers))
            + "</row>"
        )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        + "".join(lines)
        + "</sheetData></worksheet>"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def _split(prefix: str, count: int = EXPECTED_ROWS_PER_SPLIT) -> list[dict[str, str]]:
    return [
        {"Index": str(i), "category": "Adult Content", "query": f"{prefix} prompt {i}", "source": "DoNotAnswer"}
        for i in range(1, count + 1)
    ]


def test_read_first_sheet_returns_header_keyed_rows() -> None:
    rows = _read_first_sheet(_workbook(_split("a", 3)))
    assert rows == _split("a", 3)


def test_render_builds_one_row_per_prompt_with_the_paper_decoding() -> None:
    output = _render({"6-12": _workbook(_split("young")), "13-17": _workbook(_split("teen"))})
    rows = [json.loads(line) for line in output.splitlines()]
    assert len(rows) == 2 * EXPECTED_ROWS_PER_SPLIT
    first = rows[0]
    assert first["safe_child_id"] == "safe-child-6-12-001" and first["age_group"] == "6-12"
    assert first["responses_create_params"] == {
        "input": [{"role": "user", "content": "young prompt 1"}],
        "temperature": 0.0,
        "max_output_tokens": 1024,
    }
    assert first["prompt"] == "young prompt 1" and first["category"] == "Adult Content"
    assert first["source"] == "DoNotAnswer" and first["upstream_revision"] == UPSTREAM_REVISION
    assert rows[EXPECTED_ROWS_PER_SPLIT]["safe_child_id"] == "safe-child-13-17-001"
    assert len({row["safe_child_id"] for row in rows}) == len(rows)


def test_render_refuses_a_split_of_the_wrong_size() -> None:
    with pytest.raises(ValueError, match="expected 100"):
        _render({"6-12": _workbook(_split("a", 99)), "13-17": _workbook(_split("b"))})


def test_render_refuses_duplicate_ids() -> None:
    rows = _split("a")
    rows[1]["Index"] = "1"
    with pytest.raises(ValueError, match="duplicate"):
        _render({"6-12": _workbook(rows), "13-17": _workbook(_split("b"))})
