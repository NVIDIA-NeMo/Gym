# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A deliverable's share of the judge's context must not depend on its filename."""

from pathlib import Path

from responses_api_agents.stirrup_agent.file_reader import (
    MAX_TEXT_BLOCK_CHARS,
    MAX_TOTAL_TEXT_BLOCK_CHARS,
    _fair_text_allowances,
    convert_deliverables_to_content_blocks,
)


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def test_small_deliverable_is_not_starved_by_a_large_one_sorting_first(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "000_huge.txt").write_text("A" * 500_000)
    (d / "zzz_report.md").write_text("THE REAL DELIVERABLE\n" * 50)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "THE REAL DELIVERABLE" in out
    assert "truncated" in out


def test_allotment_does_not_depend_on_filename_order(tmp_path: Path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "000_huge.txt").write_text("A" * 500_000)
    (a / "zzz_report.md").write_text("PAYLOAD\n" * 50)

    b = tmp_path / "b"
    b.mkdir()
    (b / "zzz_huge.txt").write_text("A" * 500_000)
    (b / "000_report.md").write_text("PAYLOAD\n" * 50)

    out_a = _text_of(convert_deliverables_to_content_blocks(str(a)))
    out_b = _text_of(convert_deliverables_to_content_blocks(str(b)))
    assert ("PAYLOAD" in out_a) == ("PAYLOAD" in out_b)


def test_aggregate_budget_bounds_the_request(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    for i in range(60):
        (d / f"f{i:03d}.txt").write_text("B" * 40_000)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert len(out) <= MAX_TOTAL_TEXT_BLOCK_CHARS * 1.15


def test_files_dropped_for_budget_are_named_not_silently_lost(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    for i in range(60):
        (d / f"f{i:03d}.txt").write_text("B" * 40_000)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert all(f"f{i:03d}.txt" in out for i in range(0, 60, 7)) or "omitted" in out


def test_surplus_from_small_files_is_redistributed():
    assert _fair_text_allowances([10, 10, 1_000_000], 300, 200) == [10, 10, 200]


def test_equal_files_split_equally():
    al = _fair_text_allowances([100, 100, 100], 150, 200)
    assert sum(al) <= 150
    assert max(al) - min(al) <= 1


def test_per_file_cap_is_never_exceeded():
    assert _fair_text_allowances([10**9], 10**9, 1234) == [1234]


def test_allowance_of_zero_files_is_empty():
    assert _fair_text_allowances([], 1000, 100) == []


def test_sniffed_text_file_receives_an_allowance(tmp_path: Path):
    """Deciding 'is this text' twice would emit it while allotting it nothing."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Makefile").write_text("all:\n\tcc x.c\n" * 20)

    assert "cc x.c" in _text_of(convert_deliverables_to_content_blocks(str(d)))


def test_a_single_file_may_use_the_per_file_cap(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "only.txt").write_text("C" * (MAX_TEXT_BLOCK_CHARS + 5_000))

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "only.txt" in out
    assert "truncated" in out
