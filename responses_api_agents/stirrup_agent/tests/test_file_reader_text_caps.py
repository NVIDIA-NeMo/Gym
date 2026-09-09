# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One oversized text deliverable must not cost the whole judgement.

``convert_deliverables_to_content_blocks`` fed text files to the judge uncapped.
A 2.8 MB file became a 720,898-token request against a 786,432-token judge; the
judge returned 400, the resources server turned that into a 500, and the task
was recorded as a *transient* failure with no score at all. Bounding the text
loses a tail; not bounding it loses the task.
"""

from pathlib import Path

from responses_api_agents.stirrup_agent.file_reader import (
    MAX_TEXT_BLOCK_CHARS,
    MAX_TOTAL_TEXT_BLOCK_CHARS,
    convert_deliverables_to_content_blocks,
)


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def test_oversized_text_deliverable_is_truncated_not_dropped(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    huge = "A" * (MAX_TEXT_BLOCK_CHARS * 3)
    (d / "Notes.md").write_text(huge)

    blocks = convert_deliverables_to_content_blocks(str(d))
    joined = _text_of(blocks)

    assert "Notes.md" in joined, "the judge must still be told the deliverable exists"
    assert "truncated" in joined
    # The whole point: the payload is bounded rather than 3x the per-file cap.
    assert len(joined) < MAX_TEXT_BLOCK_CHARS * 1.1


def test_normal_deliverable_is_passed_through_untouched(tmp_path: Path):
    """The largest genuine text deliverable in the corpus is ~101 KB — well under the cap."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    body = "Revenue grew 12%.\n" * 5_000  # ~90 KB, realistic
    (d / "Report.md").write_text(body)

    joined = _text_of(convert_deliverables_to_content_blocks(str(d)))

    assert body.strip() in joined, "a real deliverable must not be trimmed"
    assert "truncated" not in joined
    assert "omitted" not in joined


def test_many_files_cannot_blow_the_request_between_them(tmp_path: Path):
    """A per-file cap alone is not enough — N files at the cap still exceed any context."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    for i in range(6):
        (d / f"part_{i}.txt").write_text("B" * MAX_TEXT_BLOCK_CHARS)

    blocks = convert_deliverables_to_content_blocks(str(d))
    joined = _text_of(blocks)

    assert len(joined) < MAX_TOTAL_TEXT_BLOCK_CHARS * 1.2, "aggregate text budget was not enforced"
    # Every file is still named, and each gets a share rather than the first two
    # consuming everything.
    for i in range(6):
        assert f"part_{i}.txt" in joined
        assert f"part_{i}.txt:\nB" in joined, "every file should get some of the budget"


def test_a_big_file_cannot_starve_a_small_one_by_sorting_first(tmp_path: Path):
    """Allocation must not depend on filename order.

    With a spend-as-you-go budget, ``000_dump.txt`` eats everything and the real
    deliverable that sorts after it is reduced to a marker — so *renaming* a file
    would change the score.
    """
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "000_dump.txt").write_text("X" * (MAX_TOTAL_TEXT_BLOCK_CHARS * 2))
    (d / "001_dump.txt").write_text("Y" * (MAX_TOTAL_TEXT_BLOCK_CHARS * 2))
    report = "Revenue grew 12%.\n" * 100
    (d / "zzz_Report.md").write_text(report)

    joined = _text_of(convert_deliverables_to_content_blocks(str(d)))

    assert report.strip() in joined, "a small real deliverable was starved by larger files"


def test_file_exactly_at_the_cap_is_not_marked_truncated(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "atcap.txt").write_text("Z" * MAX_TEXT_BLOCK_CHARS)

    joined = _text_of(convert_deliverables_to_content_blocks(str(d)))

    assert "truncated" not in joined
    assert joined.count("Z") == MAX_TEXT_BLOCK_CHARS, "an exactly-at-cap file must pass through whole"


def test_empty_and_whitespace_only_files_produce_no_block(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Empty.txt").write_text("")
    (d / "Blank.txt").write_text("   \n\t\n")
    (d / "Real.md").write_text("content")

    joined = _text_of(convert_deliverables_to_content_blocks(str(d)))

    assert "Empty.txt" not in joined and "Blank.txt" not in joined
    assert "content" in joined


def test_office_text_fallback_is_capped_too(tmp_path: Path, monkeypatch):
    """When conversion fails the Office file falls back to text — also uncapped before."""
    from responses_api_agents.stirrup_agent import file_reader

    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Deck.pptx").write_bytes(b"not really a pptx")

    monkeypatch.setattr(file_reader, "_convert_office_to_pdf", lambda fpath, out_dir=None: None)
    monkeypatch.setattr(file_reader, "_extract_text", lambda fpath, ext: "Z" * (MAX_TEXT_BLOCK_CHARS * 3))

    joined = _text_of(file_reader.convert_deliverables_to_content_blocks(str(d)))

    assert "Deck.pptx (text fallback)" in joined
    assert "truncated" in joined
    assert len(joined) < MAX_TEXT_BLOCK_CHARS * 1.1


def test_enough_files_stop_being_named_one_by_one(tmp_path: Path):
    """Headers alone are tokens: past the budget the remainder collapses to a summary.

    Otherwise the framing for a pathological number of files blows the request even
    though every individual body stayed inside its cap.
    """
    d = tmp_path / "repeat_0"
    d.mkdir()
    for i in range(25_000):
        (d / f"f{i:05d}.txt").write_text("x")

    joined = _text_of(convert_deliverables_to_content_blocks(str(d)))

    assert "text deliverable(s) omitted, budget exhausted" in joined
    assert len(joined) < MAX_TOTAL_TEXT_BLOCK_CHARS * 1.2, "framing was not bounded"


def test_binary_deliverables_are_unaffected_by_the_text_budget(tmp_path: Path):
    """Images/PDFs go through the base64 path; the text cap must not suppress them."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Big.txt").write_text("C" * (MAX_TOTAL_TEXT_BLOCK_CHARS * 2))
    (d / "Chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    blocks = convert_deliverables_to_content_blocks(str(d))

    assert any(b.get("type") == "image_url" for b in blocks), "image block was dropped"
