# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A deliverable's share of the judge's context must not depend on its filename."""

from pathlib import Path

import pytest

from responses_api_agents.stirrup_agent.file_reader import (
    MAX_TEXT_BLOCK_CHARS,
    MAX_TOTAL_CHARS,
    MAX_TOTAL_TEXT_BLOCK_CHARS,
    _fair_text_allowances,
    _read_pptx,
    convert_deliverables_to_content_blocks,
    read_deliverable_files,
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


def test_hard_budget_counts_headers_and_truncation_markers(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    for i in range(100):
        (d / f"long_header_name_{i:03d}.txt").write_text("X" * 20_000)

    blocks = convert_deliverables_to_content_blocks(str(d))

    assert sum(len(block.get("text", "")) for block in blocks if block.get("type") == "text") <= (
        MAX_TOTAL_TEXT_BLOCK_CHARS
    )


def test_hard_budget_counts_pdf_extractor_text(monkeypatch, tmp_path: Path):
    import responses_api_agents.stirrup_agent.file_reader as file_reader

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"fake pdf")

    def _render(*args, **kwargs):
        return [
            {"type": "text", "text": "P" * (MAX_TOTAL_TEXT_BLOCK_CHARS * 2)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        ]

    monkeypatch.setattr(file_reader, "_pdf_bytes_to_image_text_blocks", _render)
    blocks = convert_deliverables_to_content_blocks(str(tmp_path), media_mode="images_and_text")
    text_blocks = [block["text"] for block in blocks if block.get("type") == "text"]

    assert sum(map(len, text_blocks)) <= MAX_TOTAL_TEXT_BLOCK_CHARS
    assert "aggregate text budget exhausted" in "".join(text_blocks)


def test_plain_text_summary_includes_its_marker_within_the_cap(tmp_path: Path):
    (tmp_path / "report.txt").write_text("R" * (MAX_TOTAL_CHARS * 2))

    output = read_deliverable_files(str(tmp_path))

    assert len(output) <= MAX_TOTAL_CHARS
    assert output.endswith("[...truncated]")


def test_empty_text_deliverable_is_not_mislabeled_as_budget_omitted(tmp_path: Path):
    (tmp_path / "empty.txt").write_bytes(b"")

    output = _text_of(convert_deliverables_to_content_blocks(str(tmp_path)))

    assert "empty.txt" in output
    assert "present but EMPTY" in output
    assert "budget exhausted" not in output


def test_pptx_text_that_fits_the_budget_is_returned(tmp_path: Path):
    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Quarterly evidence"
    path = tmp_path / "report.pptx"
    presentation.save(path)

    assert "Quarterly evidence" in _read_pptx(path, max_chars=1_000)


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


def test_text_paths_do_not_use_unbounded_path_read_text(monkeypatch, tmp_path: Path):
    import responses_api_agents.stirrup_agent.file_reader as file_reader

    (tmp_path / "large.txt").write_text("prefix evidence\n" + "X" * 1_000_000)

    def _unbounded_read_forbidden(*args, **kwargs):
        raise AssertionError("Path.read_text would load the whole deliverable")

    monkeypatch.setattr(Path, "read_text", _unbounded_read_forbidden)

    assert "prefix evidence" in _text_of(convert_deliverables_to_content_blocks(str(tmp_path)))
    assert "prefix evidence" in file_reader.read_deliverable_files(str(tmp_path))


def test_rubric_attachment_budget_rejects_before_read(monkeypatch, tmp_path: Path):
    import responses_api_agents.stirrup_agent.file_reader as file_reader

    (tmp_path / "Plan.docx").write_bytes(b"source")
    sidecar = tmp_path / "Plan.docx.pdf"
    sidecar.write_bytes(b"render-too-large")
    monkeypatch.setattr(file_reader, "MAX_TOTAL_RAW_ATTACHMENT_BYTES", 4)
    monkeypatch.setattr(file_reader, "MAX_TOTAL_ENCODED_ATTACHMENT_CHARS", 8)
    original_read_bytes = Path.read_bytes

    def _guarded_read(path: Path):
        if path == sidecar:
            raise AssertionError("rejected Office sidecar must not be read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _guarded_read)

    blocks = convert_deliverables_to_content_blocks(str(tmp_path))

    assert "attachment omitted" in _text_of(blocks)


def test_rubric_attachment_budget_is_aggregate(monkeypatch, tmp_path: Path):
    import responses_api_agents.stirrup_agent.file_reader as file_reader

    (tmp_path / "a.png").write_bytes(b"AAAA")
    (tmp_path / "b.png").write_bytes(b"BBBB")
    monkeypatch.setattr(file_reader, "MAX_TOTAL_RAW_ATTACHMENT_BYTES", 4)
    monkeypatch.setattr(file_reader, "MAX_TOTAL_ENCODED_ATTACHMENT_CHARS", 8)

    blocks = convert_deliverables_to_content_blocks(str(tmp_path))

    attachments = [block for block in blocks if block.get("type") == "image_url"]
    assert len(attachments) == 1
    assert "attachment omitted" in _text_of(blocks)


def test_pdf_source_size_does_not_consume_rendered_page_budget(monkeypatch, tmp_path: Path):
    import responses_api_agents.stirrup_agent.file_reader as file_reader

    pdf = tmp_path / "compressed.pdf"
    pdf.write_bytes(b"source-is-larger-than-output-budget")
    monkeypatch.setattr(file_reader, "MAX_TOTAL_RAW_ATTACHMENT_BYTES", 4)
    monkeypatch.setattr(file_reader, "MAX_TOTAL_ENCODED_ATTACHMENT_CHARS", 8)

    def _render(data, *, attachment_budget, **_kwargs):
        assert data == pdf.read_bytes()
        assert attachment_budget.reserve(1)
        return [{"type": "image_url", "image_url": {"url": "data:image/png;base64,UA=="}}]

    monkeypatch.setattr(file_reader, "_pdf_bytes_to_image_text_blocks", _render)

    blocks = convert_deliverables_to_content_blocks(str(tmp_path), media_mode="images_and_text")

    assert any(block.get("type") == "image_url" for block in blocks)
    assert "attachment omitted" not in _text_of(blocks)
