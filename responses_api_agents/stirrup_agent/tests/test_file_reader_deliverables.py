# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run state and derived artifacts must never be mistaken for deliverables."""

import base64
from pathlib import Path

import pytest

import responses_api_agents.stirrup_agent.file_reader as file_reader
from responses_api_agents.stirrup_agent.file_reader import (
    IGNORE_FILES,
    convert_deliverables_to_content_blocks,
    is_deliverable,
    read_deliverable_files,
)


def _run_state(d: Path) -> None:
    (d / "finish_params.json").write_text('{"summary": "did the thing"}')
    (d / "history.json").write_text('[{"role": "assistant", "content": "internal reasoning"}]')
    (d / "history.pkl").write_bytes(b"\x80\x04\x95")
    (d / "inprogress_history.json").write_text("[]")
    (d / "metadata.json").write_text('{"_model_speed": {"num_calls": 42}}')
    (d / "log.txt").write_text("2026-01-01 agent started\n")


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def test_run_state_is_not_a_deliverable(tmp_path: Path):
    _run_state(tmp_path)
    for name in IGNORE_FILES:
        p = tmp_path / name
        if p.exists():
            assert not is_deliverable(p), f"{name} would be graded as work product"


def test_real_deliverable_still_is_one(tmp_path: Path):
    (tmp_path / "report.md").write_text("# findings\n")
    assert is_deliverable(tmp_path / "report.md")


def test_run_state_never_reaches_the_judge_blocks(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    _run_state(d)
    (d / "report.md").write_text("# real work\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "report.md" in out
    for name in ("finish_params.json", "history.json", "metadata.json", "log.txt"):
        assert name not in out, f"{name} was shown to the judge"
    assert "internal reasoning" not in out
    assert "did the thing" not in out


def test_run_state_never_reaches_the_text_extractor(tmp_path: Path):
    _run_state(tmp_path)
    (tmp_path / "report.md").write_text("# real work\n")

    out = read_deliverable_files(str(tmp_path))
    assert "real work" in out
    assert "internal reasoning" not in out
    assert "finish_params.json" not in out


def test_reference_files_directory_is_not_a_deliverable(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "reference_files").mkdir()
    (d / "reference_files" / "given.pdf").write_bytes(b"%PDF-1.4\n")
    (d / "answer.md").write_text("# answer\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "answer.md" in out
    assert "given.pdf" not in out, "an input was graded as the model's output"


def test_a_directory_named_like_run_state_is_still_excluded(tmp_path: Path):
    assert not is_deliverable(tmp_path / "reference_files")


def test_office_sidecar_scan_ignores_run_state(tmp_path: Path):
    """A run-state file must not be treated as an Office source when consuming PDFs."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    _run_state(d)
    (d / "Plan.docx").write_bytes(b"PK\x03\x04")
    (d / "Plan.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "Plan.docx" in out
    assert "Plan.pdf:" not in out, "the consumed sibling was emitted standalone"


def test_ignore_list_covers_every_file_the_agent_writes(tmp_path: Path):
    """Guard against the set drifting from what the agent actually produces."""
    for name in ("finish_params.json", "history.json", "history.pkl", "metadata.json", "log.txt"):
        assert name in IGNORE_FILES


def test_same_stem_office_sidecars_are_emitted_once_and_stale_pdf_is_quarantined(tmp_path: Path):
    (tmp_path / "Plan.docx").write_bytes(b"docx source")
    (tmp_path / "Plan.pptx").write_bytes(b"pptx source")
    (tmp_path / "Plan.docx.pdf").write_bytes(b"DOCX RENDER")
    (tmp_path / "Plan.pptx.pdf").write_bytes(b"PPTX RENDER")
    (tmp_path / "Plan.pdf").write_bytes(b"STALE COLLIDED RENDER")
    (tmp_path / "Appendix.pdf").write_bytes(b"INDEPENDENT PDF")

    blocks = convert_deliverables_to_content_blocks(str(tmp_path))
    attachments = [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in blocks
        if block.get("type") == "image_url"
    ]

    assert attachments == [b"INDEPENDENT PDF", b"DOCX RENDER", b"PPTX RENDER"]
    text = _text_of(blocks)
    assert "Plan.docx" in text
    assert "Plan.pptx" in text
    assert "Plan.pdf:" not in text
    assert "Plan.docx.pdf" not in text
    assert "Plan.pptx.pdf" not in text


def test_ambiguous_plain_pdf_is_ignored_while_each_office_source_is_rendered(monkeypatch, tmp_path: Path):
    (tmp_path / "Plan.docx").write_bytes(b"docx source")
    (tmp_path / "Plan.pptx").write_bytes(b"pptx source")
    (tmp_path / "Plan.pdf").write_bytes(b"STALE COLLIDED RENDER")

    def _render(source: Path, out_dir: Path | None = None) -> Path:
        assert out_dir is not None
        rendered = out_dir / f"{source.stem}.pdf"
        rendered.write_bytes(source.suffix.upper().encode())
        return rendered

    monkeypatch.setattr(file_reader, "_convert_office_to_pdf", _render)

    blocks = convert_deliverables_to_content_blocks(str(tmp_path))
    attachments = [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in blocks
        if block.get("type") == "image_url"
    ]

    assert attachments == [b".DOCX", b".PPTX"]
    assert b"STALE COLLIDED RENDER" not in attachments
    assert "Plan.pdf:" not in _text_of(blocks)


def test_xlsx_emits_formula_text_alongside_rendered_pdf(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Forecast"
    sheet["A1"] = 4
    sheet["A2"] = 8
    sheet["A3"] = "=SUM(A1:A2)"
    xlsx = tmp_path / "Budget.xlsx"
    workbook.save(xlsx)
    workbook.close()
    (tmp_path / "Budget.xlsx.pdf").write_bytes(b"PDF RENDER")

    blocks = convert_deliverables_to_content_blocks(str(tmp_path))

    assert any(block.get("type") == "image_url" for block in blocks)
    text = _text_of(blocks)
    assert "structured spreadsheet cells" in text
    assert "Sheet: Forecast" in text
    assert "A3: formula: =SUM(A1:A2)" in text
