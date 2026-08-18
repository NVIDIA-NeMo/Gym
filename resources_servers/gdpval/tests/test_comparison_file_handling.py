# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GDPVal comparison file provenance and payload bounds."""

from __future__ import annotations

import base64
import shutil
import zipfile
from pathlib import Path

import pytest

from resources_servers.gdpval import comparison


def _attachments(blocks: list[dict]) -> list[bytes]:
    payloads: list[bytes] = []
    for block in blocks:
        if block.get("type") != "image_url":
            continue
        url = block["image_url"]["url"]
        payloads.append(base64.b64decode(url.split(",", 1)[1]))
    return payloads


def _text(blocks: list[dict]) -> str:
    return "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")


def test_same_stem_sidecars_are_used_once_and_stale_plain_pdf_is_suppressed(tmp_path: Path) -> None:
    (tmp_path / "Plan.docx").write_bytes(b"docx source")
    (tmp_path / "Plan.pptx").write_bytes(b"pptx source")
    (tmp_path / "Plan.docx.pdf").write_bytes(b"DOCX RENDER")
    (tmp_path / "Plan.pptx.pdf").write_bytes(b"PPTX RENDER")
    (tmp_path / "Plan.pdf").write_bytes(b"STALE COLLIDED RENDER")
    (tmp_path / "Appendix.pdf").write_bytes(b"INDEPENDENT PDF")

    blocks = comparison.build_file_section(str(tmp_path))

    assert _attachments(blocks) == [b"INDEPENDENT PDF", b"DOCX RENDER", b"PPTX RENDER"]
    text = _text(blocks)
    assert "Plan.docx:" in text
    assert "Plan.pptx:" in text
    assert "Plan.pdf:" not in text
    assert "Plan.docx.pdf:" not in text
    assert "Plan.pptx.pdf:" not in text


def test_ambiguous_plain_pdf_is_never_guessed_as_either_office_render(tmp_path: Path) -> None:
    (tmp_path / "Plan.docx").write_bytes(b"docx source")
    (tmp_path / "Plan.pptx").write_bytes(b"pptx source")
    (tmp_path / "Plan.pdf").write_bytes(b"STALE COLLIDED RENDER")

    blocks = comparison.build_file_section(str(tmp_path))

    assert _attachments(blocks) == []
    assert "Plan.pdf:" not in _text(blocks)


def test_xlsx_emits_formula_cell_text_alongside_pdf(tmp_path: Path) -> None:
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

    blocks = comparison.build_file_section(str(tmp_path))

    assert b"PDF RENDER" in _attachments(blocks)
    text = _text(blocks)
    assert "structured spreadsheet cells" in text
    assert "Sheet: Forecast" in text
    assert "A3: formula: =SUM(A1:A2)" in text
    assert len(text) < comparison.MAX_XLSX_TEXT_CHARS_FOR_JUDGE + 500


def test_oversize_xlsx_does_not_attempt_structured_extraction(tmp_path: Path, monkeypatch) -> None:
    xlsx = tmp_path / "huge.xlsx"
    xlsx.touch()
    xlsx.write_bytes(b"x")
    with xlsx.open("r+b") as stream:
        stream.truncate(comparison.MAX_FILE_BYTES_FOR_JUDGE + 1)

    def _unexpected(*args, **kwargs):
        raise AssertionError("oversize XLSX must not be parsed")

    monkeypatch.setattr(comparison, "extract_xlsx_structured_text", _unexpected)

    blocks = comparison.build_file_section(str(tmp_path))

    assert "oversize" in _text(blocks)


def _data_block(payload: bytes) -> dict:
    encoded = base64.b64encode(payload).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{encoded}"}}


def test_all_fit_attachments_do_not_require_payload_fingerprints(monkeypatch) -> None:
    def _unexpected_fingerprint(_block):
        raise AssertionError("all-fit payloads must not be hashed")

    monkeypatch.setattr(comparison, "_attachment_fingerprint", _unexpected_fingerprint)

    messages = comparison.construct_judge_messages(
        "task",
        [_data_block(b"reference")],
        [_data_block(b"submission-a")],
        [_data_block(b"submission-b")],
    )

    assert _attachments(messages[0]["content"]) == [b"reference", b"submission-a", b"submission-b"]


def test_pdf_source_size_does_not_consume_rendered_page_budget(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "compressed.pdf"
    pdf.write_bytes(b"source-is-larger-than-output-budget")
    budget = comparison.AttachmentBudget(raw_limit=4, encoded_limit=8)

    def _render(_path, *, attachment_budget, **_kwargs):
        assert attachment_budget.reserve(1)
        return [_data_block(b"P")]

    monkeypatch.setattr(comparison, "_pdf_path_to_image_text_blocks", _render)

    blocks = comparison.get_file_image_text_blocks(
        str(tmp_path),
        pdf.name,
        render_dpi=72,
        max_pages=1,
        include_text=True,
        attachment_budget=budget,
    )

    assert _attachments(blocks) == [b"P"]


def test_zip_extraction_rejects_unsafe_and_oversize_members_before_open(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("safe.txt", "OK")
        archive.writestr("oversize.txt", "TOO LARGE")
        archive.writestr("../escape.txt", "ESCAPE")

    monkeypatch.setattr(comparison, "MAX_ZIP_MEMBER_BYTES_FOR_JUDGE", 4)
    monkeypatch.setattr(comparison, "MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES_FOR_JUDGE", 4)
    real_open = zipfile.ZipFile.open
    opened: list[str] = []

    def _guarded_open(self, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
        if name in {"oversize.txt", "../escape.txt"}:
            raise AssertionError(f"rejected member was opened: {name}")
        opened.append(name)
        return real_open(self, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", _guarded_open)
    cleanup: list[Path] = []
    try:
        blocks = comparison.build_file_section(str(tmp_path), cleanup)
    finally:
        for directory in cleanup:
            shutil.rmtree(directory, ignore_errors=True)

    assert opened == ["safe.txt"]
    text = _text(blocks)
    assert "OK" in text
    assert "TOO LARGE" not in text
    assert "ESCAPE" not in text


def test_zip_aggregate_limit_rejects_member_before_open(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("first.txt", "ONE")
        archive.writestr("second.txt", "TWO")

    monkeypatch.setattr(comparison, "MAX_ZIP_MEMBER_BYTES_FOR_JUDGE", 10)
    monkeypatch.setattr(comparison, "MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES_FOR_JUDGE", 3)
    real_open = zipfile.ZipFile.open
    opened: list[str] = []

    def _guarded_open(self, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else str(member)
        if name == "second.txt":
            raise AssertionError("aggregate-budget rejection happened after opening the member")
        opened.append(name)
        return real_open(self, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", _guarded_open)
    extract_dir, paths = comparison._maybe_unzip(archive_path)
    try:
        assert opened == ["first.txt"]
        assert [path.name for path in paths] == ["first.txt"]
        assert paths[0].read_text() == "ONE"
    finally:
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)


def test_zip_compressed_size_limit_rejects_before_parsing(tmp_path: Path, monkeypatch) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("file.txt", "content")

    monkeypatch.setattr(comparison, "MAX_ZIP_ARCHIVE_BYTES_FOR_JUDGE", 1)

    def _unexpected_parse(*_args, **_kwargs):
        raise AssertionError("oversize archive must be rejected before ZipFile parses it")

    monkeypatch.setattr(zipfile, "ZipFile", _unexpected_parse)

    assert comparison._maybe_unzip(archive_path) == (None, [])


def test_construct_messages_enforces_one_attachment_budget_across_all_sections(monkeypatch) -> None:
    monkeypatch.setattr(comparison, "MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE", 12)
    monkeypatch.setattr(comparison, "MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE", 16)
    refs = [{"type": "text", "text": "reference.pdf:"}, _data_block(b"ABCDEF")]
    submission_a = [{"type": "text", "text": "a.pdf:"}, _data_block(b"GHIJKL")]
    submission_b = [{"type": "text", "text": "b.pdf:"}, _data_block(b"MNOPQR")]

    messages = comparison.construct_judge_messages("task", refs, submission_a, submission_b)
    content = messages[0]["content"]
    costs = [comparison._attachment_cost(block) for block in content]

    assert sum(raw for raw, _encoded in costs) <= 12
    assert sum(encoded for _raw, encoded in costs) <= 16
    assert sum(1 for _raw, encoded in costs if encoded) == 2
    assert _attachments(content) == [b"GHIJKL", b"MNOPQR"]
    assert "attachment omitted" in _text(content)


def test_office_sidecar_is_budgeted_before_reading(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "Plan.docx").write_bytes(b"source")
    (tmp_path / "Plan.docx.pdf").write_bytes(b"render-too-large")
    monkeypatch.setattr(comparison, "MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE", 4)
    monkeypatch.setattr(comparison, "MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE", 8)

    def _unexpected_read(_path):
        raise AssertionError("rejected sidecar must not be read")

    monkeypatch.setattr(comparison, "_load_media", _unexpected_read)

    blocks = comparison.build_file_section(str(tmp_path))

    assert "attachment omitted" in _text(blocks)


def test_cached_provenance_avoids_one_rescan_per_office_file(tmp_path: Path, monkeypatch) -> None:
    for index in range(5):
        (tmp_path / f"Plan{index}.docx").write_bytes(b"source")
        (tmp_path / f"Plan{index}.docx.pdf").write_bytes(f"render-{index}".encode())

    real_resolver = comparison.resolve_pdf_provenance
    calls = 0

    def _counted(paths):
        nonlocal calls
        calls += 1
        return real_resolver(paths)

    monkeypatch.setattr(comparison, "resolve_pdf_provenance", _counted)

    comparison.build_file_section(str(tmp_path))

    assert calls == 1


def test_unsupported_document_keeps_direct_sidecar_fallback(tmp_path: Path) -> None:
    (tmp_path / "Notes.odt").write_bytes(b"source")
    (tmp_path / "Notes.odt.pdf").write_bytes(b"render")

    blocks = comparison.build_file_section(str(tmp_path))

    assert b"render" in _attachments(blocks)


def test_full_payload_fingerprint_makes_a_b_swap_selection_invariant(monkeypatch) -> None:
    left = b"A" * 1024 + b"L" * 1024 + b"Z" * 1024
    right = b"A" * 1024 + b"R" * 1024 + b"Z" * 1024
    monkeypatch.setattr(comparison, "MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE", len(left))
    monkeypatch.setattr(
        comparison,
        "MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE",
        len(base64.b64encode(left)),
    )

    first = comparison.construct_judge_messages("task", [], [_data_block(left)], [_data_block(right)])
    second = comparison.construct_judge_messages("task", [], [_data_block(right)], [_data_block(left)])

    assert _attachments(first[0]["content"]) == _attachments(second[0]["content"])
    assert len(_attachments(first[0]["content"])) == 1


def test_total_serialized_bound_includes_task_and_section_text(monkeypatch) -> None:
    monkeypatch.setattr(comparison, "MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE", 100_000)
    huge_text = {"type": "text", "text": "x" * 100_000}

    messages = comparison.construct_judge_messages(
        "task" * 100_000,
        [huge_text, _data_block(b"r" * 20_000)],
        [huge_text, _data_block(b"a" * 20_000)],
        [huge_text, _data_block(b"b" * 20_000)],
    )

    assert comparison._content_serialized_upper_bound(messages[0]["content"]) <= 100_000


@pytest.mark.parametrize(
    "message",
    [
        "Request size is too large. Max size is 500 MB",
        "litellm.ContextWindowExceededError: maximum number of tokens allowed 1048576",
        "HTTP 413: request entity too large",
        "503 upstream error: context window exceeded",
    ],
)
def test_deterministic_payload_errors_are_not_retryable(message: str) -> None:
    assert comparison._is_retryable(RuntimeError(message)) is False


def test_transient_upstream_error_remains_retryable() -> None:
    assert comparison._is_retryable(RuntimeError("503 service unavailable")) is True
