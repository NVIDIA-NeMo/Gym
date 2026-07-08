# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the ``images_and_text`` judge media conversion path.

Covers the shared PDF→page-image + text rasterizer
(:mod:`resources_servers.gdpval.media_conversion`) and its wiring into both
content-block builders — the pairwise comparison scorer
(:func:`resources_servers.gdpval.comparison.build_file_section`) and the rubric
visual scorer
(:func:`responses_api_agents.stirrup_agent.file_reader.convert_deliverables_to_content_blocks`).

All tests skip gracefully when PyMuPDF (``fitz``) isn't installed.
"""

from __future__ import annotations

import pytest


fitz = pytest.importorskip("fitz", reason="PyMuPDF (fitz) required for media-conversion tests")


def _make_pdf_bytes(pages: int = 1, text: str = "Hello GDPVal judge") -> bytes:
    """Build a tiny multi-page PDF in memory."""
    doc = fitz.open()
    try:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"{text} (page {i + 1})")
        return doc.tobytes()
    finally:
        doc.close()


def _is_png_image_block(block: dict) -> bool:
    return block.get("type") == "image_url" and block.get("image_url", {}).get("url", "").startswith(
        "data:image/png;base64,"
    )


class TestPdfRasterization:
    def test_pdf_bytes_to_image_blocks_one_per_page(self) -> None:
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_image_blocks

        blocks = pdf_bytes_to_image_blocks(_make_pdf_bytes(pages=3), dpi=72, max_pages=50)
        image_blocks = [b for b in blocks if _is_png_image_block(b)]
        assert len(image_blocks) == 3
        # No truncation marker when under the page cap.
        assert not any(b.get("type") == "text" and "truncated" in b.get("text", "") for b in blocks)

    def test_pdf_bytes_to_image_blocks_truncates_and_marks(self) -> None:
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_image_blocks

        blocks = pdf_bytes_to_image_blocks(_make_pdf_bytes(pages=4), dpi=72, max_pages=2)
        image_blocks = [b for b in blocks if _is_png_image_block(b)]
        assert len(image_blocks) == 2
        markers = [b for b in blocks if b.get("type") == "text" and "truncated" in b.get("text", "")]
        assert len(markers) == 1
        assert "2 of 4 pages" in markers[0]["text"]

    def test_pdf_bytes_to_image_blocks_bad_pdf_returns_empty(self) -> None:
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_image_blocks

        assert pdf_bytes_to_image_blocks(b"not a pdf at all", dpi=72) == []

    def test_pdf_bytes_to_text_extracts(self) -> None:
        pytest.importorskip("pdfminer.high_level", reason="pdfminer.six required for text extraction")
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_text

        text = pdf_bytes_to_text(_make_pdf_bytes(pages=1, text="UNIQUE_MARKER_TOKEN"))
        assert "UNIQUE_MARKER_TOKEN" in text

    def test_pdf_bytes_to_text_truncates(self) -> None:
        pytest.importorskip("pdfminer.high_level", reason="pdfminer.six required for text extraction")
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_text

        text = pdf_bytes_to_text(_make_pdf_bytes(pages=1, text="x" * 500), max_chars=50)
        assert text.endswith("[...text truncated]")

    def test_pdf_bytes_to_blocks_text_first_then_images(self) -> None:
        pytest.importorskip("pdfminer.high_level", reason="pdfminer.six required for text extraction")
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_blocks

        blocks = pdf_bytes_to_blocks(_make_pdf_bytes(pages=2), dpi=72, max_pages=50, include_text=True)
        # First block is the extracted-text block, then the page images.
        assert blocks[0]["type"] == "text" and "extracted text" in blocks[0]["text"]
        assert len([b for b in blocks if _is_png_image_block(b)]) == 2

    def test_pdf_bytes_to_blocks_images_only_when_text_disabled(self) -> None:
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_blocks

        blocks = pdf_bytes_to_blocks(_make_pdf_bytes(pages=1), dpi=72, include_text=False)
        assert all(b["type"] == "image_url" for b in blocks)


class TestComparisonImagesAndText:
    def test_build_file_section_rasterizes_pdf(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import build_file_section

        (tmp_path / "deliverable.pdf").write_bytes(_make_pdf_bytes(pages=2))
        section = build_file_section(str(tmp_path), media_mode="images_and_text", render_dpi=72, max_pages=50)

        # PDF is rendered to PNG page images, NOT sent as an application/pdf URL.
        assert any(_is_png_image_block(b) for b in section)
        assert not any(
            b.get("type") == "image_url" and "application/pdf" in b.get("image_url", {}).get("url", "")
            for b in section
        )

    def test_build_file_section_native_mode_keeps_pdf_url(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import build_file_section

        (tmp_path / "deliverable.pdf").write_bytes(_make_pdf_bytes(pages=1))
        section = build_file_section(str(tmp_path))  # default native_pdf

        assert any(
            b.get("type") == "image_url" and "application/pdf" in b.get("image_url", {}).get("url", "")
            for b in section
        )

    def test_build_file_section_images_mode_text_passthrough(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import build_file_section

        (tmp_path / "note.txt").write_text("plain text deliverable")
        section = build_file_section(str(tmp_path), media_mode="images_and_text")
        texts = [b.get("text", "") for b in section if b.get("type") == "text"]
        assert any(t == "plain text deliverable" for t in texts)

    def test_get_file_image_text_blocks_audio_marker(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import get_file_image_text_blocks

        (tmp_path / "clip.mp3").write_bytes(b"ID3fakeaudio")
        blocks = get_file_image_text_blocks(str(tmp_path), "clip.mp3", render_dpi=72, max_pages=10, include_text=True)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "not readable by this judge" in blocks[0]["text"]

    def test_get_file_image_text_blocks_oversize_marker(self, tmp_path) -> None:
        import os as _os

        from resources_servers.gdpval.comparison import MAX_FILE_BYTES_FOR_JUDGE, get_file_image_text_blocks

        big = tmp_path / "huge.pdf"
        big.touch()
        _os.truncate(big, MAX_FILE_BYTES_FOR_JUDGE + 1)
        blocks = get_file_image_text_blocks(str(tmp_path), "huge.pdf", render_dpi=72, max_pages=10, include_text=True)
        assert len(blocks) == 1
        assert "oversize" in blocks[0]["text"]


class TestRubricImagesAndText:
    def test_convert_deliverables_rasterizes_pdf(self, tmp_path) -> None:
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "report.pdf").write_bytes(_make_pdf_bytes(pages=2))
        blocks = convert_deliverables_to_content_blocks(
            str(tmp_path), media_mode="images_and_text", render_dpi=72, max_pages=50
        )
        assert any(_is_png_image_block(b) for b in blocks)
        assert not any(
            b.get("type") == "image_url" and "application/pdf" in b.get("image_url", {}).get("url", "") for b in blocks
        )

    def test_convert_deliverables_native_mode_keeps_pdf_url(self, tmp_path) -> None:
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "report.pdf").write_bytes(_make_pdf_bytes(pages=1))
        blocks = convert_deliverables_to_content_blocks(str(tmp_path))  # default native_pdf
        assert any(
            b.get("type") == "image_url" and "application/pdf" in b.get("image_url", {}).get("url", "") for b in blocks
        )
