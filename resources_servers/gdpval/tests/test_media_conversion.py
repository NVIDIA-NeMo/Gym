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

import base64

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


class TestAudioVideoBlock:
    def test_frontier_dialect_is_image_url(self) -> None:
        from resources_servers.gdpval.media_conversion import audio_video_block

        block = audio_video_block("video/mp4", b"vid", ext="mp4", file_type="VIDEO", openai_native=False)
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:video/mp4;base64,")

    def test_vllm_video_is_video_url(self) -> None:
        from resources_servers.gdpval.media_conversion import audio_video_block

        block = audio_video_block("video/mp4", b"vid", ext="mp4", file_type="VIDEO", openai_native=True)
        assert block["type"] == "video_url"
        assert block["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_vllm_audio_is_input_audio(self) -> None:
        from resources_servers.gdpval.media_conversion import audio_video_block

        block = audio_video_block("audio/wav", b"snd", ext=".wav", file_type="AUDIO", openai_native=True)
        assert block["type"] == "input_audio"
        assert block["input_audio"]["format"] == "wav"
        assert base64.b64decode(block["input_audio"]["data"]) == b"snd"

    def test_audio_format_token_maps_extensions(self) -> None:
        from resources_servers.gdpval.media_conversion import audio_format_token

        assert audio_format_token(".mp3") == "mp3"
        assert audio_format_token("wav") == "wav"
        assert audio_format_token(".UNKNOWN") == "unknown"


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

    def test_get_file_image_text_blocks_audio_passthrough_when_av_capable(self, tmp_path) -> None:
        # images_and_text == self-hosted vLLM judge -> OpenAI-standard input_audio,
        # NOT an image_url wrapper (which vLLM won't route to the audio tower).
        from resources_servers.gdpval.comparison import get_file_image_text_blocks

        payload = b"ID3fakeaudio"
        (tmp_path / "clip.mp3").write_bytes(payload)
        blocks = get_file_image_text_blocks(
            str(tmp_path), "clip.mp3", render_dpi=72, max_pages=10, include_text=True, audio_capable=True
        )
        assert len(blocks) == 1
        assert blocks[0]["type"] == "input_audio"
        assert blocks[0]["input_audio"]["format"] == "mp3"
        assert base64.b64decode(blocks[0]["input_audio"]["data"]) == payload

    def test_get_file_image_text_blocks_video_passthrough_when_av_capable(self, tmp_path) -> None:
        # Self-hosted vLLM judge -> OpenAI-standard video_url data URL.
        from resources_servers.gdpval.comparison import get_file_image_text_blocks

        payload = b"\x00\x00\x00\x18ftypmp42"
        (tmp_path / "clip.mp4").write_bytes(payload)
        blocks = get_file_image_text_blocks(
            str(tmp_path), "clip.mp4", render_dpi=72, max_pages=10, include_text=True, video_capable=True
        )
        assert len(blocks) == 1
        assert blocks[0]["type"] == "video_url"
        url = blocks[0]["video_url"]["url"]
        assert url.startswith("data:video/mp4;base64,")

    def test_build_file_section_av_passthrough_when_av_capable(self, tmp_path) -> None:
        from resources_servers.gdpval.comparison import build_file_section

        (tmp_path / "clip.mp3").write_bytes(b"ID3fakeaudio")
        section = build_file_section(str(tmp_path), media_mode="images_and_text", audio_capable=True)
        assert any(b.get("type") == "input_audio" and b.get("input_audio", {}).get("format") == "mp3" for b in section)
        assert not any("not readable by this judge" in b.get("text", "") for b in section)

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

    def test_convert_deliverables_skips_audio_when_not_av_capable(self, tmp_path) -> None:
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "clip.mp3").write_bytes(b"ID3fakeaudio")
        blocks = convert_deliverables_to_content_blocks(
            str(tmp_path), media_mode="images_and_text", audio_capable=False
        )
        assert blocks == []

    def test_convert_deliverables_audio_passthrough_when_av_capable(self, tmp_path) -> None:
        # images_and_text == self-hosted vLLM judge -> input_audio (not image_url).
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "clip.mp3").write_bytes(b"ID3fakeaudio")
        blocks = convert_deliverables_to_content_blocks(
            str(tmp_path), media_mode="images_and_text", audio_capable=True
        )
        assert any(b.get("type") == "input_audio" and b.get("input_audio", {}).get("format") == "mp3" for b in blocks)

    def test_convert_deliverables_video_passthrough_when_av_capable(self, tmp_path) -> None:
        # Self-hosted vLLM judge -> video_url data URL.
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        blocks = convert_deliverables_to_content_blocks(
            str(tmp_path), media_mode="images_and_text", video_capable=True
        )
        assert any(
            b.get("type") == "video_url" and b.get("video_url", {}).get("url", "").startswith("data:video/mp4;base64,")
            for b in blocks
        )

    def test_convert_deliverables_video_capable_stubs_audio(self, tmp_path) -> None:
        # MiniMax-M3 case: reads video but NOT audio -> video passes through as a
        # native block, audio is dropped (skipped) even in the same directory.
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        (tmp_path / "voice.mp3").write_bytes(b"ID3fakeaudio")
        blocks = convert_deliverables_to_content_blocks(
            str(tmp_path), media_mode="images_and_text", video_capable=True, audio_capable=False
        )
        assert any(b.get("type") == "video_url" for b in blocks)
        assert not any(b.get("type") == "input_audio" for b in blocks)
        # The audio file is skipped entirely (only its name marker, if any).
        assert not any(b.get("type") == "input_audio" for b in blocks)

    def test_get_file_image_text_blocks_video_capable_stubs_audio(self, tmp_path) -> None:
        # Same asymmetry via the comparison path: audio -> marker, not a block.
        from resources_servers.gdpval.comparison import get_file_image_text_blocks

        (tmp_path / "voice.mp3").write_bytes(b"ID3fakeaudio")
        blocks = get_file_image_text_blocks(
            str(tmp_path), "voice.mp3", render_dpi=72, max_pages=10, include_text=True, video_capable=True
        )
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "not readable by this judge" in blocks[0]["text"]

    def test_convert_deliverables_native_mode_av_uses_image_url(self, tmp_path) -> None:
        # native_pdf == frontier judge (Gemini) -> AV stays an image_url data URL.
        from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks

        (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        blocks = convert_deliverables_to_content_blocks(str(tmp_path), media_mode="native_pdf", video_capable=True)
        assert any(
            b.get("type") == "image_url" and b.get("image_url", {}).get("url", "").startswith("data:video/mp4;base64,")
            for b in blocks
        )
        assert not any(b.get("type") in ("video_url", "input_audio") for b in blocks)
