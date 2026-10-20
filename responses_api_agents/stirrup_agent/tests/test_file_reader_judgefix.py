# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDPval judge reader: subfolders, archive members, PDF-in-image, nothing-gradable refusal.

Regressions from the 2026-10-12/13 Super training audit: files saved under a
subfolder or inside a zip were invisible to the judge (same task scored 0.0 with a
subfolder and 1.0 without), and audio-only submissions were graded on nothing.
"""
import struct
import wave
import zipfile
from pathlib import Path

import pytest

from responses_api_agents.stirrup_agent import file_reader as fr

MINIMAL_PDF = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
               b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n")


def _labels(blocks):
    return [b["text"].strip().split("\n", 1)[0] for b in blocks if b["type"] == "text"]


def test_subfolder_files_are_judged_with_relative_labels(tmp_path: Path):
    (tmp_path / "deliverables" / "scripts").mkdir(parents=True)
    (tmp_path / "deliverables" / "report.md").write_text("# report")
    (tmp_path / "deliverables" / "scripts" / "run.sql").write_text("SELECT 1;")
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path))
    labels = _labels(blocks)
    assert "deliverables/report.md:" in labels
    assert "deliverables/scripts/run.sql:" in labels
    assert fr._has_gradable_content(blocks)


def test_reference_files_and_run_state_stay_hidden_recursively(tmp_path: Path):
    (tmp_path / "reference_files" / "nested").mkdir(parents=True)
    (tmp_path / "reference_files" / "nested" / "input.csv").write_text("a,b")
    (tmp_path / "history.json").write_text("{}")
    (tmp_path / "out.txt").write_text("answer")
    labels = _labels(fr.convert_deliverables_to_content_blocks(str(tmp_path)))
    assert labels == ["out.txt:"]


def test_archive_members_are_rendered_after_the_listing(tmp_path: Path):
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as zf:
        zf.writestr("docs/notes.md", "# inside")
        zf.writestr("docs/page.pdf", MINIMAL_PDF)
        zf.writestr("../escape.pdf", MINIMAL_PDF)          # zip-slip attempt: never extracted
        zf.writestr("docs/inner.zip", b"PK\x05\x06" + b"\x00" * 18)
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path))
    labels = _labels(blocks)
    assert labels[0] == "bundle.zip:"                       # manifest (with the text member) comes first
    assert "# inside" in blocks[0]["text"]
    assert "bundle.zip/docs/page.pdf:" in labels            # PDF member rendered via the normal PDF path
    assert any(b["type"] == "image_url" for b in blocks)
    assert not any("escape" in lbl for lbl in labels)
    assert not (tmp_path / "escape.pdf").exists()


def test_pdf_with_image_extension_goes_through_pdf_path(tmp_path: Path):
    (tmp_path / "chart.png").write_bytes(MINIMAL_PDF)
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path))
    assert _labels(blocks)[0] == "chart.png (PDF content):"
    assert blocks[1]["image_url"]["url"].startswith("data:application/pdf;")


def test_audio_only_submission_is_refused_not_guessed(tmp_path: Path):
    with wave.open(str(tmp_path / "jingle.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(struct.pack("<h", 0) * 800)
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path), audio_capable=False)
    assert not fr._has_gradable_content(blocks[:-1])
    assert blocks[-1]["text"].lstrip().startswith("deliverables: [Error:")


def test_mixed_submission_with_one_readable_file_is_not_refused(tmp_path: Path):
    (tmp_path / "model.npy").write_bytes(b"\x93NUMPY" + b"\x00" * 64)
    (tmp_path / "summary.txt").write_text("results")
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path))
    assert not any("[Error:" in b["text"] for b in blocks)
    assert "model.npy:" in _labels(blocks)                   # explicitly present, explicitly unreadable


def test_file_cap_trims_deep_vendored_tree_not_top_level(tmp_path: Path):
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    for i in range(fr.MAX_DELIVERABLE_FILES + 50):
        (tmp_path / "node_modules" / "x" / f"f{i}.js").write_text("1;")
    (tmp_path / "answer.sql").write_text("SELECT 1;")
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path))
    labels = _labels(blocks)
    assert "answer.sql:" in labels
    assert labels[-1].startswith("[51 more files not shown")   # 250 vendored + answer.sql - cap 200


def test_zip_of_text_members_is_gradable_not_refused(tmp_path: Path):
    # The manifest carries the members' contents, so the judge can grade it.
    with zipfile.ZipFile(tmp_path / "code.zip", "w") as zf:
        zf.writestr("src/query.sql", "SELECT 1;")
        zf.writestr("README.md", "how to run")
    blocks = fr.convert_deliverables_to_content_blocks(str(tmp_path))
    texts = "\n".join(b["text"] for b in blocks if b["type"] == "text")
    assert "no gradable deliverable content" not in texts
    assert "SELECT 1;" in texts
