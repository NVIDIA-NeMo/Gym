# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Silence must never read as absence: every deliverable is named, whatever its type."""

import zipfile
from pathlib import Path

from responses_api_agents.stirrup_agent.file_reader import (
    HANDLED_EXTS,
    LEGACY_OFFICE_EXTS,
    OFFICE_EXTS,
    TEXT_EXTS,
    convert_deliverables_to_content_blocks,
)


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def test_every_deliverable_is_named_whatever_its_extension(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "proxy.ts").write_text("export const x = 1;\n")
    (d / "pyproject.toml").write_text("[project]\nname='x'\n")
    (d / "notebook.ipynb").write_text('{"cells": []}')
    (d / "diagram.svg").write_text("<svg></svg>")
    (d / "Makefile").write_text("all:\n\tcc x.c\n")
    (d / "blob.bin").write_bytes(b"\x00\x01\x02" * 500)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    for name in ("proxy.ts", "pyproject.toml", "notebook.ipynb", "diagram.svg", "Makefile", "blob.bin"):
        assert name in out, f"{name} reached the judge as nothing"


def test_source_file_contents_reach_the_judge_not_just_the_name(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "handler.ts").write_text("export function handler() { return 42; }\n")

    assert "export function handler" in _text_of(convert_deliverables_to_content_blocks(str(d)))


def test_unreadable_binary_is_announced_as_present(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"\x00\x01\x02\x03" * 400)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "model.onnx" in out
    assert "NOT missing" in out


def test_zip_deliverable_lists_its_members(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    with zipfile.ZipFile(d / "bundle.zip", "w") as zf:
        zf.writestr("src/main.py", "print('hi')\n")
        zf.writestr("README.md", "# hi\n")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "bundle.zip" in out
    assert "src/main.py" in out


def test_empty_file_is_announced_as_empty(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "report.md").write_text("")
    (d / "notes.unknownext").write_bytes(b"")

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "report.md" in out and "EMPTY" in out
    assert "notes.unknownext" in out


def test_legacy_office_is_office_not_an_unknown_blob(tmp_path: Path):
    assert LEGACY_OFFICE_EXTS <= OFFICE_EXTS

    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "memo.doc").write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 200)

    assert "memo.doc" in _text_of(convert_deliverables_to_content_blocks(str(d)))


def test_audio_is_disclosed_when_the_judge_cannot_decode_it(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "mix.wav").write_bytes(b"RIFF" + b"\x00" * 2048)

    out = _text_of(convert_deliverables_to_content_blocks(str(d), audio_capable=False))
    assert "mix.wav" in out and "AUDIO" in out
    assert "Do NOT treat it as missing" in out
    assert "unverifiable rather than unmet" in out


def test_undecodable_artifact_says_accompanying_prose_is_not_evidence(tmp_path: Path):
    """Otherwise the judge credits the agent's own claims about a file it cannot inspect."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "mix.wav").write_bytes(b"RIFF" + b"\x00" * 2048)
    (d / "model.onnx").write_bytes(b"\x00\x01\x02\x03" * 400)

    out = _text_of(convert_deliverables_to_content_blocks(str(d), audio_capable=False))
    assert out.count("unverified assertions") == 2, "both undecodable artifacts must carry the warning"
    assert "do not award credit for a property you cannot observe directly" in out


def test_audio_still_forwarded_when_the_judge_can_decode_it(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "mix.wav").write_bytes(b"RIFF" + b"\x00" * 2048)

    blocks = convert_deliverables_to_content_blocks(str(d), audio_capable=True)
    assert any(b.get("type") != "text" for b in blocks)
    assert "Do NOT treat it as missing" not in _text_of(blocks)


def test_handled_exts_is_derived_not_hand_maintained(tmp_path: Path):
    assert TEXT_EXTS <= HANDLED_EXTS
    assert OFFICE_EXTS <= HANDLED_EXTS
    assert ".pdf" in HANDLED_EXTS


def test_legacy_encoded_text_is_shown_not_withheld(tmp_path: Path):
    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "notes.oldtxt").write_bytes("Café — naïve résumé\n".encode("cp1252") * 20)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "notes.oldtxt" in out
    assert "NOT missing" not in out


def test_zip_member_contents_are_read_not_just_named(tmp_path: Path):
    """Rubrics ask whether what is inside the bundle is correct."""
    d = tmp_path / "repeat_0"
    d.mkdir()
    with zipfile.ZipFile(d / "bundle.zip", "w") as zf:
        zf.writestr("src/main.py", "def handler():\n    return 42\n")
        zf.writestr("logo.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "src/main.py" in out
    assert "def handler()" in out, "member listed but its contents withheld"
    assert "\x89PNG" not in out, "a binary member was dumped as text"


def test_zip_container_documents_are_not_treated_as_archives(tmp_path: Path):
    """.odt/.xlsm open cleanly as zips; summarising them lists XML internals."""
    import io

    d = tmp_path / "repeat_0"
    d.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", "<office:document-content/>")
        zf.writestr("META-INF/manifest.xml", "<manifest/>")
    (d / "Report.odt").write_bytes(buf.getvalue())

    out = _text_of(convert_deliverables_to_content_blocks(str(d)))
    assert "Report.odt" in out
    assert "content.xml" not in out
    assert "META-INF" not in out
