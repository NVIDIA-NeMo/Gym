# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only agent-produced deliverables are shown to the judge, and judging is read-only.

Stirrup writes its run state (finish payload, message history, metadata) into the
same directory as the deliverables. Feeding those back to the judge lets it grade
the agent's reasoning trace instead of the artefact.
"""

import subprocess
from pathlib import Path

from responses_api_agents.stirrup_agent.file_reader import (
    IGNORE_FILES,
    convert_deliverables_to_content_blocks,
    is_deliverable,
    read_deliverable_files,
)


def _mk_run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "repeat_0"
    d.mkdir()
    # The actual deliverable.
    (d / "Report.md").write_text("# Findings\nRevenue grew 12%.\n")
    # Stirrup run state that must never reach the judge.
    (d / "finish_params.json").write_text('{"paths": ["/root/Report.md"], "reason": "Completed."}')
    (d / "history.json").write_text('[{"role": "assistant", "content": "I will claim revenue grew 99%."}]')
    (d / "metadata.json").write_text('{"token_usage": ["input=1 answer=2"]}')
    (d / "inprogress_history.json").write_text("[]")
    (d / "log.txt").write_text("verbose agent log\n")
    (d / "reference_files").mkdir()
    return d


def test_run_state_is_not_read_as_deliverable_text(tmp_path):
    d = _mk_run_dir(tmp_path)

    text = read_deliverable_files(str(d))

    assert "Revenue grew 12%" in text, "the real deliverable must still be read"
    # None of the agent's own trace may leak into the judged submission.
    assert "revenue grew 99%" not in text.lower()
    assert "Completed." not in text
    assert "token_usage" not in text
    assert "verbose agent log" not in text


def test_run_state_is_not_sent_as_content_blocks(tmp_path):
    d = _mk_run_dir(tmp_path)

    blocks = convert_deliverables_to_content_blocks(str(d))

    joined = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    assert "Revenue grew 12%" in joined
    for leaked in ("history.json", "finish_params.json", "metadata.json", "log.txt"):
        assert leaked not in joined


def test_is_deliverable_rejects_every_ignored_name(tmp_path):
    for name in IGNORE_FILES:
        p = tmp_path / name
        if name == "reference_files":
            p.mkdir()
        else:
            p.write_text("x")
        assert not is_deliverable(p), f"{name} must not count as a deliverable"
    real = tmp_path / "Deliverable.docx"
    real.write_text("x")
    assert is_deliverable(real)


def test_preexisting_sibling_pdf_is_reused_and_not_deleted(tmp_path, monkeypatch):
    """A judging pass must not destroy a preconverted corpus it did not create.

    The converter is stubbed to *succeed*: without a stub it fails wherever
    LibreOffice is absent, nothing is registered for cleanup, and the test would
    pass against the unfixed code for the wrong reason.
    """
    from responses_api_agents.stirrup_agent import file_reader

    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Deck.pptx").write_bytes(b"not really a pptx")
    sibling = d / "Deck.pdf"
    sibling.write_bytes(b"%PDF-1.4 preconverted\n")

    calls: list[Path] = []

    def _fake_convert(fpath: Path, out_dir: Path | None = None):
        calls.append(fpath)
        out = (out_dir or fpath.parent) / (fpath.stem + ".pdf")
        out.write_bytes(b"%PDF-1.4 REGENERATED\n")  # clobbers, as the real one does
        return out

    monkeypatch.setattr(file_reader, "_convert_office_to_pdf", _fake_convert)

    file_reader.convert_deliverables_to_content_blocks(str(d))

    assert calls == [], "should not reconvert when a sibling PDF already exists"
    assert sibling.is_file(), "pre-existing sibling PDF was deleted by a judging pass"
    assert sibling.read_bytes() == b"%PDF-1.4 preconverted\n", "sibling PDF was overwritten"


def test_judging_never_writes_into_the_deliverables_dir(tmp_path, monkeypatch):
    """Conversions land in a tempdir, so reading a corpus never mutates it."""
    from responses_api_agents.stirrup_agent import file_reader

    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Sheet.xlsx").write_bytes(b"not really an xlsx")
    before = {p.name for p in d.iterdir()}

    def _fake_convert(fpath: Path, out_dir: Path | None = None):
        assert out_dir is not None, "judging must convert into a tempdir, not in place"
        out = out_dir / (fpath.stem + ".pdf")
        out.write_bytes(b"%PDF-1.4 generated\n")
        return out

    monkeypatch.setattr(file_reader, "_convert_office_to_pdf", _fake_convert)

    blocks = file_reader.convert_deliverables_to_content_blocks(str(d))

    assert any(b.get("type") == "image_url" for b in blocks), "converted PDF was not sent"
    assert {p.name for p in d.iterdir()} == before, "judging pass altered the deliverables dir"


def test_convert_office_to_pdf_honours_out_dir(tmp_path, monkeypatch):
    """``out_dir`` is what keeps a judging pass out of the deliverables tree."""
    from responses_api_agents.stirrup_agent import file_reader

    src = tmp_path / "Deck.pptx"
    src.write_bytes(b"not really a pptx")
    dest = tmp_path / "scratch"
    dest.mkdir()

    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["outdir"] = cmd[cmd.index("--outdir") + 1]
        (Path(captured["outdir"]) / "Deck.pdf").write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(file_reader.subprocess, "run", _fake_run)

    out = file_reader._convert_office_to_pdf(src, out_dir=dest)

    assert out == dest / "Deck.pdf"
    assert captured["outdir"] == str(dest)
    assert not (tmp_path / "Deck.pdf").exists(), "conversion leaked into the source dir"


def test_convert_office_to_pdf_defaults_to_writing_beside_the_source(tmp_path, monkeypatch):
    """Preconversion still depends on the in-place default."""
    from responses_api_agents.stirrup_agent import file_reader

    src = tmp_path / "Sheet.xlsx"
    src.write_bytes(b"not really an xlsx")

    def _fake_run(cmd, **kwargs):
        (Path(cmd[cmd.index("--outdir") + 1]) / "Sheet.pdf").write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(file_reader.subprocess, "run", _fake_run)

    assert file_reader._convert_office_to_pdf(src) == tmp_path / "Sheet.pdf"


def test_same_stem_office_files_do_not_share_one_render(tmp_path, monkeypatch):
    """``Report.docx`` and ``Report.pptx`` both map to ``Report.pdf``.

    Reusing that one sibling for both would show the judge the first file's content
    twice and the second file's content never. One such collision exists in the
    200-task Mercor corpus (``Pineloten_Capital_Structure.{docx,pptx}``), where the
    preconverted PDF is the .docx render — 612x792 portrait — so the .pptx has no
    render of its own to reuse.
    """
    from responses_api_agents.stirrup_agent import file_reader

    d = tmp_path / "repeat_0"
    d.mkdir()
    (d / "Report.docx").write_bytes(b"docx bytes")
    (d / "Report.pptx").write_bytes(b"pptx bytes")
    (d / "Report.pdf").write_bytes(b"%PDF-1.4 render of the DOCX\n")

    converted: list[str] = []

    def _fake_convert(fpath: Path, out_dir: Path | None = None):
        converted.append(fpath.name)
        out = (out_dir or fpath.parent) / (fpath.stem + ".pdf")
        out.write_bytes(f"%PDF-1.4 render of {fpath.name}\n".encode())
        return out

    monkeypatch.setattr(file_reader, "_convert_office_to_pdf", _fake_convert)

    blocks = file_reader.convert_deliverables_to_content_blocks(str(d))
    pdf_payloads = [b["image_url"]["url"] for b in blocks if b.get("type") == "image_url"]

    assert "Report.pptx" in converted and "Report.docx" in converted, (
        "an ambiguous sibling PDF must not be reused for either file"
    )
    assert len(set(pdf_payloads)) == len(pdf_payloads), "two deliverables were sent the same render"
    assert (d / "Report.pdf").read_bytes() == b"%PDF-1.4 render of the DOCX\n", "sibling PDF was overwritten"
