# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A judge host without LibreOffice must degrade, not fabricate a missing artifact.

``convert_deliverables_to_content_blocks`` reaches for LibreOffice whenever an Office
deliverable has no unambiguous PDF beside it. When the binary is absent that call used
to raise ``FileNotFoundError``, which the per-file handler turned into a lone
``[Error: ...]`` block *in place of the deliverable* — indistinguishable to a judge from
a missing work product, and worth a near-zero score. Text extraction is the correct
degradation, and it already exists.
"""

import subprocess
from pathlib import Path

import responses_api_agents.stirrup_agent.file_reader as file_reader
from responses_api_agents.stirrup_agent.file_reader import convert_deliverables_to_content_blocks


def _docx(path: Path, body: str) -> None:
    """Minimal but real .docx: a zip whose document.xml holds one paragraph."""
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument'
            '.wordprocessingml.document.main+xml"/></Types>',
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'officeDocument" Target="word/document.xml"/></Relationships>',
        )
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{body}</w:t></w:r></w:p></w:body></w:document>",
        )


def _text_of(blocks) -> str:
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _no_libreoffice(monkeypatch) -> None:
    """Make any attempt to exec LibreOffice fail the way a missing binary does."""

    def _boom(cmd, *args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "libreoffice")

    monkeypatch.setattr(subprocess, "run", _boom)
    # raising=False so these tests fail on their own assertions against code that predates
    # the warn-once flag, rather than erroring out here and hiding what regressed.
    monkeypatch.setattr(file_reader, "_libreoffice_unavailable_warned", False, raising=False)


def test_missing_libreoffice_does_not_replace_the_deliverable_with_an_error(monkeypatch, tmp_path: Path):
    _docx(tmp_path / "report.docx", "Quarterly throughput rose 12 percent.")
    _no_libreoffice(monkeypatch)

    text = _text_of(convert_deliverables_to_content_blocks(tmp_path))

    assert "[Error:" not in text, "a missing binary was reported as a broken deliverable"
    assert "report.docx" in text


def test_missing_libreoffice_falls_back_to_extracted_text(monkeypatch, tmp_path: Path):
    _docx(tmp_path / "report.docx", "Quarterly throughput rose 12 percent.")
    _no_libreoffice(monkeypatch)

    text = _text_of(convert_deliverables_to_content_blocks(tmp_path))

    assert "Quarterly throughput rose 12 percent." in text, (
        "the document's own content never reached the judge, so it would score as absent"
    )


def test_conversion_helper_returns_none_rather_than_raising(monkeypatch, tmp_path: Path):
    _docx(tmp_path / "report.docx", "body")
    _no_libreoffice(monkeypatch)

    assert file_reader._convert_office_to_pdf(tmp_path / "report.docx", out_dir=tmp_path) is None


def test_unavailability_is_announced_once(monkeypatch, capsys, tmp_path: Path):
    for name in ("a.docx", "b.docx", "c.docx"):
        _docx(tmp_path / name, "body")
    _no_libreoffice(monkeypatch)

    convert_deliverables_to_content_blocks(tmp_path)

    warnings = [ln for ln in capsys.readouterr().out.splitlines() if "LibreOffice is unavailable" in ln]
    assert len(warnings) == 1, f"expected exactly one warning for three files, got {len(warnings)}"


def test_conversion_failure_retries_roundtripped_copy_without_mutating_original(monkeypatch, tmp_path: Path):
    source = tmp_path / "report.docx"
    original = b"original OOXML package"
    source.write_bytes(original)
    destination = tmp_path / "render"
    destination.mkdir()
    calls: list[Path] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = "Error: source file could not be loaded"

    def _roundtrip(src: Path, dst: Path) -> None:
        assert src == source
        dst.write_bytes(b"roundtripped package")

    def _run(command, *args, **kwargs):
        input_path = Path(command[-1])
        calls.append(input_path)
        if len(calls) == 2:
            outdir = Path(command[command.index("--outdir") + 1])
            (outdir / f"{input_path.stem}.pdf").write_bytes(b"%PDF retry")
        return _Completed()

    monkeypatch.setattr(file_reader, "roundtrip_ooxml_copy", _roundtrip)
    monkeypatch.setattr(subprocess, "run", _run)

    result = file_reader._convert_office_to_pdf(source, out_dir=destination)

    assert result == destination / "report.pdf"
    assert result.read_bytes() == b"%PDF retry"
    assert len(calls) == 2
    assert "gdpval-roundtrip-" in str(calls[1])
    assert source.read_bytes() == original
