# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
from pathlib import Path

import pytest

from resources_servers.gdpval import preconvert as pcv


class TestNeedsConversion:
    def test_office_without_pdf_needs_conversion(self, tmp_path: Path) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")
        assert pcv.needs_conversion(f) is True

    def test_office_with_sibling_pdf_does_not(self, tmp_path: Path) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")
        (tmp_path / "a.pdf").write_text("p")
        assert pcv.needs_conversion(f) is False

    def test_non_office_skipped(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert pcv.needs_conversion(f) is False


class TestFindConvertibleFiles:
    def test_finds_recursively_and_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "b.docx").write_text("b")
        (tmp_path / "sub" / "a.xlsx").write_text("a")
        (tmp_path / "ignore.txt").write_text("t")
        # already-converted should be skipped
        (tmp_path / "c.pptx").write_text("c")
        (tmp_path / "c.pdf").write_text("p")
        files = pcv.find_convertible_files(str(tmp_path))
        # Sorted by full path: top-level b.docx < sub/a.xlsx; c.pptx skipped (sibling .pdf exists).
        assert [p.name for p in files] == ["b.docx", "a.xlsx"]


class TestConvertToPdfErrors:
    def test_returns_message_when_libreoffice_not_found(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")

        def _raise(*_a, **_kw):
            raise FileNotFoundError("libreoffice")

        monkeypatch.setattr(subprocess, "run", _raise)
        path, ok, msg = pcv.convert_to_pdf(f)
        assert ok is False
        assert "libreoffice not found" in msg

    def test_returns_message_when_libreoffice_runs_but_no_pdf(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")

        class _CompletedNoPdf:
            returncode = 1
            stdout = ""
            stderr = "Some libreoffice error"

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _CompletedNoPdf())
        path, ok, msg = pcv.convert_to_pdf(f)
        assert ok is False
        assert "did not produce" in msg
        assert "Some libreoffice error" in msg

    def test_passes_user_installation_flag(self, tmp_path: Path, monkeypatch) -> None:
        f = tmp_path / "a.docx"
        f.write_text("x")
        captured: list[list[str]] = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(cmd, *_a, **_kw):
            captured.append(cmd)
            f.with_suffix(".pdf").write_bytes(b"%PDF-1.4 fake\n")
            return _Completed()

        monkeypatch.setattr(subprocess, "run", _run)
        path, ok, _ = pcv.convert_to_pdf(f)
        assert ok is True
        assert len(captured) == 1
        env_flags = [arg for arg in captured[0] if arg.startswith("-env:UserInstallation=")]
        assert len(env_flags) == 1, f"expected one -env:UserInstallation flag, got {env_flags}"
        assert env_flags[0].startswith("-env:UserInstallation=file://")
        # The path should be a unique tempdir (one per call); just sanity-check it points to a path.
        assert "/lo-profile-" in env_flags[0]


class TestPreconvertDirSurfacesFailures:
    def test_returns_error_messages(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "a.docx").write_text("x")
        (tmp_path / "b.xlsx").write_text("x")

        def _convert(path: Path) -> tuple[Path, bool, str]:
            return path, False, f"forced fail on {path.name}"

        monkeypatch.setattr(pcv, "convert_to_pdf", _convert)

        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert ok == 0
        assert fail == 2
        assert sorted(errors) == sorted(["forced fail on a.docx", "forced fail on b.xlsx"])

    def test_empty_dir_returns_zeros(self, tmp_path: Path) -> None:
        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert (ok, fail, errors) == (0, 0, [])

    def test_success_path_returns_no_errors(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "a.docx").write_text("x")

        def _convert(path: Path) -> tuple[Path, bool, str]:
            (path.with_suffix(".pdf")).write_text("p")
            return path, True, "ok"

        monkeypatch.setattr(pcv, "convert_to_pdf", _convert)

        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert (ok, fail, errors) == (1, 0, [])


@pytest.mark.asyncio
async def test_preconvert_dir_async_propagates_results(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.docx").write_text("x")

    monkeypatch.setattr(pcv, "convert_to_pdf", lambda p: (p, False, "boom"))
    ok, fail, errors = await pcv.preconvert_dir_async(str(tmp_path))
    assert (ok, fail) == (0, 1)
    assert errors == ["boom"]


# Fixtures + tests for the ns0-namespace normalization (Mode A in
# the GDPVal corpus). See module docstring on preconvert.py for the
# background on why python-docx-style ns0 prefixing breaks LibreOffice.

NS0_RELS = (
    b"<?xml version='1.0' encoding='utf-8'?>\n"
    b'<ns0:Relationships xmlns:ns0="http://schemas.openxmlformats.org/package/2006/relationships">'
    b'<ns0:Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    b'relationships/officeDocument" Target="word/document.xml" />'
    b"</ns0:Relationships>"
)

NS0_CONTENT_TYPES = (
    b"<?xml version='1.0' encoding='utf-8'?>\n"
    b'<ns0:Types xmlns:ns0="http://schemas.openxmlformats.org/package/2006/content-types">'
    b'<ns0:Default Extension="rels" '
    b'ContentType="application/vnd.openxmlformats-package.relationships+xml" />'
    b'<ns0:Override PartName="/word/document.xml" '
    b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml" />'
    b"</ns0:Types>"
)

DEFAULT_NS_RELS = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    b'<Relationship Id="rId1" Type="x" Target="word/document.xml"/></Relationships>'
)

DEFAULT_NS_CONTENT_TYPES = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>'
)

DOCUMENT_XML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document '
    b'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"></w:document>'
)


def _make_zip(path: Path, parts: dict[str, bytes]) -> Path:
    import zipfile as _zip

    with _zip.ZipFile(path, "w", _zip.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return path


class TestRewriteNs0Namespace:
    def test_rewrites_root_to_default_namespace(self) -> None:
        out = pcv._rewrite_ns0_namespace(NS0_RELS.decode("utf-8"))
        assert "<ns0:" not in out
        assert "</ns0:" not in out
        assert "xmlns:ns0=" not in out
        assert '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"' in out

    def test_rewrites_content_types(self) -> None:
        out = pcv._rewrite_ns0_namespace(NS0_CONTENT_TYPES.decode("utf-8"))
        assert "<ns0:" not in out
        assert "</ns0:" not in out
        assert '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"' in out
        # Override children must remain (just unprefixed).
        assert "<Override PartName=" in out

    def test_idempotent_on_default_namespace(self) -> None:
        out = pcv._rewrite_ns0_namespace(DEFAULT_NS_RELS.decode("utf-8"))
        assert out == DEFAULT_NS_RELS.decode("utf-8")


class TestOoxmlHasNs0Prefix:
    def test_true_when_rels_has_ns0(self, tmp_path: Path) -> None:
        zp = _make_zip(
            tmp_path / "a.docx",
            {
                "[Content_Types].xml": DEFAULT_NS_CONTENT_TYPES,
                "_rels/.rels": NS0_RELS,
                "word/document.xml": DOCUMENT_XML,
            },
        )
        assert pcv._ooxml_has_ns0_prefix(zp) is True

    def test_true_when_only_content_types_has_ns0(self, tmp_path: Path) -> None:
        zp = _make_zip(
            tmp_path / "a.docx",
            {
                "[Content_Types].xml": NS0_CONTENT_TYPES,
                "_rels/.rels": DEFAULT_NS_RELS,
                "word/document.xml": DOCUMENT_XML,
            },
        )
        assert pcv._ooxml_has_ns0_prefix(zp) is True

    def test_false_when_default_namespace(self, tmp_path: Path) -> None:
        zp = _make_zip(
            tmp_path / "a.docx",
            {
                "[Content_Types].xml": DEFAULT_NS_CONTENT_TYPES,
                "_rels/.rels": DEFAULT_NS_RELS,
                "word/document.xml": DOCUMENT_XML,
            },
        )
        assert pcv._ooxml_has_ns0_prefix(zp) is False

    def test_false_on_non_zip(self, tmp_path: Path) -> None:
        bogus = tmp_path / "a.docx"
        bogus.write_bytes(b"not a zip")
        assert pcv._ooxml_has_ns0_prefix(bogus) is False


class TestNormalizeOoxmlZip:
    def test_rewrites_rels_and_content_types_only(self, tmp_path: Path) -> None:
        import zipfile as _zip

        src = _make_zip(
            tmp_path / "in.docx",
            {
                "[Content_Types].xml": NS0_CONTENT_TYPES,
                "_rels/.rels": NS0_RELS,
                "word/_rels/document.xml.rels": NS0_RELS,
                "word/document.xml": DOCUMENT_XML,
            },
        )
        dst = tmp_path / "out.docx"
        pcv._normalize_ooxml_zip(src, dst)

        with _zip.ZipFile(dst) as z:
            for part in ("[Content_Types].xml", "_rels/.rels", "word/_rels/document.xml.rels"):
                text = z.read(part).decode("utf-8")
                assert "<ns0:" not in text, f"ns0 still present in {part}"
                assert "xmlns:ns0=" not in text, f"xmlns:ns0 still in {part}"
            # non-package XML must be byte-identical
            assert z.read("word/document.xml") == DOCUMENT_XML


class TestConvertToPdfNormalization:
    def test_calls_libreoffice_with_normalized_copy_when_ns0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_zip(
            tmp_path / "src.docx",
            {
                "[Content_Types].xml": NS0_CONTENT_TYPES,
                "_rels/.rels": NS0_RELS,
                "word/document.xml": DOCUMENT_XML,
            },
        )
        captured: list[list[str]] = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(cmd, *_a, **_kw):
            captured.append(cmd)
            # libreoffice would write the PDF to the original outdir using the input stem.
            (src.with_suffix(".pdf")).write_bytes(b"%PDF-1.4 fake\n")
            return _Completed()

        monkeypatch.setattr(subprocess, "run", _run)
        path, ok, msg = pcv.convert_to_pdf(src)
        assert ok is True
        assert "(after ns0 normalization)" in msg
        # The input arg passed to libreoffice should NOT be the original file: it must come
        # from the gdpval-norm- tempdir, but with the same basename so output stem is preserved.
        assert len(captured) == 1
        input_arg = captured[0][-1]
        assert input_arg.endswith("/src.docx")
        assert "/gdpval-norm-" in input_arg
        assert input_arg != str(src)

    def test_calls_libreoffice_with_original_when_not_ns0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_zip(
            tmp_path / "src.docx",
            {
                "[Content_Types].xml": DEFAULT_NS_CONTENT_TYPES,
                "_rels/.rels": DEFAULT_NS_RELS,
                "word/document.xml": DOCUMENT_XML,
            },
        )
        captured: list[list[str]] = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def _run(cmd, *_a, **_kw):
            captured.append(cmd)
            (src.with_suffix(".pdf")).write_bytes(b"%PDF-1.4 fake\n")
            return _Completed()

        monkeypatch.setattr(subprocess, "run", _run)
        path, ok, msg = pcv.convert_to_pdf(src)
        assert ok is True
        assert "(after ns0 normalization)" not in msg
        assert captured[0][-1] == str(src)
