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
        # Entries are (source, explicit_destination); destination is None unless the stem is ambiguous.
        assert [src.name for src, _dest in files] == ["b.docx", "a.xlsx"]
        assert all(dest is None for _src, dest in files)


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

    def test_failed_conversion_retries_roundtripped_copy_without_mutating_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "report.docx"
        original = b"original OOXML package"
        source.write_bytes(original)
        calls: list[Path] = []

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = "Error: source file could not be loaded"

        def _roundtrip(src: Path, dst: Path) -> None:
            assert src == source
            dst.write_bytes(b"roundtripped package")

        def _run(command, *_args, **_kwargs):
            input_path = Path(command[-1])
            calls.append(input_path)
            if len(calls) == 2:
                outdir = Path(command[command.index("--outdir") + 1])
                (outdir / f"{input_path.stem}.pdf").write_bytes(b"%PDF retry")
            return _Completed()

        monkeypatch.setattr(pcv, "roundtrip_ooxml_copy", _roundtrip)
        monkeypatch.setattr(subprocess, "run", _run)

        _path, ok, message = pcv.convert_to_pdf(source)

        assert ok is True
        assert "round-trip retry" in message
        assert len(calls) == 2
        assert calls[0] == source
        assert "gdpval-roundtrip-" in str(calls[1])
        assert source.read_bytes() == original
        assert source.with_suffix(".pdf").read_bytes() == b"%PDF retry"


class TestPreconvertDirSurfacesFailures:
    def test_returns_error_messages(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "a.docx").write_text("x")
        (tmp_path / "b.xlsx").write_text("x")

        def _convert(path: Path, output_pdf: Path | None = None) -> tuple[Path, bool, str]:
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

        def _convert(path: Path, output_pdf: Path | None = None) -> tuple[Path, bool, str]:
            (output_pdf or path.with_suffix(".pdf")).write_text("p")
            return path, True, "ok"

        monkeypatch.setattr(pcv, "convert_to_pdf", _convert)

        ok, fail, errors = pcv.preconvert_dir(str(tmp_path))
        assert (ok, fail, errors) == (1, 0, [])


class TestStructuredXlsxText:
    def test_sparse_extreme_coordinate_does_not_scan_empty_rectangle(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sparse"
        sheet["A1"] = "start"
        sheet["XFD1048576"] = "far corner"
        path = tmp_path / "sparse.xlsx"
        workbook.save(path)
        workbook.close()

        text = pcv.extract_xlsx_structured_text(path, max_chars=500)

        assert "A1: value=start" in text
        assert "XFD1048576: value=far corner" in text
        assert len(text) <= 500

    def test_formula_and_truncation_marker_fit_within_bound(self, tmp_path: Path) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = 2
        sheet["A2"] = 3
        sheet["A3"] = "=SUM(A1:A2)"
        sheet["A4"] = "X" * 500
        path = tmp_path / "bounded.xlsx"
        workbook.save(path)
        workbook.close()

        text = pcv.extract_xlsx_structured_text(path, max_chars=100)

        assert "A3: formula: =SUM(A1:A2)" in text
        assert text.endswith("[...spreadsheet text truncated]")
        assert len(text) <= 100

    def test_formula_includes_cached_value_when_package_provides_one(self, tmp_path: Path) -> None:
        import zipfile

        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet["A1"] = 2
        sheet["A2"] = 3
        sheet["A3"] = "=SUM(A1:A2)"
        original = tmp_path / "formula.xlsx"
        cached = tmp_path / "formula_cached.xlsx"
        workbook.save(original)
        workbook.close()

        replaced = False
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(cached, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "xl/worksheets/sheet1.xml":
                    updated = data.replace(
                        b"<f>SUM(A1:A2)</f><v></v>",
                        b"<f>SUM(A1:A2)</f><v>5</v>",
                    )
                    replaced = updated != data
                    data = updated
                target.writestr(item, data)
        assert replaced, "test fixture did not install a formula cache"

        text = pcv.extract_xlsx_structured_text(cached)

        assert "A3: formula: =SUM(A1:A2); cached/display value: 5" in text

    def test_first_oversize_shared_string_keeps_cell_evidence(self, tmp_path: Path) -> None:
        import zipfile

        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = "seed"
        original = tmp_path / "inline.xlsx"
        shared = tmp_path / "shared.xlsx"
        workbook.save(original)
        workbook.close()

        spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        sheet_xml = (
            f'<worksheet xmlns="{spreadsheet_ns}"><sheetData><row r="1">'
            '<c r="A1" t="s"><v>0</v></c></row></sheetData></worksheet>'
        ).encode()
        shared_xml = (
            f'<sst xmlns="{spreadsheet_ns}" count="1" uniqueCount="1"><si><t>' + "X" * 10_000 + "</t></si></sst>"
        ).encode()
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(shared, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "xl/worksheets/sheet1.xml":
                    data = sheet_xml
                target.writestr(item, data)
            target.writestr("xl/sharedStrings.xml", shared_xml)

        text = pcv.extract_xlsx_structured_text(shared, max_chars=100)

        assert "A1: value=" in text
        assert text.endswith("[...spreadsheet text truncated]")
        assert len(text) <= 100


@pytest.mark.parametrize("extension", [".docx", ".pptx", ".xlsx"])
def test_roundtrip_ooxml_copy_uses_format_library_and_preserves_source(tmp_path: Path, extension: str) -> None:
    source = tmp_path / f"source{extension}"
    destination = tmp_path / f"roundtripped{extension}"

    if extension == ".docx":
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_paragraph("GDPVal document")
        document.save(source)
    elif extension == ".pptx":
        pptx = pytest.importorskip("pptx")
        presentation = pptx.Presentation()
        presentation.slides.add_slide(presentation.slide_layouts[6])
        presentation.save(source)
    else:
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = "GDPVal workbook"
        workbook.save(source)
        workbook.close()

    original = source.read_bytes()
    pcv.roundtrip_ooxml_copy(source, destination)

    assert source.read_bytes() == original
    assert destination.is_file()
    assert destination.read_bytes().startswith(b"PK")


@pytest.mark.asyncio
async def test_preconvert_dir_async_propagates_results(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.docx").write_text("x")

    monkeypatch.setattr(pcv, "convert_to_pdf", lambda p, output_pdf=None: (p, False, "boom"))
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
            # libreoffice writes <--outdir>/<input_stem>.pdf — convert_to_pdf
            # now stages into a tempdir and moves the PDF back.
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            input_arg = Path(cmd[-1])
            (outdir / (input_arg.stem + ".pdf")).write_bytes(b"%PDF-1.4 fake\n")
            return _Completed()

        monkeypatch.setattr(subprocess, "run", _run)
        path, ok, msg = pcv.convert_to_pdf(src)
        assert ok is True
        assert "(after ns0 normalization)" in msg
        # The input arg passed to libreoffice should NOT be the original file: it must come
        # from the gdpval-stage- tempdir, but with the same basename so output stem is preserved.
        assert len(captured) == 1
        input_arg = captured[0][-1]
        assert input_arg.endswith("/src.docx")
        assert "/gdpval-stage-" in input_arg
        assert input_arg != str(src)
        # PDF should land at the caller's expected location after stage→move.
        assert src.with_suffix(".pdf").exists()

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

    def test_stages_when_path_has_whitespace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # GDPVal HF filenames such as ``Population v2.xlsx`` make LibreOffice's
        # batch-convert mode silently drop the input; ``convert_to_pdf`` must
        # stage to a tempdir with a sanitized basename and move the PDF back.
        src = _make_zip(
            tmp_path / "Population v2.xlsx",
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
            outdir = Path(cmd[cmd.index("--outdir") + 1])
            input_arg = Path(cmd[-1])
            (outdir / (input_arg.stem + ".pdf")).write_bytes(b"%PDF-1.4 fake\n")
            return _Completed()

        monkeypatch.setattr(subprocess, "run", _run)
        path, ok, msg = pcv.convert_to_pdf(src)
        assert ok is True
        assert len(captured) == 1
        input_arg = captured[0][-1]
        assert " " not in Path(input_arg).name
        assert Path(input_arg).name == "Population_v2.xlsx"
        assert "/gdpval-stage-" in input_arg
        # PDF must end up next to the original with the original stem.
        assert (tmp_path / "Population v2.pdf").exists()
