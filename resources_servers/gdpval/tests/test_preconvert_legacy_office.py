# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy binary Office deliverables are preconverted too.

A deliverable the preconverter does not recognise silently never gets a
companion PDF, and the judge — which reads the PDF, not the Office file — scores
the task as having produced nothing. That is a scoring bug, not a rendering one.
"""

from pathlib import Path

from resources_servers.gdpval.preconvert import (
    LEGACY_OFFICE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    OOXML_EXTENSIONS,
    find_convertible_files,
    needs_conversion,
)


def test_legacy_binary_formats_are_convertible():
    for ext in (".doc", ".ppt", ".xls"):
        assert ext in OFFICE_EXTENSIONS, f"{ext} deliverables would never get a PDF"
    assert OFFICE_EXTENSIONS == OOXML_EXTENSIONS | LEGACY_OFFICE_EXTENSIONS


def test_needs_conversion_covers_legacy_and_respects_existing_pdf(tmp_path: Path):
    legacy = tmp_path / "Report.doc"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0")  # OLE compound-file magic
    assert needs_conversion(legacy)

    # An already-preconverted sibling means there is nothing to do.
    (tmp_path / "Report.pdf").write_bytes(b"%PDF-1.4\n")
    assert not needs_conversion(legacy)


def test_extension_match_is_case_insensitive(tmp_path: Path):
    shouty = tmp_path / "MINUTES.DOC"
    shouty.write_bytes(b"\xd0\xcf\x11\xe0")
    assert needs_conversion(shouty), "uppercase extensions must still convert"


def test_find_convertible_files_picks_up_legacy(tmp_path: Path):
    (tmp_path / "a.docx").write_bytes(b"PK\x03\x04")
    (tmp_path / "b.ppt").write_bytes(b"\xd0\xcf\x11\xe0")
    (tmp_path / "c.txt").write_text("not office")

    found = {p.name for p in find_convertible_files(tmp_path)}

    assert found == {"a.docx", "b.ppt"}


def test_same_stem_office_files_are_left_to_the_judging_path(tmp_path: Path):
    """Both would convert to ``Report.pdf`` -- one racing the other in the thread pool.

    Observed once in the 200-task Mercor corpus
    (``Pineloten_Capital_Structure.{docx,pptx}``): the single PDF present is the
    .docx render (612x792 portrait), so the .pptx was silently never converted.
    """
    (tmp_path / "Report.docx").write_bytes(b"PK\x03\x04")
    (tmp_path / "Report.pptx").write_bytes(b"PK\x03\x04")
    (tmp_path / "Unique.docx").write_bytes(b"PK\x03\x04")

    found = {p.name for p in find_convertible_files(tmp_path)}

    assert found == {"Unique.docx"}, "ambiguous stems must not be preconverted"
