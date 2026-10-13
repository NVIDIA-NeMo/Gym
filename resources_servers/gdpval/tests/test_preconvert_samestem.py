# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two Office files sharing a stem must not race for one PDF name."""

from pathlib import Path

from resources_servers.gdpval.preconvert import (
    LEGACY_OFFICE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    find_convertible_files,
    needs_conversion,
    resolve_pdf_provenance,
    sidecar_pdf,
)


def test_sidecar_name_is_injective():
    assert sidecar_pdf(Path("/d/Plan.pptx")).name == "Plan.pptx.pdf"
    assert sidecar_pdf(Path("/d/Plan.xlsx")).name == "Plan.xlsx.pdf"


def test_same_stem_pair_gets_distinct_destinations(tmp_path: Path):
    (tmp_path / "Plan.pptx").write_bytes(b"PK\x03\x04")
    (tmp_path / "Plan.xlsx").write_bytes(b"PK\x03\x04")

    dests = {src.name: dest for src, dest in find_convertible_files(tmp_path)}
    assert dests["Plan.pptx"].name == "Plan.pptx.pdf"
    assert dests["Plan.xlsx"].name == "Plan.xlsx.pdf"
    assert dests["Plan.pptx"] != dests["Plan.xlsx"], "both would overwrite one PDF"


def test_unambiguous_stem_keeps_the_plain_sibling_name(tmp_path: Path):
    (tmp_path / "Report.docx").write_bytes(b"PK\x03\x04")

    dests = dict(find_convertible_files(tmp_path))
    assert dests[tmp_path / "Report.docx"] is None, "unambiguous files keep Report.pdf"


def test_existing_sidecar_means_no_reconversion(tmp_path: Path):
    (tmp_path / "Plan.pptx").write_bytes(b"PK\x03\x04")
    (tmp_path / "Plan.xlsx").write_bytes(b"PK\x03\x04")
    (tmp_path / "Plan.pptx.pdf").write_bytes(b"%PDF-1.4")

    srcs = {src.name for src, _ in find_convertible_files(tmp_path)}
    assert "Plan.pptx" not in srcs
    assert "Plan.xlsx" in srcs


def test_existing_sidecar_remains_valid_if_stem_becomes_unambiguous(tmp_path: Path):
    source = tmp_path / "Plan.pptx"
    source.write_bytes(b"PK\x03\x04")
    sidecar_pdf(source).write_bytes(b"%PDF-1.4")

    assert find_convertible_files(tmp_path) == []


def test_ambiguous_conversion_checks_the_sidecar_not_the_sibling(tmp_path: Path):
    p = tmp_path / "Plan.pptx"
    p.write_bytes(b"PK\x03\x04")
    (tmp_path / "Plan.pdf").write_bytes(b"%PDF-1.4")

    assert needs_conversion(p, ambiguous=True), "a shared Plan.pdf says nothing about this file"
    assert not needs_conversion(p, ambiguous=False)


def test_legacy_office_is_converted_too():
    assert LEGACY_OFFICE_EXTENSIONS <= OFFICE_EXTENSIONS
    for ext in (".doc", ".ppt", ".xls"):
        assert ext in OFFICE_EXTENSIONS


def test_ambiguous_plain_pdf_is_quarantined_but_independent_pdf_survives(tmp_path: Path):
    docx = tmp_path / "Plan.docx"
    pptx = tmp_path / "Plan.pptx"
    stale = tmp_path / "Plan.pdf"
    independent = tmp_path / "Appendix.pdf"
    for path in (docx, pptx):
        path.write_bytes(b"PK\x03\x04")
    stale.write_bytes(b"stale")
    independent.write_bytes(b"independent")

    provenance = resolve_pdf_provenance(tmp_path.iterdir())

    assert provenance.office_pdfs == {}
    assert stale in provenance.ambiguous_pdfs
    assert stale in provenance.suppressed_pdfs
    assert independent not in provenance.suppressed_pdfs


def test_injective_sidecars_win_and_every_derived_pdf_is_consumed(tmp_path: Path):
    docx = tmp_path / "Plan.docx"
    pptx = tmp_path / "Plan.pptx"
    stale = tmp_path / "Plan.pdf"
    docx_sidecar = sidecar_pdf(docx)
    pptx_sidecar = sidecar_pdf(pptx)
    for path in (docx, pptx):
        path.write_bytes(b"PK\x03\x04")
    stale.write_bytes(b"stale")
    docx_sidecar.write_bytes(b"docx render")
    pptx_sidecar.write_bytes(b"pptx render")

    provenance = resolve_pdf_provenance(tmp_path.iterdir())

    assert provenance.office_pdfs == {docx: docx_sidecar, pptx: pptx_sidecar}
    assert provenance.suppressed_pdfs == frozenset({stale, docx_sidecar, pptx_sidecar})
