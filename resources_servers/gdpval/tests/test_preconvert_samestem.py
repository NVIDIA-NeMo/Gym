# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two Office files sharing a stem must not race for one PDF name."""

from pathlib import Path

from resources_servers.gdpval.preconvert import (
    LEGACY_OFFICE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    find_convertible_files,
    needs_conversion,
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
