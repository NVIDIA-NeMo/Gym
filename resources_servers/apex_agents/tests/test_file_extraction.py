# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from pypdf import PdfWriter

from resources_servers.apex_agents.file_extraction import (
    DocumentConverterError,
    ExtractedContent,
    _changed_office_pages,
    _convert_office_to_pdf,
    _DocumentTool,
    _docx_review_text,
    _resolve_document_command,
    _run_document_tool,
    conversion_timeouts,
    converter_setup_problem,
    extract_file_content,
    extract_file_text,
    resolve_converter_image,
    start_conversion_record,
    visual_blocks_for_file,
    visual_content_blocks,
)


_MODULE = "resources_servers.apex_agents.file_extraction"
_HOST_OFFICE = shutil.which("soffice") or shutil.which("libreoffice")
needs_host_office = pytest.mark.skipif(
    _HOST_OFFICE is None or shutil.which("pdftoppm") is None, reason="LibreOffice and Poppler are not installed"
)


def test_pptx_extraction_includes_table_cells(tmp_path: Path) -> None:
    path = tmp_path / "deliverable.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(7), Inches(2)).table
    table.cell(0, 0).text = "Country–Region"
    table.cell(0, 1).text = "Avoided Curtailment"
    table.cell(1, 0).text = "Germany–North"
    table.cell(1, 1).text = "228,025.94"
    presentation.save(path)

    text = extract_file_text(path)

    assert "Country–Region\tAvoided Curtailment" in text
    assert "Germany–North\t228,025.94" in text


def test_pptx_extraction_preserves_archipelago_slide_records(tmp_path: Path) -> None:
    path = tmp_path / "deliverable.pptx"
    presentation = Presentation()
    first = presentation.slides.add_slide(presentation.slide_layouts[1])
    first.shapes.title.text = "Summary"
    first.placeholders[1].text = "First slide body"
    second = presentation.slides.add_slide(presentation.slide_layouts[1])
    second.shapes.title.text = "Appendix"
    second.placeholders[1].text = "Second slide body"
    presentation.save(path)

    extracted = extract_file_content(path)

    assert [(slide.index, slide.type, slide.title) for slide in extracted.sub_artifacts] == [
        (0, "slide", "Summary"),
        (1, "slide", "Appendix"),
    ]
    assert "First slide body" in extracted.sub_artifacts[0].content
    assert "Second slide body" in extracted.sub_artifacts[1].content


def test_docx_extraction_includes_tables(tmp_path: Path) -> None:
    path = tmp_path / "deliverable.docx"
    document = Document()
    document.add_paragraph("Executive summary")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Finding"
    table.cell(1, 0).text = "Thermal Overload"
    table.cell(1, 1).text = "42.4%"
    document.save(path)

    text = extract_file_text(path)

    assert "Executive summary" in text
    assert "Metric\tFinding" in text
    assert "Thermal Overload\t42.4%" in text


def test_docx_review_extraction_includes_redlines_and_anchored_comments() -> None:
    document_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
      <w:p><w:r><w:t>7.7 Termination</w:t></w:r></w:p>
      <w:p><w:commentRangeStart w:id="0"/><w:r><w:t>30 days</w:t></w:r><w:commentRangeEnd w:id="0"/>
        <w:del w:author="Reviewer"><w:r><w:delText>30 days</w:delText></w:r></w:del>
        <w:ins w:author="Agent"><w:r><w:t>45 days</w:t></w:r></w:ins></w:p>
    </w:body></w:document>"""
    comments_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:comment w:id="0" w:author="Agent"><w:p><w:r><w:t>Consider a longer cure period.</w:t></w:r></w:p></w:comment>
    </w:comments>"""
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/comments.xml", comments_xml)

    text = _docx_review_text(source.getvalue())

    assert '[7.7 Termination] DELETED: "30 days"' in text
    assert '[7.7 Termination] INSERTED: "45 days"' in text
    assert '[7.7 Termination] on "30 days": Consider a longer cure period.' in text


def test_xlsx_extraction_includes_formulas_formats_and_skips_hidden_sheets(tmp_path: Path, monkeypatch) -> None:
    # No converter on this host: exercise the fallback even where LibreOffice is installed.
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction._resolve_document_command", lambda *_args, **_kwargs: None
    )
    path = tmp_path / "deliverable.xlsx"
    workbook = Workbook()
    visible = workbook.active
    visible.title = "Summary"
    visible["A1"] = "Revenue"
    visible["B1"] = 125000
    visible["B1"].number_format = "$#,##0"
    visible["B2"] = "=B1*2"
    hidden = workbook.create_sheet("Raw")
    hidden["A1"] = "should not be graded"
    hidden.sheet_state = "hidden"
    workbook.save(path)

    text = extract_file_text(path)

    assert "=== Sheet: Summary ===" in text
    assert "B1: [fmt: $#,##0]" in text
    assert "B2: =B1*2" in text
    assert "formula cells could not be recalculated" in text
    assert "should not be graded" not in text

    extracted = extract_file_content(path)
    assert [(sheet.index, sheet.type, sheet.title) for sheet in extracted.sub_artifacts] == [(0, "sheet", "Summary")]
    assert "=== Sheet: Summary ===" in extracted.sub_artifacts[0].content
    assert "formula cells could not be recalculated" in extracted.sub_artifacts[0].content


def test_xlsx_extraction_uses_libreoffice_recalculated_values(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "formula.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = 125000
    workbook.active["A2"] = "=A1*2"
    workbook.save(path)

    recalculated_path = tmp_path / "recalculated.xlsx"
    recalculated = Workbook()
    recalculated.active["A1"] = 125000
    recalculated.active["A2"] = 250000
    recalculated.save(recalculated_path)
    calls: list[tuple[Path, str | None]] = []

    def fake_recalculate(actual_path: Path, *, document_converter_image: str | None, **_kwargs) -> bytes:
        calls.append((actual_path, document_converter_image))
        return recalculated_path.read_bytes()

    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction._evaluate_excel_formulas_with_libreoffice",
        fake_recalculate,
    )

    extracted = extract_file_content(path, document_converter_image="/images/archipelago.sif")

    assert calls == [(path, "/images/archipelago.sif")]
    assert "250000" in extracted.text
    assert "A2: =A1*2" in extracted.text
    assert "formula cells could not be recalculated" not in extracted.text


def test_xlsx_extraction_attaches_archipelago_chart_metadata(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "chart.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Quarter", "Revenue"])
    sheet.append(["Q1", 10])
    sheet.append(["Q2", 15])
    chart = BarChart()
    chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=3), titles_from_data=True)
    sheet.add_chart(chart, "D2")
    workbook.save(path)

    pdf_dir = tmp_path / "rendered"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "chart.pdf"
    pdf_path.write_bytes(b"rendered-pdf")
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction._convert_office_to_pdf",
        lambda *_args, **_kwargs: pdf_path,
    )
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction._pdf_to_chart_images",
        lambda *_args, **_kwargs: [
            {
                "url": "data:image/png;base64,Y2hhcnQ=",
                "placeholder": "[CHART_1]",
                "type": "Chart",
                "caption": "Chart from Excel (Page 1)",
            }
        ],
    )

    extracted = extract_file_content(path, document_converter_image="/images/archipelago.sif")

    assert extracted.images[0]["placeholder"] == "[CHART_1]"
    assert "=== Charts ===\n[CHART_1] - Chart from Excel (Page 1)" in extracted.text


def test_pdf_extraction_marks_image_only_documents(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as stream:
        writer.write(stream)

    text = extract_file_text(path)

    assert "likely scanned/image-only" in text
    assert "=== Page 1 ===" in text
    extracted = extract_file_content(path)
    assert [(page.index, page.type, page.title) for page in extracted.sub_artifacts] == [(0, "page", "Page 1")]
    assert "likely scanned/image-only" in extracted.sub_artifacts[0].content


def test_notebook_extraction_drops_base64_images(tmp_path: Path) -> None:
    path = tmp_path / "analysis.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Findings"]},
                    {
                        "cell_type": "code",
                        "source": ["print('ok')"],
                        "outputs": [
                            {
                                "output_type": "display_data",
                                "data": {"text/plain": ["<Figure>"], "image/png": "very-large-base64"},
                            }
                        ],
                    },
                ]
            }
        )
    )

    text = extract_file_text(path)

    assert "# Findings" in text
    assert "print('ok')" in text
    assert "<Figure>" in text
    assert "very-large-base64" not in text


def test_pdf_visuals_are_png_pages_not_pdf_image_urls(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "deliverable.pdf"
    path.write_bytes(b"not parsed because rendering is mocked")
    monkeypatch.setattr("pdf2image.convert_from_path", lambda *args, **kwargs: [Image.new("RGB", (2, 2))])
    # Host tools are allowed but pdftoppm is not found: exercise the pdf2image fallback.
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction._resolve_document_command", lambda *_args, **_kwargs: None
    )

    blocks = visual_content_blocks(tmp_path, allow_host_document_tools=True)

    image_blocks = [block for block in blocks if block["type"] == "image_url"]
    assert len(image_blocks) == 1
    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")


def test_pdf_visuals_render_only_requested_page(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "deliverable.pdf"
    path.write_bytes(b"not parsed because rendering is mocked")
    calls: list[tuple[int | None, int | None]] = []

    def fake_convert(*_args, **kwargs):
        calls.append((kwargs.get("first_page"), kwargs.get("last_page")))
        return [Image.new("RGB", (2, 2))]

    monkeypatch.setattr("pdf2image.convert_from_path", fake_convert)
    # Host tools are allowed but pdftoppm is not found: exercise the pdf2image fallback.
    monkeypatch.setattr(
        "resources_servers.apex_agents.file_extraction._resolve_document_command", lambda *_args, **_kwargs: None
    )

    blocks = visual_content_blocks(tmp_path, pdf_page_indices={path.name: {2}}, allow_host_document_tools=True)

    assert calls == [(3, 3)]
    assert any(block == {"type": "text", "text": "Page 3"} for block in blocks)
    assert sum(block["type"] == "image_url" for block in blocks) == 1


def _which(*present: str):
    return lambda executable: f"/usr/bin/{executable}" if executable in present else None


def test_document_tools_run_only_in_the_converter_image_by_default(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "archipelago.sif"
    image.write_bytes(b"sif")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    # LibreOffice is installed on the host, and is still not used.
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("libreoffice", "soffice", "pdftoppm", "apptainer"))

    assert _resolve_document_command(("libreoffice", "soffice"), None) is None

    tool = _resolve_document_command(("libreoffice", "soffice"), str(image))
    command, root = tool.command(scratch)

    assert tool.containerized
    assert root == "/apex-convert"
    assert command == [
        "/usr/bin/apptainer",
        "exec",
        "--cleanenv",
        "--containall",
        "--workdir",
        str(scratch / ".apptainer-workdir"),
        "--bind",
        f"{scratch}:/apex-convert",
        str(image),
        "libreoffice",
    ]
    # The only host path the container is given is the scratch directory.
    assert [argument for argument in command if str(tmp_path) in argument and str(scratch) not in argument] == [
        str(image)
    ]


def test_host_document_tools_are_opt_in_and_then_preferred(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "archipelago.sif"
    image.write_bytes(b"sif")
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("soffice", "apptainer"))

    tool = _resolve_document_command(("libreoffice", "soffice"), str(image), allow_host_document_tools=True)

    assert not tool.containerized
    assert tool.command(tmp_path) == (["/usr/bin/soffice"], str(tmp_path))
    # Allowed but not installed: the image is used.
    assert _resolve_document_command(("pdftoppm",), str(image), allow_host_document_tools=True).containerized


@pytest.mark.parametrize("missing", ["apptainer", "image"])
def test_a_configured_converter_that_cannot_run_is_a_converter_error(
    monkeypatch, tmp_path: Path, missing: str
) -> None:
    image = tmp_path / "archipelago.sif"
    if missing != "image":
        image.write_bytes(b"sif")
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which(*([] if missing == "apptainer" else ["apptainer"])))

    with pytest.raises(DocumentConverterError, match="apptainer is not on PATH|image not found"):
        _resolve_document_command(("libreoffice", "soffice"), str(image))


def test_a_container_that_cannot_start_is_a_converter_error_but_a_rejected_file_is_not(
    monkeypatch, tmp_path: Path
) -> None:
    tool = _DocumentTool("libreoffice", apptainer="/usr/bin/apptainer", image="/images/archipelago.sif")
    outcomes = iter(
        [
            subprocess.CompletedProcess([], 255, b"", b"FATAL:   could not open image /images/archipelago.sif"),
            subprocess.CompletedProcess([], 1, b"", b"Error: source file could not be loaded"),
        ]
    )
    monkeypatch.setattr(f"{_MODULE}.subprocess.run", lambda *_args, **_kwargs: next(outcomes))

    with pytest.raises(DocumentConverterError, match="could not open image"):
        _run_document_tool(tool, tmp_path, ["--headless"])
    # LibreOffice refusing the file is the file's fault: the caller falls back to the file's text.
    assert _run_document_tool(tool, tmp_path, ["--headless"]).returncode == 1

    def cannot_start(*_args, **_kwargs):
        raise FileNotFoundError("apptainer")

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", cannot_start)
    with pytest.raises(DocumentConverterError, match="could not start"):
        _run_document_tool(tool, tmp_path, ["--headless"])


def test_a_converter_fault_is_raised_not_replaced_by_a_text_fallback(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "memo.docx"
    document = Document()
    document.add_paragraph("Quarterly memo")
    document.save(path)
    image = tmp_path / "archipelago.sif"
    image.write_bytes(b"sif")
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which())

    with pytest.raises(DocumentConverterError):
        visual_blocks_for_file(path, document_converter_image=str(image))
    with pytest.raises(DocumentConverterError):
        visual_content_blocks(tmp_path, document_converter_image=str(image))

    # A file LibreOffice itself cannot convert is still graded on its text.
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("apptainer"))
    monkeypatch.setattr(
        f"{_MODULE}.subprocess.run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b"Error")
    )
    blocks = visual_blocks_for_file(path, document_converter_image=str(image))
    assert blocks == [{"type": "text", "text": "\nmemo.docx (text fallback):\nQuarterly memo"}]

    # With no converter configured there is nothing to fail: text only, as before.
    assert visual_blocks_for_file(path) == blocks


def test_the_shipped_config_defaults_the_converter_to_the_agent_image() -> None:
    from omegaconf import OmegaConf

    config_path = Path(__file__).parents[1] / "configs" / "apex_agents.yaml"
    default_image = "responses_api_agents/apex_agent/images/archipelago.sif"

    def server_config(**overrides: str):
        config = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.create(overrides))
        return config.apex_agents.resources_servers.apex_agents

    assert server_config().document_converter_image == default_image
    assert server_config().allow_host_document_tools is False
    assert server_config(apex_agents_image="/images/other.sif").document_converter_image == "/images/other.sif"
    assert server_config(apex_allow_host_document_tools="true").allow_host_document_tools == "true"
    # The agent's own default is the same image.
    agent_config = Path(__file__).parents[3] / "responses_api_agents" / "apex_agent" / "configs" / "apex_agent.yaml"
    assert f"apex_agents_image,{default_image}" in agent_config.read_text(encoding="utf-8")


def _pdf_with_pages(path: Path, count: int, sizes: dict[int, tuple[int, int]] | None = None) -> Path:
    writer = PdfWriter()
    for index in range(count):
        width, height = (sizes or {}).get(index, (72, 72))
        writer.add_blank_page(width=width, height=height)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _deck(path: Path, count: int, hidden: set[int] = frozenset()) -> Path:
    presentation = Presentation()
    for index in range(count):
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = f"Slide {index + 1}"
        if index in hidden:
            slide._element.set("show", "0")
    presentation.save(path)
    return path


def test_changed_slides_map_to_their_pdf_pages_past_hidden_slides(tmp_path: Path) -> None:
    deck = _deck(tmp_path / "deck.pptx", 16, hidden={1})

    # Slide 2 is hidden and gets no page, so slide 14 is page 13 (index 12).
    pages = _changed_office_pages(deck, _pdf_with_pages(tmp_path / "deck.pdf", 15), {13, 1}, single_page_sheets=False)
    assert pages == {12: "Slide 14"}
    # A page count that does not match the shown slides ties no page to a slide.
    assert (
        _changed_office_pages(deck, _pdf_with_pages(tmp_path / "odd.pdf", 9), {13}, single_page_sheets=False) is None
    )


def test_a_change_on_slide_14_is_shown_as_slide_14_not_the_first_pages(monkeypatch, tmp_path: Path) -> None:
    deck = _deck(tmp_path / "deck.pptx", 16)
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    pdf_path = _pdf_with_pages(rendered / "document.pdf", 16)
    requested: list[set[int] | None] = []

    def page_urls(_pdf: Path, *, page_indices: set[int] | None = None, **_converter):
        requested.append(page_indices)
        return [(index + 1, "data:image/png;base64,cGFnZQ==") for index in sorted(page_indices or range(10))]

    monkeypatch.setattr(f"{_MODULE}._convert_office_to_pdf", lambda *_args, **_kwargs: pdf_path)
    monkeypatch.setattr(f"{_MODULE}._pdf_page_data_urls", page_urls)

    blocks = visual_blocks_for_file(deck, document_converter_image="/images/archipelago.sif", changed_indices={13})

    assert requested == [{13}]
    assert [block.get("text") for block in blocks if block["type"] == "text"] == [
        "\ndeck.pptx (rendered from Office):",
        "Slide 14",
    ]
    assert sum(block["type"] == "image_url" for block in blocks) == 1
    assert not rendered.exists()


def test_changed_sheets_use_the_one_sheet_per_page_export(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "model.xlsx"
    workbook = Workbook()
    workbook.active.title = "Summary"
    for title in ("Inputs", "Scenarios", "Outputs"):
        workbook.create_sheet(title).append([title])
    workbook.save(path)
    exports: list[bool] = []
    requested: list[set[int] | None] = []

    def convert(_path: Path, *, single_page_sheets: bool = False, **_converter) -> Path:
        exports.append(single_page_sheets)
        rendered = tmp_path / f"rendered-{len(exports)}"
        rendered.mkdir()
        return _pdf_with_pages(rendered / "document.pdf", 4 if single_page_sheets else 9)

    def page_urls(_pdf: Path, *, page_indices: set[int] | None = None, **_converter):
        requested.append(page_indices)
        return [(index + 1, "data:image/png;base64,cGFnZQ==") for index in sorted(page_indices or range(9))]

    monkeypatch.setattr(f"{_MODULE}._convert_office_to_pdf", convert)
    monkeypatch.setattr(f"{_MODULE}._pdf_page_data_urls", page_urls)

    blocks = visual_blocks_for_file(path, document_converter_image="/images/archipelago.sif", changed_indices={2})

    assert exports == [True]
    assert requested == [{2}]
    assert {"type": "text", "text": "Sheet 3"} in blocks

    # With no known changed sheet the ordinary paginated export and its leading pages are used, as before.
    visual_blocks_for_file(path, document_converter_image="/images/archipelago.sif")
    assert exports == [True, False]
    assert requested == [{2}, None]


def test_a_changed_sheet_too_long_for_one_page_falls_back_to_the_paginated_export(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "model.xlsx"
    workbook = Workbook()
    workbook.active.title = "Summary"
    workbook.create_sheet("Ledger").append(["Ledger"])
    workbook.save(path)
    # Sheet 1 fits a page; sheet 2 has some 300 rows, exported as one page 96 pt wide and 4500 pt tall.
    single_page = _pdf_with_pages(tmp_path / "single.pdf", 2, sizes={0: (385, 900), 1: (96, 4500)})

    assert _changed_office_pages(path, single_page, {0}, single_page_sheets=True) == {0: "Sheet 1"}
    assert _changed_office_pages(path, single_page, {1}, single_page_sheets=True) is None
    # One unreadable changed sheet sends the whole file to the paginated export.
    assert _changed_office_pages(path, single_page, {0, 1}, single_page_sheets=True) is None
    # A sheet many columns wide is as unreadable as a long one.
    wide = _pdf_with_pages(tmp_path / "wide.pdf", 2, sizes={0: (2889, 75)})
    assert _changed_office_pages(path, wide, {0}, single_page_sheets=True) is None

    exports: list[bool] = []
    requested: list[set[int] | None] = []
    rendered: list[Path] = []

    def convert(_path: Path, *, single_page_sheets: bool = False, **_converter) -> Path:
        exports.append(single_page_sheets)
        directory = tmp_path / f"rendered-{len(exports)}"
        directory.mkdir()
        rendered.append(directory)
        if single_page_sheets:
            return _pdf_with_pages(directory / "document.pdf", 2, sizes={1: (96, 4500)})
        return _pdf_with_pages(directory / "document.pdf", 8)

    def page_urls(_pdf: Path, *, page_indices: set[int] | None = None, **_converter):
        requested.append(page_indices)
        return [(index + 1, "data:image/png;base64,cGFnZQ==") for index in sorted(page_indices or range(8))]

    monkeypatch.setattr(f"{_MODULE}._convert_office_to_pdf", convert)
    monkeypatch.setattr(f"{_MODULE}._pdf_page_data_urls", page_urls)

    blocks = visual_blocks_for_file(path, document_converter_image="/images/archipelago.sif", changed_indices={1})

    assert exports == [True, False]
    assert requested == [None]
    assert [block["text"] for block in blocks if block["type"] == "text"][1:] == [f"Page {n}" for n in range(1, 9)]
    assert not any(directory.exists() for directory in rendered)


@needs_host_office
def test_libreoffice_paginates_a_changed_sheet_too_long_for_one_page(tmp_path: Path) -> None:
    from pypdf import PdfReader

    path = tmp_path / "ledger.xlsx"
    workbook = Workbook()
    workbook.active.title = "Summary"
    workbook.active.append(["Total", 300])
    ledger = workbook.create_sheet("Ledger")
    for row in range(300):
        ledger.append([f"entry {row}", row])
    workbook.save(path)

    pdf_path = _convert_office_to_pdf(path, allow_host_document_tools=True, single_page_sheets=True)
    try:
        heights = [float(page.mediabox.height) for page in PdfReader(str(pdf_path)).pages]
        # The measured rule: the short sheet's page is small, the 300-row sheet's page is far past the limit.
        assert heights[0] < 1200 < heights[1]
        assert _changed_office_pages(path, pdf_path, {0}, single_page_sheets=True) == {0: "Sheet 1"}
        assert _changed_office_pages(path, pdf_path, {1}, single_page_sheets=True) is None
    finally:
        shutil.rmtree(pdf_path.parent)

    short = visual_blocks_for_file(path, allow_host_document_tools=True, changed_indices={0})
    long = visual_blocks_for_file(path, allow_host_document_tools=True, changed_indices={1})

    assert [block["text"] for block in short if block["type"] == "text"][1:] == ["Sheet 1"]
    # The long sheet is shown as ordinary pages, several of them, each a readable page image.
    labels = [block["text"] for block in long if block["type"] == "text"][1:]
    assert len(labels) > 1 and labels[0] == "Page 1"
    for block in long:
        if block["type"] == "image_url":
            image = Image.open(io.BytesIO(base64.b64decode(block["image_url"]["url"].split(",", 1)[1])))
            assert max(image.size) <= 2500


@needs_host_office
def test_libreoffice_renders_the_changed_slide_and_sheet(tmp_path: Path) -> None:
    deck = _deck(tmp_path / "deck.pptx", 14, hidden={1})
    workbook_path = tmp_path / "model.xlsx"
    workbook = Workbook()
    workbook.active.title = "Summary"
    for row in range(120):
        workbook.active.append([row] * 12)
    for title in ("Inputs", "Outputs"):
        workbook.create_sheet(title).append([title])
    workbook.save(workbook_path)

    slide_blocks = visual_blocks_for_file(deck, allow_host_document_tools=True, changed_indices={13})
    sheet_blocks = visual_blocks_for_file(workbook_path, allow_host_document_tools=True, changed_indices={2})

    for blocks, label in ((slide_blocks, "Slide 14"), (sheet_blocks, "Sheet 3")):
        assert [block["text"] for block in blocks if block["type"] == "text"][1:] == [label]
        images = [block for block in blocks if block["type"] == "image_url"]
        assert len(images) == 1
        assert base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG")

    # The export really is one page per shown slide and per sheet, which is what the labels rely on.
    from pypdf import PdfReader

    for path, single_page_sheets, pages in ((deck, False, 13), (workbook_path, True, 3)):
        pdf_path = _convert_office_to_pdf(path, allow_host_document_tools=True, single_page_sheets=single_page_sheets)
        try:
            assert len(PdfReader(str(pdf_path)).pages) == pages
        finally:
            shutil.rmtree(pdf_path.parent)


def test_a_text_file_is_read_only_up_to_the_extraction_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MODULE}._MAX_EXTRACTED_TEXT_CHARS", 1_000)
    log = tmp_path / "run.log"
    log.write_text("line of log output\n" * 10_000, encoding="utf-8")
    table = tmp_path / "rows.csv"
    table.write_text("a,b\n" * 10_000, encoding="utf-8")

    log_text = extract_file_text(log)
    table_text = extract_file_text(table)

    for text in (log_text, table_text):
        assert text.endswith("[...file content truncated: extraction limit reached]")
        assert len(text) < 1_100
    assert log_text.startswith("line of log output")
    assert table_text.startswith("a\tb\na\tb")
    # A file within the limit is read whole and unmarked.
    small = tmp_path / "note.txt"
    small.write_text("short note", encoding="utf-8")
    assert extract_file_text(small) == "short note"


@pytest.mark.parametrize("extension", [".docx", ".xlsx", ".pptx"])
def test_an_office_file_with_an_oversized_part_is_not_opened(monkeypatch, tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"bomb{extension}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("media/padding.bin", b"\0" * 200_000)
    assert path.stat().st_size < 5_000
    monkeypatch.setattr(f"{_MODULE}._MAX_OFFICE_PART_BYTES", 100_000)

    def opened(*_args, **_kwargs):
        raise AssertionError("the package was opened")

    for extractor in ("_extract_docx", "_extract_xlsx", "_extract_pptx"):
        monkeypatch.setattr(f"{_MODULE}.{extractor}", opened)

    text = extract_file_text(path)

    assert text.startswith("[File not extracted: ")
    assert "'media/padding.bin' expands to 200000 bytes" in text


def test_an_oversized_pdf_or_notebook_is_not_parsed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MODULE}._MAX_DOCUMENT_BYTES", 100)
    for name in ("report.pdf", "analysis.ipynb", "legacy.xls"):
        path = tmp_path / name
        path.write_bytes(b"x" * 500)
        assert extract_file_text(path) == f"[File not extracted: {name} is 500 bytes; the extraction limit is 100]"


def _legacy_source(tmp_path: Path, target: str) -> Path:
    path = tmp_path / f"source.{target}"
    if target == "docx":
        document = Document()
        document.add_paragraph("Net revenue was 4217 thousand")
        document.save(path)
    elif target == "xlsx":
        workbook = Workbook()
        workbook.active.append(["Net revenue", 4217])
        workbook.save(path)
    else:
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Net revenue 4217"
        presentation.save(path)
    return path


@pytest.mark.parametrize(
    ("extension", "target"),
    [(".doc", "docx"), (".odt", "docx"), (".ods", "xlsx"), (".ppt", "pptx"), (".odp", "pptx")],
)
def test_legacy_office_formats_are_read_through_a_conversion(
    monkeypatch, tmp_path: Path, extension: str, target: str
) -> None:
    path = tmp_path / f"deliverable{extension}"
    path.write_bytes(b"legacy bytes")
    conversions: list[tuple[Path, str]] = []

    def convert(source: Path, convert_to: str, **_kwargs) -> Path:
        conversions.append((source, convert_to))
        converted_dir = tmp_path / "converted"
        converted_dir.mkdir()
        return shutil.copyfile(_legacy_source(tmp_path, target), converted_dir / f"document.{target}")

    monkeypatch.setattr(f"{_MODULE}._convert_with_libreoffice", convert)

    text = extract_file_text(path, document_converter_image="/images/archipelago.sif")

    assert conversions == [(path, target)]
    assert "4217" in text
    assert "Binary file" not in text
    assert not (tmp_path / "converted").exists()

    # Without a converter the judge is told the file was not read, rather than given an unexplained placeholder.
    monkeypatch.setattr(f"{_MODULE}._convert_with_libreoffice", lambda *_args, **_kwargs: None)
    assert extract_file_text(path) == (
        f"[Binary file: deliverable{extension}, 12 bytes; not converted to a readable format]"
    )


@needs_host_office
@pytest.mark.parametrize(
    ("extension", "target"),
    [(".doc", "docx"), (".odt", "docx"), (".ods", "xlsx"), (".ppt", "pptx"), (".odp", "pptx")],
)
def test_libreoffice_makes_legacy_office_formats_readable(tmp_path: Path, extension: str, target: str) -> None:
    source = _legacy_source(tmp_path, target)
    profile = tmp_path / "profile"
    subprocess.run(
        [
            _HOST_OFFICE,
            "--headless",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to",
            extension.lstrip("."),
            "--outdir",
            str(tmp_path),
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    path = tmp_path / f"source{extension}"
    assert path.is_file()

    text = extract_file_text(path, allow_host_document_tools=True)
    blocks = visual_blocks_for_file(path, allow_host_document_tools=True)

    assert "4217" in text
    assert sum(block["type"] == "image_url" for block in blocks) >= 1
    # Host tools are opt-in: by default the same file is not handed to the host's LibreOffice.
    assert "not converted to a readable format" in extract_file_text(path)


def test_converter_image_references_resolve_as_apptainer_takes_them(tmp_path: Path) -> None:
    squashfs = tmp_path / "images" / "converter.squashfs"
    squashfs.parent.mkdir()
    squashfs.write_bytes(b"image")

    assert resolve_converter_image(None, tmp_path) is None
    assert resolve_converter_image("", tmp_path) is None
    # Registry references are not local paths: a URI passes through, a bare OCI reference is a docker:// one.
    for uri in (
        "docker://registry.example/apex/archipelago:1.0",
        "oras://registry.example/apex:sif",
        "library://a/b:1",
    ):
        assert resolve_converter_image(uri, tmp_path) == uri
    assert resolve_converter_image("registry.example/apex/archipelago:1.0", tmp_path) == (
        "docker://registry.example/apex/archipelago:1.0"
    )
    # Local images: by suffix whether or not they exist yet, by path syntax, or any existing file.
    assert resolve_converter_image("images/archipelago.sif", tmp_path) == str(tmp_path / "images" / "archipelago.sif")
    assert resolve_converter_image("images/world.sqsh", tmp_path) == str(tmp_path / "images" / "world.sqsh")
    assert resolve_converter_image("./images/world", tmp_path) == str(tmp_path / "images" / "world")
    assert resolve_converter_image("images/converter.squashfs", tmp_path) == str(squashfs)
    assert resolve_converter_image(str(squashfs), tmp_path / "elsewhere") == str(squashfs)


def test_a_registry_image_is_handed_to_apptainer_without_a_file_check(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("apptainer"))
    image = "docker://registry.example/apex/archipelago:1.0"

    assert converter_setup_problem(image) is None
    command, _ = _resolve_document_command(("libreoffice", "soffice"), image).command(tmp_path)

    assert command[-2:] == [image, "libreoffice"]
    # A local image of any extension is accepted once it exists, and reported when it does not.
    local = tmp_path / "converter.sqsh"
    assert "image not found" in converter_setup_problem(str(local))
    local.write_bytes(b"image")
    assert converter_setup_problem(str(local)) is None
    # Nothing configured is never a problem, with or without apptainer.
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which())
    assert converter_setup_problem(None) is None
    assert "apptainer is not on PATH" in converter_setup_problem(image)
    # Host tools that are all installed and allowed stand in for the image.
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("soffice", "pdftoppm"))
    assert converter_setup_problem(image, allow_host_document_tools=True) is None
    assert "apptainer is not on PATH" in converter_setup_problem(image)
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("soffice"))
    assert "apptainer is not on PATH" in converter_setup_problem(image, allow_host_document_tools=True)


def test_a_conversion_timeout_falls_back_to_text_and_is_recorded(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "memo.docx"
    document = Document()
    document.add_paragraph("Quarterly memo")
    document.save(path)
    image = tmp_path / "archipelago.sif"
    image.write_bytes(b"sif")
    monkeypatch.setattr(f"{_MODULE}.shutil.which", _which("apptainer"))

    def hangs(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", hangs)
    recorded = start_conversion_record()

    blocks = visual_blocks_for_file(path, document_converter_image=str(image))

    # A file the model wrote can make a conversion hang, so this is not a converter fault: text, and a record.
    assert blocks == [{"type": "text", "text": "\nmemo.docx (text fallback):\nQuarterly memo"}]
    assert recorded == [{"tool": "libreoffice", "file": "memo.docx", "timeout_seconds": 120}]
    assert conversion_timeouts() is recorded
    assert start_conversion_record() == []


def _chartsheet_workbook(path: Path) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Revenue"
    for row in [("Quarter", "Revenue"), ("Q1", 10), ("Q2", 15)]:
        sheet.append(row)
    chart = BarChart()
    chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=3), titles_from_data=True)
    workbook.create_chartsheet("Revenue chart").add_chart(chart)
    workbook.create_sheet("Notes").append(["Prepared for the board"])
    workbook.save(path)
    return path


def test_a_workbook_with_a_chartsheet_is_extracted_with_its_chart(monkeypatch, tmp_path: Path) -> None:
    path = _chartsheet_workbook(tmp_path / "charted.xlsx")
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    monkeypatch.setattr(f"{_MODULE}._convert_office_to_pdf", lambda *_args, **_kwargs: rendered / "document.pdf")
    monkeypatch.setattr(
        f"{_MODULE}._pdf_to_chart_images",
        lambda *_args, **_kwargs: [
            {
                "url": "data:image/png;base64,Y2hhcnQ=",
                "placeholder": "[CHART_1]",
                "type": "Chart",
                "caption": "Chart from Excel (Page 2)",
            }
        ],
    )

    extracted = extract_file_content(path, document_converter_image="/images/archipelago.sif")

    assert isinstance(extracted, ExtractedContent)
    assert "Q2\t15" in extracted.text and "Prepared for the board" in extracted.text
    assert "Error" not in extracted.text
    # The chartsheet has no cells to walk, and is still a sheet of the workbook with its chart fingerprinted ...
    assert [(artifact.index, artifact.title) for artifact in extracted.sub_artifacts] == [
        (0, "Revenue"),
        (1, "Revenue chart"),
        (2, "Notes"),
    ]
    chartsheet = extracted.sub_artifacts[1]
    assert chartsheet.content == "=== Sheet: Revenue chart ===\n[chart sheet: no cells]"
    assert len(chartsheet.images) == 1
    # ... and the workbook's chart images are still rendered and attached.
    assert extracted.images[0]["placeholder"] == "[CHART_1]"
    assert "=== Charts ===\n[CHART_1] - Chart from Excel (Page 2)" in extracted.text


@needs_host_office
def test_libreoffice_renders_the_chart_of_a_chartsheet(tmp_path: Path) -> None:
    extracted = extract_file_content(_chartsheet_workbook(tmp_path / "charted.xlsx"), allow_host_document_tools=True)

    assert "Q2\t15" in extracted.text
    assert extracted.images and extracted.images[0]["url"].startswith("data:image/png;base64,")
