# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local extraction and rendering for APEX grading artifacts.

The document extraction behavior in this module is adapted from the Mercor Archipelago grader. Only its offline
artifact-processing layer is carried here; model invocation, artifact-selection LLMs, cloud extractors, and scoring
remain owned by NeMo Gym.

Adapted source (https://github.com/Mercor-Intelligence/archipelago, revision 0cb5c476c219a9df637e0bd37fb86b2361f4ab89, Apache-2.0):
  grading/runner/utils/file_extraction/methods/local_extractor.py
  grading/runner/utils/file_extraction/utils/chart_extraction.py
  grading/runner/utils/file_extraction/utils/docx_tracked_changes.py
The upstream files carry no per-file copyright notice. NVIDIA modifications are licensed under Apache-2.0.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


_TEXT_EXTENSIONS = {
    ".html",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".sql",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_OFFICE_EXTENSIONS = {".doc", ".docx", ".odp", ".ods", ".odt", ".ppt", ".pptx", ".xls", ".xlsm", ".xlsx"}
# Formats with no extractor of their own: LibreOffice converts them to the format whose extractor reads them.
_LEGACY_OFFICE_TARGETS = {".doc": "docx", ".odt": "docx", ".ods": "xlsx", ".ppt": "pptx", ".odp": "pptx"}
_SPREADSHEET_EXTENSIONS = {".xls", ".xlsm", ".xlsx"}
_IMAGE_MIME_TYPES = {
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DOCUMENT_XML_PATH = "word/document.xml"
_COMMENTS_XML_PATH = "word/comments.xml"
_SECTION_HEADING_RE = re.compile(r"^(SECTION|ARTICLE)\s+[0-9IVXLC]+\b|^[0-9]+\.[0-9]+")
_MAX_VISUAL_PAGES = 10
# Bounds on what one file may cost the verifier. A file over a bound is described to the judge, not loaded.
_MAX_EXTRACTED_TEXT_CHARS = 4_000_000
_MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
_MAX_OFFICE_PART_BYTES = 256 * 1024 * 1024
_MAX_OFFICE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
_MAX_INLINE_IMAGE_BYTES = 12 * 1024 * 1024
_TEXT_TRUNCATION_NOTE = "\n[...file content truncated: extraction limit reached]"
_DOCUMENT_TOOL_TIMEOUT_SECONDS = 120
# Where a containerized converter sees its scratch directory, and the private /tmp and home inside that directory.
_CONTAINER_SCRATCH = "/apex-convert"
_CONTAINER_WORKDIR = ".apptainer-workdir"
# A sheet exported as one page is shown only while that page stays readable as one image: about a page and a half
# of rows or columns at most. A longer sheet is rasterized far taller than a judge model reads at full size.
_MAX_SINGLE_PAGE_SHEET_POINTS = 1200
_LOCAL_IMAGE_SUFFIXES = (".sif", ".sqsh", ".sqfs", ".img")
LOG = logging.getLogger(__name__)
# Conversions that timed out during the current verify, for the grading record; None outside a verify.
_CONVERSION_TIMEOUTS: ContextVar[list[dict[str, Any]] | None] = ContextVar("apex_conversion_timeouts", default=None)
_SINGLE_PAGE_SHEETS_PDF = 'pdf:calc_pdf_Export:{"SinglePageSheets":{"type":"boolean","value":"true"}}'


class ExtractionLimitError(ValueError):
    """A file is too large to load for extraction."""


class DocumentConverterError(RuntimeError):
    """The configured document converter could not run: a fault of the verifier's setup, not of the file."""


def start_conversion_record() -> list[dict[str, Any]]:
    """Start a fresh record of the conversions that time out from here on, in the caller's context.

    Worker threads that run with a copy of that context append to the same list.
    """
    timeouts: list[dict[str, Any]] = []
    _CONVERSION_TIMEOUTS.set(timeouts)
    return timeouts


def conversion_timeouts() -> list[dict[str, Any]]:
    """The current record of timed-out conversions, started if the caller's context has none."""
    timeouts = _CONVERSION_TIMEOUTS.get()
    return start_conversion_record() if timeouts is None else timeouts


def resolve_converter_image(image: str | None, base_dir: Path) -> str | None:
    """The converter image as apptainer takes it: an absolute path for a local image, else a registry URI.

    A reference is local when it is written as a path (``/``, ``.``, ``~``), names an image file by its suffix,
    or exists as a file under ``base_dir``. A URI (``docker://``, ``oras://``, ``library://``) passes through
    unchanged, and a bare OCI reference such as ``registry/name:tag`` is addressed as ``docker://``.
    """
    image = (image or "").strip()
    if not image:
        return None
    if "://" in image:
        return image
    path = Path(image).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    if image.startswith(("/", ".", "~")) or image.lower().endswith(_LOCAL_IMAGE_SUFFIXES) or path.is_file():
        return str(path.resolve())
    return f"docker://{image}"


def converter_setup_problem(
    document_converter_image: str | None, *, allow_host_document_tools: bool = False
) -> str | None:
    """Why the configured converter cannot run on this host, or None when it can or none is configured.

    ``document_converter_image`` is a resolved reference (see ``resolve_converter_image``).
    """
    if not document_converter_image:
        return None
    if allow_host_document_tools and all(
        any(shutil.which(executable) for executable in executables)
        for executables in (("libreoffice", "soffice"), ("pdftoppm",))
    ):
        # Every tool is installed on the host and preferred, so the image is never entered.
        return None
    if shutil.which("apptainer") is None:
        return f"document converter image {document_converter_image} is configured but apptainer is not on PATH"
    if "://" not in document_converter_image and not Path(document_converter_image).is_file():
        return f"document converter image not found: {document_converter_image}"
    return None


@dataclass(frozen=True)
class SubArtifact:
    """One Archipelago-style slide, sheet, or page within an artifact."""

    index: int
    type: str
    title: str | None
    content: str
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedContent:
    """Structured extraction result with concatenated text for compatibility."""

    text: str
    images: list[dict[str, Any]] = field(default_factory=list)
    sub_artifacts: list[SubArtifact] = field(default_factory=list)


def extract_file_text(
    path: Path,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
) -> str:
    """Extract judge-readable text using Archipelago's offline format semantics."""
    return extract_file_content(
        path,
        document_converter_image=document_converter_image,
        allow_host_document_tools=allow_host_document_tools,
    ).text


def extract_file_content(
    path: Path,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
) -> ExtractedContent:
    """Extract text and Archipelago-compatible sub-artifact records, bounded per file."""
    try:
        content = _extract_file_content(
            path,
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
    except ExtractionLimitError as exc:
        return ExtractedContent(text=f"[File not extracted: {exc}]")
    return ExtractedContent(
        text=_bounded_text(content.text),
        images=content.images,
        sub_artifacts=[
            SubArtifact(
                index=artifact.index,
                type=artifact.type,
                title=artifact.title,
                content=_bounded_text(artifact.content),
                images=artifact.images,
            )
            for artifact in content.sub_artifacts
        ],
    )


def _extract_file_content(
    path: Path,
    *,
    document_converter_image: str | None,
    allow_host_document_tools: bool,
) -> ExtractedContent:
    extension = path.suffix.lower()
    if extension in _TEXT_EXTENSIONS:
        return ExtractedContent(text=read_text_bounded(path))
    if extension == ".csv":
        return ExtractedContent(text=_extract_csv(path))
    if extension == ".ipynb":
        _check_document_size(path)
        return ExtractedContent(text=_extract_ipynb(path))
    if extension == ".docx":
        _check_office_parts(path)
        return ExtractedContent(text=_extract_docx(path))
    if extension == ".pdf":
        _check_document_size(path)
        return _extract_pdf(path)
    if extension in {".xlsx", ".xlsm"}:
        _check_office_parts(path)
        return _extract_xlsx(
            path,
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
    if extension == ".xls":
        _check_document_size(path)
        return _extract_xls(path)
    if extension == ".pptx":
        _check_office_parts(path)
        return _extract_pptx(path)
    if target := _LEGACY_OFFICE_TARGETS.get(extension):
        _check_document_size(path)
        converted = _convert_with_libreoffice(
            path,
            target,
            scratch_prefix="apex-legacy-office-",
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
        if converted is None:
            return ExtractedContent(
                text=f"[Binary file: {path.name}, {path.stat().st_size} bytes; not converted to a readable format]"
            )
        try:
            return _extract_file_content(
                converted,
                document_converter_image=document_converter_image,
                allow_host_document_tools=allow_host_document_tools,
            )
        finally:
            shutil.rmtree(converted.parent, ignore_errors=True)
    return ExtractedContent(text=f"[Binary file: {path.name}, {path.stat().st_size} bytes]")


def _bounded_text(text: str) -> str:
    if len(text) <= _MAX_EXTRACTED_TEXT_CHARS:
        return text
    return text[:_MAX_EXTRACTED_TEXT_CHARS] + _TEXT_TRUNCATION_NOTE


def _read_bounded(path: Path) -> tuple[bytes, bool]:
    """The start of a file, at most one byte per allowed character, and whether more was left unread."""
    with path.open("rb") as stream:
        data = stream.read(_MAX_EXTRACTED_TEXT_CHARS + 1)
    return data[:_MAX_EXTRACTED_TEXT_CHARS], len(data) > _MAX_EXTRACTED_TEXT_CHARS


def read_text_bounded(path: Path) -> str:
    """Read a text file without loading more of it than the judge can be shown."""
    data, truncated = _read_bounded(path)
    text = data.decode("utf-8", errors="replace").strip()
    return text + _TEXT_TRUNCATION_NOTE if truncated else text


def _check_document_size(path: Path) -> None:
    size = path.stat().st_size
    if size > _MAX_DOCUMENT_BYTES:
        raise ExtractionLimitError(f"{path.name} is {size} bytes; the extraction limit is {_MAX_DOCUMENT_BYTES}")


def _check_office_parts(path: Path) -> None:
    """Refuse an Office package whose parts would not fit in memory, before any library opens them."""
    _check_document_size(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            parts = archive.infolist()
    except zipfile.BadZipFile:
        # Not a package: the format's own reader reports that.
        return
    total = 0
    for part in parts:
        if part.file_size > _MAX_OFFICE_PART_BYTES:
            raise ExtractionLimitError(
                f"{path.name} part {part.filename!r} expands to {part.file_size} bytes; "
                f"the limit is {_MAX_OFFICE_PART_BYTES}"
            )
        total += part.file_size
    if total > _MAX_OFFICE_UNCOMPRESSED_BYTES:
        raise ExtractionLimitError(
            f"{path.name} expands to {total} bytes; the limit is {_MAX_OFFICE_UNCOMPRESSED_BYTES}"
        )


def _extract_csv(path: Path) -> str:
    data, truncated = _read_bounded(path)
    text: str | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError as exc:
            if truncated and encoding == "utf-8" and exc.start >= len(data) - 3:
                # The read stopped inside a multi-byte character; the bytes before it are still UTF-8.
                text = data[: exc.start].decode(encoding)
                break
    if text is None:
        return data.decode("utf-8", errors="replace")
    rendered = "\n".join("\t".join(row) for row in csv.reader(io.StringIO(text, newline="")))
    return rendered + _TEXT_TRUNCATION_NOTE if truncated else rendered


def _join_ipynb_source(source: Any) -> str:
    return "".join(str(line) for line in source) if isinstance(source, list) else str(source or "")


def _ipynb_text_outputs(outputs: Any) -> str:
    if not isinstance(outputs, list):
        return ""
    lines: list[str] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        output_type = output.get("output_type")
        if output_type == "stream":
            lines.append(_join_ipynb_source(output.get("text")))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data")
            if isinstance(data, dict) and "text/plain" in data:
                lines.append(_join_ipynb_source(data["text/plain"]))
        elif output_type == "error":
            traceback = output.get("traceback")
            if isinstance(traceback, list):
                lines.append("\n".join(str(line) for line in traceback))
    return "\n".join(line.strip() for line in lines if line.strip())


def _extract_ipynb(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    parts: list[str] = []
    for cell in data.get("cells", []) if isinstance(data, dict) else []:
        if not isinstance(cell, dict):
            continue
        source = _join_ipynb_source(cell.get("source")).rstrip()
        if cell.get("cell_type") == "markdown" and source:
            parts.append(source)
        elif cell.get("cell_type") == "code":
            if source:
                parts.append(f"```python\n{source}\n```")
            output = _ipynb_text_outputs(cell.get("outputs"))
            if output:
                parts.append(f"Output:\n{output}")
    return "\n\n".join(parts)


def _extract_text_from_pptx_shape(shape: Any) -> list[str]:
    """Recursively extract PowerPoint text, including grouped shapes and tables."""
    text_parts: list[str] = []
    if hasattr(shape, "shapes"):
        for child_shape in shape.shapes:
            text_parts.extend(_extract_text_from_pptx_shape(child_shape))
        return text_parts

    try:
        table = shape.table
        for row in table.rows:
            row_texts = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if row_texts:
                text_parts.append("\t".join(row_texts))
        return text_parts
    except (ValueError, AttributeError):
        pass

    text_frame_succeeded = False
    if hasattr(shape, "text_frame"):
        try:
            for paragraph in shape.text_frame.paragraphs:
                paragraph_text = "".join(run.text for run in paragraph.runs if run.text)
                if paragraph_text.strip():
                    text_parts.append(paragraph_text.strip())
                    text_frame_succeeded = True
        except Exception:
            pass
    if not text_frame_succeeded and hasattr(shape, "text") and shape.text:
        text = shape.text.strip()
        if text:
            text_parts.append(text)
    return text_parts


def _visual_key(data: bytes) -> dict[str, Any]:
    """A fingerprint entry for visual content that text extraction does not see.

    Sub-artifact matching compares these keys, so a slide, page or sheet whose picture or chart changed counts as
    changed even when its text is identical.
    """
    return {"hash": hashlib.sha256(data).hexdigest()}


def _pptx_shape_visual_keys(shape: Any) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    try:
        for child in getattr(shape, "shapes", None) or []:
            keys.extend(_pptx_shape_visual_keys(child))
        image = getattr(shape, "image", None)
        if image is not None:
            keys.append(_visual_key(image.blob))
        if getattr(shape, "has_chart", False):
            keys.append(_visual_key(shape.chart.part.blob))
    except Exception:
        # A shape whose picture or chart cannot be read keeps its text-only fingerprint.
        pass
    return keys


def _pdf_page_visual_keys(page: Any) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    try:
        resources = page.get("/Resources")
        xobjects = resources.get_object().get("/XObject") if resources is not None else None
        if xobjects is None:
            return keys
        for name in sorted(xobjects.get_object()):
            xobject = xobjects.get_object()[name].get_object()
            if xobject.get("/Subtype") == "/Image":
                keys.append(_visual_key(bytes(getattr(xobject, "_data", b"") or b"")))
    except Exception:
        return []
    return keys


_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _part_relationships(archive: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """Relationship id -> (type suffix, target part) for one package part; empty when it has none."""
    directory, _, name = part.rpartition("/")
    rels_path = f"{directory}/_rels/{name}.rels"
    if rels_path not in archive.namelist():
        return {}
    relationships: dict[str, tuple[str, str]] = {}
    for relationship in ET.fromstring(archive.read(rels_path)).findall(f"{{{_REL_NS}}}Relationship"):
        if relationship.get("TargetMode") == "External":
            continue
        target = str(relationship.get("Target") or "")
        segments: list[str] = [] if target.startswith("/") else directory.split("/")
        for segment in target.split("/"):
            if segment == "..":
                if segments:
                    segments.pop()
            elif segment and segment != ".":
                segments.append(segment)
        relationships[str(relationship.get("Id"))] = (
            str(relationship.get("Type") or "").rsplit("/", 1)[-1],
            "/".join(segments),
        )
    return relationships


def _xlsx_sheet_visual_keys(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Sheet name -> fingerprints of the charts and pictures drawn on that sheet."""
    keys_by_sheet: dict[str, list[dict[str, Any]]] = {}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
            workbook_rels = _part_relationships(archive, "xl/workbook.xml")
            namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall(".//main:sheet", namespace):
                _, sheet_part = workbook_rels.get(str(sheet.get(f"{{{_OFFICE_REL_NS}}}id")), ("", ""))
                keys: list[dict[str, Any]] = []
                for kind, drawing_part in sorted(_part_relationships(archive, sheet_part).values()):
                    if kind != "drawing" or drawing_part not in names:
                        continue
                    for visual_kind, visual_part in sorted(_part_relationships(archive, drawing_part).values()):
                        if visual_kind in {"chart", "image"} and visual_part in names:
                            keys.append(_visual_key(archive.read(visual_part)))
                if keys and sheet.get("name"):
                    keys_by_sheet[str(sheet.get("name"))] = keys
    except Exception:
        return {}
    return keys_by_sheet


def _extract_pptx(path: Path) -> ExtractedContent:
    from pptx import Presentation

    presentation = Presentation(str(path))
    full_text_parts: list[str] = []
    sub_artifacts: list[SubArtifact] = []
    for slide_index, slide in enumerate(presentation.slides):
        slide_text_parts: list[str] = []
        slide_visual_keys: list[dict[str, Any]] = []
        slide_title: str | None = None
        for shape in slide.shapes:
            if slide_title is None:
                try:
                    if hasattr(shape, "placeholder_format") and shape.placeholder_format.type == 1 and shape.text:
                        slide_title = shape.text.strip()
                except Exception:
                    pass
            slide_text_parts.extend(_extract_text_from_pptx_shape(shape))
            slide_visual_keys.extend(_pptx_shape_visual_keys(shape))
        if slide_title is None and slide_text_parts:
            slide_title = slide_text_parts[0][:100]
        slide_text = "\n".join(slide_text_parts)
        sub_artifacts.append(
            SubArtifact(
                index=slide_index,
                type="slide",
                title=slide_title or f"Slide {slide_index + 1}",
                content=slide_text,
                images=slide_visual_keys,
            )
        )
        full_text_parts.append(f"=== Slide {slide_index + 1}: {slide_title or 'Untitled'} ===\n{slide_text}")
    return ExtractedContent(text="\n\n".join(full_text_parts), sub_artifacts=sub_artifacts)


def _extract_pdf(path: Path) -> ExtractedContent:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    pages: list[str] = []
    sub_artifacts: list[SubArtifact] = []
    for page_index, page in enumerate(reader.pages):
        try:
            page_text = (page.extract_text() or "").strip()
        except Exception as exc:
            page_text = f"[page {page_index + 1} text extraction failed: {type(exc).__name__}]"
        sub_artifacts.append(
            SubArtifact(
                index=page_index,
                type="page",
                title=f"Page {page_index + 1}",
                content=page_text,
                images=_pdf_page_visual_keys(page),
            )
        )
        pages.append(f"=== Page {page_index + 1} ===\n{page_text}")
    scanned_banner = ""
    if sub_artifacts and not any(page.content.strip() for page in sub_artifacts):
        scanned_banner = (
            "=== Extraction Note ===\n"
            "[no extractable text — the PDF is likely scanned/image-only and no OCR is available offline]\n\n"
        )
        sub_artifacts = [
            SubArtifact(
                index=page.index,
                type=page.type,
                title=page.title,
                content=scanned_banner + page.content,
                images=page.images,
            )
            for page in sub_artifacts
        ]
    return ExtractedContent(
        text=scanned_banner + "\n\n".join(pages),
        sub_artifacts=sub_artifacts,
    )


def _accepted_docx_bytes(raw_bytes: bytes) -> bytes:
    """Return DOCX bytes with revisions accepted, matching Archipelago text transformation."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as source:
            if _DOCUMENT_XML_PATH not in source.namelist():
                return raw_bytes
            from lxml import etree

            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            tree = etree.fromstring(source.read(_DOCUMENT_XML_PATH), parser=parser)
            for local_name in ("del", "moveFrom"):
                for element in list(tree.iter(f"{{{_WORD_NS}}}{local_name}")):
                    if (parent := element.getparent()) is not None:
                        parent.remove(element)
            for local_name in ("ins", "moveTo"):
                for element in list(tree.iter(f"{{{_WORD_NS}}}{local_name}")):
                    parent = element.getparent()
                    if parent is None:
                        continue
                    index = list(parent).index(element)
                    for child in reversed(list(element)):
                        parent.insert(index, child)
                    parent.remove(element)
            modified_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
            destination = io.BytesIO()
            with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as output:
                for item in source.infolist():
                    output.writestr(
                        item, modified_xml if item.filename == _DOCUMENT_XML_PATH else source.read(item.filename)
                    )
            return destination.getvalue()
    except Exception:
        return raw_bytes


def _docx_review_text(raw_bytes: bytes) -> str:
    """Expose tracked changes and anchored comments that python-docx omits."""
    try:
        from lxml import etree

        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as archive:
            if _DOCUMENT_XML_PATH not in archive.namelist():
                return ""
            document = etree.fromstring(archive.read(_DOCUMENT_XML_PATH), parser=parser)
            comments: dict[str, tuple[str, str]] = {}
            if _COMMENTS_XML_PATH in archive.namelist():
                comments_root = etree.fromstring(archive.read(_COMMENTS_XML_PATH), parser=parser)
                for comment in comments_root.iter(f"{{{_WORD_NS}}}comment"):
                    comment_id = comment.get(f"{{{_WORD_NS}}}id")
                    if comment_id is not None:
                        author = comment.get(f"{{{_WORD_NS}}}author") or "Unknown author"
                        text = "".join(node.text or "" for node in comment.iter(f"{{{_WORD_NS}}}t")).strip()
                        comments[comment_id] = (author, text)

        redlines_by_author: dict[str, list[str]] = {}
        anchors: dict[str, dict[str, Any]] = {}
        open_comment_ids: set[str] = set()
        current_section = ""
        paragraph_index = 0
        revision_tags = {"ins": "INSERTED", "del": "DELETED", "moveFrom": "MOVED FROM", "moveTo": "MOVED TO"}
        for paragraph in document.iter(f"{{{_WORD_NS}}}p"):
            paragraph_index += 1
            paragraph_text = "".join(node.text or "" for node in paragraph.iter(f"{{{_WORD_NS}}}t")).strip()
            if paragraph_text and _SECTION_HEADING_RE.match(paragraph_text):
                current_section = paragraph_text[:60].strip()
            location = current_section or f"Para {paragraph_index}"
            for element in paragraph.iter():
                local_name = element.tag.rsplit("}", 1)[-1]
                if local_name == "commentRangeStart":
                    comment_id = element.get(f"{{{_WORD_NS}}}id")
                    if comment_id is not None:
                        anchors[comment_id] = {"location": location, "parts": []}
                        open_comment_ids.add(comment_id)
                elif local_name == "commentRangeEnd":
                    comment_id = element.get(f"{{{_WORD_NS}}}id")
                    if comment_id is not None:
                        open_comment_ids.discard(comment_id)
                elif local_name == "t" and element.text:
                    for comment_id in open_comment_ids:
                        anchors[comment_id]["parts"].append(element.text)
            for revision_name, label in revision_tags.items():
                text_tag = "delText" if revision_name == "del" else "t"
                for revision in paragraph.iter(f"{{{_WORD_NS}}}{revision_name}"):
                    text = "".join(node.text or "" for node in revision.iter(f"{{{_WORD_NS}}}{text_tag}"))
                    if text:
                        author = revision.get(f"{{{_WORD_NS}}}author") or "Unknown author"
                        redlines_by_author.setdefault(author, []).append(f'  [{location}] {label}: "{text}"')

        parts: list[str] = []
        if redlines_by_author:
            parts.append("=== DOCUMENT REDLINES ===")
            for author, lines in redlines_by_author.items():
                parts.extend((f"\nAuthor: {author}", *lines))
        if comments:
            comment_lines = ["=== DOCUMENT COMMENTS ==="]
            for comment_id, (author, text) in comments.items():
                anchor = anchors.get(comment_id)
                if anchor:
                    anchored_text = "".join(anchor["parts"]).strip()
                    on = f' on "{anchored_text}"' if anchored_text else ""
                    comment_lines.append(f"\nAuthor: {author}\n  [{anchor['location']}]{on}: {text}")
                else:
                    comment_lines.append(f"\nAuthor: {author}\n  [no anchor]: {text}")
            parts.append("\n".join(comment_lines))
        return "\n\n".join(parts)
    except Exception:
        return ""


def _extract_docx(path: Path) -> str:
    from docx import Document

    raw_bytes = path.read_bytes()
    document = Document(io.BytesIO(_accepted_docx_bytes(raw_bytes)))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in getattr(document, "tables", []):
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    review_text = _docx_review_text(raw_bytes)
    if review_text:
        parts.append(review_text)
    return "\n".join(parts)


def _hidden_xlsx_sheets(path: Path) -> set[str]:
    hidden: set[str] = set()
    try:
        with zipfile.ZipFile(path, "r") as archive, archive.open("xl/workbook.xml") as stream:
            root = ET.parse(stream).getroot()
        namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for sheet in root.findall(".//main:sheet", namespace):
            if sheet.get("state", "visible") in {"hidden", "veryHidden"} and sheet.get("name"):
                hidden.add(str(sheet.get("name")))
    except Exception:
        pass
    return hidden


@dataclass(frozen=True)
class _DocumentTool:
    """One LibreOffice or Poppler executable: inside the converter image, or on this host when that is allowed."""

    executable: str
    apptainer: str | None = None
    image: str | None = None

    @property
    def containerized(self) -> bool:
        return self.apptainer is not None

    def command(self, scratch: Path) -> tuple[list[str], str]:
        """The command prefix, and the path under which the tool sees ``scratch``.

        A containerized tool sees nothing of the host but ``scratch``: no host /tmp, home or working directory.
        """
        if self.apptainer is None or self.image is None:
            return [self.executable], str(scratch)
        workdir = scratch / _CONTAINER_WORKDIR
        workdir.mkdir(mode=0o700, exist_ok=True)
        return [
            self.apptainer,
            "exec",
            "--cleanenv",
            "--containall",
            # Keep the container's /tmp and home on disk inside the scratch directory, not in a small session tmpfs.
            "--workdir",
            str(workdir),
            "--bind",
            f"{scratch}:{_CONTAINER_SCRATCH}",
            self.image,
            self.executable,
        ], _CONTAINER_SCRATCH


def _resolve_document_command(
    executables: tuple[str, ...],
    document_converter_image: str | None,
    *,
    allow_host_document_tools: bool = False,
) -> _DocumentTool | None:
    """Where to run a document tool, or None when no converter is configured.

    The tools open files the model wrote, so by default they run only inside the converter image. Tools installed
    on this host are used, and preferred, only when ``allow_host_document_tools`` is set. A converter image that
    is configured but cannot be used is a ``DocumentConverterError``, never a silent skip.
    """
    if allow_host_document_tools:
        for executable in executables:
            if local_executable := shutil.which(executable):
                return _DocumentTool(local_executable)
    if not document_converter_image:
        return None
    if problem := converter_setup_problem(document_converter_image):
        raise DocumentConverterError(problem)
    return _DocumentTool(executables[0], apptainer=shutil.which("apptainer"), image=document_converter_image)


def _run_document_tool(
    tool: _DocumentTool, scratch: Path, arguments: list[str], *, source: str = ""
) -> subprocess.CompletedProcess[bytes] | None:
    """Run a tool on files under ``scratch``; None when it timed out or, on the host, could not start.

    ``arguments`` name files by the root ``tool.command(scratch)`` returns. A non-zero exit is the caller's to
    read: the tool rejected the file. A container that could not start is a ``DocumentConverterError``. A timeout
    is not: a file the model wrote can cause one, so the caller falls back to the file's text, and the timeout is
    logged and recorded for the grading record under ``source``, the name of the file being converted.
    """
    command, _ = tool.command(scratch)
    try:
        process = subprocess.run(
            [*command, *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_DOCUMENT_TOOL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        executable = Path(tool.executable).name
        LOG.warning("%s timed out after %ss converting %s", executable, _DOCUMENT_TOOL_TIMEOUT_SECONDS, source)
        timeouts = _CONVERSION_TIMEOUTS.get()
        if timeouts is not None:
            timeouts.append({"tool": executable, "file": source, "timeout_seconds": _DOCUMENT_TOOL_TIMEOUT_SECONDS})
        return None
    except OSError as exc:
        if tool.containerized:
            raise DocumentConverterError(f"document converter could not start: {exc}") from exc
        return None
    if tool.containerized and process.returncode == 255 and b"FATAL" in process.stderr:
        # Apptainer's own failure (an image it cannot mount, a missing executable), not the tool's verdict on a file.
        detail = process.stderr.decode("utf-8", errors="replace").strip()[-500:]
        raise DocumentConverterError(f"document converter container failed: {detail}")
    return process


def _convert_with_libreoffice(
    path: Path,
    convert_to: str,
    *,
    scratch_prefix: str,
    extra_arguments: tuple[str, ...] = (),
    document_converter_image: str | None,
    allow_host_document_tools: bool,
) -> Path | None:
    """Convert a copy of ``path`` with LibreOffice, using an isolated profile.

    Returns the converted file, alone with its inputs in a scratch directory (its parent) that the caller removes;
    None when no converter is configured or LibreOffice did not convert the file.
    """
    tool = _resolve_document_command(
        ("libreoffice", "soffice"),
        document_converter_image,
        allow_host_document_tools=allow_host_document_tools,
    )
    if tool is None:
        return None
    scratch = Path(tempfile.mkdtemp(prefix=scratch_prefix))
    output: Path | None = None
    try:
        (scratch / "in").mkdir()
        shutil.copyfile(path, scratch / "in" / f"document{path.suffix.lower()}")
        _, root = tool.command(scratch)
        process = _run_document_tool(
            tool,
            scratch,
            [
                "--headless",
                *extra_arguments,
                f"-env:UserInstallation=file://{root}/profile",
                "--convert-to",
                convert_to,
                "--outdir",
                root,
                f"{root}/in/document{path.suffix.lower()}",
            ],
            source=path.name,
        )
        candidate = scratch / f"document.{convert_to.split(':', 1)[0]}"
        if process is not None and process.returncode == 0 and candidate.is_file():
            output = candidate
    finally:
        if output is None:
            shutil.rmtree(scratch, ignore_errors=True)
    return output


def _sheet_rows(sheet: Any) -> Any:
    """A worksheet's rows; none for a chartsheet, which holds a chart and no cells."""
    return sheet.iter_rows() if hasattr(sheet, "iter_rows") else ()


def _xlsx_has_uncached_formulas(path: Path) -> bool:
    """Port Archipelago's formula/cache coordinate comparison."""
    from openpyxl import load_workbook

    try:
        formulas = load_workbook(path, data_only=False, read_only=True)
        try:
            formula_coordinates = {
                (sheet_name, cell.coordinate)
                for sheet_name in formulas.sheetnames
                for row in _sheet_rows(formulas[sheet_name])
                for cell in row
                if cell.data_type == "f" or (cell.value is not None and str(cell.value).startswith("="))
            }
        finally:
            formulas.close()
        if not formula_coordinates:
            return False

        values = load_workbook(path, data_only=True, read_only=True)
        try:
            return any(
                cell.value is None and (sheet_name, getattr(cell, "coordinate", None)) in formula_coordinates
                for sheet_name in values.sheetnames
                for row in _sheet_rows(values[sheet_name])
                for cell in row
            )
        finally:
            values.close()
    except Exception:
        return True


def _evaluate_excel_formulas_with_libreoffice(
    path: Path,
    *,
    document_converter_image: str | None,
    allow_host_document_tools: bool = False,
) -> bytes | None:
    """Re-save a workbook through LibreOffice so formula cells gain cached values."""
    try:
        recalculated = _convert_with_libreoffice(
            path,
            "xlsx",
            scratch_prefix="apex-formula-eval-",
            extra_arguments=("--calc",),
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
    except OSError:
        return None
    if recalculated is None:
        return None
    try:
        return recalculated.read_bytes()
    finally:
        shutil.rmtree(recalculated.parent, ignore_errors=True)


def _has_charts_in_xlsx(path: Path) -> bool:
    from openpyxl import load_workbook

    workbook = None
    try:
        workbook = load_workbook(path, data_only=True)
        return any(bool(getattr(workbook[sheet_name], "_charts", None)) for sheet_name in workbook.sheetnames)
    except Exception:
        return False
    finally:
        if workbook is not None:
            workbook.close()


def _png_data_url(data: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def _pdf_page_data_urls(
    pdf_path: Path,
    *,
    document_converter_image: str | None,
    allow_host_document_tools: bool = False,
    page_indices: set[int] | None = None,
    source: str | None = None,
) -> list[tuple[int, str]]:
    """Rasterize the requested pages (the first pages when none are requested), capped at the visual page limit.

    ``source`` names the file the PDF stands for, for the record of a rasterization that timed out.
    """
    source = source or pdf_path.name
    page_numbers = sorted(index + 1 for index in page_indices)[:_MAX_VISUAL_PAGES] if page_indices else None
    tool = _resolve_document_command(
        ("pdftoppm",), document_converter_image, allow_host_document_tools=allow_host_document_tools
    )
    if tool is None:
        if not allow_host_document_tools:
            # No converter configured: the PDF is graded on its extracted text.
            return []
        from pdf2image import convert_from_path

        ranges = (
            [(page_number, page_number) for page_number in page_numbers] if page_numbers else [(1, _MAX_VISUAL_PAGES)]
        )
        results: list[tuple[int, str]] = []
        for first_page, last_page in ranges:
            pages = convert_from_path(str(pdf_path), dpi=150, first_page=first_page, last_page=last_page)
            for offset, page in enumerate(pages):
                buffer = io.BytesIO()
                page.save(buffer, format="PNG")
                results.append((first_page + offset, _png_data_url(buffer.getvalue())))
        return results

    scratch = Path(tempfile.mkdtemp(prefix="apex-pdf-pages-"))
    try:
        (scratch / "in").mkdir()
        shutil.copyfile(pdf_path, scratch / "in" / "document.pdf")
        _, root = tool.command(scratch)
        document = f"{root}/in/document.pdf"
        if page_numbers:
            selected: list[tuple[int, str]] = []
            for page_number in page_numbers:
                process = _run_document_tool(
                    tool,
                    scratch,
                    [
                        "-png",
                        "-r",
                        "150",
                        "-f",
                        str(page_number),
                        "-l",
                        str(page_number),
                        "-singlefile",
                        document,
                        f"{root}/selected-{page_number}",
                    ],
                    source=source,
                )
                image_path = scratch / f"selected-{page_number}.png"
                if process is not None and process.returncode == 0 and image_path.is_file():
                    selected.append((page_number, _png_data_url(image_path.read_bytes())))
            return selected

        process = _run_document_tool(
            tool,
            scratch,
            ["-png", "-r", "150", "-f", "1", "-l", str(_MAX_VISUAL_PAGES), document, f"{root}/page"],
            source=source,
        )
        if process is None or process.returncode != 0:
            return []
        return [
            (page_number, _png_data_url(image_path.read_bytes()))
            for page_number, image_path in enumerate(sorted(scratch.glob("page-*.png")), 1)
        ]
    except OSError:
        return []
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _pdf_to_chart_images(
    pdf_path: Path,
    *,
    document_converter_image: str | None,
    allow_host_document_tools: bool = False,
) -> list[dict[str, Any]]:
    try:
        return [
            {
                "url": url,
                "placeholder": f"[CHART_{index}]",
                "type": "Chart",
                "caption": f"Chart from Excel (Page {page_number})",
            }
            for index, (page_number, url) in enumerate(
                _pdf_page_data_urls(
                    pdf_path,
                    document_converter_image=document_converter_image,
                    allow_host_document_tools=allow_host_document_tools,
                ),
                1,
            )
        ]
    except DocumentConverterError:
        raise
    except Exception:
        return []


def _extract_xlsx(
    path: Path,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
) -> ExtractedContent:
    from openpyxl import load_workbook

    hidden_sheets = _hidden_xlsx_sheets(path)
    formulas: dict[tuple[str, str], str] = {}
    formula_workbook = load_workbook(path, data_only=False, read_only=True)
    try:
        for sheet_name in formula_workbook.sheetnames:
            if sheet_name in hidden_sheets:
                continue
            for row in _sheet_rows(formula_workbook[sheet_name]):
                for cell in row:
                    if cell.data_type == "f" and isinstance(cell.value, str) and cell.value.startswith("="):
                        formulas[(sheet_name, cell.coordinate)] = cell.value
    finally:
        formula_workbook.close()

    unresolved = _xlsx_has_uncached_formulas(path)
    recalculated = (
        _evaluate_excel_formulas_with_libreoffice(
            path,
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
        if unresolved
        else None
    )
    if recalculated is not None:
        unresolved = False
    workbook_source: Path | io.BytesIO = io.BytesIO(recalculated) if recalculated is not None else path
    workbook = load_workbook(workbook_source, data_only=True, read_only=True)
    parts: list[str] = []
    sheet_values: list[tuple[int, str, str]] = []
    try:
        for sheet_index, sheet_name in enumerate(workbook.sheetnames):
            if sheet_name in hidden_sheets:
                continue
            rows: list[str] = []
            formula_lines: list[str] = []
            format_lines: list[str] = []
            sheet = workbook[sheet_name]
            if not hasattr(sheet, "iter_rows"):
                # A chartsheet stays a sheet of the workbook, so a change to its chart is still a sheet change.
                rows.append("[chart sheet: no cells]")
            for row in _sheet_rows(sheet):
                values: list[str] = []
                for cell in row:
                    coordinate = getattr(cell, "coordinate", "")
                    formula = formulas.get((sheet_name, coordinate))
                    if formula:
                        formula_lines.append(f"{coordinate}: {formula}")
                    if cell.value is None:
                        continue
                    values.append(str(cell.value))
                    if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                        number_format = cell.number_format
                        if number_format and number_format != "General" and re.search(r"[$€£¥%]|#,##0", number_format):
                            format_lines.append(f"{coordinate}: [fmt: {number_format}]")
                if values:
                    rows.append("\t".join(values))
            sheet_text = f"=== Sheet: {sheet_name} ===\n" + "\n".join(rows)
            if format_lines:
                sheet_text += "\n\n=== Number Formats ===\n" + "\n".join(format_lines)
            if formula_lines:
                sheet_text += "\n\n=== Formulas ===\n" + "\n".join(formula_lines)
            parts.append(sheet_text)
            sheet_values.append((sheet_index, sheet_name, sheet_text))
    finally:
        workbook.close()
    banner = ""
    if unresolved:
        banner = (
            "=== Extraction Note ===\n[formula cells could not be recalculated — LibreOffice unavailable; "
            "blanks may be uncomputed formulas, not missing values]\n\n"
        )
    visual_keys = _xlsx_sheet_visual_keys(path)
    sub_artifacts = [
        SubArtifact(
            index=index, type="sheet", title=title, content=banner + content, images=visual_keys.get(title, [])
        )
        for index, title, content in sheet_values
    ]
    chart_images: list[dict[str, Any]] = []
    if _has_charts_in_xlsx(path):
        pdf_path = _convert_office_to_pdf(
            path,
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
        try:
            if pdf_path is not None:
                chart_images = _pdf_to_chart_images(
                    pdf_path,
                    document_converter_image=document_converter_image,
                    allow_host_document_tools=allow_host_document_tools,
                )
        finally:
            if pdf_path is not None:
                shutil.rmtree(pdf_path.parent, ignore_errors=True)
    chart_text = ""
    if chart_images:
        chart_text = "\n\n=== Charts ===\n" + "\n".join(
            f"{image['placeholder']} - {image['caption']}" for image in chart_images
        )
    return ExtractedContent(
        text=banner + "\n\n".join(parts) + chart_text,
        images=chart_images,
        sub_artifacts=sub_artifacts,
    )


def _extract_xls(path: Path) -> ExtractedContent:
    import xlrd

    workbook = xlrd.open_workbook(str(path))
    parts: list[str] = []
    sub_artifacts: list[SubArtifact] = []
    for sheet_index in range(workbook.nsheets):
        sheet = workbook.sheet_by_index(sheet_index)
        if sheet.visibility != 0:
            continue
        rows: list[str] = []
        for row_index in range(sheet.nrows):
            values: list[str] = []
            for column_index in range(sheet.ncols):
                cell = sheet.cell(row_index, column_index)
                if cell.ctype == xlrd.XL_CELL_EMPTY:
                    continue
                if cell.ctype == xlrd.XL_CELL_NUMBER:
                    value = cell.value
                    if (
                        isinstance(value, float)
                        and not (math.isnan(value) or math.isinf(value))
                        and value == int(value)
                    ):
                        values.append(str(int(value)))
                    else:
                        values.append(str(value))
                elif cell.ctype == xlrd.XL_CELL_DATE:
                    date = xlrd.xldate_as_tuple(float(cell.value), workbook.datemode)
                    values.append(f"{date[0]}-{date[1]:02d}-{date[2]:02d}")
                elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                    values.append("TRUE" if cell.value else "FALSE")
                elif str(cell.value).strip():
                    values.append(str(cell.value).strip())
            if values:
                rows.append("\t".join(values))
        sheet_text = f"=== Sheet: {sheet.name} ===\n" + "\n".join(rows)
        parts.append(sheet_text)
        sub_artifacts.append(SubArtifact(index=sheet_index, type="sheet", title=sheet.name, content=sheet_text))
    return ExtractedContent(text="\n\n".join(parts), sub_artifacts=sub_artifacts)


def _convert_office_to_pdf(
    path: Path,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
    single_page_sheets: bool = False,
) -> Path | None:
    """Convert Office input to a temporary PDF; the caller removes the PDF's directory.

    With ``single_page_sheets`` a workbook is exported one sheet per page, so a page number names a sheet.
    """
    try:
        return _convert_with_libreoffice(
            path,
            _SINGLE_PAGE_SHEETS_PDF if single_page_sheets else "pdf",
            scratch_prefix="apex-office-pdf-",
            document_converter_image=document_converter_image,
            allow_host_document_tools=allow_host_document_tools,
        )
    except OSError:
        return None


def _pdf_visual_blocks(
    display_name: str,
    path: Path,
    *,
    converted: bool = False,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
    page_indices: set[int] | None = None,
    page_labels: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    pages = _pdf_page_data_urls(
        path,
        document_converter_image=document_converter_image,
        allow_host_document_tools=allow_host_document_tools,
        page_indices=page_indices,
        source=display_name,
    )
    label = f"\n{display_name} (rendered from Office):" if converted else f"\n{display_name}:"
    blocks: list[dict[str, Any]] = [{"type": "text", "text": label}]
    for page_number, url in pages:
        blocks.append({"type": "text", "text": (page_labels or {}).get(page_number - 1, f"Page {page_number}")})
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def _pptx_shown_slides(path: Path) -> list[int]:
    """Indices of the slides LibreOffice exports: a hidden slide gets no PDF page."""
    from pptx import Presentation

    return [index for index, slide in enumerate(Presentation(str(path)).slides) if slide._element.get("show") != "0"]


def _workbook_sheet_count(path: Path) -> int:
    if path.suffix.lower() == ".xls":
        import xlrd

        return xlrd.open_workbook(str(path), on_demand=True).nsheets
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True)
    try:
        return len(workbook.sheetnames)
    finally:
        workbook.close()


def _changed_office_pages(
    path: Path, pdf_path: Path, indices: set[int], *, single_page_sheets: bool
) -> dict[int, str] | None:
    """PDF page index -> label for the changed slides or sheets of a converted Office file.

    None when the pages cannot be tied to slides or sheets (the export did not produce one page per shown slide
    or per sheet), or when a changed sheet's single page is too large to read as one image; the caller then shows
    the leading pages of the ordinary paginated export, as for a file with no known changed part.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        if path.suffix.lower() == ".pptx":
            shown = _pptx_shown_slides(path)
            if len(shown) != page_count:
                return None
            pages = {shown.index(index): f"Slide {index + 1}" for index in sorted(indices) if index in shown}
        elif single_page_sheets:
            if _workbook_sheet_count(path) != page_count:
                return None
            pages = {index: f"Sheet {index + 1}" for index in sorted(indices) if index < page_count}
            for index in pages:
                box = reader.pages[index].mediabox
                if max(float(box.width), float(box.height)) > _MAX_SINGLE_PAGE_SHEET_POINTS:
                    # A long or wide sheet on one page is an image too large to read.
                    return None
        else:
            return None
    except Exception:
        return None
    return pages or None


def _office_visual_blocks(
    display_name: str,
    path: Path,
    *,
    document_converter_image: str | None,
    allow_host_document_tools: bool,
    indices: set[int] | None,
) -> list[dict[str, Any]]:
    converter = {
        "document_converter_image": document_converter_image,
        "allow_host_document_tools": allow_host_document_tools,
    }
    pdf_path: Path | None = None
    page_labels: dict[int, str] | None = None
    if indices and path.suffix.lower() in _SPREADSHEET_EXTENSIONS:
        # One sheet per page ties a page to a changed sheet. It is kept only when that works and the changed
        # sheets stay readable; otherwise the workbook is shown by the leading pages of the paginated export.
        pdf_path = _convert_office_to_pdf(path, single_page_sheets=True, **converter)
        if pdf_path is not None:
            page_labels = _changed_office_pages(path, pdf_path, indices, single_page_sheets=True)
            if page_labels is None:
                shutil.rmtree(pdf_path.parent, ignore_errors=True)
                pdf_path = None
    if pdf_path is None:
        pdf_path = _convert_office_to_pdf(path, **converter)
        if pdf_path is not None and indices:
            page_labels = _changed_office_pages(path, pdf_path, indices, single_page_sheets=False)
    if pdf_path is None:
        text = extract_file_text(path, **converter)
        return [{"type": "text", "text": f"\n{display_name} (text fallback):\n{text}"}] if text else []
    try:
        return _pdf_visual_blocks(
            display_name,
            pdf_path,
            converted=True,
            page_indices=set(page_labels) if page_labels else None,
            page_labels=page_labels,
            **converter,
        )
    finally:
        shutil.rmtree(pdf_path.parent, ignore_errors=True)


def renders_as_images(path: Path) -> bool:
    """Whether a file can yield image evidence: a document that is rasterized, or an image itself."""
    extension = path.suffix.lower()
    return extension == ".pdf" or extension in _OFFICE_EXTENSIONS or extension in _IMAGE_MIME_TYPES


def visual_blocks_for_file(
    path: Path,
    *,
    display_name: str | None = None,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
    changed_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Render one artifact as text and real image blocks; documents are rasterized page by page.

    ``changed_indices`` are the zero-based pages (PDF), slides (PPTX) or sheets (workbooks) that changed: those are
    the pages rendered, so a change late in a file is shown rather than the file's first pages.
    """
    name = display_name or path.name
    extension = path.suffix.lower()
    try:
        if extension in _TEXT_EXTENSIONS or extension in {".csv", ".ipynb"}:
            text = extract_file_text(path)
            return [{"type": "text", "text": f"\n{name}:\n{text}"}] if text else []
        if extension in _OFFICE_EXTENSIONS:
            return _office_visual_blocks(
                name,
                path,
                document_converter_image=document_converter_image,
                allow_host_document_tools=allow_host_document_tools,
                indices=changed_indices,
            )
        if extension == ".pdf":
            return _pdf_visual_blocks(
                name,
                path,
                document_converter_image=document_converter_image,
                allow_host_document_tools=allow_host_document_tools,
                page_indices=changed_indices,
            )
        if mime_type := _IMAGE_MIME_TYPES.get(extension):
            size = path.stat().st_size
            if size > _MAX_INLINE_IMAGE_BYTES:
                return [{"type": "text", "text": f"\n{name}: [image of {size} bytes is too large to attach]"}]
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return [
                {"type": "text", "text": f"\n{name}:"},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ]
    except DocumentConverterError:
        raise
    except Exception as exc:
        return [{"type": "text", "text": f"\n{name}: [Error: {exc}]"}]
    return []


def visual_content_blocks(
    output_dir: Path,
    *,
    document_converter_image: str | None = None,
    allow_host_document_tools: bool = False,
    pdf_page_indices: dict[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    """Render a directory's artifacts; ``pdf_page_indices`` maps a file name to its changed pages, slides or sheets."""
    if not output_dir.is_dir():
        return []
    blocks: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            blocks.extend(
                visual_blocks_for_file(
                    path,
                    document_converter_image=document_converter_image,
                    allow_host_document_tools=allow_host_document_tools,
                    changed_indices=(pdf_page_indices or {}).get(path.name),
                )
            )
    return blocks
