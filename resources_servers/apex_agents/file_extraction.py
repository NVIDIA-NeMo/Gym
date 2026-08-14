# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local text extraction and visual rendering for APEX grading artifacts."""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_TEXT_EXTENSIONS = {
    ".csv",
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
_OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
_IMAGE_MIME_TYPES = {
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def extract_file_text(path: Path) -> str:
    """Extract judge-readable text from one local artifact."""
    extension = path.suffix.lower()
    if extension in _TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    if extension == ".docx":
        from docx import Document

        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    if extension == ".pdf":
        from pdfminer.high_level import extract_text

        return extract_text(str(path)).strip()
    if extension == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(str(path), read_only=True, data_only=True)
        try:
            parts: list[str] = []
            for sheet_name in workbook.sheetnames:
                rows = []
                for row in workbook[sheet_name].iter_rows(values_only=True):
                    cells = [str(cell) if cell is not None else "" for cell in row]
                    if any(cells):
                        rows.append(", ".join(cells))
                if rows:
                    parts.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))
            return "\n\n".join(parts)
        finally:
            workbook.close()
    if extension == ".pptx":
        from pptx import Presentation

        presentation = Presentation(str(path))
        parts = []
        for index, slide in enumerate(presentation.slides, 1):
            texts = [
                paragraph.text
                for shape in slide.shapes
                if shape.has_text_frame
                for paragraph in shape.text_frame.paragraphs
                if paragraph.text.strip()
            ]
            if texts:
                parts.append(f"Slide {index}:\n" + "\n".join(texts))
        return "\n\n".join(parts)
    return f"[Binary file: {path.name}, {path.stat().st_size} bytes]"


def _convert_office_to_pdf(path: Path) -> Path | None:
    """Render an Office artifact to PDF with an isolated LibreOffice profile."""
    profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
    output_pdf = path.with_suffix(".pdf")
    stage_dir: Path | None = None
    input_path = path
    try:
        if any(character.isspace() for character in path.name):
            stage_dir = Path(tempfile.mkdtemp(prefix="lo-stage-"))
            input_path = stage_dir / (re.sub(r"\s+", "_", path.stem) + path.suffix)
            shutil.copy2(path, input_path)
            output_dir = stage_dir
        else:
            output_dir = path.parent
        process = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--nologo",
                "--nolockcheck",
                "--nodefault",
                "--norestore",
                f"-env:UserInstallation=file://{profile_dir.as_posix()}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(input_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        if stage_dir is not None:
            staged_pdf = stage_dir / f"{input_path.stem}.pdf"
            if staged_pdf.exists():
                shutil.move(str(staged_pdf), str(output_pdf))
        if process.returncode != 0 or not output_pdf.exists():
            return None
        return output_pdf
    except subprocess.TimeoutExpired:
        return None
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def visual_content_blocks(output_dir: Path) -> list[dict[str, Any]]:
    """Render local artifacts as OpenAI-compatible text and image blocks."""
    if not output_dir.is_dir():
        return []

    blocks: list[dict[str, Any]] = []
    converted_pdfs: list[Path] = []
    try:
        for path in sorted(output_dir.iterdir()):
            if not path.is_file():
                continue
            extension = path.suffix.lower()
            try:
                if extension in _TEXT_EXTENSIONS:
                    text = path.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        blocks.append({"type": "text", "text": f"\n{path.name}:\n{text}"})
                elif extension in _OFFICE_EXTENSIONS:
                    pdf_path = _convert_office_to_pdf(path)
                    if pdf_path is None:
                        text = extract_file_text(path)
                        if text:
                            blocks.append({"type": "text", "text": f"\n{path.name} (text fallback):\n{text}"})
                        continue
                    converted_pdfs.append(pdf_path)
                    blocks.extend(_binary_visual_blocks(path.name, pdf_path, "application/pdf", converted=True))
                elif extension == ".pdf":
                    blocks.extend(_binary_visual_blocks(path.name, path, "application/pdf"))
                elif mime_type := _IMAGE_MIME_TYPES.get(extension):
                    blocks.extend(_binary_visual_blocks(path.name, path, mime_type))
            except Exception as exc:
                blocks.append({"type": "text", "text": f"\n{path.name}: [Error: {exc}]"})
    finally:
        for path in converted_pdfs:
            path.unlink(missing_ok=True)
    return blocks


def _binary_visual_blocks(
    display_name: str,
    path: Path,
    mime_type: str,
    *,
    converted: bool = False,
) -> list[dict[str, Any]]:
    label = f"\n{display_name} (converted to PDF):" if converted else f"\n{display_name}:"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": label},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
    ]
