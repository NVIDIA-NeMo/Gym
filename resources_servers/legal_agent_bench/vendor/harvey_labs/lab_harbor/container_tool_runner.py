# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Container-side implementation for non-bash Harbor agent tools.

LegalAgentBenchHarborAgent invokes this script inside the Harbor environment
with a tool name and a JSON argument payload. It returns one JSON object on
stdout:

    {"result": "...", "metrics": {...}}

Keeping this logic in the image means document parsing uses the preinstalled
runtime tools and libraries instead of whatever happens to be on the host.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


VDR_DIR = Path(os.environ.get("VDR_DIR", "/workspace/vdr")).resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output")).resolve()
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace/workspace")).resolve()
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", "/workspace/skills")).resolve()

READABLE_ROOTS = (WORKSPACE_DIR, OUTPUT_DIR, VDR_DIR, SKILLS_DIR)
WRITABLE_ROOTS = (OUTPUT_DIR, WORKSPACE_DIR)
MAX_READ_LINES = 200


def main() -> None:
    if len(sys.argv) != 3:
        _emit("Error: expected tool name and JSON arguments", {})
        return

    tool_name = sys.argv[1]
    try:
        arguments = json.loads(sys.argv[2])
    except json.JSONDecodeError as exc:
        _emit(f"Error: invalid JSON arguments: {exc}", {})
        return

    metrics: dict[str, Any] = {}
    try:
        if tool_name == "preflight":
            result = _preflight()
        elif tool_name == "read":
            result = _read(arguments, metrics)
        elif tool_name == "write":
            result = _write(arguments, metrics)
        elif tool_name == "write_docx":
            result = _write_docx(arguments, metrics)
        elif tool_name == "edit":
            result = _edit(arguments, metrics)
        elif tool_name == "glob":
            result = _glob(arguments, metrics)
        elif tool_name == "grep":
            result = _grep(arguments, metrics)
        else:
            result = f"Error: unknown container tool: {tool_name}"
    except Exception as exc:  # defensive boundary for model-facing tools
        result = f"Error executing {tool_name} in container: {exc}"

    _emit(result, metrics)


def _emit(result: str, metrics: dict[str, Any]) -> None:
    print(json.dumps({"result": result, "metrics": metrics}, ensure_ascii=False))


def _expand_tool_path(path_str: str) -> str:
    if not path_str:
        return path_str
    return (
        path_str.replace("${VDR_DIR}", str(VDR_DIR))
        .replace("$VDR_DIR", str(VDR_DIR))
        .replace("${OUTPUT_DIR}", str(OUTPUT_DIR))
        .replace("$OUTPUT_DIR", str(OUTPUT_DIR))
        .replace("${WORKSPACE_DIR}", str(WORKSPACE_DIR))
        .replace("$WORKSPACE_DIR", str(WORKSPACE_DIR))
        .replace("${SKILLS_DIR}", str(SKILLS_DIR))
        .replace("$SKILLS_DIR", str(SKILLS_DIR))
    )


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _ensure_allowed(path: Path, roots: tuple[Path, ...], action: str) -> Path:
    resolved = path.resolve()
    if not any(_is_relative_to(resolved, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(f"{action} path is outside mounted workspace: {resolved}. Allowed roots: {allowed}")
    return resolved


def _resolve_read_path(path_str: str) -> Path:
    path_str = _expand_tool_path(path_str)
    p = Path(path_str)
    if p.is_absolute():
        return _ensure_allowed(p, READABLE_ROOTS, "read")
    for base in (WORKSPACE_DIR, OUTPUT_DIR, VDR_DIR, SKILLS_DIR):
        candidate = base / p
        if candidate.exists():
            return _ensure_allowed(candidate, READABLE_ROOTS, "read")
    return _ensure_allowed(VDR_DIR / p, READABLE_ROOTS, "read")


def _resolve_write_path(path_str: str) -> Path:
    path_str = _expand_tool_path(path_str)
    p = Path(path_str)
    if p.is_absolute():
        return _ensure_allowed(p, WRITABLE_ROOTS, "write")
    return _ensure_allowed(OUTPUT_DIR / p, WRITABLE_ROOTS, "write")


def _resolve_search_path(path_str: str | None) -> Path:
    if path_str:
        expanded = _expand_tool_path(path_str)
        p = Path(expanded)
        if p.is_absolute():
            return _ensure_allowed(p, READABLE_ROOTS, "search")
        return _ensure_allowed(VDR_DIR / p, READABLE_ROOTS, "search")
    return VDR_DIR


def _read(arguments: dict[str, Any], metrics: dict[str, Any]) -> str:
    file_path = arguments.get("file_path", "")
    if not file_path:
        return "Error: file_path is required"

    resolved = _resolve_read_path(file_path)
    if not resolved.exists():
        return f"Error: file not found: {file_path}"
    if resolved.is_dir():
        return f"Error: {file_path} is a directory, not a file"

    if _is_relative_to(resolved, VDR_DIR):
        metrics["file_read"] = str(resolved.relative_to(VDR_DIR))
    else:
        metrics["file_read"] = str(resolved)

    content = _read_file_content(resolved)
    if arguments.get("line_count_only", False):
        return f"{file_path}: {_count_lines(content)} lines"

    offset = arguments.get("offset")
    limit = arguments.get("limit")
    return _slice_read_content(file_path, content, offset, limit, "limit" in arguments)


def _count_lines(content: str) -> int:
    return 0 if content == "" else len(content.splitlines())


def _slice_read_content(
    file_path: str,
    content: str,
    offset: int | None,
    limit: int | None,
    limit_provided: bool = False,
) -> str:
    lines = content.split("\n")
    total_lines = _count_lines(content)
    start = max(int(offset or 0), 0)

    if _full_read_requested(limit, limit_provided):
        end = len(lines)
        page = "\n".join(lines[start:end])
        if start == 0:
            return page
        return (
            f"[Read {file_path}: showing lines {start}-{max(end - 1, start)} "
            f"of {total_lines}. Full read requested. End of file.]\n\n{page}"
        )

    requested_limit = int(limit) if limit is not None else MAX_READ_LINES
    page_size = max(min(requested_limit, MAX_READ_LINES), 0)
    end = min(start + page_size, len(lines))
    page = "\n".join(lines[start:end])

    if start == 0 and end >= len(lines) and limit is None and not limit_provided:
        return content

    limit_note = _read_limit_note(limit, limit_provided, page_size)
    note = f"[Read {file_path}: showing lines {start}-{max(end - 1, start)} of {total_lines}. {limit_note} "
    if end < len(lines):
        note += f"Use offset={end}, limit={MAX_READ_LINES} to continue.]"
    else:
        note += "End of file.]"
    return f"{note}\n\n{page}" if page else note


def _read_limit_note(
    limit: int | None,
    limit_provided: bool,
    page_size: int,
) -> str:
    if not limit_provided:
        return f"Limit applied: default {MAX_READ_LINES} lines."
    if limit is None:
        return "Limit applied: explicit null means read to end."

    requested_limit = int(limit)
    if requested_limit > MAX_READ_LINES:
        return f"Limit applied: requested {requested_limit} lines, capped to {page_size} lines."
    return f"Limit applied: requested {requested_limit} lines."


def _full_read_requested(limit: int | None, limit_provided: bool) -> bool:
    if not limit_provided:
        return False
    if limit is None:
        return True
    return not isinstance(limit, bool) and limit == 0


def _read_file_content(target: Path) -> str:
    suffix = target.suffix.lower()
    if suffix == ".docx":
        return _parse_docx(target)
    if suffix == ".pptx":
        return _parse_pptx(target)
    if suffix == ".xlsx":
        return _parse_xlsx(target)
    if suffix == ".pdf":
        return _parse_pdf(target)
    return target.read_text(encoding="utf-8", errors="replace")


def _parse_docx(path: Path) -> str:
    try:
        result = subprocess.run(
            ["pandoc", str(path), "-t", "markdown", "--wrap=none"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        pass

    try:
        from docx import Document

        document = Document(path)
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
            parts.append("")
        return "\n".join(parts)
    except Exception as exc:
        return f"Error parsing .docx file: {exc}"


def _parse_pptx(path: Path) -> str:
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(path))
    return result.text_content


def _parse_xlsx(path: Path) -> str:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"=== Sheet: {sheet_name} ===")
        parts.append(df.to_string(index=False))
    return "\n".join(parts)


def _parse_pdf(path: Path) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
            for table in page.extract_tables():
                for row in table:
                    parts.append("\t".join(cell if cell else "" for cell in row))
                parts.append("")
    return "\n".join(parts)


def _rich_document_write_error(file_path: str) -> str | None:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".docx":
        return (
            "Error: write/edit cannot create or modify .docx files as plain text. "
            "Use write_docx with Markdown content to create a valid Word document "
            "under OUTPUT_DIR."
        )
    if suffix in {".pptx", ".xlsx", ".pdf"}:
        return (
            f"Error: write/edit cannot create or modify {suffix} files as plain text. "
            "Use the preinstalled document libraries or skill scripts to create a "
            f"valid {suffix} file under OUTPUT_DIR."
        )
    return None


def _write(arguments: dict[str, Any], metrics: dict[str, Any]) -> str:
    file_path = arguments.get("file_path", "")
    content = arguments.get("content", "")
    if not file_path:
        return "Error: file_path is required"
    if error := _rich_document_write_error(file_path):
        return error

    resolved = _resolve_write_path(file_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    metrics["files_written"] = 1
    return f"Wrote {len(content)} bytes to {file_path}"


def _write_docx(arguments: dict[str, Any], metrics: dict[str, Any]) -> str:
    file_path = arguments.get("file_path", "")
    markdown = arguments.get("markdown", "")
    if not file_path:
        return "Error: file_path is required"
    if not markdown:
        return "Error: markdown is required"

    normalized = _normalize_docx_path(file_path)
    if normalized.startswith("Error:"):
        return normalized

    resolved = _resolve_write_path(normalized)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "pandoc",
                "--from=gfm",
                "--to=docx",
                "--standalone",
                "--output",
                str(resolved),
            ],
            input=markdown,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return "Error: pandoc is required to create .docx files"
    except subprocess.TimeoutExpired:
        return "Error: write_docx timed out after 60s"
    except Exception as exc:
        return f"Error creating .docx file: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return f"Error creating .docx file with pandoc: {detail}"

    metrics["files_written"] = 1
    return f"Wrote DOCX to {normalized} ({resolved.stat().st_size} bytes)"


def _normalize_docx_path(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix and suffix != ".docx":
        return "Error: write_docx file_path must end with .docx"
    if suffix == ".docx":
        return file_path
    return str(path.with_suffix(".docx"))


def _edit(arguments: dict[str, Any], metrics: dict[str, Any]) -> str:
    file_path = arguments.get("file_path", "")
    old_string = arguments.get("old_string", "")
    new_string = arguments.get("new_string", "")
    replace_all = arguments.get("replace_all", False)
    if not file_path:
        return "Error: file_path is required"
    if error := _rich_document_write_error(file_path):
        return error

    resolved = _resolve_write_path(file_path)
    if not resolved.exists():
        resolved = _resolve_read_path(file_path)
        if not resolved.exists():
            return f"Error: file not found: {file_path}"
    if not any(_is_relative_to(resolved, root) for root in WRITABLE_ROOTS):
        return f"Error: cannot edit read-only mounted file: {file_path}"

    text = resolved.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {file_path}"
    if count > 1 and not replace_all:
        return f"Error: old_string found {count} times in {file_path}. Use replace_all=true to replace all."

    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    resolved.write_text(new_text, encoding="utf-8")
    metrics["files_edited"] = 1
    replaced = count if replace_all else 1
    return f"Replaced {replaced} occurrence(s) in {file_path}"


def _glob(arguments: dict[str, Any], metrics: dict[str, Any]) -> str:
    pattern = arguments.get("pattern", "")
    if not pattern:
        return "Error: pattern is required"
    metrics["glob_count"] = 1

    resolved = _resolve_search_path(arguments.get("path"))
    if not resolved.exists():
        return f"Error: path does not exist: {arguments.get('path')}"

    matches = sorted(
        (m for m in resolved.glob(pattern) if m.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return f"No files matching '{pattern}' in {resolved}"
    return "\n".join(str(m.relative_to(resolved)) for m in matches[:100])


def _grep(arguments: dict[str, Any], metrics: dict[str, Any]) -> str:
    pattern_str = arguments.get("pattern", "")
    if not pattern_str:
        return "Error: pattern is required"
    metrics["grep_count"] = 1

    resolved = _resolve_search_path(arguments.get("path"))
    if not resolved.exists():
        return f"Error: path does not exist: {arguments.get('path')}"

    try:
        regex = re.compile(pattern_str)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"

    results = []
    for fpath in resolved.glob(arguments.get("glob") or "**/*"):
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        matches = list(regex.finditer(text))
        if not matches:
            continue
        rel = str(fpath.relative_to(resolved))
        output_mode = arguments.get("output_mode", "files_with_matches")
        if output_mode == "files_with_matches":
            results.append(rel)
        elif output_mode == "count":
            results.append(f"{rel}: {len(matches)}")
        elif output_mode == "content":
            for i, line in enumerate(text.split("\n")):
                if regex.search(line):
                    results.append(f"{rel}:{i + 1}: {line}")

    return "\n".join(results[:250]) if results else f"No matches for '{pattern_str}'"


def _preflight() -> str:
    required_binaries = {
        "pandoc": ("pandoc",),
        "LibreOffice": ("soffice", "libreoffice"),
        "pdftoppm": ("pdftoppm",),
        "qpdf": ("qpdf",),
        "tesseract": ("tesseract",),
        "node": ("node",),
        "npm": ("npm",),
        "marp": ("marp",),
    }
    missing = []
    for label, candidates in required_binaries.items():
        if not any(shutil.which(candidate) for candidate in candidates):
            missing.append(label)

    imports = [
        "pandas",
        "pdfplumber",
        "docx",
        "markitdown",
        "pypdf",
        "reportlab",
        "pdf2image",
        "pytesseract",
        "PIL",
        "lxml",
        "defusedxml",
        "diff_match_patch",
        "docxtpl",
        "xlcalculator",
    ]
    for module in imports:
        try:
            __import__(module)
        except Exception:
            missing.append(f"python:{module}")

    node_check = (
        subprocess.run(
            ["node", "-e", "require('docx'); require('pptxgenjs')"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if shutil.which("node")
        else None
    )
    if node_check is not None and node_check.returncode != 0:
        missing.append("node:docx/pptxgenjs")

    if missing:
        return "Error: agent runtime image is missing required tools: " + ", ".join(sorted(missing))
    return "OK: agent runtime tools are installed"


if __name__ == "__main__":
    main()
