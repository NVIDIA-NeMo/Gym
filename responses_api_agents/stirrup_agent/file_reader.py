# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract text content and convert deliverable files for reward computation.

Reads common document formats (.docx, .pdf, .xlsx, .pptx, .txt, etc.)
and returns their combined text so the LLM judge can score actual content
rather than just the agent's finish-tool summary.

Also provides PDF conversion via LibreOffice headless for visual judging
with multimodal models (e.g., Gemini 3 Pro).
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# judge_panel owns the canonical audio/video extension sets and the GDPVal
# resources server routes tasks to judges from them. Importing rather than
# redeclaring keeps detection and emission in lockstep: a local copy that missed
# an extension would send a routed deliverable to a capable judge and then drop
# it here with no warning. judge_panel is stdlib-only, so this costs nothing.
from resources_servers.gdpval.judge_panel import AUDIO_EXTS, VIDEO_EXTS


MAX_TOTAL_CHARS = 20_000


def read_deliverable_files(output_dir: str) -> str:
    """Read text from all deliverable files in *output_dir*.

    Returns a single string with sections like::

        === report.docx ===
        <extracted text>

        === data.xlsx ===
        <extracted text>

    Truncated to ~20k chars to stay within judge context limits.
    """
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return ""

    files = sorted(output_path.iterdir())
    if not files:
        return ""

    parts: list[str] = []
    total_len = 0

    for fpath in files:
        if not fpath.is_file():
            continue

        ext = fpath.suffix.lower()
        try:
            text = _extract_text(fpath, ext)
        except Exception as exc:
            text = f"[Error reading {fpath.name}: {exc}]"

        if not text:
            continue

        section = f"=== {fpath.name} ===\n{text}"
        if total_len + len(section) > MAX_TOTAL_CHARS:
            remaining = MAX_TOTAL_CHARS - total_len
            if remaining > 200:
                section = section[:remaining] + "\n[...truncated]"
                parts.append(section)
            break

        parts.append(section)
        total_len += len(section)

    return "\n\n".join(parts)


def _extract_text(fpath: Path, ext: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    if ext in (".txt", ".md", ".csv", ".json", ".html", ".xml", ".log"):
        return _read_text(fpath)
    elif ext == ".docx":
        return _read_docx(fpath)
    elif ext == ".pdf":
        return _read_pdf(fpath)
    elif ext == ".xlsx":
        return _read_xlsx(fpath)
    elif ext == ".pptx":
        return _read_pptx(fpath)
    else:
        size = os.path.getsize(fpath)
        return f"[Binary file: {fpath.name}, {size} bytes]"


def _read_text(fpath: Path) -> str:
    return fpath.read_text(encoding="utf-8", errors="replace").strip()


def _read_docx(fpath: Path) -> str:
    from docx import Document

    doc = Document(str(fpath))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_pdf(fpath: Path) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(str(fpath)).strip()


def _read_xlsx(fpath: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(fpath), read_only=True, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                rows.append(", ".join(cells))
        if rows:
            parts.append(f"Sheet: {sheet_name}\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts)


def _read_pptx(fpath: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(fpath))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(para.text)
        if texts:
            parts.append(f"Slide {i}:\n" + "\n".join(texts))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PDF conversion for visual judging (Gemini 3 Pro)
# ---------------------------------------------------------------------------

OOXML_EXTS = {".docx", ".pptx", ".xlsx"}
# LibreOffice renders these like OOXML; preconvert.py already converts them.
LEGACY_OFFICE_EXTS = {".doc", ".ppt", ".xls"}
OFFICE_EXTS = OOXML_EXTS | LEGACY_OFFICE_EXTS
TEXT_EXTS = (
    {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".html", ".htm", ".svg"}
    | {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties", ".env"}
    | {".py", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".sql", ".r", ".tex", ".ipynb"}
    | {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".css", ".scss", ".less"}
    | {".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".kt", ".swift"}
    | {".rb", ".php", ".pl", ".lua", ".scala", ".jl", ".m", ".log", ".diff", ".patch"}
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}

# Derived, so it cannot drift from the branches below.
HANDLED_EXTS = TEXT_EXTS | OFFICE_EXTS | IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS | {".pdf"}

TEXT_SNIFF_BYTES = 8192
MAX_TEXT_CHARS_PER_FILE = 20_000
MAX_ARCHIVE_MEMBERS = 200
# Audio/video (AUDIO_EXTS / VIDEO_EXTS, imported above) are only passed through
# when the judge is AV-capable (e.g. MiniMax M3); image-only judges can't decode
# them, so they are otherwise skipped. Every extension in those sets needs an
# entry here, or it routes to a capable judge and is then dropped silently.
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".wma": "audio/x-ms-wma",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".aac": "audio/aac",
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".3gp": "video/3gpp",
}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1024:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _sniffs_as_text(fpath: Path) -> bool:
    """True if an unknown-extension file looks like text (Makefile, Dockerfile, ...)."""
    try:
        with fpath.open("rb") as fh:
            chunk = fh.read(TEXT_SNIFF_BYTES)
    except OSError:
        return False
    if not chunk or b"\x00" in chunk:
        return False
    text = chunk.decode("utf-8", errors="replace")
    ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r\f\v")
    if (text.count("�") + ctrl) / len(text) < 0.01:
        return True
    # cp1252/latin-1 is still text; only control characters make it binary.
    return ctrl / len(text) < 0.01


def _archive_manifest(fpath: Path) -> str | None:
    """Member listing, or None if not a readable archive."""
    import zipfile

    try:
        with zipfile.ZipFile(fpath) as zf:
            names = zf.namelist()
    except Exception:
        return None
    shown = names[:MAX_ARCHIVE_MEMBERS]
    lines = "\n".join(f"  {n}" for n in shown)
    if len(names) > len(shown):
        lines += f"\n  ... and {len(names) - len(shown)} more member(s)"
    return f"{len(names)} member(s):\n{lines}"


def _truncated(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS_PER_FILE:
        return text
    return text[:MAX_TEXT_CHARS_PER_FILE] + f"\n[... truncated at {MAX_TEXT_CHARS_PER_FILE:,} characters]"


def _convert_office_to_pdf(fpath: Path) -> Path | None:
    """Convert a .docx/.xlsx/.pptx file to PDF using LibreOffice headless.

    Returns the path to the generated PDF, or None on failure.
    Uses a unique user profile to avoid lock conflicts in concurrent workers.

    Whitespace in the input filename makes LibreOffice's batch-convert mode
    silently drop the file (the URI it builds isn't percent-encoded), so we
    stage the input to a tempdir with a sanitized basename and move the PDF
    back to the original location.
    """
    profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
    user_install = f"file://{profile_dir.as_posix()}"
    out_pdf = fpath.with_suffix(".pdf")
    stage_dir: Path | None = None
    input_path = fpath
    has_whitespace = any(c.isspace() for c in fpath.name)

    try:
        if has_whitespace:
            stage_dir = Path(tempfile.mkdtemp(prefix="lo-stage-"))
            safe_name = re.sub(r"\s+", "_", fpath.stem) + fpath.suffix
            input_path = stage_dir / safe_name
            shutil.copy2(fpath, input_path)
            lo_outdir = str(stage_dir)
        else:
            lo_outdir = str(fpath.parent)

        cmd = [
            "libreoffice",
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            f"-env:UserInstallation={user_install}",
            "--convert-to",
            "pdf",
            "--outdir",
            lo_outdir,
            str(input_path),
        ]
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        if stage_dir is not None:
            staged_pdf = stage_dir / (input_path.stem + ".pdf")
            if staged_pdf.exists():
                shutil.move(str(staged_pdf), str(out_pdf))
        if p.returncode != 0 or not out_pdf.exists():
            print(f"[file_reader] LibreOffice conversion failed for {fpath.name}: {p.stderr[:200]}", flush=True)
            return None
        return out_pdf
    except subprocess.TimeoutExpired:
        print(f"[file_reader] LibreOffice conversion timed out for {fpath.name}", flush=True)
        return None
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
        if stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _pdf_bytes_to_image_text_blocks(
    pdf_bytes: bytes,
    *,
    render_dpi: int,
    max_pages: int,
    include_text: bool,
) -> list[dict[str, Any]] | None:
    """Rasterize PDF bytes to page-image + text blocks for image-only judges.

    Delegates to :mod:`resources_servers.gdpval.media_conversion` (imported
    lazily so this module stays importable even where that package isn't on the
    path). Returns ``None`` when the helper is unavailable or yields nothing, so
    the caller can fall back to the native ``application/pdf`` data URL.
    """
    try:
        from resources_servers.gdpval.media_conversion import pdf_bytes_to_blocks
    except ImportError:
        return None
    blocks = pdf_bytes_to_blocks(
        pdf_bytes,
        dpi=render_dpi,
        max_pages=max_pages,
        include_text=include_text,
    )
    return blocks or None


def _av_block(mime: str, data: bytes, *, ext: str, file_type: str, openai_native: bool) -> dict[str, Any]:
    """Build an audio/video content block in the judge's dialect.

    Delegates to :mod:`resources_servers.gdpval.media_conversion` (imported
    lazily so this module stays importable where that package isn't on the
    path). Falls back to the legacy ``image_url`` data URL if the helper is
    unavailable — correct for frontier judges, and a benign degradation for
    self-hosted ones (which is only reachable when the gdpval package is
    present anyway).
    """
    try:
        from resources_servers.gdpval.media_conversion import audio_video_block

        return audio_video_block(mime, data, ext=ext, file_type=file_type, openai_native=openai_native)
    except ImportError:
        b64 = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def convert_deliverables_to_content_blocks(
    output_dir: str,
    *,
    media_mode: str = "native_pdf",
    render_dpi: int = 150,
    max_pages: int = 50,
    include_text: bool = True,
    audio_capable: bool = False,
    video_capable: bool = False,
) -> list[dict[str, Any]]:
    """Convert deliverable files to OpenAI-compatible content blocks for multimodal judging.

    Returns a list of content blocks suitable for the ``content`` field of an
    OpenAI chat message. Each file becomes either:

    - A text block (for .txt/.md/.csv etc.)
    - An image_url block with base64 data URL (for PDFs, images, converted Office docs)

    Office documents (.docx/.xlsx/.pptx) are first converted to PDF via
    LibreOffice headless so the judge sees rendered formatting/tables/charts.

    *media_mode* selects how PDFs/Office docs are presented: ``"native_pdf"``
    (default) sends them as ``application/pdf`` data URLs for frontier judges;
    ``"images_and_text"`` rasterizes each page to a PNG block and attaches the
    extracted text, for image-only local VLM judges (e.g. a gym-spawned Kimi
    K2.6). *render_dpi*, *max_pages*, and *include_text* tune that rendering.

    *audio_capable* / *video_capable* forward audio / video files (respectively)
    to judges that read that modality — tracked SEPARATELY because MiniMax-M3
    reads video but not audio (a video-only judge keeps video, skips audio). An
    unreadable modality's file is still announced by name and size rather than
    dropped. The AV block dialect follows
    *media_mode*: ``native_pdf`` (frontier judges, e.g. Gemini) uses an
    ``image_url`` data URL, while ``images_and_text`` (self-hosted vLLM judges)
    uses the standard ``video_url`` / ``input_audio`` content types vLLM routes to
    the media tower.
    """
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return []

    images_and_text = media_mode == "images_and_text"

    blocks: list[dict[str, Any]] = []
    converted_pdfs: list[Path] = []  # track for cleanup

    entries = sorted(output_path.iterdir())
    # A PDF an Office file renders from must not also be emitted standalone, or
    # the judge sees the same pages twice. Both spellings count: the same-stem
    # sidecar Plan.pptx.pdf, and the ordinary sibling Plan.pdf.
    consumed_pdfs: set[Path] = set()
    for entry in entries:
        if entry.is_file() and entry.suffix.lower() in OFFICE_EXTS:
            sidecar = entry.with_name(entry.name + ".pdf")
            consumed_pdfs.add(sidecar if sidecar.is_file() else entry.with_suffix(".pdf"))

    for fpath in entries:
        if not fpath.is_file() or fpath in consumed_pdfs:
            continue

        ext = fpath.suffix.lower()

        try:
            if ext in TEXT_EXTS:
                text = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    blocks.append({"type": "text", "text": f"\n{fpath.name}:\n{_truncated(text)}"})
                else:
                    blocks.append(
                        {"type": "text", "text": f"\n{fpath.name}: [present but EMPTY (0 bytes of content)]"}
                    )

            elif ext in OFFICE_EXTS:
                # Prefer a PDF preconvert already rendered. Reconverting is
                # wasteful, needs LibreOffice on the judge host, and the cleanup
                # below would delete someone else's artifact.
                sidecar = fpath.with_name(fpath.name + ".pdf")
                sibling = fpath.with_suffix(".pdf")
                pdf_path = sidecar if sidecar.is_file() else (sibling if sibling.is_file() else None)
                if pdf_path is None:
                    pdf_path = _convert_office_to_pdf(fpath)
                    if pdf_path and pdf_path.exists():
                        converted_pdfs.append(pdf_path)
                if pdf_path and pdf_path.exists():
                    data = pdf_path.read_bytes()
                    if images_and_text:
                        rendered = _pdf_bytes_to_image_text_blocks(
                            data, render_dpi=render_dpi, max_pages=max_pages, include_text=include_text
                        )
                        if rendered is not None:
                            blocks.append({"type": "text", "text": f"\n{fpath.name} (rendered from PDF):"})
                            blocks.extend(rendered)
                            continue
                    b64 = base64.b64encode(data).decode("ascii")
                    blocks.append({"type": "text", "text": f"\n{fpath.name} (converted to PDF):"})
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:application/pdf;base64,{b64}"},
                        }
                    )
                else:
                    # Fallback to text extraction
                    text = _extract_text(fpath, ext)
                    if text:
                        blocks.append({"type": "text", "text": f"\n{fpath.name} (text fallback):\n{text}"})

            elif ext == ".pdf":
                data = fpath.read_bytes()
                if images_and_text:
                    rendered = _pdf_bytes_to_image_text_blocks(
                        data, render_dpi=render_dpi, max_pages=max_pages, include_text=include_text
                    )
                    if rendered is not None:
                        blocks.append({"type": "text", "text": f"\n{fpath.name}:"})
                        blocks.extend(rendered)
                        continue
                b64 = base64.b64encode(data).decode("ascii")
                blocks.append({"type": "text", "text": f"\n{fpath.name}:"})
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:application/pdf;base64,{b64}"},
                    }
                )

            elif ext in IMAGE_EXTS:
                mime = MIME_TYPES.get(ext, "image/png")
                data = fpath.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                blocks.append({"type": "text", "text": f"\n{fpath.name}:"})
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )

            elif ext in AUDIO_EXTS or ext in VIDEO_EXTS:
                # Forward AV only to judges that read that SPECIFIC modality, gated
                # independently: video needs *video_capable*, audio *audio_capable*
                # — so MiniMax-M3 (video yes, audio no) keeps video and skips audio.
                # Among frontier judges only Gemini reads AV. Gemini (native_pdf)
                # accepts AV as an image_url data URL; self-hosted vLLM judges
                # (images_and_text) need the standard video_url / input_audio types,
                # which vLLM routes to the media tower.
                is_video = ext in VIDEO_EXTS
                file_type = "VIDEO" if is_video else "AUDIO"
                if video_capable if is_video else audio_capable:
                    mime = MIME_TYPES.get(ext, "application/octet-stream")
                    data = fpath.read_bytes()
                    blocks.append({"type": "text", "text": f"\n{fpath.name}:"})
                    blocks.append(_av_block(mime, data, ext=ext, file_type=file_type, openai_native=images_and_text))
                else:
                    # Gating is correct; emitting nothing is not -- the judge
                    # then grades against a file it believes was never produced.
                    size = fpath.stat().st_size
                    blocks.append(
                        {
                            "type": "text",
                            "text": (
                                f"\n{fpath.name}: [{file_type} deliverable, {_human_size(size)} "
                                f"({size:,} bytes) — present on disk, but this judge cannot decode "
                                f"{file_type.lower()}. Do NOT treat it as missing or unproduced. Grade the "
                                f"criteria that do not require {'viewing' if is_video else 'listening'}; "
                                f"mark the rest unverifiable rather than unmet. "
                                f"Statements in other files about this file's contents are unverified "
                                f"assertions, not evidence: do not award credit for a property you "
                                f"cannot observe directly.]"
                            ),
                        }
                    )

            else:
                # Any allowlist lags reality; an unlisted extension used to emit
                # nothing, so the judge scored it as never produced.
                size = fpath.stat().st_size
                if size == 0:
                    body = "[present but EMPTY (0 bytes)]"
                elif _sniffs_as_text(fpath):
                    text = fpath.read_text(encoding="utf-8", errors="replace").strip()
                    body = f"\n{_truncated(text)}" if text else "[present but EMPTY (0 bytes of content)]"
                else:
                    manifest = _archive_manifest(fpath)
                    if manifest is not None:
                        body = f"[archive, {_human_size(size)} — NOT missing] {manifest}"
                    else:
                        body = (
                            f"[deliverable present, {_human_size(size)} ({size:,} bytes) — NOT missing; "
                            f"content not extractable in this format. Statements in other files about "
                            f"its contents are unverified assertions, not evidence.]"
                        )
                blocks.append({"type": "text", "text": f"\n{fpath.name}: {body}"})
        except Exception as exc:
            blocks.append({"type": "text", "text": f"\n{fpath.name}: [Error: {exc}]"})

    # Clean up converted PDFs (they live next to the originals)
    for pdf_path in converted_pdfs:
        pdf_path.unlink(missing_ok=True)

    return blocks
