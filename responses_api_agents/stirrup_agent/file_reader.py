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

# Run state the agent writes into the SAME directory as its deliverables. These
# are not model output, and `.json` is in TEXT_EXTS, so without this the judge is
# shown the agent's own transcript and grades it as work product. Single source:
# every judging path imports it from here.
IGNORE_FILES = frozenset(
    {
        "finish_params.json",
        "history.json",
        "history.pkl",
        "inprogress_history.json",
        "metadata.json",
        "log.txt",
        "reference_files",
    }
)


def is_deliverable(path: Path) -> bool:
    """True if *path* is agent-produced output rather than run state."""
    return path.is_file() and path.name not in IGNORE_FILES


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
        if not is_deliverable(fpath):
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
    # TEXT_EXTS, not a second shorter list: a .ts/.py/.yaml deliverable was source
    # on the block path and "[Binary file: ...]" here.
    if ext in TEXT_EXTS:
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
# Per-file ceiling, and the aggregate across all text blocks in one request.
# Upstream has no cap at all: a big generated log produced a 720k-token request
# that the judge rejected outright, leaving the task silently unjudged.
MAX_TEXT_BLOCK_CHARS = 200_000
MAX_TOTAL_TEXT_BLOCK_CHARS = 400_000
# Header and truncation notice per block; reserved before allotting because
# it is tokens too.
TEXT_BLOCK_OVERHEAD_CHARS = 120
MAX_ARCHIVE_ENTRIES = 200
MAX_ARCHIVE_TEXT_CHARS = 120_000
MAX_ARCHIVE_MEMBER_BYTES = 2_000_000

# Document formats that happen to BE zip containers. Without this they open
# cleanly as archives and the judge is handed a listing of OOXML/ODF internals
# (`word/document.xml`, `[Content_Types].xml`) instead of the document.
ARCHIVE_DOC_EXTS = {
    ".xlsm",
    ".docm",
    ".pptm",
    ".xlsb",
    ".odt",
    ".ods",
    ".odp",
    ".odg",
    ".epub",
    ".jar",
    ".whl",
    ".apk",
    ".ipa",
}
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


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
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
    """Member listing plus text-member contents, or None if not a readable archive.

    Rubrics ask both whether a bundle contains X and whether what is inside is
    correct, so names alone cannot answer them.
    """
    import zipfile

    # Must come first: these open cleanly as zips and would otherwise be
    # summarised as their own XML internals.
    if fpath.suffix.lower() in ARCHIVE_DOC_EXTS:
        return None

    try:
        with zipfile.ZipFile(fpath) as zf:
            infos = zf.infolist()
            lines = [f"  {i.filename} ({i.file_size:,} bytes)" for i in infos[:MAX_ARCHIVE_ENTRIES]]
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                lines.append(f"  ... and {len(infos) - MAX_ARCHIVE_ENTRIES:,} more entries")

            # Read IN MEMORY, never extracted: nothing is written to disk, so
            # there is no zip-slip to defend against. Bounded by per-member size,
            # aggregate characters, and the caller's own text budget.
            bodies: list[str] = []
            spent = 0
            for info in infos:
                if info.is_dir() or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    continue
                if Path(info.filename).suffix.lower() not in TEXT_EXTS:
                    continue
                if spent >= MAX_ARCHIVE_TEXT_CHARS:
                    bodies.append("[archive text budget exhausted; remaining members listed above but not shown]")
                    break
                try:
                    raw = zf.read(info)
                except Exception:
                    continue
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                take = text[: MAX_ARCHIVE_TEXT_CHARS - spent]
                spent += len(take)
                truncated = "\n[...member truncated]" if len(take) < len(text) else ""
                bodies.append(f"--- {info.filename} ---\n{take}{truncated}")
    except Exception:
        return None

    out = f"{len(infos)} member(s):\n" + "\n".join(lines)
    if bodies:
        out += "\n\n" + "\n\n".join(bodies)
    return out


def _fair_text_allowances(sizes: list[int], total_budget: int, per_file_cap: int) -> list[int]:
    """Max-min fair split of *total_budget* across text deliverables of *sizes*.

    Smallest first: each file may take an equal share of what is left divided by
    how many still need one. Files under their share take only what they need and
    hand back the surplus, so a small real deliverable is never starved by a large
    one that happens to sort earlier -- renaming a file must not change the score.
    Returned in the order *sizes* was given.
    """
    allowances = [0] * len(sizes)
    remaining = total_budget
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    for position, idx in enumerate(order):
        still_to_serve = len(order) - position
        share = remaining // still_to_serve
        take = max(0, min(sizes[idx], per_file_cap, share))
        allowances[idx] = take
        remaining -= take
    return allowances


def _convert_office_to_pdf(fpath: Path, out_dir: Path | None = None) -> Path | None:
    """Convert a .docx/.xlsx/.pptx file to PDF using LibreOffice headless.

    With *out_dir* the PDF is written there instead of beside *fpath*, so a
    judging pass never writes into the directory it is reading. The in-place
    default is kept because ``preconvert.py`` relies on it.

    Returns the path to the generated PDF, or None on failure.
    Uses a unique user profile to avoid lock conflicts in concurrent workers.

    Whitespace in the input filename makes LibreOffice's batch-convert mode
    silently drop the file (the URI it builds isn't percent-encoded), so we
    stage the input to a tempdir with a sanitized basename and move the PDF
    back to the original location.
    """
    profile_dir = Path(tempfile.mkdtemp(prefix="lo-profile-"))
    user_install = f"file://{profile_dir.as_posix()}"
    dest_dir = out_dir if out_dir is not None else fpath.parent
    out_pdf = dest_dir / (fpath.stem + ".pdf")
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
            lo_outdir = str(dest_dir)

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
    scratch_dirs: list[Path] = []  # tempdirs holding PDFs this pass converted

    entries = sorted(output_path.iterdir())
    # A PDF an Office file renders from must not also be emitted standalone, or
    # the judge sees the same pages twice. Both spellings count: the same-stem
    # sidecar Plan.pptx.pdf, and the ordinary sibling Plan.pdf.
    # How many Office files share each stem. `Report.docx` and `Report.pptx` both
    # point at `Report.pdf`, which cannot say which of them it renders, so the
    # plain sibling is only trustworthy when the stem is unambiguous. preconvert
    # writes an injective `Report.docx.pdf` sidecar for the ambiguous case.
    office_stem_counts: dict[str, int] = {}
    for entry in entries:
        if is_deliverable(entry) and entry.suffix.lower() in OFFICE_EXTS:
            office_stem_counts[entry.stem] = office_stem_counts.get(entry.stem, 0) + 1

    consumed_pdfs: set[Path] = set()
    for entry in entries:
        if is_deliverable(entry) and entry.suffix.lower() in OFFICE_EXTS:
            sidecar = entry.with_name(entry.name + ".pdf")
            if sidecar.is_file():
                consumed_pdfs.add(sidecar)
            elif office_stem_counts.get(entry.stem, 0) == 1:
                consumed_pdfs.add(entry.with_suffix(".pdf"))

    # Which files the judge receives AS TEXT. Decided once and reused by both the
    # allotment and the dispatch loop: deciding it twice lets a file be emitted as
    # text while holding no allowance, i.e. truncated to nothing.
    texty = {
        p
        for p in entries
        if is_deliverable(p)
        and p not in consumed_pdfs
        and (p.suffix.lower() in TEXT_EXTS or (p.suffix.lower() not in HANDLED_EXTS and _sniffs_as_text(p)))
    }
    text_files = [p for p in entries if p in texty]
    body_budget = max(0, MAX_TOTAL_TEXT_BLOCK_CHARS - len(text_files) * TEXT_BLOCK_OVERHEAD_CHARS)
    allowance_by_name = dict(
        zip(
            (p.name for p in text_files),
            _fair_text_allowances([p.stat().st_size for p in text_files], body_budget, MAX_TEXT_BLOCK_CHARS),
        )
    )

    emitted_chars = 0
    omitted_names: list[str] = []

    def _text_block(header: str, body: str, allowance: int) -> dict[str, Any] | None:
        """A text block for *body*, trimmed to *allowance*.

        None once the aggregate budget is spent: with enough files even the headers
        alone would blow the request, so past that point they are summarised in one
        block rather than named individually.
        """
        nonlocal emitted_chars
        if emitted_chars >= MAX_TOTAL_TEXT_BLOCK_CHARS:
            return None
        cap = max(0, min(MAX_TEXT_BLOCK_CHARS, allowance))
        if len(body) > cap:
            shown = body[:cap] + f"\n[...truncated: {len(body) - cap:,} of {len(body):,} chars omitted]"
        else:
            shown = body
        block = {"type": "text", "text": f"\n{header}\n{shown}"}
        emitted_chars += len(block["text"])
        return block

    for fpath in entries:
        if not is_deliverable(fpath) or fpath in consumed_pdfs:
            continue

        ext = fpath.suffix.lower()

        try:
            if fpath in texty:
                text = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    block = _text_block(f"{fpath.name}:", text, allowance_by_name.get(fpath.name, 0))
                else:
                    block = _text_block(f"{fpath.name}:", "[present but EMPTY (0 bytes of content)]", 200)
                blocks.append(block) if block else omitted_names.append(fpath.name)

            elif ext in OFFICE_EXTS:
                # Prefer a PDF preconvert already rendered. Reconverting is
                # wasteful, needs LibreOffice on the judge host, and the cleanup
                # below would delete someone else's artifact.
                sidecar = fpath.with_name(fpath.name + ".pdf")
                sibling = fpath.with_suffix(".pdf")
                if sidecar.is_file():
                    pdf_path = sidecar
                elif sibling.is_file() and office_stem_counts.get(fpath.stem, 0) == 1:
                    pdf_path = sibling
                else:
                    # Ambiguous stem with no sidecar: rendering it ourselves is the
                    # only way to know which file the PDF belongs to.
                    pdf_path = None
                if pdf_path is None:
                    scratch = Path(tempfile.mkdtemp(prefix="gdpval-render-"))
                    scratch_dirs.append(scratch)
                    pdf_path = _convert_office_to_pdf(fpath, out_dir=scratch)
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
                        block = _text_block(f"{fpath.name} (text fallback):", text, MAX_TEXT_BLOCK_CHARS)
                        blocks.append(block) if block else omitted_names.append(fpath.name)

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
                    body = text if text else "[present but EMPTY (0 bytes of content)]"
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
                block = _text_block(f"{fpath.name}:", body, MAX_TEXT_BLOCK_CHARS)
                blocks.append(block) if block else omitted_names.append(fpath.name)
        except Exception as exc:
            blocks.append({"type": "text", "text": f"\n{fpath.name}: [Error: {exc}]"})

    if omitted_names:
        shown = ", ".join(omitted_names[:20])
        if len(omitted_names) > 20:
            shown += f", and {len(omitted_names) - 20} more"
        blocks.append(
            {
                "type": "text",
                "text": f"\n[{len(omitted_names)} text deliverable(s) omitted, budget exhausted: {shown}]",
            }
        )

    # Clean up only what this pass created; it all lives in tempdirs.
    for scratch in scratch_dirs:
        shutil.rmtree(scratch, ignore_errors=True)

    return blocks
