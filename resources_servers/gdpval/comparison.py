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
"""GDPVal pairwise comparison judging.

Used by the GDPVal resources server's ``verify`` (per-task pairwise judge
between the eval model and a reference model's deliverables) and
``aggregate_metrics`` (turns win/loss/tie counts into an ELO rating).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from openai import APITimeoutError

from resources_servers.gdpval.judge_panel import AUDIO_EXTS, VIDEO_EXTS, merge_create_kwargs, sample_judge
from resources_servers.gdpval.preconvert import (
    OFFICE_EXTENSIONS,
    AttachmentBudget,
    extract_xlsx_structured_text,
    resolve_pdf_provenance,
    sidecar_pdf,
)
from resources_servers.gdpval.scoring import is_permanent_judge_error


LOGGER = logging.getLogger(__name__)


def _ignore_files() -> frozenset[str]:
    """Run-state names that must never be judged as a submission.

    Owned by the agent that writes them rather than copied into each judging
    path. Imported lazily: ``stirrup_agent/__init__`` eagerly imports Ray and
    FastAPI, and ``multistage_elo`` imports this module at module scope.
    ``sys.modules`` caches it, so repeat calls are a dict lookup.
    """
    from responses_api_agents.stirrup_agent.file_reader import IGNORE_FILES

    return IGNORE_FILES


JUDGE_PROMPT = (
    "Given a task description and reference files, select which of two submission file(s) "
    "better completed the task. "
    "Explain your reasoning then answer BOXED[A], BOXED[B], or BOXED[TIE].\n"
)
FINAL_VERDICT_REMINDER = (
    "End your response with exactly one verdict token on its own final line: "
    "BOXED[A], BOXED[B], or BOXED[TIE]. A response without one is invalid.\n"
)

A_WIN_RESPONSE = "BOXED[A]"
B_WIN_RESPONSE = "BOXED[B]"
TIE_RESPONSE = "BOXED[TIE]"

TASK_TEMPLATE = "<TASK_DESCRIPTION_START>\n{task}\n<TASK_DESCRIPTION_END>\n\n"

REFERENCES_OPEN = "<REFERENCES_FILES_START>\n"
REFERENCES_CLOSE = "\n<REFERENCES_FILES_END>\n\n"

SUBMISSION_A_OPEN = "<SUBMISSION_A_START>\n"
SUBMISSION_A_CLOSE = "\n<SUBMISSION_A_END>\n\n"
SUBMISSION_B_OPEN = "<SUBMISSION_B_START>\n"
SUBMISSION_B_CLOSE = "\n<SUBMISSION_B_END>\n\n"

REQUEST_MAX_ATTEMPTS = 5
REQUEST_INITIAL_BACKOFF_SECONDS = 5.0
REQUEST_BACKOFF_MULTIPLIER = 2.0
REQUEST_MAX_BACKOFF_SECONDS = 60.0
# Per-request OpenAI client timeout. A local multimodal VLM judge must prefill
# the whole payload (dozens of rasterized pages + extracted text per side)
# before the first token, which for image-dense deliverables can exceed the
# 120 s originally tuned for a flaky frontier proxy -- silently dropping
# slow-but-healthy local judgements and biasing the ELO fit. Default 300 s;
# override with ``GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS``. Kept non-retryable
# (below) so a genuinely dead upstream still bounds wall-clock rather than
# retrying N x timeout.
JUDGE_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS", "300"))


def _byte_limit(name: str, default: int) -> int:
    """Read a positive byte limit while retaining the hardened default."""
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


# Per-file size cap on multimodal content blocks. Above this the file is
# replaced with a one-line text marker so the judge still knows what was
# claimed without us pushing 100s of MB of base64 through the proxy.
# Set high enough (250 MB) that normal multi-stem audio and reference
# videos in the GDPVal task set still go through; only catches the truly
# pathological cases (e.g. ``task_a941b6d8`` 657 MB overlay clip).
MAX_FILE_BYTES_FOR_JUDGE = _byte_limit("GDPVAL_MAX_FILE_BYTES_FOR_JUDGE", 250 * 1024 * 1024)
# Aggregate encoded payload is what the HTTP server sees. Keeping it below
# 400 MiB leaves roughly 80 MB even if the model server's advertised 500 MB
# ceiling is decimal, for JSON, prompts, spreadsheet text, and framing. The raw
# ceiling separately prevents pathological media accumulation before base64.
MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE = _byte_limit(
    "GDPVAL_MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE", 300 * 1024 * 1024
)
MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE = _byte_limit(
    "GDPVAL_MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE", 400 * 1024 * 1024
)
# Each directory is built independently. Keeping every section below an equal,
# conservative share means references can never consume the bytes needed to
# show both submissions, and limits peak memory before message assembly.
MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE = _byte_limit(
    "GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE", 96 * 1024 * 1024
)
MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE = _byte_limit(
    "GDPVAL_MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE", 128 * 1024 * 1024
)
MAX_TEXT_FILE_CHARS_FOR_JUDGE = 200_000
MAX_SECTION_TEXT_CHARS_FOR_JUDGE = 1_000_000
MAX_XLSX_TEXT_CHARS_FOR_JUDGE = 120_000
# Includes base64, all text, data-URL prefixes, and conservative JSON framing.
# This stays below endpoints configured with a 500 MB request-body ceiling.
MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE = _byte_limit(
    "GDPVAL_MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE", 420 * 1024 * 1024
)
MAX_TASK_PROMPT_CHARS_FOR_JUDGE = 1_000_000
# Archives expand before their members enter the normal payload builders, so
# bound that work independently. Extracting more than one section's raw budget
# cannot make more binary evidence visible to the judge.
MAX_ZIP_ARCHIVE_BYTES_FOR_JUDGE = MAX_FILE_BYTES_FOR_JUDGE
MAX_ZIP_MEMBERS_FOR_JUDGE = 1_000
MAX_ZIP_MEMBER_BYTES_FOR_JUDGE = MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES_FOR_JUDGE = MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE
ZIP_COPY_CHUNK_BYTES = 1024 * 1024
RETRYABLE_ERROR_MARKERS = (
    "429",
    "502",
    "503",
    "504",
    "rate limit",
    "ratelimit",
    "resource_exhausted",
    "resource has been exhausted",
    "throttling",
    "bad gateway",
    "gateway timeout",
    "gateway time-out",
    "service unavailable",
    "upstream",
    "temporarily unavailable",
    "connection error",
)
# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


def _data_url(mime_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _bounded_text(text: str, cap: int, marker: str = "\n[...truncated]") -> str:
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    if cap <= len(marker):
        return marker[-cap:]
    return text[: cap - len(marker)] + marker


def _load_raw_text(path: str | Path, max_chars: int = MAX_TEXT_FILE_CHARS_FOR_JUDGE) -> str:
    """Read only the prefix that can be emitted to the judge."""

    if max_chars <= 0:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read(max_chars + 1)
    return _bounded_text(text, max_chars)


def _load_media(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _resolve_office_pdf_path(
    path: str | Path,
    *,
    cached_pdf_path: Path | None = None,
    provenance_resolved: bool = False,
) -> Path | None:
    """Return a provenance-safe Office render without rereading known listings."""

    input_path = Path(path).resolve()
    if provenance_resolved:
        return cached_pdf_path.resolve() if cached_pdf_path is not None else None
    try:
        entries = [entry.resolve() for entry in input_path.parent.iterdir() if entry.is_file()]
    except OSError:
        return None
    output_path = resolve_pdf_provenance(entries).office_pdfs.get(input_path)
    if output_path is None and input_path.suffix.lower() not in OFFICE_EXTENSIONS:
        # Preserve the historical fallback for less-common document formats
        # that are not preconverted by GDPVal but may already have a render.
        injective = sidecar_pdf(input_path)
        plain = input_path.with_suffix(".pdf")
        output_path = injective if injective.is_file() else plain if plain.is_file() else None
    return output_path


def _convert_to_pdf(path: str | Path) -> bytes | None:
    """Load the provenance-safe pre-converted PDF for an Office source."""

    output_path = _resolve_office_pdf_path(path)
    return _load_media(output_path) if output_path is not None else None


def _attachment_omission(file_name: str, size_bytes: int, reason: str) -> dict[str, str]:
    return {
        "type": "text",
        "text": f"[attachment omitted for {file_name}: {_human_attachment_size(size_bytes)} raw; {reason}]",
    }


def _reserve_attachment_path(
    path: Path,
    file_name: str,
    budget: AttachmentBudget | None,
) -> tuple[int, dict[str, str] | None]:
    """Stat and reserve an attachment before any read or base64 allocation."""

    try:
        size_bytes = path.stat().st_size
    except OSError:
        return 0, {"type": "text", "text": f"[attachment unavailable for {file_name}]"}
    if size_bytes > MAX_FILE_BYTES_FOR_JUDGE:
        return size_bytes, _attachment_omission(file_name, size_bytes, "per-file judge payload limit")
    if budget is not None and not budget.reserve(size_bytes):
        return size_bytes, _attachment_omission(file_name, size_bytes, "section judge payload budget")
    return size_bytes, None


def _maybe_unzip(path: str | Path) -> tuple[Path | None, list[Path]]:
    """Extract a bounded zip into a per-call tempdir.

    The reference deliverables tree is mounted read-only in production, so the
    previous behaviour of ``extractall(path.parent)`` raised ``PermissionError``
    and failed /verify outright. Member count, individual expanded size, and
    total expanded size are checked before opening a member; the streaming copy
    enforces the declared size as a second line of defence. Absolute, parent-
    traversing, duplicate, and symlink entries are ignored.

    Returns ``(extract_dir, member_paths)``. Callers are responsible for
    ``shutil.rmtree(extract_dir)`` after reading the members.
    """
    path = Path(path)
    extract_dir: Path | None = None
    try:
        if path.stat().st_size > MAX_ZIP_ARCHIVE_BYTES_FOR_JUDGE:
            LOGGER.warning("zip %s exceeds the compressed archive limit; ignoring it", path)
            return None, []
        with zipfile.ZipFile(path, "r") as zip_ref:
            extract_dir = Path(tempfile.mkdtemp(prefix="gdpval_unzip_"))
            extract_root = extract_dir.resolve()
            extracted_paths: list[Path] = []
            extracted_targets: set[Path] = set()
            total_uncompressed = 0
            examined_files = 0

            for info in zip_ref.infolist():
                if info.is_dir():
                    continue
                examined_files += 1
                if examined_files > MAX_ZIP_MEMBERS_FOR_JUDGE:
                    LOGGER.warning(
                        "zip %s exceeds the %d-member extraction limit; ignoring remaining members",
                        path,
                        MAX_ZIP_MEMBERS_FOR_JUDGE,
                    )
                    break

                # ZIP member names are POSIX paths regardless of host OS. Treat
                # backslashes as separators too so a Windows consumer cannot turn
                # a benign-looking name into traversal later.
                member_path = PurePosixPath(info.filename.replace("\\", "/"))
                parts = member_path.parts
                mode = info.external_attr >> 16
                if (
                    member_path.is_absolute()
                    or not parts
                    or any(part in {"", ".", ".."} for part in parts)
                    or re.match(r"^[A-Za-z]:", parts[0])
                    or stat.S_ISLNK(mode)
                ):
                    LOGGER.warning("ignoring unsafe zip member %r in %s", info.filename, path)
                    continue

                member_size = max(0, info.file_size)
                if member_size > MAX_ZIP_MEMBER_BYTES_FOR_JUDGE:
                    LOGGER.warning("ignoring oversize zip member %r in %s", info.filename, path)
                    continue
                if total_uncompressed + member_size > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES_FOR_JUDGE:
                    LOGGER.warning("ignoring zip member %r after aggregate expansion limit", info.filename)
                    continue

                target = (extract_root / Path(*parts)).resolve()
                if not target.is_relative_to(extract_root) or target in extracted_targets or target.exists():
                    LOGGER.warning("ignoring duplicate or escaping zip member %r in %s", info.filename, path)
                    continue

                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                except OSError:
                    LOGGER.warning("ignoring conflicting zip member %r in %s", info.filename, path)
                    continue
                written = 0
                try:
                    with zip_ref.open(info, "r") as source, target.open("xb") as destination:
                        while written < member_size:
                            chunk = source.read(min(ZIP_COPY_CHUNK_BYTES, member_size - written))
                            if not chunk:
                                break
                            destination.write(chunk)
                            written += len(chunk)
                        if written != member_size or source.read(1):
                            raise ValueError("expanded size does not match ZIP metadata")
                except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
                    if target.is_file() or target.is_symlink():
                        target.unlink(missing_ok=True)
                    LOGGER.warning("failed bounded extraction of zip member %r in %s", info.filename, path)
                    continue

                total_uncompressed += written
                extracted_targets.add(target)
                extracted_paths.append(target)
        return extract_dir, extracted_paths
    except (zipfile.BadZipFile, zipfile.LargeZipFile, FileNotFoundError, OSError):
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)
        return None, []


# Keyed by bare (dot-less) extension, mirroring FILE_TYPE_MAP's lookup. Media
# entries are generated from judge_panel's canonical sets below, so an extension
# that routing treats as audio/video can never be missing an emitter here. Note
# these are the mime strings the frontier judges document (e.g. "video/mov",
# "audio/mp3"), which differ from the IANA names used on the rubric path.
_AUDIO_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "opus": "audio/opus",
    "wma": "audio/x-ms-wma",
    "aiff": "audio/aiff",
    "aif": "audio/aiff",
    "aac": "audio/aac",
    "flac": "audio/flac",
}
_VIDEO_MIME_TYPES: dict[str, str] = {
    "mp4": "video/mp4",
    "m4v": "video/x-m4v",
    "mov": "video/mov",
    "avi": "video/avi",
    "mkv": "video/x-matroska",
    "webm": "video/webm",
    "wmv": "video/wmv",
    "flv": "video/x-flv",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
    "3gp": "video/3gpp",
}


def _media_entries(exts: frozenset[str], mime_types: dict[str, str], file_type: str) -> dict[str, dict[str, Any]]:
    """Build FILE_TYPE_MAP entries for every extension routing can detect."""
    return {
        ext.lstrip("."): {
            "type": file_type,
            "converter": _load_media,
            "mime_type": mime_types.get(ext.lstrip("."), "application/octet-stream"),
        }
        for ext in exts
    }


FILE_TYPE_MAP: dict[str, dict[str, Any]] = {
    "pdf": {"type": "PDF", "converter": None, "mime_type": "application/pdf"},
    "jpg": {"type": "IMG", "converter": _load_media, "mime_type": "image/jpeg"},
    "jpeg": {"type": "IMG", "converter": _load_media, "mime_type": "image/jpeg"},
    "png": {"type": "IMG", "converter": _load_media, "mime_type": "image/png"},
    "webp": {"type": "IMG", "converter": _load_media, "mime_type": "image/webp"},
    "heic": {"type": "IMG", "converter": _load_media, "mime_type": "image/heic"},
    "heif": {"type": "IMG", "converter": _load_media, "mime_type": "image/heif"},
    **_media_entries(AUDIO_EXTS, _AUDIO_MIME_TYPES, "AUDIO"),
    **_media_entries(VIDEO_EXTS, _VIDEO_MIME_TYPES, "VIDEO"),
    "docx": {"type": "DOC", "converter": _convert_to_pdf, "mime_type": "application/pdf"},
    "pptx": {"type": "DOC", "converter": _convert_to_pdf, "mime_type": "application/pdf"},
    "xlsx": {"type": "DOC", "converter": _convert_to_pdf, "mime_type": "application/pdf"},
    "txt": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "csv": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "json": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "xml": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "html": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "md": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "yaml": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "yml": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "py": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "sh": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "bash": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "c": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "cpp": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "java": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "js": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "tsx": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "sol": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
    "ts": {"type": "TXT", "converter": _load_raw_text, "mime_type": None},
}


def get_file_content_block(
    file_dir: str,
    file_name: str,
    *,
    attachment_budget: AttachmentBudget | None = None,
    cached_office_pdf: Path | None = None,
    provenance_resolved: bool = False,
    max_text_chars: int = MAX_TEXT_FILE_CHARS_FOR_JUDGE,
) -> dict | None:
    """Return a single OpenAI content block (dict) for a file, or ``None``."""
    file_extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if file_extension not in FILE_TYPE_MAP:
        file_type = "DOC"
        file_mime_type = "application/pdf"
    else:
        file_type = FILE_TYPE_MAP[file_extension]["type"]
        file_mime_type = FILE_TYPE_MAP[file_extension]["mime_type"]

    full_path = Path(file_dir) / file_name

    try:
        size_bytes = full_path.stat().st_size
    except OSError:
        return None
    if size_bytes > MAX_FILE_BYTES_FOR_JUDGE:
        size_mb = size_bytes / (1024 * 1024)
        return {
            "type": "text",
            "text": f"[oversize: {file_name} {size_mb:.1f}MB — not included]",
        }

    try:
        if file_type == "TXT":
            raw_text = _load_raw_text(full_path, max_text_chars)
            return {"type": "text", "text": raw_text}

        if file_type == "DOC":
            pdf_path = _resolve_office_pdf_path(
                full_path,
                cached_pdf_path=cached_office_pdf,
                provenance_resolved=provenance_resolved,
            )
            if pdf_path is None:
                return None
            _size, omitted = _reserve_attachment_path(pdf_path, file_name, attachment_budget)
            if omitted is not None:
                return omitted
            data = _load_media(pdf_path)
            return {"type": "image_url", "image_url": {"url": _data_url(file_mime_type, data)}}

        if file_type == "PDF":
            _size, omitted = _reserve_attachment_path(full_path, file_name, attachment_budget)
            if omitted is not None:
                return omitted
            data = _load_media(full_path)
            return {"type": "image_url", "image_url": {"url": _data_url(file_mime_type, data)}}

        if file_type in ("IMG", "AUDIO", "VIDEO"):
            _size, omitted = _reserve_attachment_path(full_path, file_name, attachment_budget)
            if omitted is not None:
                return omitted
            media_bytes = _load_media(full_path)
            return {"type": "image_url", "image_url": {"url": _data_url(file_mime_type, media_bytes)}}

    except Exception as e:
        raise RuntimeError(f"Error getting file: {file_name} in directory: {file_dir}: {e}") from e

    return None


MAX_RASTER_PAGE_PIXELS_FOR_JUDGE = 40_000_000


def _pdf_path_to_image_text_blocks(
    pdf_path: Path,
    *,
    render_dpi: int,
    max_pages: int,
    include_text: bool,
    attachment_budget: AttachmentBudget,
    max_text_chars: int,
    file_name: str,
) -> list[dict]:
    """Render one PDF page at a time, reserving PNG bytes before base64."""

    try:
        import fitz
    except ImportError:
        LOGGER.warning("PyMuPDF (fitz) not installed; cannot rasterize PDF for image-only judge")
        return []

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        LOGGER.warning("failed to open PDF %s for rasterization: %r", file_name, exc)
        return []

    images: list[dict] = []
    text_parts: list[str] = []
    text_remaining = max(0, max_text_chars)
    text_truncated = False
    try:
        total_pages = document.page_count
        page_limit = min(total_pages, max(0, max_pages))
        zoom = render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(page_limit):
            try:
                page = document.load_page(page_index)
                if include_text and text_remaining > 0:
                    page_text = (page.get_text("text") or "").strip()
                    if page_text:
                        separator = 2 if text_parts else 0
                        if separator < text_remaining:
                            take = min(len(page_text), text_remaining - separator)
                            text_parts.append(page_text[:take])
                            text_remaining -= separator + take
                            if take < len(page_text):
                                text_truncated = True
                elif include_text and text_parts:
                    text_truncated = True

                rect = page.rect
                width = math.ceil(rect.width * zoom)
                height = math.ceil(rect.height * zoom)
                if width * height > MAX_RASTER_PAGE_PIXELS_FOR_JUDGE:
                    images.append(
                        {
                            "type": "text",
                            "text": f"[page {page_index + 1} omitted for {file_name}: raster dimensions too large]",
                        }
                    )
                    continue
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                png = pixmap.tobytes("png")
            except Exception as exc:
                LOGGER.warning("failed to render PDF page %d for %s: %r", page_index, file_name, exc)
                continue
            if not attachment_budget.reserve(len(png)):
                images.append(_attachment_omission(file_name, len(png), "section judge payload budget"))
                break
            images.append({"type": "image_url", "image_url": {"url": _data_url("image/png", png)}})

        if total_pages > page_limit:
            images.append({"type": "text", "text": f"[truncated: rendered {page_limit} of {total_pages} pages]"})
    finally:
        document.close()

    blocks: list[dict] = []
    if text_parts:
        extracted = "\n\n".join(text_parts)
        if text_truncated:
            extracted = _bounded_text(extracted + "x", max_text_chars, "\n[...text truncated]")
        blocks.append({"type": "text", "text": f"[extracted text]\n{extracted}"})
    blocks.extend(images)
    return blocks


def get_file_image_text_blocks(
    file_dir: str,
    file_name: str,
    *,
    render_dpi: int,
    max_pages: int,
    include_text: bool,
    audio_capable: bool = False,
    video_capable: bool = False,
    attachment_budget: AttachmentBudget | None = None,
    cached_office_pdf: Path | None = None,
    provenance_resolved: bool = False,
    max_text_chars: int = MAX_TEXT_FILE_CHARS_FOR_JUDGE,
) -> list[dict]:
    """``images_and_text`` variant of :func:`get_file_content_block`.

    For image-only local VLM judges (e.g. a gym-spawned Kimi K2.6) PDFs and
    (preconverted) Office docs are rasterized to per-page PNG image blocks with
    the extracted text attached, instead of being sent as an ``application/pdf``
    data URL the judge can't decode. Native images and text files pass through
    unchanged.

    Audio and video are gated INDEPENDENTLY because judges differ per modality:
    a video file passes through only when *video_capable*, an audio file only
    when *audio_capable* — so a MiniMax-M3 judge (video yes, audio no) keeps video
    but stubs audio. Passed-through media uses the vLLM-standard ``video_url`` /
    ``input_audio`` content types (not the ``image_url`` wrapper used for frontier
    judges, which vLLM won't route to the video/audio tower). An unreadable
    modality is replaced with a one-line marker. Returns a list of 0+ blocks.
    """
    from resources_servers.gdpval.media_conversion import audio_video_block

    file_extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    info = FILE_TYPE_MAP.get(file_extension)
    file_type = info["type"] if info else "DOC"
    full_path = Path(file_dir) / file_name
    budget = attachment_budget or AttachmentBudget(
        raw_limit=MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE,
        encoded_limit=MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE,
    )

    try:
        size_bytes = full_path.stat().st_size
    except OSError:
        return []
    if size_bytes > MAX_FILE_BYTES_FOR_JUDGE:
        size_mb = size_bytes / (1024 * 1024)
        return [{"type": "text", "text": f"[oversize: {file_name} {size_mb:.1f}MB — not included]"}]

    try:
        # PDFs and Office docs (preconverted to a sibling .pdf) → page images + text.
        if file_type == "PDF":
            pdf_path: Path | None = full_path
        elif file_type == "DOC":
            pdf_path = _resolve_office_pdf_path(
                full_path,
                cached_pdf_path=cached_office_pdf,
                provenance_resolved=provenance_resolved,
            )
            if pdf_path is None:
                # No sibling .pdf render exists, so this Office deliverable's content
                # is invisible to an image-only judge (only the filename is shown).
                # Warn so an incomplete-preconvert reference cache surfaces instead of
                # silently scoring empty -- run preconvert on the cache to include it.
                LOGGER.warning(
                    "office file %r has no sibling .pdf render; its content is invisible "
                    "to this judge -- preconvert the deliverables cache to include it",
                    file_name,
                )
        else:
            pdf_path = None

        if pdf_path is not None:
            try:
                pdf_size = pdf_path.stat().st_size
            except OSError:
                return [{"type": "text", "text": f"[attachment unavailable for {file_name}]"}]
            if pdf_size > MAX_FILE_BYTES_FOR_JUDGE:
                return [_attachment_omission(file_name, pdf_size, "per-file judge payload limit")]
            # The PDF is only an input to rasterization; it is not part of the
            # request. Its compressed source size says nothing about whether a
            # rendered PNG page fits the remaining attachment-output budget.
            if not budget.can_fit(1):
                return [_attachment_omission(file_name, pdf_size, "section judge payload budget")]
            blocks = _pdf_path_to_image_text_blocks(
                pdf_path,
                render_dpi=render_dpi,
                max_pages=max_pages,
                include_text=include_text,
                attachment_budget=budget,
                max_text_chars=max_text_chars,
                file_name=file_name,
            )
            if blocks:
                return blocks
            # Office doc with no preconverted PDF (or an unrenderable PDF): fall
            # back to raw text extraction so the judge still sees the content.
            if file_type == "DOC":
                return [{"type": "text", "text": f"[no PDF render available for {file_name}]"}]
            return []

        # Audio/video: forward with the vLLM-standard content type when the judge
        # can read that specific modality; otherwise advertise the file by name.
        # This ``images_and_text`` path is only ever a self-hosted vLLM judge, so
        # AV is emitted as ``video_url`` / ``input_audio`` (not the ``image_url``
        # wrapper frontier judges use), which is what vLLM routes to the media
        # tower. Audio and video are gated separately (MiniMax-M3: video yes,
        # audio no).
        if file_type in ("AUDIO", "VIDEO"):
            can_read = video_capable if file_type == "VIDEO" else audio_capable
            if not can_read:
                return [
                    {"type": "text", "text": f"[{file_type.lower()} file not readable by this judge: {file_name}]"}
                ]
            info = FILE_TYPE_MAP.get(file_extension) or {}
            mime = info.get("mime_type") or "application/octet-stream"
            _size, omitted = _reserve_attachment_path(full_path, file_name, budget)
            if omitted is not None:
                return [omitted]
            data = _load_media(full_path)
            return [audio_video_block(mime, data, ext=file_extension, file_type=file_type, openai_native=True)]

        # Text and native images pass through exactly as in native mode.
        block = get_file_content_block(
            file_dir,
            file_name,
            attachment_budget=budget,
            cached_office_pdf=cached_office_pdf,
            provenance_resolved=provenance_resolved,
            max_text_chars=max_text_chars,
        )
        return [block] if block is not None else []
    except Exception as e:
        raise RuntimeError(f"Error getting file: {file_name} in directory: {file_dir}: {e}") from e


def build_file_section(
    file_dir: str | None,
    clean_up_list: list[Path] | None = None,
    *,
    media_mode: str = "native_pdf",
    render_dpi: int = 144,
    max_pages: int = 50,
    include_text: bool = True,
    audio_capable: bool = False,
    video_capable: bool = False,
) -> list[dict]:
    """Build OpenAI content blocks from all files in a directory.

    Skips run-state files (see :func:`_ignore_files`). Extracts zips into per-call tempdirs
    (the dirs are appended to ``clean_up_list`` for the caller to ``rmtree``).
    Returns a list of content block dicts suitable for OpenAI messages.

    *media_mode* selects how PDFs/Office docs are presented: ``"native_pdf"``
    (default) sends them as ``application/pdf`` data URLs for frontier judges;
    ``"images_and_text"`` rasterizes each page to a PNG block and attaches the
    extracted text, for image-only local VLM judges (see
    :func:`get_file_image_text_blocks`). *render_dpi*, *max_pages*, and
    *include_text* tune the ``images_and_text`` rendering. *audio_capable* /
    *video_capable* keep audio / video files (respectively) as native media
    blocks (vs stubbing them) when the judge reads that modality — they are
    independent so a video-only judge (MiniMax-M3) keeps video but stubs audio.
    """
    if clean_up_list is None:
        clean_up_list = []

    section: list[dict] = []
    no_files = True
    attachment_budget = AttachmentBudget(
        raw_limit=MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE,
        encoded_limit=MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE,
    )
    text_used = 0
    text_omitted = False
    text_marker = "\n[...additional text omitted: section judge text budget exhausted]"
    retained_av_payloads: dict[tuple[int, str], str] = {}

    def _av_identity(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = path.stat().st_size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return size, digest.hexdigest()

    def _append_block(block: dict) -> None:
        nonlocal text_used, text_omitted
        if block.get("type") != "text":
            section.append(block)
            return
        value = str(block.get("text", ""))
        remaining = max(0, MAX_SECTION_TEXT_CHARS_FOR_JUDGE - text_used)
        if not value:
            return
        if remaining <= 0:
            text_omitted = True
            return
        shown = _bounded_text(value, remaining, text_marker)
        if len(shown) < len(value):
            text_omitted = True
        copied = dict(block)
        copied["text"] = shown
        section.append(copied)
        text_used += len(shown)

    def _append_blocks(blocks: list[dict]) -> None:
        for block in blocks:
            _append_block(block)

    extracted_dirs: list[Path] = []
    if file_dir is not None and os.path.exists(file_dir):
        for file_name in os.listdir(file_dir):
            if file_name.lower().endswith(".zip"):
                extract_dir, _ = _maybe_unzip(os.path.join(file_dir, file_name))
                if extract_dir is not None:
                    clean_up_list.append(extract_dir)
                    extracted_dirs.append(extract_dir)

    ignore_files = _ignore_files()
    provenance_by_dir: dict[Path, Any] = {}
    fallback_pdfs_by_dir: dict[Path, dict[Path, Path]] = {}
    fallback_suppressed_by_dir: dict[Path, frozenset[Path]] = {}

    def _provenance(directory: str):
        parent = Path(directory)
        if parent not in provenance_by_dir:
            entries = [entry for entry in parent.iterdir() if entry.is_file() and entry.name not in ignore_files]
            provenance_by_dir[parent] = resolve_pdf_provenance(entries)
            entry_set = set(entries)
            fallback_sources = [
                entry
                for entry in entries
                if entry.suffix.lower() != ".zip" and (entry.suffix.lower().lstrip(".") not in FILE_TYPE_MAP)
            ]
            stem_counts: dict[str, int] = {}
            for source in fallback_sources:
                stem_counts[source.stem] = stem_counts.get(source.stem, 0) + 1
            fallback_pdfs: dict[Path, Path] = {}
            fallback_suppressed: set[Path] = set()
            for source in fallback_sources:
                injective = sidecar_pdf(source)
                plain = source.with_suffix(".pdf")
                if injective in entry_set:
                    fallback_pdfs[source] = injective
                    fallback_suppressed.add(injective)
                elif plain in entry_set and stem_counts[source.stem] == 1:
                    fallback_pdfs[source] = plain
                    fallback_suppressed.add(plain)
                elif plain in entry_set:
                    # Same ambiguity rule as recognized Office sources: never
                    # emit a stale collided render as independent evidence.
                    fallback_suppressed.add(plain)
            fallback_pdfs_by_dir[parent] = fallback_pdfs
            fallback_suppressed_by_dir[parent] = frozenset(fallback_suppressed)
        return provenance_by_dir[parent]

    def _structured_xlsx_text(full_path: Path) -> str | None:
        if full_path.suffix.lower() != ".xlsx":
            return None
        try:
            if full_path.stat().st_size > MAX_FILE_BYTES_FOR_JUDGE:
                return None
            remaining = max(0, MAX_SECTION_TEXT_CHARS_FOR_JUDGE - text_used)
            if remaining <= 0:
                return None
            return extract_xlsx_structured_text(
                full_path,
                max_chars=min(MAX_XLSX_TEXT_CHARS_FOR_JUDGE, remaining),
            )
        except Exception as exc:
            return f"[structured spreadsheet extraction failed: {exc}]"

    def _emit(directory: str, file_name: str) -> None:
        nonlocal no_files
        if file_name in ignore_files:
            return
        full_path = Path(directory) / file_name
        provenance = _provenance(directory)
        parent = Path(directory)
        if full_path in provenance.suppressed_pdfs or full_path in fallback_suppressed_by_dir[parent]:
            return
        _append_block({"type": "text", "text": f"\n{file_name}:\n"})
        info = FILE_TYPE_MAP.get(full_path.suffix.lower().lstrip(".")) or {}
        av_identity: tuple[int, str] | None = None
        if info.get("type") in {"AUDIO", "VIDEO"}:
            av_identity = _av_identity(full_path)
            retained_as = retained_av_payloads.get(av_identity)
            if retained_as is not None:
                _append_block(
                    {
                        "type": "text",
                        "text": f"[byte-identical duplicate attachment; content retained once as "
                        f"{retained_as}; sha256={av_identity[1]}]",
                    }
                )
                no_files = False
                return
        cached_office_pdf = provenance.office_pdfs.get(full_path) or fallback_pdfs_by_dir[parent].get(full_path)
        remaining_text = max(0, MAX_SECTION_TEXT_CHARS_FOR_JUDGE - text_used)
        if media_mode == "images_and_text":
            blocks = get_file_image_text_blocks(
                directory,
                file_name,
                render_dpi=render_dpi,
                max_pages=max_pages,
                include_text=include_text,
                audio_capable=audio_capable,
                video_capable=video_capable,
                attachment_budget=attachment_budget,
                cached_office_pdf=cached_office_pdf,
                provenance_resolved=True,
                max_text_chars=min(MAX_TEXT_FILE_CHARS_FOR_JUDGE, remaining_text),
            )
            if blocks:
                _append_blocks(blocks)
                if av_identity is not None and any(_attachment_payload(block)[0] for block in blocks):
                    retained_av_payloads[av_identity] = file_name
                no_files = False
            sheet_text = _structured_xlsx_text(full_path)
            if sheet_text:
                _append_block(
                    {
                        "type": "text",
                        "text": f"\n{file_name} (structured spreadsheet cells):\n{sheet_text}",
                    }
                )
                no_files = False
            return
        block = get_file_content_block(
            directory,
            file_name,
            attachment_budget=attachment_budget,
            cached_office_pdf=cached_office_pdf,
            provenance_resolved=True,
            max_text_chars=min(MAX_TEXT_FILE_CHARS_FOR_JUDGE, remaining_text),
        )
        if block is not None:
            _append_block(block)
            if av_identity is not None and _attachment_payload(block)[0]:
                retained_av_payloads[av_identity] = file_name
            no_files = False
        sheet_text = _structured_xlsx_text(full_path)
        if sheet_text:
            _append_block({"type": "text", "text": f"\n{file_name} (structured spreadsheet cells):\n{sheet_text}"})
            no_files = False

    if file_dir is not None and os.path.exists(file_dir):
        for file_name in sorted(os.listdir(file_dir)):
            full_path = os.path.join(file_dir, file_name)
            if os.path.isdir(full_path) or file_name.lower().endswith(".zip"):
                continue
            _emit(file_dir, file_name)

    for extract_dir in extracted_dirs:
        for member in sorted(extract_dir.rglob("*")):
            if not member.is_file():
                continue
            _emit(str(member.parent), member.name)

    if no_files:
        _append_block({"type": "text", "text": "None"})

    if text_omitted and text_marker not in "".join(
        str(block.get("text", "")) for block in section if block.get("type") == "text"
    ):
        for block in reversed(section):
            if block.get("type") == "text":
                previous = str(block.get("text", ""))
                block["text"] = _bounded_text(previous + "x", len(previous), text_marker)
                break

    return section


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------


def _attachment_payload(block: dict) -> tuple[str, int]:
    """Return the backing string and base64 start offset without copying it."""

    block_type = block.get("type")
    if block_type in {"image_url", "video_url"}:
        value = block.get(block_type, {})
        url = value.get("url", "") if isinstance(value, dict) else ""
        if isinstance(url, str) and url.startswith("data:"):
            comma = url.find(",")
            if comma >= 0:
                return url, comma + 1
    elif block_type == "input_audio":
        value = block.get("input_audio", {})
        data = value.get("data", "") if isinstance(value, dict) else ""
        if isinstance(data, str):
            return data, 0
    return "", 0


def _attachment_cost(block: dict) -> tuple[int, int]:
    """Return estimated raw bytes and encoded characters for a data attachment."""

    payload, start = _attachment_payload(block)
    encoded_chars = len(payload) - start
    if encoded_chars <= 0:
        return 0, 0

    padding = 0
    if payload[-1] == "=":
        padding = 1
        if encoded_chars > 1 and payload[-2] == "=":
            padding = 2
    raw_bytes = (encoded_chars // 4) * 3 - padding
    return max(0, raw_bytes), encoded_chars


def _attachment_fingerprint(block: dict) -> str:
    """Hash the complete payload in bounded chunks for A/B-stable tie breaks."""

    payload, start = _attachment_payload(block)
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    for offset in range(start, len(payload), chunk_size):
        digest.update(payload[offset : offset + chunk_size].encode("ascii", errors="ignore"))
    return digest.hexdigest()


def _block_serialized_upper_bound(block: dict) -> int:
    """Conservative JSON byte bound without copying large base64 strings."""

    _raw_bytes, encoded_chars = _attachment_cost(block)
    if encoded_chars:
        payload, start = _attachment_payload(block)
        # Base64 is JSON-safe ASCII. Account exactly for the data URL prefix and
        # generously for keys, quotes, optional audio format, and punctuation.
        return encoded_chars + start + 512
    if block.get("type") == "text":
        # Six bytes per character covers JSON escaping (including ensure_ascii)
        # for BMP text; non-BMP can require two \uXXXX escapes, hence twelve.
        value = str(block.get("text", ""))
        non_bmp = sum(ord(character) > 0xFFFF for character in value)
        return len(value) * 6 + non_bmp * 6 + 256
    return 4096


def _content_serialized_upper_bound(content: list[dict]) -> int:
    """Conservative serialized request size including outer JSON framing."""

    return 4096 + sum(_block_serialized_upper_bound(block) for block in content)


def _bound_section_text(section: list[dict], cap: int) -> list[dict]:
    result: list[dict] = []
    remaining = max(0, cap)
    marker = "\n[...additional text omitted: section judge text budget exhausted]"
    omitted = False
    last_text: int | None = None
    for block in section:
        if block.get("type") != "text":
            result.append(block)
            continue
        value = str(block.get("text", ""))
        if remaining <= 0:
            omitted = omitted or bool(value)
            continue
        shown = _bounded_text(value, remaining, marker)
        omitted = omitted or len(shown) < len(value)
        copied = dict(block)
        copied["text"] = shown
        result.append(copied)
        last_text = len(result) - 1
        remaining -= len(shown)
    if omitted and last_text is not None and marker not in str(result[last_text].get("text", "")):
        previous = str(result[last_text].get("text", ""))
        result[last_text]["text"] = _bounded_text(previous + "x", len(previous), marker)
    return result


def _enforce_attachment_budget(
    sections: list[list[dict]],
    *,
    raw_limit: int,
    encoded_limit: int,
    serialized_limit: int | None = None,
) -> list[list[dict]]:
    """Keep the request's aggregate binary payload below both hard ceilings.

    Attachments are selected smallest-first, maximizing the number of visible
    artifacts and preventing one huge reference from starving both submissions.
    Complete payload hashes break equal-size ties independently of A/B position,
    so swapping the trial does not change which underlying artifact is kept.
    Omitted blocks are replaced by explicit text evidence.
    """

    candidates: list[dict[str, Any]] = []
    costs: dict[tuple[int, int], tuple[int, int]] = {}
    for section_index, section in enumerate(sections):
        for block_index, block in enumerate(section):
            raw_bytes, encoded_chars = _attachment_cost(block)
            if encoded_chars == 0:
                continue
            key = (section_index, block_index)
            costs[key] = (raw_bytes, encoded_chars)
            candidates.append(
                {
                    "raw": raw_bytes,
                    "encoded": encoded_chars,
                    "serialized": _block_serialized_upper_bound(block),
                    "section": section_index,
                    "block": block_index,
                }
            )

    serialized_ceiling = encoded_limit + 1024 * len(candidates) if serialized_limit is None else serialized_limit
    if (
        sum(int(candidate["raw"]) for candidate in candidates) <= raw_limit
        and sum(int(candidate["encoded"]) for candidate in candidates) <= encoded_limit
        and sum(int(candidate["serialized"]) for candidate in candidates) <= serialized_ceiling
    ):
        # Normal requests are already bounded per section and fit the aggregate
        # ceilings. Avoid hashing hundreds of MiB of base64 on every repeated
        # judge trial when no selection decision is required.
        return [list(section) for section in sections]

    for candidate in candidates:
        section_index = int(candidate["section"])
        block_index = int(candidate["block"])
        candidate["fingerprint"] = _attachment_fingerprint(sections[section_index][block_index])

    def _sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
        return (
            int(candidate["serialized"]),
            int(candidate["encoded"]),
            int(candidate["raw"]),
            str(candidate["fingerprint"]),
        )

    selected: set[tuple[int, int]] = set()
    used_raw = 0
    used_encoded = 0
    used_serialized = 0

    def _select(candidate: dict[str, Any], local: AttachmentBudget | None = None, local_serialized: int = 0) -> bool:
        nonlocal used_raw, used_encoded, used_serialized
        key = (int(candidate["section"]), int(candidate["block"]))
        if key in selected:
            return True
        raw_bytes = int(candidate["raw"])
        encoded_chars = int(candidate["encoded"])
        serialized_bytes = int(candidate["serialized"])
        if (
            used_raw + raw_bytes > raw_limit
            or used_encoded + encoded_chars > encoded_limit
            or used_serialized + serialized_bytes > serialized_ceiling
        ):
            return False
        if local is not None:
            if local_serialized + serialized_bytes > serialized_ceiling // max(1, len(sections)):
                return False
            if not local.reserve(raw_bytes, encoded_chars):
                return False
        selected.add(key)
        used_raw += raw_bytes
        used_encoded += encoded_chars
        used_serialized += serialized_bytes
        return True

    # Reserve equal shares for A and B before references. Both passes use the
    # same limits and payload-derived ordering, so swapping A/B is invariant.
    section_raw = raw_limit // max(1, len(sections))
    section_encoded = encoded_limit // max(1, len(sections))
    local_budgets = [AttachmentBudget(section_raw, section_encoded) for _ in sections]
    local_serialized_used = [0 for _ in sections]
    section_order = list(range(1, len(sections))) + ([0] if sections else [])
    for section_index in section_order:
        for candidate in sorted(
            (item for item in candidates if item["section"] == section_index),
            key=_sort_key,
        ):
            before = used_serialized
            if _select(candidate, local_budgets[section_index], local_serialized_used[section_index]):
                local_serialized_used[section_index] += used_serialized - before

    # If an attachment was larger than its equal share, give each nonempty
    # submission one deterministic chance before references use spare capacity.
    submission_seeds: list[dict[str, Any]] = []
    for section_index in range(1, len(sections)):
        if any(key[0] == section_index for key in selected):
            continue
        choices = [item for item in candidates if item["section"] == section_index]
        if choices:
            submission_seeds.append(min(choices, key=_sort_key))
    for candidate in sorted(submission_seeds, key=_sort_key):
        _select(candidate)

    # Redistribute every unused byte globally, with complete-payload hashes as
    # the final tie-break rather than section position.
    for candidate in sorted(candidates, key=_sort_key):
        _select(candidate)

    bounded: list[list[dict]] = []
    for section_index, section in enumerate(sections):
        output: list[dict] = []
        last_header = "attachment"
        omitted_count = 0
        omitted_raw = 0
        omitted_encoded = 0
        omitted_names: list[str] = []
        for block_index, block in enumerate(section):
            if block.get("type") == "text":
                candidate_header = str(block.get("text", "")).strip().splitlines()
                if candidate_header:
                    last_header = candidate_header[0].rstrip(":")
            raw_bytes, encoded_chars = costs.get((section_index, block_index), (0, 0))
            if encoded_chars and (section_index, block_index) not in selected:
                omitted_count += 1
                omitted_raw += raw_bytes
                omitted_encoded += encoded_chars
                if len(omitted_names) < 10:
                    omitted_names.append(last_header)
            else:
                output.append(block)
        if omitted_count:
            names = ", ".join(omitted_names)
            if omitted_count > len(omitted_names):
                names += f", and {omitted_count - len(omitted_names)} more"
            output.append(
                {
                    "type": "text",
                    "text": (
                        f"[attachment omitted: {omitted_count} file(s) ({names}), "
                        f"{_human_attachment_size(omitted_raw)} raw, "
                        f"{_human_attachment_size(omitted_encoded)} base64; aggregate judge payload budget]"
                    ),
                }
            )
        bounded.append(output)
    return bounded


def _human_attachment_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def construct_judge_messages(
    task_prompt: str,
    refs: list[dict],
    submission_a: list[dict],
    submission_b: list[dict],
) -> list[dict]:
    """Assemble OpenAI messages for the judge: prompt + task + refs + submissions."""
    verdict_reminder_block = {"type": "text", "text": FINAL_VERDICT_REMINDER}
    # Text can require up to twelve JSON bytes per Unicode code point. Give the
    # task and three sections equal worst-case shares after reserving framing,
    # including the trailing verdict reminder, so the final serialized bound
    # cannot exceed the request cap once every share is fully used.
    framing_reserve = 16 * 1024 + _block_serialized_upper_bound(verdict_reminder_block)
    available_text_chars = max(0, MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE - framing_reserve) // 12
    fair_text_cap = available_text_chars // 4
    section_text_cap = min(MAX_SECTION_TEXT_CHARS_FOR_JUDGE, fair_text_cap)
    refs = _bound_section_text(refs, section_text_cap)
    submission_a = _bound_section_text(submission_a, section_text_cap)
    submission_b = _bound_section_text(submission_b, section_text_cap)
    bounded_task = _bounded_text(task_prompt, min(MAX_TASK_PROMPT_CHARS_FOR_JUDGE, fair_text_cap))
    prompt_block = {"type": "text", "text": JUDGE_PROMPT + TASK_TEMPLATE.format(task=bounded_task)}
    framing = [
        prompt_block,
        {"type": "text", "text": REFERENCES_OPEN},
        {"type": "text", "text": REFERENCES_CLOSE},
        {"type": "text", "text": SUBMISSION_A_OPEN},
        {"type": "text", "text": SUBMISSION_A_CLOSE},
        {"type": "text", "text": SUBMISSION_B_OPEN},
        {"type": "text", "text": SUBMISSION_B_CLOSE},
        verdict_reminder_block,
    ]
    sections = [refs, submission_a, submission_b]
    fixed_blocks = framing + [block for section in sections for block in section if _attachment_cost(block)[1] == 0]
    # Reserve one small aggregate omission notice per section. Everything else
    # is bounded conservatively at the JSON-string level, including escaped text.
    fixed_serialized = (
        4096 + 4096 * len(sections) + sum(_block_serialized_upper_bound(block) for block in fixed_blocks)
    )
    attachment_serialized_limit = max(
        0,
        MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE - fixed_serialized,
    )
    refs, submission_a, submission_b = _enforce_attachment_budget(
        sections,
        raw_limit=MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE,
        encoded_limit=MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE,
        serialized_limit=attachment_serialized_limit,
    )
    content: list[dict] = []
    content.append(prompt_block)
    content.append({"type": "text", "text": REFERENCES_OPEN})
    content.extend(refs)
    content.append({"type": "text", "text": REFERENCES_CLOSE})
    content.append({"type": "text", "text": SUBMISSION_A_OPEN})
    content.extend(submission_a)
    content.append({"type": "text", "text": SUBMISSION_A_CLOSE})
    content.append({"type": "text", "text": SUBMISSION_B_OPEN})
    content.extend(submission_b)
    content.append({"type": "text", "text": SUBMISSION_B_CLOSE})
    content.append(verdict_reminder_block)

    serialized_bound = _content_serialized_upper_bound(content)
    if serialized_bound > MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE:
        raise ValueError(
            "GDPVal judge request size budget exhausted before dispatch: "
            f"{serialized_bound} > {MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE} bytes"
        )

    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Judge API call
# ---------------------------------------------------------------------------


def _is_retryable(error: Exception) -> bool:
    # Timeouts on multimodal payloads are deterministic — the payload is too
    # large for the judge endpoint to digest in time, and retrying just burns
    # another full timeout window per attempt. Fail the trial fast instead.
    if isinstance(error, APITimeoutError):
        return False
    if is_permanent_judge_error(error):
        return False
    error_text = str(error).lower()
    return any(marker in error_text for marker in RETRYABLE_ERROR_MARKERS)


def send_judge_request(
    client: Any,
    model: str,
    messages: list[dict],
    max_output_tokens: int = 65535,
    create_overrides: Optional[dict] = None,
) -> str:
    """Send a judge request with exponential-backoff retry.  Returns response text.

    *create_overrides* (a panel member's reasoning/generation knobs) is merged
    over the default create kwargs; a ``None`` value removes the matching
    default (e.g. to drop ``temperature`` for a reasoning model that rejects it).
    """
    backoff = REQUEST_INITIAL_BACKOFF_SECONDS
    create_kwargs = merge_create_kwargs(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": 1.0,
        },
        create_overrides,
    )

    for attempt in range(1, REQUEST_MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(**create_kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as error:
            retryable = _is_retryable(error)
            is_last = attempt == REQUEST_MAX_ATTEMPTS
            if not retryable or is_last:
                raise
            print(
                f"  Judge request attempt {attempt}/{REQUEST_MAX_ATTEMPTS} failed "
                f"(retryable={retryable}), retrying in {backoff:.1f}s...",
                flush=True,
            )
            time.sleep(backoff)
            backoff = min(backoff * REQUEST_BACKOFF_MULTIPLIER, REQUEST_MAX_BACKOFF_SECONDS)

    raise RuntimeError("Unreachable retry loop exit")


# ---------------------------------------------------------------------------
# Judgement parsing and tallying
# ---------------------------------------------------------------------------


_BOXED_RE = re.compile(r"BOXED\[(A|B|TIE)\]")

_BOXED_TO_RESPONSE = {"A": A_WIN_RESPONSE, "B": B_WIN_RESPONSE, "TIE": TIE_RESPONSE}


def parse_judgement(response_text: str) -> Optional[str]:
    """Extract the judge's verdict (``BOXED[A|B|TIE]``) from its response.

    Uses the **last** boxed token so a verbose/reasoning judge that writes e.g.
    ``"not BOXED[A]; the answer is BOXED[B]"`` is scored on its conclusion rather
    than the first token mentioned (the old first-substring match scored A here).
    When no boxed verdict is present the response is mis-formatted; return
    ``None`` so the caller can exclude it from the ELO calculation. Treating an
    invalid response as a tie would award the eval model half a win.
    """
    matches = _BOXED_RE.findall(response_text or "")
    if not matches:
        LOGGER.warning(
            "judge response has no BOXED[A|B|TIE] verdict; dropping vote. head=%r",
            (response_text or "")[:200],
        )
        return None
    return _BOXED_TO_RESPONSE[matches[-1]]


def tally_result(
    judgement: str,
    swapped: bool,
    win_count_a: int,
    win_count_b: int,
    tie_count: int,
) -> tuple[int, int, int]:
    """Update win/loss/tie counters, accounting for position swap."""
    if swapped:
        if B_WIN_RESPONSE in judgement:
            win_count_a += 1
        elif A_WIN_RESPONSE in judgement:
            win_count_b += 1
        elif TIE_RESPONSE in judgement:
            tie_count += 1
    else:
        if A_WIN_RESPONSE in judgement:
            win_count_a += 1
        elif B_WIN_RESPONSE in judgement:
            win_count_b += 1
        elif TIE_RESPONSE in judgement:
            tie_count += 1
    return win_count_a, win_count_b, tie_count


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


@dataclass
class Judge:
    """A panel member bound to a live (sync) OpenAI client for the trial loop.

    Built by the resources server from a
    :class:`resources_servers.gdpval.judge_panel.ResolvedJudge` (with one OpenAI
    client per distinct upstream, so members that share a proxy reuse a client).
    ``run_trials`` samples one of these per trial.
    """

    name: str
    client: Any
    model: str
    create_overrides: Optional[dict] = None
    weight: float = 1.0
    # Per-modality media capability, tracked separately: a MiniMax-M3 judge reads
    # video but not audio.
    handles_audio: bool = False
    handles_video: bool = False
    media_mode: str = "native_pdf"
    max_native_pdf_pages: Optional[int] = None
    max_native_pdf_documents: Optional[int] = None
    max_native_pdf_bytes: Optional[int] = None
    max_native_pdf_bytes_per_document: Optional[int] = None
    raster_dpi_tiers: tuple[int, ...] = ()
    max_serialized_request_bytes: Optional[int] = None


def preview_trial_judges(judges: list[Judge], num_trials: int, rng: random.Random) -> list[Judge]:
    """Preview the seeded schedule without consuming *rng*."""
    if not judges:
        raise ValueError("preview_trial_judges requires a non-empty judge panel")
    clone = random.Random()
    clone.setstate(rng.getstate())
    return [sample_judge(judges, clone) for _ in range(num_trials)]


def filter_media_eligible_judges(
    judges: list[Judge],
    *,
    native_stats: dict[str, int],
    estimated_images: int,
    image_cap: int,
    overflow_plan: Optional[dict] = None,
) -> tuple[list[Judge], list[dict]]:
    """Capability-gate the complete panel before seeded trial sampling."""
    eligible: list[Judge] = []
    exclusions: list[dict] = []
    for judge in judges:
        if judge.media_mode == "native_pdf_overflow_images":
            if overflow_plan is None or not overflow_plan.get("eligible", False):
                exclusions.append(
                    {
                        "mode": judge.media_mode,
                        "judges": [judge.name],
                        "overflow_plan": overflow_plan,
                        "reason": "native_pdf_overflow_unavailable",
                    }
                )
                continue
            eligible.append(judge)
            continue
        if judge.media_mode == "images_and_text" and estimated_images > image_cap:
            exclusions.append(
                {
                    "mode": judge.media_mode,
                    "judges": [judge.name],
                    "estimated_image_count": estimated_images,
                    "pdf_stats": dict(native_stats),
                    "cap": image_cap,
                    "reason": "request_image_cap_preflight",
                }
            )
            continue
        if judge.media_mode == "native_pdf":
            over = (
                (judge.max_native_pdf_pages is not None and native_stats["pages"] > judge.max_native_pdf_pages)
                or (
                    judge.max_native_pdf_documents is not None
                    and native_stats["documents"] > judge.max_native_pdf_documents
                )
                or (judge.max_native_pdf_bytes is not None and native_stats["bytes"] > judge.max_native_pdf_bytes)
            )
            if over:
                exclusions.append(
                    {
                        "mode": judge.media_mode,
                        "judges": [judge.name],
                        "pdf_stats": dict(native_stats),
                        "reason": "native_pdf_cap",
                    }
                )
                continue
        eligible.append(judge)
    return eligible, exclusions


def plan_native_pdf_overflow(
    sections: dict[str, list[dict]],
    *,
    native_page_cap: int,
    native_pdf_bytes_per_document: int,
    image_cap: int,
) -> dict:
    """Rasterize oversize PDFs and exactly enough pages for the page cap.

    Any PDF above the provider's per-file limit is fully rasterized. For an
    additional page-count overflow, whole small PDFs are rasterized first and
    only the required prefix of the final document is rasterized. Every source
    page remains represented while avoiding unnecessary image expansion.
    """
    from resources_servers.gdpval.media_conversion import pdf_page_count

    prefix = "data:application/pdf;base64,"
    documents: list[dict] = []
    existing_images = 0
    for section_name, blocks in sections.items():
        for block_index, block in enumerate(blocks):
            if block.get("type") != "image_url":
                continue
            url = str((block.get("image_url") or {}).get("url", ""))
            if not url.startswith(prefix):
                if url.startswith("data:image/"):
                    existing_images += 1
                continue
            payload = base64.b64decode(url[len(prefix) :], validate=True)
            pages = pdf_page_count(payload)
            if pages <= 0:
                raise ValueError(f"native PDF has no pages: {section_name}/{block_index}")
            documents.append(
                {
                    "section": section_name,
                    "block_index": block_index,
                    "pages": pages,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    total_pages = sum(item["pages"] for item in documents)
    excess = max(0, total_pages - native_page_cap)
    forced = {index for index, item in enumerate(documents) if item["bytes"] > native_pdf_bytes_per_document}
    selected: list[dict] = [
        {
            **documents[index],
            "raster_page_start": 0,
            "raster_page_count": int(documents[index]["pages"]),
            "native_page_count": 0,
            "reason": "native_pdf_bytes_per_document",
        }
        for index in sorted(forced)
    ]
    forced_pages = sum(int(documents[index]["pages"]) for index in forced)
    remaining = max(0, excess - forced_pages)
    for index in sorted(range(len(documents)), key=lambda value: (documents[value]["pages"], value)):
        if index in forced:
            continue
        if remaining <= 0:
            break
        document = documents[index]
        raster_page_count = min(int(document["pages"]), remaining)
        selected.append(
            {
                **document,
                "raster_page_start": 0,
                "raster_page_count": raster_page_count,
                "native_page_count": int(document["pages"]) - raster_page_count,
                "reason": "native_pdf_page_cap",
            }
        )
        remaining -= raster_page_count
    raster_pages = sum(int(item["raster_page_count"]) for item in selected)
    native_pages = total_pages - raster_pages
    total_images_after = existing_images + raster_pages
    return {
        "eligible": (remaining == 0 and native_pages <= native_page_cap and total_images_after <= image_cap),
        "total_pdf_pages": total_pages,
        "native_page_cap": native_page_cap,
        "native_pdf_bytes_per_document": native_pdf_bytes_per_document,
        "native_pages_after": native_pages,
        "raster_pages": raster_pages,
        "image_cap": image_cap,
        "existing_images": existing_images,
        "total_images_after": total_images_after,
        "selected": selected,
    }


def _split_pdf_prefix(pdf_bytes: bytes, raster_page_count: int) -> tuple[bytes, bytes]:
    """Return PDF byte streams for a non-empty prefix and suffix."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for native-PDF overflow splitting") from exc

    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    prefix = fitz.open()
    suffix = fitz.open()
    try:
        pages = int(source.page_count)
        if not 0 < raster_page_count < pages:
            raise ValueError(f"invalid PDF prefix split: {raster_page_count}/{pages}")
        prefix.insert_pdf(source, from_page=0, to_page=raster_page_count - 1, links=True, annots=True)
        suffix.insert_pdf(source, from_page=raster_page_count, to_page=pages - 1, links=True, annots=True)
        return prefix.tobytes(garbage=0, deflate=False), suffix.tobytes(garbage=0, deflate=False)
    finally:
        suffix.close()
        prefix.close()
        source.close()


def apply_native_pdf_overflow(
    sections: dict[str, list[dict]],
    plan: dict,
    *,
    render_dpi: int,
    max_pages: int,
    include_text: bool,
) -> dict[str, list[dict]]:
    """Apply a preflighted overflow plan while retaining every PDF page."""
    from resources_servers.gdpval.media_conversion import pdf_bytes_to_blocks, pdf_page_count

    selected = {(x["section"], int(x["block_index"])): x for x in plan.get("selected", [])}
    if len(selected) != len(plan.get("selected", [])):
        raise ValueError("overflow plan selects a PDF block more than once")
    prefix = "data:application/pdf;base64,"
    output: dict[str, list[dict]] = {}
    for section_name, blocks in sections.items():
        converted: list[dict] = []
        for block_index, block in enumerate(blocks):
            item = selected.get((section_name, block_index))
            if item is None:
                converted.append(block)
                continue
            url = str((block.get("image_url") or {}).get("url", ""))
            if not url.startswith(prefix):
                raise ValueError(f"overflow plan block drift: {section_name}/{block_index}")
            payload = base64.b64decode(url[len(prefix) :], validate=True)
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"overflow plan hash drift: {section_name}/{block_index}")
            pages = pdf_page_count(payload)
            if pages != int(item["pages"]):
                raise ValueError(f"overflow plan page-count drift: {section_name}/{block_index}")
            raster_page_start = int(item.get("raster_page_start", 0))
            raster_page_count = int(item.get("raster_page_count", pages))
            native_page_count = int(item.get("native_page_count", pages - raster_page_count))
            if raster_page_start != 0 or raster_page_count <= 0 or raster_page_count > pages:
                raise ValueError(f"invalid overflow page range: {section_name}/{block_index}")
            if native_page_count != pages - raster_page_count:
                raise ValueError(f"overflow plan page partition drift: {section_name}/{block_index}")
            if native_page_count:
                raster_payload, native_payload = _split_pdf_prefix(payload, raster_page_count)
                if pdf_page_count(raster_payload) != raster_page_count:
                    raise ValueError(f"overflow prefix page-count drift: {section_name}/{block_index}")
                if pdf_page_count(native_payload) != native_page_count:
                    raise ValueError(f"overflow suffix page-count drift: {section_name}/{block_index}")
            else:
                raster_payload = payload
                native_payload = b""
            rendered = pdf_bytes_to_blocks(
                raster_payload,
                dpi=render_dpi,
                max_pages=max_pages,
                include_text=include_text,
            )
            image_count = sum(1 for x in rendered if x.get("type") == "image_url")
            if image_count != raster_page_count or any(
                str(x.get("text", "")).startswith("[truncated: rendered") for x in rendered
            ):
                raise ValueError(
                    f"overflow render mismatch {section_name}/{block_index}: "
                    f"images={image_count} pages={raster_page_count}"
                )
            converted.extend(rendered)
            if native_payload:
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": prefix + base64.b64encode(native_payload).decode("ascii")},
                    }
                )
        output[section_name] = converted
    return output


_LOSSY_TRANSPORT_PREFIXES = (
    "[attachment omitted",
    "[oversize:",
    "[truncated: rendered",
)


def preflight_judge_transport(
    judge: Judge,
    task_prompt: str,
    sections: dict[str, list[dict]],
) -> dict[str, Any]:
    """Project one provider request and reject lossy or oversized transport."""
    messages = construct_judge_messages(
        task_prompt=task_prompt,
        refs=sections["refs"],
        submission_a=sections["submission_a"],
        submission_b=sections["submission_b"],
    )
    markers: list[str] = []
    for message in messages:
        for block in message.get("content") or []:
            text = str(block.get("text", ""))
            if text.startswith(_LOSSY_TRANSPORT_PREFIXES):
                markers.append(text.splitlines()[0][:240])
    create_kwargs = merge_create_kwargs(
        {
            "model": judge.model,
            "messages": messages,
            "max_tokens": 65535,
            "temperature": 1.0,
        },
        judge.create_overrides,
    )
    serialized_bytes = len(json.dumps(create_kwargs, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    cap = judge.max_serialized_request_bytes
    reasons: list[str] = []
    if markers:
        reasons.append("lossy_attachment_omission")
    if cap is not None and serialized_bytes >= cap:
        reasons.append("provider_wire_cap")
    return {
        "eligible": not reasons,
        "judge": judge.name,
        "media_mode": judge.media_mode,
        "serialized_request_bytes": serialized_bytes,
        "max_serialized_request_bytes": cap,
        "loss_markers": markers,
        "reasons": reasons,
    }


def run_trials(
    judges: list[Judge],
    task_prompt: str,
    refs: list[dict],
    submission_a: list[dict],
    submission_b: list[dict],
    sections_by_judge: Optional[dict[str, dict[str, list[dict]]]] = None,
    num_trials: int = 4,
    max_output_tokens: int = 65535,
    return_raw_responses: bool = False,
    rng: Optional[random.Random] = None,
) -> dict:
    """Run ``num_trials`` judge calls, alternating swapped/unswapped positions.

    For each trial one member of *judges* is sampled (see
    ``judge_panel.sample_judge``) — the "sample between the judges for each
    comparison" panel behavior. With a single-member panel this reduces to the
    historical single-judge loop. Pass *rng* (a seeded ``random.Random``) for
    reproducible judge selection.

    Invalid responses without a boxed verdict are excluded rather than scored
    as ties. If every response is invalid, the matchup fails so its caller can
    retry or drop it explicitly.

    Returns a dict with ``winner``, ``win_count_a``, ``win_count_b``,
    ``tie_count``, ``task_count`` (valid votes only), ``invalid_count``,
    ``per_judge`` (per-member a/b/tie/valid/invalid counts keyed by judge name),
    and ``trial_judges`` (the judge name that graded each attempted trial,
    ordered by trial index — always present so the grader of every match is
    documented).

    When ``return_raw_responses`` is True, the dict also carries
    ``raw_responses`` (per-trial judge completion strings, same ordering as
    ``trial_judges`` — trial ``i`` was swapped iff ``i % 2 != 0``).
    """
    if not judges:
        raise ValueError("run_trials requires a non-empty judge panel")
    rng = rng or random.Random()

    win_count_a = 0
    win_count_b = 0
    tie_count = 0
    invalid_count = 0
    raw_responses: list[str] = []
    trial_judges: list[str] = []
    per_judge: dict[str, dict] = {}

    for i in range(num_trials):
        judge = sample_judge(judges, rng)
        trial_judges.append(judge.name)
        if sections_by_judge is not None:
            if judge.name not in sections_by_judge:
                raise ValueError(f"missing sections for judge {judge.name!r}")
            judge_sections = sections_by_judge[judge.name]
            refs = judge_sections["refs"]
            submission_a = judge_sections["submission_a"]
            submission_b = judge_sections["submission_b"]
        swapped = i % 2 != 0
        current_a = submission_b if swapped else submission_a
        current_b = submission_a if swapped else submission_b

        messages = construct_judge_messages(
            task_prompt=task_prompt,
            refs=refs,
            submission_a=current_a,
            submission_b=current_b,
        )
        response_text = send_judge_request(
            judge.client, judge.model, messages, max_output_tokens, judge.create_overrides
        )
        if return_raw_responses:
            raw_responses.append(response_text)
        judgement = parse_judgement(response_text)

        # Per-judge tally (same A=submission_a / B=submission_b convention as the
        # global counts) so the panel's per-member balance is auditable.
        jc = per_judge.setdefault(
            judge.name,
            {"win_count_a": 0, "win_count_b": 0, "tie_count": 0, "trials": 0, "invalid_count": 0},
        )
        if judgement is None:
            invalid_count += 1
            jc["invalid_count"] += 1
            continue

        win_count_a, win_count_b, tie_count = tally_result(judgement, swapped, win_count_a, win_count_b, tie_count)
        jc["win_count_a"], jc["win_count_b"], jc["tie_count"] = tally_result(
            judgement, swapped, jc["win_count_a"], jc["win_count_b"], jc["tie_count"]
        )
        jc["trials"] += 1

    valid_count = win_count_a + win_count_b + tie_count
    if valid_count == 0:
        raise ValueError(f"All {num_trials} pairwise judge responses were invalid")

    if win_count_a > win_count_b:
        winner = A_WIN_RESPONSE
    elif win_count_b > win_count_a:
        winner = B_WIN_RESPONSE
    else:
        winner = TIE_RESPONSE

    result: dict = {
        "winner": winner,
        "win_count_a": win_count_a,
        "win_count_b": win_count_b,
        "tie_count": tie_count,
        "task_count": valid_count,
        "invalid_count": invalid_count,
        "per_judge": per_judge,
        # Always recorded (just judge names, ordered by trial) so every match's
        # per-trial grader is documented even when raw responses aren't kept.
        "trial_judges": trial_judges,
    }
    if return_raw_responses:
        result["raw_responses"] = raw_responses
    return result


# ---------------------------------------------------------------------------
# ELO calculation
# ---------------------------------------------------------------------------


def calculate_elo(win_rate: float, ref_elo: float) -> tuple[float, float]:
    """Compute ELO from win rate against a reference model.

    Returns ``(elo, normalized_elo)`` where normalized is ``(elo - 500) / 2000``.
    """
    if win_rate <= 0.0 or win_rate >= 1.0:
        win_rate = max(0.001, min(0.999, win_rate))
    elo = ref_elo - 400.0 * (math.log10(1.0 - win_rate) - math.log10(win_rate))
    normalized_elo = (elo - 500.0) / 2000.0
    return elo, normalized_elo


def calculate_mle_elo(
    battles: list[tuple[float, float, float, float]],
    scale: float = 400.0,
    base: float = 10.0,
) -> tuple[float, float] | None:
    """Anchored Bradley-Terry MLE ELO for one eval model vs N fixed references.

    This is the multi-reference generalization of ``calculate_elo``. It applies
    the traditional ELO rating system (logistic / Bradley-Terry) to the pooled
    pairwise comparisons, estimating the eval model's rating globally rather
    than inverting a single win rate against a single anchor.

    ``battles`` is a list of ``(reference_elo, wins, losses, ties)`` where the
    counts are the eval model's win / loss / tie vote totals against that
    reference model (ties counted as half a win). The reference ratings are
    held **fixed** at their known ELOs (e.g. published Arena/AA numbers); the
    eval model's rating ``R`` is the single free parameter, found by maximizing
    the Bradley-Terry log-likelihood

        L(R) = sum_i [ s_i * log(p_i) + (n_i - s_i) * log(1 - p_i) ]

    with ``p_i = 1 / (1 + base**((reference_elo_i - R) / scale))``, ``n_i`` the
    number of games vs reference ``i`` and ``s_i = wins_i + 0.5 * ties_i``.

    For a single reference this reduces exactly to ``calculate_elo``. Returns
    ``(elo, normalized_elo)`` with ``normalized_elo = (elo - 500) / 2000``, or
    ``None`` when there are no games to fit.
    """
    data: list[tuple[float, float, float]] = []
    for ref_elo, wins, losses, ties in battles:
        n = float(wins) + float(losses) + float(ties)
        if n <= 0:
            continue
        s = float(wins) + 0.5 * float(ties)
        data.append((float(ref_elo), s, n))

    if not data:
        return None

    total_s = sum(s for _, s, _ in data)
    total_n = sum(n for _, _, n in data)
    eps = 1e-3

    overall_win_rate = total_s / total_n
    if overall_win_rate <= eps or overall_win_rate >= 1.0 - eps:
        # Degenerate: the eval model won (or lost) every battle, so the MLE
        # rating diverges to ±inf. Clamp exactly like ``calculate_elo`` does,
        # anchored to the game-weighted mean reference ELO.
        clamped = min(max(overall_win_rate, eps), 1.0 - eps)
        mean_ref = sum(ref_elo * n for ref_elo, _, n in data) / total_n
        elo = mean_ref - scale * (math.log10(1.0 - clamped) - math.log10(clamped))
        return elo, (elo - 500.0) / 2000.0

    def gradient(rating: float) -> float:
        # dL/dR up to the positive constant ln(base)/scale: sum_i (s_i - n_i*p_i).
        # Strictly decreasing in ``rating``, so the root is unique.
        total = 0.0
        for ref_elo, s, n in data:
            p = 1.0 / (1.0 + base ** ((ref_elo - rating) / scale))
            total += s - n * p
        return total

    # gradient(lo) > 0 and gradient(hi) < 0 are guaranteed once the overall win
    # rate is strictly inside (0, 1); bisect for the unique root.
    lo = min(ref_elo for ref_elo, _, _ in data) - 4000.0
    hi = max(ref_elo for ref_elo, _, _ in data) + 4000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if gradient(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    elo = 0.5 * (lo + hi)
    return elo, (elo - 500.0) / 2000.0


def predict_win_rate(eval_elo: float, ref_elo: float, scale: float = 400.0, base: float = 10.0) -> float:
    """Expected eval-model win probability vs a reference at ``ref_elo``."""
    return 1.0 / (1.0 + base ** ((ref_elo - eval_elo) / scale))


def compute_comparison_reward(winner: str) -> float:
    """Convert a BOXED winner string to a reward float.

    - Reference model (A) wins → 0.0  (eval model lost)
    - Eval model (B) wins → 1.0
    - Tie → 0.5
    """
    if winner == B_WIN_RESPONSE:
        return 1.0
    if winner == A_WIN_RESPONSE:
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Convenience: check if a task was attempted
# ---------------------------------------------------------------------------


def task_attempted(task_dir: str) -> bool:
    """Return True if the task directory has a ``finish_params.json`` (completed run)."""
    return os.path.exists(task_dir) and os.path.exists(os.path.join(task_dir, "finish_params.json"))


def clean_up_paths(paths: list[Path]) -> None:
    """Remove extracted zip artifacts."""
    for path in paths:
        try:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        except Exception:
            pass
