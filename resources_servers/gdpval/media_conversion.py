# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Media conversion for image-only multimodal judges (e.g. locally-served Kimi K2.6).

The frontier judges (Gemini / GPT / Claude) accept PDFs directly as an
``application/pdf`` data URL and rasterize server-side. Locally-served
vision-language judges (Kimi K2.6 on vLLM, Qwen-VL, InternVL, …) only accept
**image** inputs — they cannot decode a raw PDF data URL. For those judges we
must pre-render each PDF/Office page to a raster image (PNG) and, additionally,
extract the underlying text so the judge gets both the visual layout and a
clean textual copy it can quote from.

This module is the single home for that ``images_and_text`` conversion so the
two content-block builders — the pairwise comparison scorer
(:mod:`resources_servers.gdpval.comparison`) and the rubric visual scorer
(:mod:`responses_api_agents.stirrup_agent.file_reader`) — share one code path.

Everything here is pure-python: PDF rasterization uses PyMuPDF (``fitz``, already
a GDPVal dependency) so there is no poppler/system dependency, and text
extraction reuses ``pdfminer.six``.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Literal


LOGGER = logging.getLogger(__name__)

# ``native_pdf``   — send PDFs/Office docs as ``application/pdf`` data URLs
#                    (frontier judges that decode PDFs natively). Legacy default.
# ``images_and_text`` — rasterize each page to a PNG image block AND attach the
#                    extracted text (image-only local VLM judges, e.g. Kimi K2.6).
JudgeMediaMode = Literal["native_pdf", "images_and_text"]

# Defaults chosen so a typical multi-page GDPVal deliverable stays well within a
# local VLM's context: 144 DPI renders text legibly for a judge without the
# 4-8x payload of 300 DPI, and 50 pages covers essentially every deliverable
# while capping the pathological (a 500-page appendix would otherwise explode
# the request).
DEFAULT_RENDER_DPI = 144
DEFAULT_MAX_PAGES = 50
# Per-file extracted-text cap (characters). The judge already sees the rendered
# pages; the text copy is a legibility aid, not the primary signal, so it is
# bounded to keep the prompt from ballooning on text-dense PDFs.
DEFAULT_MAX_TEXT_CHARS = 20_000

# PDF page rasters are always PNG (lossless — keeps small text crisp for the
# judge). One shared MIME constant so callers don't re-specify it.
_PNG_MIME = "image/png"


def _data_url(mime_type: str, data: bytes) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def pdf_bytes_to_image_blocks(
    pdf_bytes: bytes,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> List[Dict[str, Any]]:
    """Rasterize a PDF (as bytes) to a list of OpenAI ``image_url`` content blocks.

    One block per page (PNG data URL), capped at *max_pages*. When the PDF has
    more pages than the cap, a trailing text block records the truncation so the
    judge knows it did not see the whole document. Returns ``[]`` (and logs) when
    PyMuPDF is unavailable or the PDF can't be opened, so callers can fall back.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        LOGGER.warning("PyMuPDF (fitz) not installed; cannot rasterize PDF for image-only judge")
        return []

    blocks: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — malformed/encrypted PDF → fall back
        LOGGER.warning("failed to open PDF for rasterization: %r", exc)
        return []

    try:
        total_pages = doc.page_count
        n = min(total_pages, max_pages)
        # 72 DPI is PDF user-space; scale the render matrix to hit the target DPI.
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(n):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                png = pix.tobytes("png")
            except Exception as exc:  # noqa: BLE001 — skip an unrenderable page, keep the rest
                LOGGER.warning("failed to render PDF page %d: %r", i, exc)
                continue
            blocks.append({"type": "image_url", "image_url": {"url": _data_url(_PNG_MIME, png)}})
        if total_pages > max_pages:
            blocks.append({"type": "text", "text": f"[truncated: rendered {max_pages} of {total_pages} pages]"})
    finally:
        doc.close()

    return blocks


def pdf_bytes_to_text(pdf_bytes: bytes, *, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    """Extract text from a PDF (as bytes), truncated to *max_chars*.

    Returns ``""`` when pdfminer is unavailable or extraction fails — the image
    blocks are the primary signal, so a missing text copy is non-fatal.
    """
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        LOGGER.warning("pdfminer.six not installed; skipping PDF text extraction")
        return ""

    try:
        text = (extract_text(io.BytesIO(pdf_bytes)) or "").strip()
    except Exception as exc:  # noqa: BLE001 — extraction is best-effort
        LOGGER.warning("failed to extract PDF text: %r", exc)
        return ""

    if len(text) > max_chars:
        return text[:max_chars] + "\n[...text truncated]"
    return text


def pdf_bytes_to_blocks(
    pdf_bytes: bytes,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
    include_text: bool = True,
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> List[Dict[str, Any]]:
    """Full ``images_and_text`` conversion of a PDF: page images + optional text.

    The extracted-text block (when *include_text*) is emitted **first** so the
    judge reads the machine-readable copy before the rendered pages. Returns
    ``[]`` when the PDF produced neither images nor text.
    """
    blocks: List[Dict[str, Any]] = []
    if include_text:
        text = pdf_bytes_to_text(pdf_bytes, max_chars=max_text_chars)
        if text:
            blocks.append({"type": "text", "text": f"[extracted text]\n{text}"})
    blocks.extend(pdf_bytes_to_image_blocks(pdf_bytes, dpi=dpi, max_pages=max_pages))
    return blocks
