# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve benchmark reference images at the agent-side trust boundary."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse


_PASSTHROUGH_SCHEMES = frozenset({"data", "http", "https"})
_MIME_OVERRIDES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def resolve_task_image_url(
    image_reference: str,
    *,
    image_root: str | Path | None,
    max_bytes: int,
) -> str:
    """Return a model-safe URL for one task reference image.

    Public WebArena-family JSONL records store relative image paths. The model
    endpoint cannot consume those controller paths directly, so the web agent
    reads them from an explicitly mounted, read-only root and emits a data URL.
    Existing data and HTTP(S) URLs pass through unchanged.

    Local references are accepted only when they resolve inside ``image_root``.
    This prevents benchmark metadata from becoming an arbitrary file-read API.
    """

    reference = str(image_reference).strip()
    if not reference:
        raise ValueError("task image reference must not be empty")
    parsed = urlparse(reference)
    if parsed.scheme in _PASSTHROUGH_SCHEMES:
        return reference
    if parsed.scheme:
        raise ValueError(f"unsupported task image URL scheme: {parsed.scheme!r}")
    candidate, mime_type = resolve_local_task_image_path(
        reference,
        image_root=image_root,
        max_bytes=max_bytes,
    )
    payload = base64.b64encode(candidate.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def resolve_local_task_image_path(
    image_reference: str,
    *,
    image_root: str | Path | None,
    max_bytes: int,
) -> tuple[Path, str]:
    """Resolve and validate one local reference below a read-only image root."""

    reference = str(image_reference).strip()
    if image_root is None:
        raise ValueError("task_image_root is required for local benchmark image references")
    if max_bytes <= 0:
        raise ValueError("max task image bytes must be positive")

    root = Path(image_root).expanduser().resolve(strict=True)
    candidate = Path(reference).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("task image path resolves outside task_image_root") from exc
    if not candidate.is_file():
        raise ValueError("task image reference must resolve to a regular file")
    size = candidate.stat().st_size
    if size > max_bytes:
        raise ValueError(f"task image exceeds configured byte limit ({size} > {max_bytes})")

    mime_type = _MIME_OVERRIDES.get(candidate.suffix.lower())
    if mime_type is None:
        mime_type = mimetypes.guess_type(candidate.name)[0]
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"unsupported task image type for {candidate.name!r}")
    return candidate, mime_type
