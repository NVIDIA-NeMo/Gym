# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-session screenshot persistence for web rollouts."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from pathlib import Path
from typing import Any

from nemo_gym.web.models import WebArtifactRef, WebImage


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("_", value).strip("._")
    return cleaned or "session"


def _png_bytes(image: Any) -> bytes:
    if isinstance(image, bytes):
        return image
    try:
        from PIL import Image

        buffer = io.BytesIO()
        if isinstance(image, Image.Image):
            image.save(buffer, format="PNG")
        else:
            Image.fromarray(image).save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001 - normalize optional imaging dependency failures.
        raise RuntimeError("failed to encode BrowserGym screenshot as PNG") from exc


class WebArtifactStore:
    def __init__(self, root: str | Path, *, inline_screenshots: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.inline_screenshots = inline_screenshots
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        path = self.root / _safe_component(session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_screenshot(self, session_id: str, step: int, image: Any, *, suffix: str = "") -> WebImage:
        payload = _png_bytes(image)
        suffix_part = f"-{_safe_component(suffix)}" if suffix else ""
        path = self.session_dir(session_id) / f"step-{step:04d}{suffix_part}.png"
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        artifact = WebArtifactRef(
            uri=path.as_uri(),
            mime_type="image/png",
            size_bytes=len(payload),
            sha256=digest,
        )
        data_url = None
        if self.inline_screenshots:
            data_url = f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
        return WebImage(data_url=data_url, artifact=artifact)
