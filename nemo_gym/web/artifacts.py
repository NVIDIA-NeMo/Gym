# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral artifact persistence for web rollouts."""

from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
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
        raise RuntimeError("failed to encode web screenshot as PNG") from exc


class WebArtifactStore:
    """Persist screenshots and recordings without depending on a browser backend."""

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

    def recording_artifacts(self, session_id: str) -> list[WebArtifactRef]:
        """Return finalized, non-empty recordings for a closed browser session."""

        video_dir = self.root / _safe_component(session_id) / "video"
        if not video_dir.is_dir():
            return []

        artifacts: list[WebArtifactRef] = []
        for path in sorted(video_dir.rglob("*")):
            if not path.is_file():
                continue
            size_bytes = path.stat().st_size
            if size_bytes == 0:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            mime_type, _encoding = mimetypes.guess_type(path.name)
            artifacts.append(
                WebArtifactRef(
                    uri=path.resolve().as_uri(),
                    mime_type=mime_type or "application/octet-stream",
                    size_bytes=size_bytes,
                    sha256=digest.hexdigest(),
                )
            )
        return artifacts
