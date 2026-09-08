# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-turn GDP.pdf agent with AA v4.3 document delivery."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Request
from PIL import Image, ImageDraw
from pydantic import Field, model_validator

from nemo_gym import PARENT_DIR
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyResponse,
)


class GdpPdfAgentConfig(SimpleAgentConfig):
    documents_base_dir: str = Field(
        description="Base directory for document manifests referenced by verifier_metadata.document_manifest."
    )
    include_page_images: bool = True
    source_dpi: int = Field(default=150, ge=72, le=150)
    image_dpi: int = Field(default=150, ge=72, le=150)
    pages_per_image: Literal[1, 2, 4] = 1
    max_images: Optional[int] = Field(default=None, ge=1)
    image_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = Field(default=90, ge=1, le=100)
    strip_images_from_output: bool = True

    @model_validator(mode="after")
    def validate_dpi(self) -> "GdpPdfAgentConfig":
        if self.image_dpi > self.source_dpi:
            raise ValueError("image_dpi cannot exceed the prepared source_dpi")
        return self


def _resolve_under(base_dir: Path, relative_path: str) -> Path:
    base = base_dir.resolve()
    candidate = (base / relative_path).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"document path escapes documents_base_dir: {relative_path!r}")
    return candidate


def _open_page_image(path: Path, *, source_dpi: int, image_dpi: int) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if image_dpi == source_dpi:
        return image
    scale = image_dpi / source_dpi
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _encode_image(image: Image.Image, *, image_format: str, jpeg_quality: int) -> tuple[str, str]:
    output = io.BytesIO()
    if image_format == "jpeg":
        image.convert("RGB").save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        image.save(output, format="PNG", optimize=True)
        mime = "image/png"
    return mime, base64.standard_b64encode(output.getvalue()).decode("ascii")


def _compose_pages(
    pages: list[tuple[int, Path]],
    *,
    source_dpi: int,
    image_dpi: int,
    image_format: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    images = [
        (page_number, _open_page_image(path, source_dpi=source_dpi, image_dpi=image_dpi))
        for page_number, path in pages
    ]

    if len(images) == 1:
        composed = images[0][1]
    else:
        columns = 2
        rows = math.ceil(len(images) / columns)
        label_height = max(24, round(28 * image_dpi / 150))
        cell_width = max(image.width for _, image in images)
        cell_height = max(image.height for _, image in images) + label_height
        composed = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(composed)
        for index, (page_number, image) in enumerate(images):
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            draw.text((x + 8, y + 6), f"Page {page_number}", fill="black")
            composed.paste(image, (x, y + label_height))

    mime, payload = _encode_image(composed, image_format=image_format, jpeg_quality=jpeg_quality)
    return {"type": "input_image", "image_url": f"data:{mime};base64,{payload}", "detail": "high"}


def _document_content(
    *,
    task_prompt: str,
    manifest_path: Path,
    config: GdpPdfAgentConfig,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_pages = manifest.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError(f"document manifest has no pages: {manifest_path}")

    manifest_dir = manifest_path.parent
    manifest_source_dpi = int(manifest.get("source_dpi", config.source_dpi))
    if manifest_source_dpi != config.source_dpi:
        raise ValueError(
            f"document was prepared at {manifest_source_dpi} DPI, but source_dpi is configured as {config.source_dpi}"
        )

    pages: list[tuple[int, str, Optional[Path]]] = []
    for raw_page in raw_pages:
        page_number = int(raw_page["page_number"])
        text = str(raw_page.get("text", ""))
        image_path = None
        if config.include_page_images:
            image_path = _resolve_under(manifest_dir, str(raw_page["image"]))
        pages.append((page_number, text, image_path))
    pages.sort(key=lambda page: page[0])
    if [number for number, _, _ in pages] != list(range(1, len(pages) + 1)):
        raise ValueError(f"document manifest has invalid page numbering: {manifest_path}")

    image_pages = pages
    if config.max_images is not None:
        image_pages = image_pages[: config.max_images * config.pages_per_image]
    image_batches = [
        image_pages[index : index + config.pages_per_image]
        for index in range(0, len(image_pages), config.pages_per_image)
    ]

    image_description = "No page images are included; use the complete extracted text."
    if config.include_page_images:
        shown = len(image_pages)
        image_description = (
            f"Ordered page images are included at {config.image_dpi} DPI, {config.pages_per_image} page(s) per image."
        )
        if shown < len(pages):
            image_description += f" Images cover only pages 1-{shown}; extracted text covers all {len(pages)} pages."
        else:
            image_description += f" Images cover all {len(pages)} pages."

    instructions = (
        "Complete the professional task using only the supplied source document. "
        "The full LiteParse-extracted text of every page is included and page boundaries are labeled. "
        f"{image_description}\n\nTASK:\n{task_prompt}"
    )
    content: list[dict[str, Any]] = [{"type": "input_text", "text": instructions}]

    if config.include_page_images:
        for batch in image_batches:
            page_images = []
            for number, _, image_path in batch:
                if image_path is None:
                    raise ValueError(f"missing image path for page {number}")
                page_images.append((number, image_path))
            content.append(
                _compose_pages(
                    page_images,
                    source_dpi=config.source_dpi,
                    image_dpi=config.image_dpi,
                    image_format=config.image_format,
                    jpeg_quality=config.jpeg_quality,
                )
            )

    page_text = "\n\n".join(f"<page {number}>\n{text}\n</page {number}>" for number, text, _ in pages)
    content.append({"type": "input_text", "text": f"SOURCE DOCUMENT TEXT:\n{page_text}"})
    return content


def materialize_document(row: dict[str, Any], base_dir: Path, config: GdpPdfAgentConfig) -> dict[str, Any]:
    """Inject complete extracted text and the configured page-image profile."""
    metadata = row.get("verifier_metadata") or {}
    relative_manifest = metadata.get("document_manifest")
    if not relative_manifest:
        return row

    task_prompt = str(metadata.get("task_prompt", "")).strip()
    if not task_prompt:
        raise ValueError("GDP.pdf verifier_metadata.task_prompt is required")
    manifest_path = _resolve_under(base_dir, str(relative_manifest))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"GDP.pdf document manifest not found: {manifest_path}")

    enriched = deepcopy(row)
    params = enriched["responses_create_params"]
    params["input"] = [
        {
            "role": "user",
            "content": _document_content(task_prompt=task_prompt, manifest_path=manifest_path, config=config),
        }
    ]
    params["tools"] = []
    params["parallel_tool_calls"] = False
    return enriched


def _strip_image_blocks(result: SimpleAgentVerifyResponse) -> SimpleAgentVerifyResponse:
    removed = False

    def scrub(value: Any) -> Any:
        nonlocal removed
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "input_image":
                    removed = True
                    continue
                cleaned.append(scrub(item))
            return cleaned
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    data = scrub(result.model_dump(mode="json"))
    if removed:
        for key in ("ng_trajectory", "ng_agent_observations"):
            observations = data.get(key)
            if isinstance(observations, dict):
                observations.setdefault("gaps", []).append({"code": "multimodal_history_redacted"})
    return SimpleAgentVerifyResponse.model_validate(data)


class GdpPdfAgent(SimpleAgent):
    config: GdpPdfAgentConfig

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        base_dir = Path(self.config.documents_base_dir)
        if not base_dir.is_absolute():
            base_dir = PARENT_DIR / base_dir

        row = body.model_dump(exclude_unset=True)
        enriched = await asyncio.to_thread(materialize_document, row, base_dir, self.config)
        result = await super().run(request, SimpleAgentRunRequest.model_validate(enriched))
        return _strip_image_blocks(result) if self.config.strip_images_from_output else result


if __name__ == "__main__":
    GdpPdfAgent.run_webserver()
