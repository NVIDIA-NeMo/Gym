# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-turn GDP.pdf agent with AA v4.3 document delivery."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Literal, Optional

from aiohttp import ClientResponseError
from fastapi import Request
from PIL import Image, ImageDraw
from pydantic import Field

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyResponse,
)


_DOCUMENT_TEXT_PREFIX = "SOURCE DOCUMENT TEXT:\n"
_DOCUMENT_REDACTION_MARKER = "[GDP.pdf document payload redacted]"


class GdpPdfAgentConfig(SimpleAgentConfig):
    max_steps: Literal[1] = 1
    documents_base_dir: str = Field(
        description="Base directory for document manifests referenced by verifier_metadata.document_manifest."
    )
    include_page_images: bool = True
    source_dpi: Literal[150] = 150
    max_images: Optional[int] = Field(default=None, ge=1)
    image_format: Literal["png", "jpeg"] = "png"
    jpeg_quality: int = Field(default=90, ge=1, le=100)
    strip_document_payloads_from_output: Literal[True] = True


class DocumentDelivery:
    """Per-request state; never mutate the shared agent config."""

    def __init__(self, max_images: Optional[int] = None):
        self.image_dpi = 150
        self.max_images = max_images
        self.pages_per_image = 1
        self.page_count = 0
        self.image_count = 0
        self.image_pages = 0
        self.attempts: list[dict[str, Any]] = []

    def record(self, limit: Optional[str] = None) -> dict[str, Any]:
        return {
            "image_dpi": self.image_dpi,
            "pages_per_image": self.pages_per_image,
            "image_count": self.image_count,
            "image_pages": self.image_pages,
            "page_count": self.page_count,
            "limit": limit,
        }

    def adapt(self, limit: str, image_cap: Optional[int]) -> bool:
        self.attempts.append(self.record(limit))
        if limit == "image_count":
            # A numeric endpoint limit avoids probing the same rejected image count.
            cap = image_cap if image_cap is not None else self.image_count - 1
            if cap < 1 or cap >= self.image_count:
                return False
            self.max_images = min(self.max_images or cap, cap)
            return True
        if self.image_dpi == 72 or not self.image_count:
            return False
        # AA publishes the endpoints (150 and 72), not a decrement schedule.
        # Reduce by 20% per rejected request, always trying the 72 DPI floor.
        self.image_dpi = max(72, math.floor(self.image_dpi * 0.8))
        return True


def _input_limit(error: ClientResponseError) -> tuple[Optional[str], Optional[int]]:
    """Recognize explicit input-limit errors, including Gym's wrapped upstream errors."""
    text = getattr(error, "response_content", b"")
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = str(text).lower()
    if error.status in (401, 403, 429) or any(
        marker in text for marker in ("ratelimiterror", "rate_limit_exceeded", "authenticationerror")
    ):
        return None, None
    for pattern in (
        r"(?:at most|maximum of|up to|more than)\s+(\d+)\s+images?",
        r"(?:maximum|max)\s+(?:number of )?images?[^\d]{0,20}(\d+)",
    ):
        match = re.search(pattern, text)
        if match:
            return "image_count", int(match[1])
    if "too many images" in text:
        return "image_count", None
    if error.status == 413 or any(
        marker in text
        for marker in (
            "request entity too large",
            "payload too large",
            "request body too large",
            "request_too_large",
            "image too large",
            "image dimensions exceed",
        )
    ):
        return "payload", None
    if any(
        marker in text
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "exceeds maximum input length",
            "input is too long",
            "prompt is too long",
            "exceeds the model's maximum context",
            "longer than the maximum model length",
        )
    ):
        return "context", None
    return None, None


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
    delivery: DocumentDelivery,
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

    delivery.page_count = len(pages)
    delivery.pages_per_image = 1
    if delivery.max_images is not None:
        while delivery.pages_per_image < 4 and math.ceil(len(pages) / delivery.pages_per_image) > delivery.max_images:
            delivery.pages_per_image *= 2
    image_pages = pages if config.include_page_images else []
    if delivery.max_images is not None:
        image_pages = image_pages[: delivery.max_images * delivery.pages_per_image]
    image_batches = [
        image_pages[index : index + delivery.pages_per_image]
        for index in range(0, len(image_pages), delivery.pages_per_image)
    ]
    delivery.image_pages = len(image_pages)
    delivery.image_count = len(image_batches)

    image_description = "No page images are included; use the complete extracted text."
    if config.include_page_images:
        shown = len(image_pages)
        image_description = f"Ordered page images are included at {delivery.image_dpi} DPI, {delivery.pages_per_image} page(s) per image."
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
                    image_dpi=delivery.image_dpi,
                    image_format=config.image_format,
                    jpeg_quality=config.jpeg_quality,
                )
            )

    page_text = "\n\n".join(f"<page {number}>\n{text}\n</page {number}>" for number, text, _ in pages)
    content.append({"type": "input_text", "text": f"SOURCE DOCUMENT TEXT:\n{page_text}"})
    return content


def materialize_document(
    row: dict[str, Any], base_dir: Path, config: GdpPdfAgentConfig, delivery: Optional[DocumentDelivery] = None
) -> dict[str, Any]:
    """Inject full text and the current delivery profile for a policy request only."""
    delivery = delivery or DocumentDelivery(config.max_images)
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
            "content": _document_content(
                task_prompt=task_prompt, manifest_path=manifest_path, config=config, delivery=delivery
            ),
        }
    ]
    params["tools"] = []
    params["parallel_tool_calls"] = False
    return enriched


def _strip_document_payloads(result: SimpleAgentVerifyResponse) -> SimpleAgentVerifyResponse:
    redacted = False

    def scrub(value: Any) -> Any:
        nonlocal redacted
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "input_image":
                    redacted = True
                    continue
                if (
                    isinstance(item, dict)
                    and item.get("type") == "input_text"
                    and isinstance(item.get("text"), str)
                    and item["text"].startswith(_DOCUMENT_TEXT_PREFIX)
                ):
                    redacted = True
                    cleaned.append({**item, "text": _DOCUMENT_REDACTION_MARKER})
                    continue
                cleaned.append(scrub(item))
            return cleaned
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    data = scrub(result.model_dump(mode="json"))
    if redacted:
        for key in ("ng_trajectory", "ng_agent_observations"):
            observations = data.get(key)
            if isinstance(observations, dict):
                gaps = observations.setdefault("gaps", [])
                if not any(gap.get("code") == "document_payload_redacted" for gap in gaps if isinstance(gap, dict)):
                    gaps.append({"code": "document_payload_redacted"})
    return SimpleAgentVerifyResponse.model_validate(data)


class GdpPdfAgent(SimpleAgent):
    config: GdpPdfAgentConfig

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        base_dir = Path(self.config.documents_base_dir)
        if not base_dir.is_absolute():
            base_dir = PARENT_DIR / base_dir

        row = body.model_dump(exclude_unset=True)
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name, url_path="/seed_session", json=row, cookies=request.cookies
        )
        await raise_for_status(seed)
        cookies = seed.cookies
        delivery = DocumentDelivery(self.config.max_images)
        trajectory = None
        terminal_limit = None
        while True:
            enriched = await asyncio.to_thread(materialize_document, row, base_dir, self.config, delivery)
            params = SimpleAgentRunRequest.model_validate(enriched).responses_create_params
            try:
                model_response, trajectory, _, cookies = await self._create_episode(
                    params,
                    model_url_path=self.url_path_for_run("/v1/responses", body),
                    resources_server_cookies=cookies,
                    task_id=str(row.get("_ng_task_index", "unknown")),
                    rollout_id=self.rollout_id_from_run(body) or "unscoped",
                    collect_trajectory=self._model_call_capture_enabled(),
                )
                break
            except ClientResponseError as error:
                limit, image_cap = _input_limit(error)
                if limit is None:
                    raise
                if delivery.adapt(limit, image_cap):
                    continue
                terminal_limit = limit
                # AA scores terminal input failures as zero. An empty answer goes
                # through the normal verifier without making any rubric judge calls.
                model_response = NeMoGymResponse(
                    id="gdp-pdf-input-failure",
                    created_at=time(),
                    model=self.config.model_server.name,
                    object="response",
                    status="failed",
                    output=[],
                    tools=[],
                    tool_choice="none",
                    parallel_tool_calls=False,
                )
                break

        result = row | {"response": model_response.model_dump(mode="json")}
        if self.config.skip_verification:
            result.update(
                reward=0.0 if terminal_limit else float(self.config.skip_verification_reward),
                verification_skipped=True,
            )
        else:
            verified = await self.server_client.post(
                server_name=self.config.resources_server.name, url_path="/verify", json=result, cookies=cookies
            )
            await raise_for_status(verified)
            result = await get_response_json(verified)
        result["document_delivery"] = delivery.record(terminal_limit) | {
            "image_format": self.config.image_format,
            "rejected_attempts": delivery.attempts,
        }
        if terminal_limit:
            result["failure_reason"] = f"GDP.pdf terminal input limit: {terminal_limit}"
        if trajectory is not None:
            result["ng_trajectory"] = trajectory.model_dump(mode="json")
        return _strip_document_payloads(SimpleAgentVerifyResponse.model_validate(result))


if __name__ == "__main__":
    GdpPdfAgent.run_webserver()
