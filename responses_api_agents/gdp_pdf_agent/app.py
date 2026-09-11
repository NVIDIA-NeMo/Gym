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
"""
GDP.pdf agent — extends simple_agent to embed a pre-rendered document at rollout time.

GDP.pdf rows reference their document by a manifest path (``verifier_metadata.document_manifest``)
rather than inlining anything: the corpus averages ~46 pages per task, so base64 in the JSONL
would make the dataset unusable. The manifest (page text + one 150 DPI PNG screenshot per page) is
produced **once**, at prepare time (``benchmarks/gdp_pdf/prepare.py``'s ``prepare_document()``),
not on every rollout -- GDP.pdf's 100 source PDFs are static, so re-running LiteParse OCR and
page rendering on every row, every repeat, every model (as an earlier version of this agent did)
redundantly repeats the same expensive parse of the same 100 documents. This agent just reads the
cached manifest and resizes the cached page PNGs for its reactive DPI backoff, rather than
re-rendering from the PDF.

DPI/compositing strategy is reactive, not estimated ahead of time: every document starts at
``dpi`` (default 150, matching the manifest's cached resolution). If the policy model rejects a
request (too many images, payload too large, context length exceeded), the agent adapts --
widening page compositing (multiple pages per image, up to 4) if the server reported an explicit
image-count limit, otherwise reducing DPI by 20% toward a ``min_dpi`` floor (default 72) -- and
retries. This mirrors Artificial Analysis's GDP.pdf methodology
(https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf), which publishes the
DPI endpoints (150, 72) and describes page compositing under tight per-request image-count limits,
not a decrement schedule or a token-budget estimate. A terminal (unrecoverable) input-limit
failure is scored as a zero attempt, matching AA's treatment of failed submissions -- it does not
raise, so the row still gets verified (and scores 0, since an empty answer earns no rubric credit)
rather than being dropped from the run entirely.
"""

import base64
import io
import json
import math
import re
from pathlib import Path
from time import time
from typing import Any, Literal, Optional

from aiohttp import ClientResponseError
from fastapi import Request
from PIL import Image, ImageDraw
from pydantic import Field

from nemo_gym import PARENT_DIR
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentVerifyResponse,
)


BASE_DPI = 150
MIN_DPI = 72
_COMPOSITE_COLUMNS = 2
_MAX_PAGES_PER_IMAGE = 4


class GdpPdfAgentConfig(SimpleAgentConfig):
    documents_base_dir: str = Field(
        description="Base directory for resolving verifier_metadata.document_manifest, relative to the Gym root.",
    )
    dpi: int = Field(
        default=BASE_DPI,
        description="Starting DPI for the reactive backoff. Must match the DPI documents were "
        "rendered at during prepare (source_dpi in manifest.json) -- pages can only be resized down "
        "from the cached screenshot, never rendered fresh at a higher DPI.",
    )
    min_dpi: int = Field(default=MIN_DPI, description="Floor DPI the reactive backoff will reduce to.")
    max_images: Optional[int] = Field(
        default=None,
        ge=1,
        description="Cap on images per request. When the page count exceeds this, multiple pages are "
        "composited into a single labeled image (up to 4 pages/image) instead of dropping pages. "
        "None disables compositing unless the policy model itself reports an image-count limit.",
    )
    max_pages: Optional[int] = Field(
        default=None,
        description="Cap on pages read from the manifest per document. None uses every page.",
    )
    include_text: bool = Field(
        default=True, description="Include the manifest's extracted text as an input_text block."
    )
    include_images: bool = Field(
        default=True, description="Include the manifest's rendered page images as input_image blocks."
    )
    image_format: Literal["png", "jpeg"] = Field(
        default="png",
        description="Wire format for page images. PNG is the default because it is lossless and the "
        "model must read fine print off these pages. Set jpeg when payload size is the binding "
        "constraint: measured over 296 pages sampled across all 100 documents, JPEG-90 is 72% of "
        "PNG's base64 size (~28% saved).",
    )
    jpeg_quality: int = Field(default=90, ge=1, le=100, description="JPEG quality when image_format is jpeg.")
    strip_images_from_output: bool = Field(
        default=True,
        description="Remove base64 input_image blocks from serialized rollout artifacts.",
    )


class DocumentDelivery:
    """Per-request state; never mutate the shared agent config."""

    def __init__(self, dpi: int, max_images: Optional[int]):
        self.image_dpi = dpi
        self.max_images = max_images
        self.pages_per_image = 1
        self.page_count = 0
        self.image_count = 0
        self.image_pages_covered = 0
        self.image_pages_omitted = 0
        self.image_coverage_end_page: Optional[int] = None
        self.attempts: list[dict[str, Any]] = []

    def record(self, limit: Optional[str] = None) -> dict[str, Any]:
        return {
            "image_dpi": self.image_dpi,
            "pages_per_image": self.pages_per_image,
            "image_count": self.image_count,
            "page_count": self.page_count,
            "image_pages_covered": self.image_pages_covered,
            "image_pages_omitted": self.image_pages_omitted,
            "image_coverage_end_page": self.image_coverage_end_page,
            "limit": limit,
        }

    def adapt(self, limit: str, image_cap: Optional[int], *, min_dpi: int) -> bool:
        """Reduce the delivery footprint after a rejected request. Returns False when nothing
        more can be reduced -- the caller should treat the request as a terminal failure."""
        self.attempts.append(self.record(limit))
        if limit == "image_count":
            # A numeric endpoint limit avoids probing the same rejected image count.
            cap = image_cap if image_cap is not None else max(1, self.image_count - 1)
            if cap < 1 or cap >= self.image_count:
                return False
            self.max_images = min(self.max_images, cap) if self.max_images is not None else cap
            return True
        if self.image_dpi <= min_dpi:
            return False
        # AA publishes the endpoints (150 and 72), not a decrement schedule.
        # Reduce by 20% per rejected request, always trying the 72 DPI floor.
        self.image_dpi = max(min_dpi, math.floor(self.image_dpi * 0.8))
        return True


def _terminal_failure_response(model_server_name: str) -> NeMoGymResponse:
    """An empty, ``status=failed`` response for a row that could not be completed. AA scores
    terminal input failures as zero; an empty answer goes through the normal verifier without
    making any rubric judge calls."""
    return NeMoGymResponse(
        id="gdp-pdf-input-failure",
        created_at=time(),
        model=model_server_name,
        object="response",
        status="failed",
        output=[],
        tools=[],
        tool_choice="none",
        parallel_tool_calls=False,
    )


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
    if error.status >= 500 and not text.strip():
        # An empty body on a server error is exactly what an oversized-request crash on the
        # inference server looks like when nothing survives to describe why (seen in practice:
        # vLLM returning a 500 with response_content=b'' under an oversized multi-image prompt).
        # Retrying via the same DPI backoff as a real "payload" limit is far cheaper than letting
        # this escape unclassified and take down the whole run over what was likely one document.
        return "payload", None
    return None, None


def _open_page_image(path: Path, *, source_dpi: int, image_dpi: int) -> "Image.Image":
    """Open a cached page screenshot, resizing down to ``image_dpi`` if the backoff has reduced
    it below the DPI it was rendered at. Pages are never re-rendered from the PDF -- only resized
    from the one screenshot cached at prepare time."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if image_dpi >= source_dpi:
        return image
    scale = image_dpi / source_dpi
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _compose_pages(
    images: list[tuple[int, "Image.Image"]],
    *,
    image_dpi: int,
    image_format: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Combine one or more page images into a single ``input_image`` block. When more than one
    page is given, pages are laid out in a labeled grid so the model can still tell them apart."""
    if len(images) == 1:
        composed = images[0][1]
    else:
        columns = _COMPOSITE_COLUMNS
        rows = math.ceil(len(images) / columns)
        label_height = max(24, round(28 * image_dpi / BASE_DPI))
        cell_width = max(image.width for _, image in images)
        cell_height = max(image.height for _, image in images) + label_height
        composed = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(composed)
        for index, (page_number, image) in enumerate(images):
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            draw.text((x + 8, y + 6), f"Page {page_number}", fill="black")
            composed.paste(image, (x, y + label_height))

    buf = io.BytesIO()
    if image_format == "jpeg":
        composed.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        composed.save(buf, format="PNG", optimize=True)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/{image_format};base64,{b64}", "detail": "high"}


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load a prepare-time document manifest, sorting pages by page number."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"] = sorted(manifest.get("pages") or [], key=lambda page: page["page_number"])
    return manifest


def _manifest_text_blocks(manifest: dict[str, Any], *, max_pages: Optional[int]) -> tuple[list[dict[str, Any]], int]:
    """Extracted text doesn't change across retry attempts (unlike page images, which get resized
    smaller on a DPI backoff), so this only needs to run once per row. Returns ``(content_blocks,
    pages_truncated)`` -- at most one ``input_text`` block, empty if nothing extractable."""
    pages = manifest["pages"]
    total_pages = len(pages)
    page_limit = total_pages if max_pages is None else min(max_pages, total_pages)
    pages_truncated = total_pages - page_limit
    text = "\n\n".join(page["text"] for page in pages[:page_limit])
    if not text.strip():
        return [], pages_truncated
    return [{"type": "input_text", "text": f"<document>\n{text.strip()}\n</document>"}], pages_truncated


def _render_manifest_images(
    manifest: dict[str, Any],
    manifest_dir: Path,
    *,
    delivery: DocumentDelivery,
    max_pages: Optional[int],
    source_dpi: int,
    image_format: str = "jpeg",
    jpeg_quality: int = 90,
) -> tuple[list[dict[str, Any]], int]:
    """Render one attempt's worth of page images at the delivery's current DPI / compositing
    settings, from the manifest's cached page screenshots. Returns ``(image_blocks,
    pages_truncated)``."""
    pages = manifest["pages"]
    total_pages = len(pages)
    page_limit = total_pages if max_pages is None else min(max_pages, total_pages)
    pages_truncated = total_pages - page_limit
    delivery.page_count = page_limit
    if not page_limit:
        delivery.image_count = 0
        delivery.image_pages_covered = 0
        delivery.image_pages_omitted = 0
        delivery.image_coverage_end_page = None
        return [], pages_truncated

    # Recomputed fresh each attempt from the current max_images -- not carried over --
    # so a later widened/narrowed max_images always takes full effect immediately.
    delivery.pages_per_image = 1
    if delivery.max_images is not None:
        while delivery.pages_per_image < _MAX_PAGES_PER_IMAGE and (
            math.ceil(page_limit / delivery.pages_per_image) > delivery.max_images
        ):
            delivery.pages_per_image *= 2

    selected = pages[:page_limit]
    if delivery.max_images is not None:
        selected = selected[: delivery.max_images * delivery.pages_per_image]

    batches = [selected[i : i + delivery.pages_per_image] for i in range(0, len(selected), delivery.pages_per_image)]
    delivery.image_count = len(batches)
    delivery.image_pages_covered = len(selected)
    delivery.image_pages_omitted = page_limit - len(selected)
    delivery.image_coverage_end_page = selected[-1]["page_number"] if selected else None

    blocks: list[dict[str, Any]] = []
    for batch in batches:
        images = [
            (
                page["page_number"],
                _open_page_image(manifest_dir / page["image"], source_dpi=source_dpi, image_dpi=delivery.image_dpi),
            )
            for page in batch
        ]
        blocks.append(
            _compose_pages(
                images,
                image_dpi=delivery.image_dpi,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
        )
    return blocks, pages_truncated


def _delivery_notice_blocks(delivery: DocumentDelivery) -> list[dict[str, Any]]:
    """Describe composite images and partial image coverage to the policy model."""
    notices = []
    if delivery.pages_per_image > 1:
        notices.append(
            f"The page images below are composites containing up to {delivery.pages_per_image} pages per image; "
            "each cell is labeled with its original page number."
        )
    if delivery.image_pages_omitted:
        notices.append(
            f"Image coverage stops after page {delivery.image_coverage_end_page}; "
            f"{delivery.image_pages_omitted} later page image(s) were omitted because of the endpoint image limit."
        )
        notices.append("Those later pages remain available in the complete extracted document text.")
    if not notices:
        return []
    return [{"type": "input_text", "text": f"<document_delivery>\n{' '.join(notices)}\n</document_delivery>"}]


def _prompt_blocks(row: dict[str, Any]) -> list[dict[str, Any]]:
    content = row["responses_create_params"]["input"][0]["content"]
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    return [dict(block) for block in content if block.get("type") == "input_text"]


def _strip_image_blocks(result: SimpleAgentVerifyResponse) -> SimpleAgentVerifyResponse:
    """Remove input_image blocks from the serialized rollout result.

    Operates on a dict dump to avoid Pydantic model mutation/serialization issues,
    then re-validates into the response model.
    """

    removed_image = False

    def scrub(value: Any) -> Any:
        nonlocal removed_image
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "input_image":
                    removed_image = True
                    continue
                cleaned.append(scrub(item))
            return cleaned
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    data = scrub(result.model_dump(mode="json"))
    if removed_image:
        for key in ("ng_trajectory", "ng_agent_observations"):
            observations = data.get(key)
            if isinstance(observations, dict):
                observations.setdefault("gaps", []).append({"code": "multimodal_history_redacted"})
    return SimpleAgentVerifyResponse.model_validate(data)


class GdpPdfAgent(SimpleAgent):
    config: GdpPdfAgentConfig

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        resolved_base = Path(self.config.documents_base_dir)
        if not resolved_base.is_absolute():
            resolved_base = PARENT_DIR / resolved_base

        row = body.model_dump(exclude_unset=True)
        cookies = request.cookies
        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=row,
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        meta = row.get("verifier_metadata") or {}
        manifest_relpath = meta.get("document_manifest")
        task_id = str(row.get("_ng_task_index", "unknown"))
        rollout_id = self.rollout_id_from_run(body) or "unscoped"
        model_url_path = self.url_path_for_run("/v1/responses", body)
        collect_trajectory = self._model_call_capture_enabled()

        delivery = DocumentDelivery(self.config.dpi, self.config.max_images)
        terminal_limit: Optional[str] = None
        pages_truncated = 0
        trajectory = None

        if manifest_relpath and (self.config.include_text or self.config.include_images):
            manifest_path = resolved_base / manifest_relpath
            if not manifest_path.is_file():
                hint = "gym eval prepare --benchmark gdp_pdf"
                raise FileNotFoundError(
                    f"Document manifest not found: {manifest_path}\n"
                    f"Source documents are not committed. Fetch and render them with:\n  {hint}"
                )
            manifest = _load_manifest(manifest_path)
            manifest_source_dpi = int(manifest.get("source_dpi", self.config.dpi))
            if manifest_source_dpi != self.config.dpi:
                raise ValueError(
                    f"{manifest_path} was rendered at {manifest_source_dpi} DPI, but this agent's "
                    f"starting dpi is configured as {self.config.dpi}. Pages can only be resized "
                    "down from the cached screenshot, never rendered fresh at a higher DPI -- "
                    "either re-render with prepare.py at the configured DPI, or match dpi to it."
                )
            manifest_dir = manifest_path.parent

            prompt_blocks = _prompt_blocks(row)
            text_blocks: list[dict[str, Any]] = []
            if self.config.include_text:
                text_blocks, pages_truncated = _manifest_text_blocks(manifest, max_pages=self.config.max_pages)

            while True:
                image_blocks: list[dict[str, Any]] = []
                if self.config.include_images:
                    image_blocks, image_pages_truncated = _render_manifest_images(
                        manifest,
                        manifest_dir,
                        delivery=delivery,
                        max_pages=self.config.max_pages,
                        source_dpi=manifest_source_dpi,
                        image_format=self.config.image_format,
                        jpeg_quality=self.config.jpeg_quality,
                    )
                    pages_truncated = max(pages_truncated, image_pages_truncated)
                delivery_notice_blocks = _delivery_notice_blocks(delivery)
                content = [*prompt_blocks, *delivery_notice_blocks, *text_blocks, *image_blocks]
                params_dict = body.responses_create_params.model_dump(exclude_unset=True)
                params_dict["input"] = [{"role": "user", "content": content}]
                params = NeMoGymResponseCreateParamsNonStreaming.model_validate(params_dict)
                try:
                    model_response, trajectory, _, cookies = await self._create_episode(
                        params,
                        model_url_path=model_url_path,
                        resources_server_cookies=cookies,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        collect_trajectory=collect_trajectory,
                    )
                except ClientResponseError as error:
                    limit, image_cap = _input_limit(error)
                    if limit is not None and delivery.adapt(limit, image_cap, min_dpi=self.config.min_dpi):
                        continue
                    if error.status in (401, 403, 429):
                        # Systemic (auth/rate-limit), not per-document -- every other row would
                        # fail identically, so surface it loudly rather than silently zeroing
                        # the whole run one row at a time.
                        raise
                    terminal_limit = limit or f"unclassified_http_{error.status}"
                    model_response = _terminal_failure_response(self.config.model_server.name)
                    break
                except Exception as error:  # noqa: BLE001
                    # A rare, per-document failure (e.g. an upstream multimodal-processor bug
                    # unrelated to input size, observed in vLLM itself: "AssertionError: Expected
                    # a cached item for mm_hash=..." and "Failed to apply NanoNemotronVLProcessor
                    # on data=...") must not take the whole evaluation run down over one document.
                    # Score this row as a zero attempt instead, same as a terminal input-limit.
                    print(
                        f"[gdp_pdf_agent] WARNING: unclassified failure on {manifest_relpath}: {error!r}", flush=True
                    )
                    terminal_limit = f"unclassified_{type(error).__name__}"
                    model_response = _terminal_failure_response(self.config.model_server.name)
                    break
                else:
                    # vLLM doesn't always reject an oversized request outright -- it can accept
                    # it and simply run out of room mid-generation instead (status=incomplete),
                    # which raises nothing for the except clauses above to catch. That's the same
                    # underlying problem as a rejected request (not enough room in the context
                    # window), so it gets the same reactive response: free up room by shrinking
                    # the image footprint and retry, rather than silently keeping the truncated
                    # answer. Only worth trying when images are actually part of the payload.
                    if (
                        model_response.status == "incomplete"
                        and self.config.include_images
                        and delivery.adapt("output_truncated", None, min_dpi=self.config.min_dpi)
                    ):
                        continue
                    break
        else:
            model_response, trajectory, _, cookies = await self._create_episode(
                body.responses_create_params,
                model_url_path=model_url_path,
                resources_server_cookies=cookies,
                task_id=task_id,
                rollout_id=rollout_id,
                collect_trajectory=collect_trajectory,
            )

        result = row | {"response": model_response.model_dump(mode="json")}
        if self.config.skip_verification:
            result.update(
                reward=0.0 if terminal_limit else float(self.config.skip_verification_reward),
                verification_skipped=True,
            )
        else:
            try:
                verified = await self.server_client.post(
                    server_name=self.config.resources_server.name, url_path="/verify", json=result, cookies=cookies
                )
                await raise_for_status(verified)
                result = await get_response_json(verified)
            except Exception as error:  # noqa: BLE001
                # Same reasoning as the model-call retry loop above: a /verify-side failure
                # (upstream judge outage, a malformed row tripping a resources-server bug, etc.)
                # must not take the whole evaluation run down over one row either. Score it as
                # a zero attempt and keep going -- the warning below is what makes a real
                # verifier bug (as opposed to a one-off transient failure) visible for follow-up.
                print(f"[gdp_pdf_agent] WARNING: /verify failed on {manifest_relpath}: {error!r}", flush=True)
                terminal_limit = terminal_limit or f"verify_failed_{type(error).__name__}"
                result["reward"] = 0.0

        result["document_delivery"] = delivery.record(terminal_limit) | {"rejected_attempts": delivery.attempts}
        if pages_truncated:
            print(f"[gdp_pdf_agent] WARNING: truncated {pages_truncated} page(s) of {manifest_relpath} (max_pages)")
            result.setdefault("verifier_metadata", meta)["pages_truncated"] = pages_truncated
        if terminal_limit:
            result["failure_reason"] = f"GDP.pdf terminal input limit: {terminal_limit}"
        if trajectory is not None:
            result["ng_trajectory"] = trajectory.model_dump(mode="json")

        result = SimpleAgentVerifyResponse.model_validate(result)
        if self.config.strip_images_from_output:
            result = _strip_image_blocks(result)
        return result


if __name__ == "__main__":
    GdpPdfAgent.run_webserver()
