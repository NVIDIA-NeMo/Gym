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

An agent session gets the manifest from its ``agent_context`` (``artifacts.document_manifest``, the ``agent`` role's
context in the task row) and runs the same delivery and backoff in the episode of its one activation; the agent loop
stays Simple Agent's. The delivery record goes into the response's ``document_delivery`` metadata, and the Environment
Server verifies the answer.
"""

import asyncio
import base64
import errno
import io
import json
import math
import os
import re
from dataclasses import dataclass, fields
from http import HTTPStatus
from pathlib import Path
from time import time
from typing import Any, Awaitable, Callable, Literal, Optional, TypeVar

from aiohttp import ClientResponseError
from fastapi import Request
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentOutcome,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    resolve_agent_context_artifact,
)
from nemo_gym.agents.agent_outcome import policy_timeout_failure
from nemo_gym.agents.utils.verify_call import masked_verify_failure, verify_with_transport_retry
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NgFailureEvent,
    failure_kinds,
)
from nemo_gym.server_runtime import AgentContextArtifact
from nemo_gym.trajectory import ObservationGap, TrajectoryRecord
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentSessionState,
    SimpleAgentVerifyResponse,
)


BASE_DPI = 150
MIN_DPI = 72
_COMPOSITE_COLUMNS = 2
_MAX_PAGES_PER_IMAGE = 4
# Document work (manifest reads, page decode, resize and encode) runs in at most this many worker threads at once. It
# is CPU-bound, and the bound leaves four threads of asyncio's default executor (min(32, cores + 4) threads) to its
# other users, such as aiohttp's DNS lookups.
_DOCUMENT_THREADS = max(1, min(32, (os.cpu_count() or 1) + 4) - 4)

_T = TypeVar("_T")


class GdpPdfAgentConfig(SimpleAgentConfig):
    documents_base_dir: str = Field(
        description="Base directory for resolving the document manifest (verifier_metadata.document_manifest on "
        "/run, the agent_context artifact in a session), relative to the Gym root. A session refuses a manifest "
        "that resolves outside it through a symlink, so name the real directory.",
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
    regenerate_on_incomplete_output: bool = Field(
        default=False,
        description="Regenerate the answer with a smaller image footprint when the policy response comes back "
        "incomplete. Off by default: an incomplete response means the model spent its output budget, so it is "
        "the row's result and goes to the verifier as is; regenerating would re-sample a model result. Turn it "
        "on only to reproduce the earlier behaviour for comparison. Each regeneration is counted in "
        "document_delivery.output_truncation_regenerations.",
    )
    mask_failed_attempts: bool = Field(
        default=False,
        description="For training: flag an attempt that scores zero because its input never fit (a context, "
        "image-count or payload limit left after the backoff) with mask_sample, so the trainer leaves it out of "
        "the loss; its zero says nothing about an answer. Keep it off for evaluation, where such an attempt counts "
        "as a zero. A failed policy call or /verify is masked either way.",
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
        self.text_only_fallback = False
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
            "text_only_fallback": self.text_only_fallback,
            "limit": limit,
        }

    def adaptation_counts(self) -> dict[str, int]:
        """Rejected requests by kind: input-limit rejections (the endpoint refused the request's size, a
        delivery negotiation) and regenerations of an incomplete answer (only with
        ``regenerate_on_incomplete_output``)."""
        regenerations = sum(attempt["limit"] == "output_truncated" for attempt in self.attempts)
        return {
            "input_limit_rejections": len(self.attempts) - regenerations,
            "output_truncation_regenerations": regenerations,
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
            # One last safety net: if pages still don't fit even at the DPI floor, drop images
            # entirely and retry text-only. Matches the reference provider's observed behavior
            # of sending long documents (e.g. 168-page 10-Ks) as pure text.
            if not self.text_only_fallback:
                self.text_only_fallback = True
                return True
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
    return _classify_input_limit(error.status, str(text))


def _session_model_error(response: NeMoGymResponse) -> Optional[tuple[int, str]]:
    """The status and upstream body of a model error that a session activation returned as a failed response.

    In a session, Simple Agent does not raise a model HTTP error: it returns a failed response classified as
    ``model_request_error`` with the status, and the upstream body in ``error.message``.
    """
    if response.status != "failed" or response.error is None:
        return None
    try:
        outcome = AgentOutcome.model_validate_json((response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY, ""))
    except ValidationError:
        return None
    if outcome.failure_kind != MODEL_REQUEST_ERROR or outcome.status_code is None:
        return None
    text = re.sub(r"^Model HTTP \d+: ", "", response.error.message)
    try:
        # Without a body, the message falls back to the reason phrase; that is the empty body _input_limit sees.
        if text.strip().lower() == HTTPStatus(outcome.status_code).phrase.lower():
            text = ""
    except ValueError:
        pass
    return outcome.status_code, text


def _is_empty_server_error(status: int, text: str) -> bool:
    """A server error without a body: taken for a payload limit, though no provider said so."""
    return status >= 500 and not text.strip()


def _classify_input_limit(status: int, text: str) -> tuple[Optional[str], Optional[int]]:
    """The input limit a model error's status and body report, and the image cap it names, if any."""
    text = text.lower()
    if status in (401, 403, 429) or any(
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
    if status == 413 or any(
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
    if _is_empty_server_error(status, text):
        # An empty body on a server error is exactly what an oversized-request crash on the
        # inference server looks like when nothing survives to describe why (seen in practice:
        # vLLM returning a 500 with response_content=b'' under an oversized multi-image prompt).
        # Retrying via the same DPI backoff as a real "payload" limit is far cheaper than letting
        # this escape unclassified and take down the whole run over what was likely one document.
        return "payload", None
    return None, None


def _failure_kind(terminal_limit: str) -> str:
    """The shared failure kind for an attempt the agent ended without a gradable answer."""
    if terminal_limit == "context":
        return failure_kinds.MODEL_CONTEXT_LIMIT
    return failure_kinds.MODEL_REQUEST_ERROR


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
    """Load a prepare-time document manifest, sorting pages by page number.

    Raises ValueError for a manifest that is not a JSON object, or that names a page image outside its own
    directory: page images go to the model, so a manifest cannot point at other files on the host.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} is not a JSON object")
    manifest["pages"] = sorted(manifest.get("pages") or [], key=lambda page: page["page_number"])
    manifest_dir = manifest_path.parent.resolve()
    for page in manifest["pages"]:
        if "image" in page and not (manifest_dir / page["image"]).resolve().is_relative_to(manifest_dir):
            raise ValueError(f"{manifest_path} names page image {page['image']!r}, which is outside {manifest_dir}")
    return manifest


# Errors the manifest path itself causes, which a retry meets again (the set resolve_agent_context_artifact refuses).
_MANIFEST_PATH_ERRNOS = frozenset(
    {errno.EACCES, errno.EPERM, errno.ENAMETOOLONG, errno.ELOOP, errno.ENOTDIR, errno.ENOENT, errno.EISDIR}
)


def _unusable_page(
    manifest: dict[str, Any],
    manifest_dir: Path,
    *,
    include_text: bool,
    include_images: bool,
    max_pages: Optional[int],
) -> Optional[str]:
    """Why a loaded manifest cannot be delivered, or None: a page whose text or page image the delivery would read
    is missing, so every attempt would fail the same way."""
    for page in manifest["pages"][:max_pages]:
        if include_text and not isinstance(page.get("text"), str):
            return f"page {page['page_number']} has no extracted text"
        if include_images and not (isinstance(page.get("image"), str) and (manifest_dir / page["image"]).is_file()):
            return f"page {page['page_number']} names no page image file under {manifest_dir}"
    return None


def _manifest_text_blocks(manifest: dict[str, Any], *, max_pages: Optional[int]) -> tuple[list[dict[str, Any]], int]:
    """Extracted text doesn't change across retry attempts (unlike page images, which get resized
    smaller on a DPI backoff), so this only needs to run once per row. Returns ``(content_blocks,
    pages_truncated)`` -- at most one ``input_text`` block, empty if nothing extractable.

    Pages are joined with explicit ``## Page N`` markers so the model can cite page numbers and
    anchor text to visual page boundaries -- matches the reference provider's formatting."""
    pages = manifest["pages"]
    total_pages = len(pages)
    page_limit = total_pages if max_pages is None else min(max_pages, total_pages)
    pages_truncated = total_pages - page_limit
    kept = pages[:page_limit]
    if not any(p["text"].strip() for p in kept):
        return [], pages_truncated
    parts = []
    for p in kept:
        page_number = p.get("page_number", len(parts) + 1)
        parts.append(f"## Page {page_number}\n\n{p['text'].strip()}")
    return [{"type": "input_text", "text": "\n\n".join(parts)}], pages_truncated


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


def _build_task_text_block(
    prompt_blocks: list[dict[str, Any]],
    delivery_notice_blocks: list[dict[str, Any]],
    page_text_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fold prompt + delivery notice + page-marked extracted text into ONE ``input_text`` block,
    matching the reference provider's request shape. Preamble+prompt+extracted text all live in
    a single string, page-marked, prepended by the standard 'You are answering...' preamble."""
    prompt_text = "\n\n".join(b["text"] for b in prompt_blocks if b.get("text")).strip()
    delivery_notice = "\n\n".join(b["text"] for b in delivery_notice_blocks if b.get("text")).strip()
    extracted = "\n\n".join(b["text"] for b in page_text_blocks if b.get("text")).strip()
    parts = ["You are answering a task using text extracted from the source PDF.", "Task:\n" + prompt_text]
    if delivery_notice:
        parts.append(delivery_notice)
    if extracted:
        parts.append("Extracted PDF text:\n" + extracted)
    return {"type": "input_text", "text": "\n\n".join(parts)}


def _scrub_image_blocks(value: Any) -> tuple[Any, bool]:
    """``value`` without its input_image blocks, and whether it had any."""
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

    return scrub(value), removed_image


def _strip_image_blocks(result: SimpleAgentVerifyResponse) -> SimpleAgentVerifyResponse:
    """Remove input_image blocks from the serialized rollout result.

    Operates on a dict dump to avoid Pydantic model mutation/serialization issues,
    then re-validates into the response model.
    """
    data, removed_image = _scrub_image_blocks(result.model_dump(mode="json"))
    if removed_image:
        for key in ("ng_trajectory", "ng_agent_observations"):
            observations = data.get(key)
            if isinstance(observations, dict):
                observations.setdefault("gaps", []).append({"code": "multimodal_history_redacted"})
    return SimpleAgentVerifyResponse.model_validate(data)


def _strip_trajectory_images(trajectory: TrajectoryRecord) -> TrajectoryRecord:
    """The trajectory without the base64 page images its conversations carry, with a gap that says so."""
    data, removed_image = _scrub_image_blocks(trajectory.model_dump(mode="json"))
    stripped = TrajectoryRecord.model_validate(data)
    if removed_image:
        stripped.gaps.append(ObservationGap(code="multimodal_history_redacted"))
    return stripped


# The response metadata key a session's delivery record goes in, the one a /run result carries as a field.
DOCUMENT_DELIVERY_METADATA_KEY = "document_delivery"


class GdpPdfAgentContext(BaseModel):
    """The ``agent_context`` a GDP.pdf task carries: its document manifest."""

    model_config = ConfigDict(extra="forbid")

    document_manifest: AgentContextArtifact


@dataclass
class GdpPdfSessionState(SimpleAgentSessionState):
    # The task's document manifest, resolved under documents_base_dir when the session is seeded.
    manifest_path: Optional[Path] = None


class GdpPdfAgent(SimpleAgent):
    config: GdpPdfAgentConfig
    agent_context_model = GdpPdfAgentContext
    _document_threads: asyncio.Semaphore = PrivateAttr(default_factory=lambda: asyncio.Semaphore(_DOCUMENT_THREADS))

    def _documents_base_dir(self) -> Path:
        base = Path(self.config.documents_base_dir)
        return base if base.is_absolute() else PARENT_DIR / base

    async def _in_thread(self, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        """Run blocking document work in a worker thread, so that rendering a long document does not stall the event
        loop that seeds, activates and closes every other session on this worker."""
        async with self._document_threads:
            return await asyncio.to_thread(function, *args, **kwargs)

    def _new_session_state(self, body: AgentSeedSessionRequest) -> GdpPdfSessionState:
        # A restore rebuilds sessions through here too, so a restored session keeps its document.
        state = super()._new_session_state(body)
        context = self.parse_agent_context(body)
        manifest_path = resolve_agent_context_artifact(
            context.document_manifest, self._documents_base_dir(), kind="file"
        )
        try:
            manifest = _load_manifest(manifest_path)
            source_dpi = int(manifest.get("source_dpi", self.config.dpi))
            problem = _unusable_page(
                manifest,
                manifest_path.parent,
                include_text=self.config.include_text,
                include_images=self.config.include_images,
                max_pages=self.config.max_pages,
            )
        except OSError as error:
            # A storage error (EIO, ESTALE) may pass, so it fails the seed as a transient error; only an error the
            # path itself causes refuses it, as resolve_agent_context_artifact does.
            if error.errno not in _MANIFEST_PATH_ERRNOS:
                raise
            raise AgentSeedRejectedError(f"document manifest {manifest_path} cannot be read: {error}") from None
        except (ValueError, KeyError, TypeError) as error:
            # Every retry would parse the same file, so a malformed manifest refuses the seed.
            raise AgentSeedRejectedError(
                f"document manifest {manifest_path} cannot be read: {type(error).__name__}: {error}"
            ) from None
        if problem is not None:
            # Every activation would fail on the same page, so refuse the seed instead of answering without it.
            raise AgentSeedRejectedError(f"document manifest {manifest_path}: {problem}")
        if source_dpi != self.config.dpi:
            raise AgentSeedRejectedError(
                f"{manifest_path} was rendered at {source_dpi} DPI, but this agent's starting dpi is "
                f"{self.config.dpi}: re-render the document at the configured DPI, or match dpi to it"
            )
        return GdpPdfSessionState(
            **{field.name: getattr(state, field.name) for field in fields(state)}, manifest_path=manifest_path
        )

    async def _create_session_episode(
        self, state: SimpleAgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming, **kwargs: Any
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        create_episode = super()._create_session_episode
        manifest_path = state.manifest_path if isinstance(state, GdpPdfSessionState) else None
        if manifest_path is None or not (self.config.include_text or self.config.include_images):
            return await create_episode(state, body, **kwargs)
        return await self._answer_with_document(
            body, manifest_path, lambda params: create_episode(state, params, **kwargs)
        )

    def _document_request(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        manifest: dict[str, Any],
        manifest_dir: Path,
        delivery: DocumentDelivery,
        prompt_blocks: list[dict[str, Any]],
        text_blocks: list[dict[str, Any]],
    ) -> tuple[NeMoGymResponseCreateParamsNonStreaming, int]:
        """The request for one delivery attempt, and the pages its images leave out (max_pages)."""
        image_blocks: list[dict[str, Any]] = []
        image_pages_truncated = 0
        # Text-only fallback (delivery.text_only_fallback=True) suppresses image render
        # after the DPI floor was hit and pages still didn't fit -- see adapt().
        if self.config.include_images and not delivery.text_only_fallback:
            image_blocks, image_pages_truncated = _render_manifest_images(
                manifest,
                manifest_dir,
                delivery=delivery,
                max_pages=self.config.max_pages,
                source_dpi=int(manifest.get("source_dpi", self.config.dpi)),
                image_format=self.config.image_format,
                jpeg_quality=self.config.jpeg_quality,
            )
        delivery_notice_blocks = _delivery_notice_blocks(delivery)
        # Reference request shape: images FIRST, then a single unified text block that
        # contains the preamble + user prompt + extracted PDF text with ## Page N markers.
        task_text_block = _build_task_text_block(prompt_blocks, delivery_notice_blocks, text_blocks)
        params_dict = params.model_dump(exclude_unset=True)
        params_dict["input"] = [{"role": "user", "content": [*image_blocks, task_text_block]}]
        return NeMoGymResponseCreateParamsNonStreaming.model_validate(params_dict), image_pages_truncated

    async def _answer_with_document(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        manifest_path: Path,
        create_episode: Callable[[NeMoGymResponseCreateParamsNonStreaming], Awaitable[tuple]],
    ) -> tuple:
        """Run a session activation's episode with the document, backing off as run() does.

        An input-limit rejection that no smaller delivery fixes returns an empty failed response, which the
        Environment Server verifies as a zero attempt, as run() scores it. Any other model error stays the
        classified failure Simple Agent returns, for the Environment Server's outcome policy; so does an empty-body
        server error on the last, text-only attempt, since it may be an outage rather than a limit.

        Every attempt passes the same checkpoint activation to ``create_episode``. That is safe because this agent
        runs no tools and so never commits a boundary past step 0: a restore restarts the backoff at full DPI.
        """
        manifest = await self._in_thread(_load_manifest, manifest_path)
        prompt_blocks = _prompt_blocks({"responses_create_params": params.model_dump(exclude_unset=True)})
        text_blocks: list[dict[str, Any]] = []
        pages_truncated = 0
        if self.config.include_text:
            text_blocks, pages_truncated = await self._in_thread(
                _manifest_text_blocks, manifest, max_pages=self.config.max_pages
            )
        delivery = DocumentDelivery(self.config.dpi, self.config.max_images)
        terminal_limit: Optional[str] = None
        while True:
            request, image_pages_truncated = await self._in_thread(
                self._document_request, params, manifest, manifest_path.parent, delivery, prompt_blocks, text_blocks
            )
            pages_truncated = max(pages_truncated, image_pages_truncated)
            model_response, trajectory, model_cookies, resources_cookies = await create_episode(request)
            model_error = _session_model_error(model_response)
            limit, image_cap = _classify_input_limit(*model_error) if model_error is not None else (None, None)
            if limit is not None:
                if delivery.adapt(limit, image_cap, min_dpi=self.config.min_dpi):
                    continue
                if not _is_empty_server_error(*model_error):
                    terminal_limit = limit
                    model_response = _terminal_failure_response(self.config.model_server.name)
                # Otherwise the last rejection was only an empty server error, which an outage looks like too:
                # the classified model_request_error stays the result, for the outcome policy to judge.
            elif (
                model_response.status == "incomplete"
                and self.config.regenerate_on_incomplete_output
                and self.config.include_images
                and delivery.adapt("output_truncated", None, min_dpi=self.config.min_dpi)
            ):
                continue
            break
        record = (
            delivery.record(terminal_limit)
            | {"rejected_attempts": delivery.attempts, "pages_truncated": pages_truncated}
            | delivery.adaptation_counts()
        )
        if terminal_limit and self.config.mask_failed_attempts:
            # The Environment Server returns a session's row, so gdp_pdf's verify() applies this mark as run() would:
            # mask_sample and failure_kind for an attempt whose input never fit.
            record |= {"mask_failed_attempt": True, "failure_kind": _failure_kind(terminal_limit)}
        model_response = model_response.model_copy(
            update={
                "metadata": {**(model_response.metadata or {}), DOCUMENT_DELIVERY_METADATA_KEY: json.dumps(record)}
            }
        )
        if trajectory is not None and self.config.strip_images_from_output:
            trajectory = _strip_trajectory_images(trajectory)
        return model_response, trajectory, model_cookies, resources_cookies

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
            manifest = await self._in_thread(_load_manifest, manifest_path)
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
                text_blocks, pages_truncated = await self._in_thread(
                    _manifest_text_blocks, manifest, max_pages=self.config.max_pages
                )

            while True:
                params, image_pages_truncated = await self._in_thread(
                    self._document_request,
                    body.responses_create_params,
                    manifest,
                    manifest_dir,
                    delivery,
                    prompt_blocks,
                    text_blocks,
                )
                pages_truncated = max(pages_truncated, image_pages_truncated)
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
                    if limit is None:
                        # Not a limit of the document: a policy call that failed is the evaluation's fault, not a model
                        # zero. A confirmed response timeout is typed as the policy's; any other failure (5xx, 429,
                        # 401/403, transport) leaves with its typed model failure, so the agent boundary masks the
                        # row and Collection retries it within its budget.
                        timeout = policy_timeout_failure(error)
                        if timeout is not None:
                            raise timeout from error
                        raise
                    terminal_limit = limit
                    model_response = _terminal_failure_response(self.config.model_server.name)
                    break
                else:
                    # An incomplete response means the model spent its output budget: an input that
                    # does not fit is rejected up front (handled above), so this is the model's
                    # result and is verified as is. Regenerating it with a smaller image footprint
                    # re-samples a model result, so it only happens when explicitly enabled.
                    if (
                        model_response.status == "incomplete"
                        and self.config.regenerate_on_incomplete_output
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
            request = result

            async def verify() -> dict[str, Any]:
                verified = await self.server_client.post(
                    server_name=self.config.resources_server.name, url_path="/verify", json=request, cookies=cookies
                )
                await raise_for_status(verified)
                return await get_response_json(verified)

            try:
                # A reply lost in transport is tried again before the row is masked (verify_with_transport_retry).
                result = await verify_with_transport_retry(
                    verify, server_client=self.server_client, server_name=self.config.resources_server.name
                )
            except NgFailureEvent as failure:
                # Return the real generated answer on the established masked-failure path. Raising
                # here would discard it at the HTTP boundary and make a resume generate again. Every
                # attempt was observed as its own agent route failure of operation verify.
                result = masked_verify_failure(result, failure)

        adaptations = delivery.adaptation_counts()
        result["document_delivery"] = (
            delivery.record(terminal_limit) | {"rejected_attempts": delivery.attempts} | adaptations
        )
        # The generic per-row counts the run's failure facts sum (recovery.failure_facts).
        result["agent_adaptations"] = adaptations
        if pages_truncated:
            print(f"[gdp_pdf_agent] WARNING: truncated {pages_truncated} page(s) of {manifest_relpath} (max_pages)")
            result.setdefault("verifier_metadata", meta)["pages_truncated"] = pages_truncated
        if terminal_limit and not result.get("mask_sample"):
            result["failure_reason"] = f"GDP.pdf terminal input limit: {terminal_limit}"
            if self.config.mask_failed_attempts:
                result["mask_sample"] = True
                result["failure_kind"] = _failure_kind(terminal_limit)
                # NeMo RL versions that predate the top-level field read instance_config.
                result["instance_config"] = {**(result.get("instance_config") or {}), "mask_sample": True}
        if trajectory is not None:
            result["ng_trajectory"] = trajectory.model_dump(mode="json")

        result = SimpleAgentVerifyResponse.model_validate(result)
        if self.config.strip_images_from_output:
            result = _strip_image_blocks(result)
        return result


if __name__ == "__main__":
    GdpPdfAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = GdpPdfAgent.run_webserver()  # noqa: F401
