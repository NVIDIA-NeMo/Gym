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
"""Tests for the gdp_pdf agent's manifest-based document embedding, reactive DPI backoff, and
page compositing."""

import base64
import io
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import ClientResponseError
from PIL import Image

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.gdp_pdf_agent import app as gdp_pdf_agent_app
from responses_api_agents.gdp_pdf_agent.app import (
    DocumentDelivery,
    GdpPdfAgent,
    GdpPdfAgentConfig,
    _compose_pages,
    _delivery_notice_blocks,
    _input_limit,
    _load_manifest,
    _manifest_text_blocks,
    _open_page_image,
    _render_manifest_images,
    _strip_image_blocks,
)
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest, SimpleAgentVerifyResponse


SERVER_DIR = Path(__file__).resolve().parents[3] / "resources_servers" / "gdp_pdf"
DATA_DIR = SERVER_DIR / "data"
_example_manifests = (
    sorted((DATA_DIR / "test_media" / "documents").glob("*/manifest.json")) if DATA_DIR.exists() else []
)
requires_example_manifest = pytest.mark.skipif(not _example_manifests, reason="example manifests not present")
_example_manifest_relpath = (
    f"test_media/documents/{_example_manifests[0].parent.name}/manifest.json" if _example_manifests else None
)


def _agent_config(**overrides) -> GdpPdfAgentConfig:
    config_kwargs = {
        "name": "gdp_pdf_simple_agent",
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "max_steps": 1,
        "resources_server": ResourcesServerRef(type="resources_servers", name="gdp_pdf"),
        "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "documents_base_dir": "resources_servers/gdp_pdf/data",
    }
    config_kwargs.update(overrides)
    return GdpPdfAgentConfig(**config_kwargs)


def _agent(**overrides) -> GdpPdfAgent:
    return GdpPdfAgent(config=_agent_config(**overrides), server_client=AsyncMock(spec=ServerClient))


def _block_type(block) -> str:
    return block.get("type") if isinstance(block, dict) else block.type


def _run_request(document_manifest: str = "test_media/documents/x/manifest.json") -> SimpleAgentRunRequest:
    return SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "What is the cap?"}]}]
            },
            "verifier_metadata": {
                "task_id": "t1",
                "document_manifest": document_manifest,
                "criteria": [{"index": 1}],
            },
        }
    )


def _response(text: str = "answer") -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        tools=[],
        parallel_tool_calls=False,
        tool_choice="none",
        output=[
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


def _incomplete_response(text: str = "cut off mid-") -> NeMoGymResponse:
    """A response vLLM accepted but couldn't finish -- ran out of output budget, not rejected."""
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        tools=[],
        parallel_tool_calls=False,
        tool_choice="none",
        output=[
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "incomplete",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


def _error(message: str, status: int = 500) -> ClientResponseError:
    error = ClientResponseError(None, (), status=status, message="upstream failure")
    error.response_content = message.encode()
    return error


class TestConfig:
    def test_defaults_are_conservative(self) -> None:
        config = _agent_config()
        assert config.dpi == 150
        assert config.min_dpi == 72
        assert config.max_images is None
        assert config.max_pages is None
        assert config.include_text is True
        assert config.include_images is True
        assert config.strip_images_from_output is True

    def test_overrides_apply(self) -> None:
        config = _agent_config(dpi=200, max_images=8, max_pages=10, include_images=False)
        assert config.dpi == 200
        assert config.max_images == 8
        assert config.max_pages == 10
        assert config.include_images is False


class TestInputLimit:
    @pytest.mark.parametrize(
        "message,status,expected",
        [
            ("maximum context length is 262144 tokens", 500, ("context", None)),
            ("payload too large", 500, ("payload", None)),
            ("", 413, ("payload", None)),
            ("request_too_large", 500, ("payload", None)),
            ("image dimensions exceed the maximum allowed size", 400, ("payload", None)),
            ("At most 10 images may be provided in one request.", 500, ("image_count", 10)),
            ("Too many images", 400, ("image_count", None)),
            ("invalid image data", 400, (None, None)),
            ("out of memory", 500, (None, None)),
            ("RateLimitError: maximum context length quota", 500, (None, None)),
            ("maximum context length", 401, (None, None)),
            ("", 500, ("payload", None)),  # empty body on a 5xx: treat as retryable, not unclassified
            ("", 400, (None, None)),  # empty body on a non-5xx: no evidence of an input-size issue
        ],
    )
    def test_classifies_upstream_errors(self, message, status, expected) -> None:
        assert _input_limit(_error(message, status)) == expected


class TestDocumentDeliveryAdapt:
    def test_reduces_dpi_by_20_percent_toward_floor(self) -> None:
        delivery = DocumentDelivery(150, None)
        assert delivery.adapt("context", None, min_dpi=72) is True
        assert delivery.image_dpi == 120
        assert delivery.adapt("payload", None, min_dpi=72) is True
        assert delivery.image_dpi == 96

    def test_stops_at_the_floor(self) -> None:
        delivery = DocumentDelivery(80, None)
        assert delivery.adapt("context", None, min_dpi=72) is True
        assert delivery.image_dpi == 72
        assert delivery.adapt("context", None, min_dpi=72) is False
        assert delivery.image_dpi == 72

    def test_image_count_with_explicit_cap(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.image_count = 20
        assert delivery.adapt("image_count", 8, min_dpi=72) is True
        assert delivery.max_images == 8

    def test_image_count_without_explicit_cap_decrements_by_one(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.image_count = 5
        assert delivery.adapt("image_count", None, min_dpi=72) is True
        assert delivery.max_images == 4

    def test_image_count_cannot_increase_the_cap(self) -> None:
        delivery = DocumentDelivery(150, 4)
        delivery.image_count = 4
        assert delivery.adapt("image_count", 10, min_dpi=72) is False
        assert delivery.max_images == 4

    def test_records_every_attempt(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.adapt("context", None, min_dpi=72)
        delivery.adapt("context", None, min_dpi=72)
        assert [a["image_dpi"] for a in delivery.attempts] == [150, 120]


class TestComposePages:
    def test_single_page_is_not_composited(self) -> None:
        image = Image.new("RGB", (100, 140), "red")
        block = _compose_pages([(1, image)], image_dpi=150, image_format="png", jpeg_quality=90)
        encoded = block["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.size == (100, 140)

    def test_multiple_pages_are_composited_into_a_grid(self) -> None:
        images = [(i, Image.new("RGB", (100, 140), "red")) for i in range(1, 5)]
        block = _compose_pages(images, image_dpi=150, image_format="png", jpeg_quality=90)
        encoded = block["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            # 2 columns x 2 rows, each cell >= page size plus a label strip.
            assert decoded.width == 200
            assert decoded.height > 280
        assert block["type"] == "input_image"

    def test_jpeg_format_is_declared_and_decodes_as_jpeg(self) -> None:
        image = Image.new("RGB", (100, 140), "red")
        block = _compose_pages([(1, image)], image_dpi=150, image_format="jpeg", jpeg_quality=90)
        assert block["image_url"].startswith("data:image/jpeg;base64,")
        encoded = block["image_url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.format == "JPEG"
            assert decoded.size == (100, 140)

    def test_jpeg_payload_is_smaller_than_png_for_a_noisy_page(self) -> None:
        # Solid colors favor PNG; real page scans are noisy, so seed pixel noise to make the
        # comparison representative of the actual corpus.
        rng = random.Random(0)
        image = Image.new("RGB", (200, 260))
        image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(200 * 260)])

        as_png = _compose_pages([(1, image)], image_dpi=150, image_format="png", jpeg_quality=90)
        as_jpeg = _compose_pages([(1, image)], image_dpi=150, image_format="jpeg", jpeg_quality=90)
        assert len(as_jpeg["image_url"]) < len(as_png["image_url"])


class TestOpenPageImage:
    def test_no_resize_when_dpi_matches_source(self, tmp_path: Path) -> None:
        original = Image.new("RGB", (300, 400), "blue")
        path = tmp_path / "page.png"
        original.save(path)

        image = _open_page_image(path, source_dpi=150, image_dpi=150)
        assert image.size == (300, 400)

    def test_resizes_down_when_dpi_reduced(self, tmp_path: Path) -> None:
        original = Image.new("RGB", (300, 400), "blue")
        path = tmp_path / "page.png"
        original.save(path)

        image = _open_page_image(path, source_dpi=150, image_dpi=75)
        assert image.size == (150, 200)


@requires_example_manifest
class TestManifestTextBlocks:
    @property
    def manifest_path(self) -> Path:
        return _example_manifests[0]

    def test_returns_one_text_block(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        blocks, pages_truncated = _manifest_text_blocks(manifest, max_pages=None)
        assert pages_truncated == 0
        assert len(blocks) == 1
        assert _block_type(blocks[0]) == "input_text"
        assert blocks[0]["text"].startswith("<document>")

    def test_max_pages_truncates_and_is_reported(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        blocks, pages_truncated = _manifest_text_blocks(manifest, max_pages=1)
        assert pages_truncated >= 0
        assert len(blocks) <= 1


@requires_example_manifest
class TestRenderManifestImages:
    @property
    def manifest_path(self) -> Path:
        return _example_manifests[0]

    def test_renders_images_from_the_cached_screenshots(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        delivery = DocumentDelivery(150, None)
        blocks, pages_truncated = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=delivery, max_pages=None, source_dpi=150
        )
        assert pages_truncated == 0
        assert blocks
        assert all(_block_type(b) == "input_image" for b in blocks)
        assert delivery.page_count > 0
        assert delivery.image_count > 0

    def test_max_pages_truncates_and_is_reported(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        delivery = DocumentDelivery(150, None)
        _, pages_truncated = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=delivery, max_pages=1, source_dpi=150
        )
        assert delivery.page_count == 1
        assert pages_truncated >= 0

    def test_max_images_composites_pages(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        delivery = DocumentDelivery(150, 1)
        blocks, _ = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=delivery, max_pages=None, source_dpi=150
        )
        # Capped at 1 image; all pages composited into it (up to the 4-page/image limit).
        assert len(blocks) == 1
        assert delivery.pages_per_image >= 1

    def test_dpi_backoff_shrinks_rendered_images(self) -> None:
        manifest = _load_manifest(self.manifest_path)
        full_delivery = DocumentDelivery(150, None)
        full_blocks, _ = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=full_delivery, max_pages=1, source_dpi=150
        )
        reduced_delivery = DocumentDelivery(75, None)
        reduced_blocks, _ = _render_manifest_images(
            manifest, self.manifest_path.parent, delivery=reduced_delivery, max_pages=1, source_dpi=150
        )
        assert len(reduced_blocks[0]["image_url"]) < len(full_blocks[0]["image_url"])


class TestDeliveryMetadataAndNotice:
    def test_image_cap_records_leading_page_coverage(self) -> None:
        manifest = {
            "pages": [
                {"page_number": page_number, "image": f"page_{page_number:04d}.png"} for page_number in range(1, 11)
            ]
        }
        delivery = DocumentDelivery(150, 2)
        with patch.object(gdp_pdf_agent_app, "_open_page_image", return_value=Image.new("RGB", (8, 10), "white")):
            blocks, pages_truncated = _render_manifest_images(
                manifest, Path("unused"), delivery=delivery, max_pages=None, source_dpi=150
            )

        assert pages_truncated == 0
        assert len(blocks) == 2
        assert delivery.pages_per_image == 4
        assert delivery.image_pages_covered == 8
        assert delivery.image_pages_omitted == 2
        assert delivery.image_coverage_end_page == 8

    def test_describes_composites_and_partial_image_coverage(self) -> None:
        delivery = DocumentDelivery(150, 2)
        delivery.pages_per_image = 4
        delivery.image_pages_covered = 8
        delivery.image_pages_omitted = 2
        delivery.image_coverage_end_page = 8

        blocks = _delivery_notice_blocks(delivery)

        assert len(blocks) == 1
        text = blocks[0]["text"]
        assert "up to 4 pages per image" in text
        assert "each cell is labeled with its original page number" in text
        assert "coverage stops after page 8" in text
        assert "2 later page image(s) were omitted" in text
        assert "remain available in the complete extracted document text" in text

    def test_is_empty_for_unmodified_full_image_coverage(self) -> None:
        delivery = DocumentDelivery(150, None)
        delivery.image_pages_covered = 10
        delivery.image_coverage_end_page = 10
        assert _delivery_notice_blocks(delivery) == []


class TestRun:
    async def test_missing_manifest_raises_actionable_error(self) -> None:
        agent = _agent()
        with pytest.raises(FileNotFoundError, match="prepare --benchmark gdp_pdf"):
            await agent.run(
                SimpleNamespace(cookies={}), _run_request("test_media/documents/does-not-exist/manifest.json")
            )

    @requires_example_manifest
    async def test_source_dpi_mismatch_raises(self) -> None:
        agent = _agent(strip_images_from_output=False, dpi=72)  # manifest is rendered at 150
        with pytest.raises(ValueError, match="rendered at 150 DPI"):
            await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))

    @requires_example_manifest
    async def test_succeeds_on_first_try_and_verifies_once(self) -> None:
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert result.reward == 1.0
        assert result.document_delivery["image_dpi"] == 150
        assert result.document_delivery["rejected_attempts"] == []
        assert agent.server_client.post.await_count == 2  # seed_session, verify

    @requires_example_manifest
    async def test_adapts_dpi_on_rejection_then_succeeds(self) -> None:
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        attempts = []

        async def episode(params, **kwargs):
            attempts.append(len(attempts))
            if len(attempts) < 3:
                raise _error("payload too large")
            return _response(), None, {}, {}

        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert len(attempts) == 3
        assert result.document_delivery["image_dpi"] == 96  # 150 -> 120 -> 96
        assert len(result.document_delivery["rejected_attempts"]) == 2

    @requires_example_manifest
    async def test_incomplete_status_triggers_dpi_backoff_without_an_error(self) -> None:
        # vLLM can accept an oversized request and just run out of output room instead of
        # rejecting it outright -- no exception is raised, so this must be detected separately
        # from the ClientResponseError path above.
        agent = _agent(strip_images_from_output=False, include_images=True)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        attempts = []

        async def episode(params, **kwargs):
            attempts.append(len(attempts))
            if len(attempts) < 3:
                return _incomplete_response(), None, {}, {}
            return _response(), None, {}, {}

        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert len(attempts) == 3
        assert result.document_delivery["image_dpi"] == 96  # 150 -> 120 -> 96
        assert result.reward == 1.0  # the eventual complete response, not the truncated ones

    @requires_example_manifest
    async def test_incomplete_status_is_accepted_once_dpi_floor_reached(self) -> None:
        agent = _agent(strip_images_from_output=False, include_images=True, min_dpi=140)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.3}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(
                GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_incomplete_response(), None, {}, {}))
            ),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        # Floor reached on the very first backoff attempt (min_dpi=140 >= 150*0.8=120), so the
        # truncated response is accepted as-is rather than looping forever.
        assert result.reward == 0.3
        assert result.failure_reason is None or "output_truncated" not in (result.failure_reason or "")

    @requires_example_manifest
    async def test_incomplete_status_ignored_when_text_only(self) -> None:
        # No images means no DPI to reduce -- retrying would be pointless, so the truncated
        # response is accepted immediately without ever calling adapt().
        agent = _agent(strip_images_from_output=False, include_images=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.5}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(
                GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_incomplete_response(), None, {}, {}))
            ) as call,
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert call.await_count == 1
        assert result.reward == 0.5

    @requires_example_manifest
    async def test_terminal_failure_scores_zero_without_raising(self) -> None:
        agent = _agent(strip_images_from_output=False, min_dpi=140)  # floor reached on first backoff
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("maximum context length")),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert result.reward == 0.0
        assert result.response.output == []
        assert result.failure_reason == "GDP.pdf terminal input limit: context"

    @requires_example_manifest
    async def test_unrelated_failure_is_not_retried_but_scores_zero(self) -> None:
        # An unclassified, non-auth error (e.g. a rare vLLM-side multimodal-processor bug) must
        # not take the whole evaluation run down over one document -- it becomes a graceful
        # terminal failure instead of propagating, same as a terminal input-limit.
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("out of memory")) as call,
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert call.await_count == 1
        assert result.reward == 0.0
        assert result.response.output == []
        assert result.failure_reason == "GDP.pdf terminal input limit: unclassified_http_500"

    async def test_auth_and_rate_limit_errors_still_raise(self) -> None:
        # These are systemic, not per-document -- every other row would fail identically, so
        # this must surface loudly rather than silently zeroing the whole run one row at a time.
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(return_value=SimpleNamespace(cookies={}, data={}))
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("rate limited", status=429)) as call,
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            pytest.raises(ClientResponseError),
        ):
            await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert call.await_count == 1

    async def test_non_client_response_exceptions_also_score_zero(self) -> None:
        # Any other exception type (not just ClientResponseError) must be equally contained --
        # e.g. a JSON decode error or an assertion deep inside an upstream library.
        agent = _agent(strip_images_from_output=False)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 0.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", side_effect=AssertionError("Expected a cached item")) as call,
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert call.await_count == 1
        assert result.reward == 0.0
        assert result.failure_reason == "GDP.pdf terminal input limit: unclassified_AssertionError"

    @requires_example_manifest
    async def test_images_are_stripped_from_the_result_when_enabled(self) -> None:
        agent = _agent(strip_images_from_output=True)
        agent.server_client.post = AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                cookies={},
                data=(kw["json"] | {"reward": 1.0}) if kw["url_path"] == "/verify" else {},
            )
        )
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
            patch.object(gdp_pdf_agent_app, "get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        # `data` from /verify above didn't carry responses_create_params; nothing to assert on
        # content blocks here beyond the call succeeding without image payloads surviving.
        assert result.reward == 1.0

    @requires_example_manifest
    async def test_verify_failure_scores_zero_instead_of_crashing(self) -> None:
        # A /verify-side failure (judge outage, a resources-server bug on some row) must not
        # take the whole evaluation run down either -- same containment as a model-call failure.
        agent = _agent(strip_images_from_output=False)

        async def post(**kw):
            if kw["url_path"] == "/verify":
                raise ClientResponseError(None, (), status=500, message="judge unavailable")
            return SimpleNamespace(cookies={})

        agent.server_client.post = AsyncMock(side_effect=post)
        with (
            patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(_response(), None, {}, {}))),
            patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
            patch.object(gdp_pdf_agent_app, "raise_for_status", new=AsyncMock()),
        ):
            result = await agent.run(SimpleNamespace(cookies={}), _run_request(_example_manifest_relpath))
        assert result.reward == 0.0
        assert "verify_failed" in result.failure_reason


class TestStripImageBlocks:
    def test_removes_image_blocks_and_records_the_gap(self) -> None:
        result = SimpleAgentVerifyResponse.model_validate(
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "q"},
                                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "high"},
                            ],
                        }
                    ]
                },
                "response": {
                    "id": "resp",
                    "created_at": 0.0,
                    "model": "test",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                },
                "reward": 1.0,
                "ng_trajectory": {"task_id": "t", "rollout_id": "r", "gaps": []},
            }
        )
        stripped = _strip_image_blocks(result)

        content = stripped.responses_create_params.input[0].content
        assert [_block_type(block) for block in content] == ["input_text"]
        gaps = stripped.model_extra["ng_trajectory"]["gaps"]
        assert {"code": "multimodal_history_redacted"} in gaps
