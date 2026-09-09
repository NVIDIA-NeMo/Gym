# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import ClientResponseError
from PIL import Image

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter
from responses_api_agents.gdp_pdf_agent.app import (
    _DOCUMENT_REDACTION_MARKER,
    DocumentDelivery,
    GdpPdfAgent,
    GdpPdfAgentConfig,
    _input_limit,
    _resolve_under,
    _strip_document_payloads,
    materialize_document,
)
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest, SimpleAgentVerifyResponse
from responses_api_models.inference_provider.app import InferenceProvider, InferenceProviderConfig


def _config(**overrides) -> GdpPdfAgentConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "gdp_pdf_agent",
        "resources_server": ResourcesServerRef(type="resources_servers", name="gdp_pdf"),
        "model_server": ModelServerRef(type="responses_api_models", name="policy"),
        "documents_base_dir": ".",
        "source_dpi": 150,
    }
    values.update(overrides)
    return GdpPdfAgentConfig(**values)


def _write_document(root: Path, page_count: int = 3) -> None:
    pages_dir = root / "document" / "pages"
    pages_dir.mkdir(parents=True)
    for page_number in range(1, page_count + 1):
        Image.new("RGB", (100, 120), "red").save(pages_dir / f"page_{page_number:04d}.png")
    manifest = {
        "source_dpi": 150,
        "page_count": page_count,
        "pages": [
            {"page_number": page_number, "text": f"text {page_number}", "image": f"pages/page_{page_number:04d}.png"}
            for page_number in range(1, page_count + 1)
        ],
    }
    (root / "document" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("page_count,cap,per_image,shown", [(3, None, 1, 3), (3, 2, 2, 3), (3, 1, 4, 3), (5, 1, 4, 4)])
def test_selects_composites_only_for_image_cap(tmp_path: Path, page_count, cap, per_image, shown) -> None:
    _write_document(tmp_path, page_count)
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {
            "task_prompt": "Do the analysis.",
            "document_manifest": "document/manifest.json",
        },
    }

    delivery = DocumentDelivery(cap)
    materialized = materialize_document(row, tmp_path, _config(), delivery)
    SimpleAgentRunRequest.model_validate(materialized)

    params = materialized["responses_create_params"]
    assert params["tools"] == []
    assert params["parallel_tool_calls"] is False
    content = params["input"][0]["content"]
    image_blocks = [block for block in content if block["type"] == "input_image"]
    assert len(image_blocks) == (shown + per_image - 1) // per_image
    assert delivery.pages_per_image == per_image
    assert delivery.image_pages == shown
    assert "150 DPI" in content[0]["text"]
    assert ("composite images" in content[0]["text"]) == (per_image > 1)
    if shown < page_count:
        assert f"Images cover only pages 1-{shown}" in content[0]["text"]
    assert all(f"<page {number}>\ntext {number}" in content[-1]["text"] for number in range(1, page_count + 1))
    assert row["responses_create_params"]["input"] == "seed"

    encoded = image_blocks[0]["image_url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == {1: (100, 120), 2: (200, 148), 4: (200, 296)}[per_image]


def test_text_only_profile_still_includes_every_page(tmp_path: Path) -> None:
    _write_document(tmp_path)
    for image_path in (tmp_path / "document" / "pages").iterdir():
        image_path.unlink()
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {
            "task_prompt": "Do the analysis.",
            "document_manifest": "document/manifest.json",
        },
    }

    materialized = materialize_document(row, tmp_path, _config(include_page_images=False))
    content = materialized["responses_create_params"]["input"][0]["content"]

    assert not any(block["type"] == "input_image" for block in content)
    assert "complete extracted text" in content[0]["text"]
    assert "text 3" in content[-1]["text"]


def test_redacts_complete_document_payload_from_artifacts() -> None:
    content = [
        {"type": "input_text", "text": "TASK:\nDo the analysis."},
        {"type": "input_image", "image_url": "data:image/png;base64,secret-image"},
        {"type": "input_text", "text": "SOURCE DOCUMENT TEXT:\nsecret document text"},
    ]
    result = SimpleAgentVerifyResponse(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": content}]),
        response=NeMoGymResponse(
            id="response",
            created_at=0,
            model="model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        reward=1.0,
        ng_trajectory={"gaps": [], "question": content},
    )

    redacted = _strip_document_payloads(result).model_dump(mode="json")
    serialized = json.dumps(redacted)

    assert "secret document text" not in serialized
    assert "secret-image" not in serialized
    assert _DOCUMENT_REDACTION_MARKER in serialized
    assert redacted["ng_trajectory"]["gaps"] == [{"code": "document_payload_redacted"}]


def test_document_paths_cannot_escape_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _resolve_under(tmp_path, "../secret.pdf")


def test_requires_aa_source_dpi() -> None:
    with pytest.raises(ValueError):
        _config(source_dpi=100)


def test_rejects_mismatched_source_dpi(tmp_path: Path) -> None:
    _write_document(tmp_path)
    manifest_path = tmp_path / "document" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_dpi"] = 100
    manifest_path.write_text(json.dumps(manifest))
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {
            "task_prompt": "Do the analysis.",
            "document_manifest": "document/manifest.json",
        },
    }

    with pytest.raises(ValueError, match="prepared at 100 DPI"):
        materialize_document(row, tmp_path, _config())


def _error(message: str, status: int = 500) -> ClientResponseError:
    error = ClientResponseError(None, (), status=status, message="upstream failure")
    error.response_content = message.encode()
    return error


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
    ],
)
def test_only_explicit_input_limits_adapt(message, status, expected) -> None:
    assert _input_limit(_error(message, status)) == expected


def _response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        tools=[],
        parallel_tool_calls=False,
        tool_choice="none",
        output=[
            {
                "type": "message",
                "id": "answer",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer", "annotations": []}],
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["maximum context length", "payload too large"])
async def test_adapts_only_policy_and_verifies_once(tmp_path: Path, failure) -> None:
    _write_document(tmp_path)
    config = _config(documents_base_dir=str(tmp_path))
    agent = GdpPdfAgent.model_construct(config=config)
    agent.server_client = SimpleNamespace(post=AsyncMock())
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "seed"},
            "verifier_metadata": {"task_prompt": "Do the analysis.", "document_manifest": "document/manifest.json"},
        }
    )
    profiles = []

    async def episode(params, **kwargs):
        content = params.input[0].content
        profiles.append(content[0]["text"])
        assert "text 3" in content[-1]["text"]
        if len(profiles) < 3:
            raise _error(failure)
        return _response(), None, {}, {}

    async def post(**kwargs):
        serialized = json.dumps(kwargs["json"])
        assert "data:image" not in serialized
        assert "SOURCE DOCUMENT TEXT" not in serialized
        if kwargs["url_path"] == "/verify":
            return SimpleNamespace(cookies={}, data=kwargs["json"] | {"reward": 1.0})
        return SimpleNamespace(cookies={})

    agent.server_client.post.side_effect = post
    with (
        patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
        patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
        patch("responses_api_agents.gdp_pdf_agent.app.raise_for_status", new=AsyncMock()),
        patch("responses_api_agents.gdp_pdf_agent.app.get_response_json", new=AsyncMock(side_effect=lambda r: r.data)),
    ):
        result = await agent.run(SimpleNamespace(cookies={}), body)
    assert [int(p.split("at ")[1].split()[0]) for p in profiles] == [150, 120, 96]
    assert result.document_delivery["image_dpi"] == 96
    assert result.document_delivery["image_count"] == 3
    assert len(result.document_delivery["rejected_attempts"]) == 2
    assert agent.server_client.post.await_count == 2
    assert not hasattr(config, "image_dpi")


@pytest.mark.asyncio
async def test_terminal_context_failure_stops_at_floor(tmp_path: Path) -> None:
    _write_document(tmp_path)
    agent = GdpPdfAgent.model_construct(config=_config(documents_base_dir=str(tmp_path), skip_verification=True))
    agent.server_client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(cookies={})))
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "seed"},
            "verifier_metadata": {"task_prompt": "Do the analysis.", "document_manifest": "document/manifest.json"},
        }
    )
    with (
        patch.object(GdpPdfAgent, "_create_episode", side_effect=_error("maximum context length")) as call,
        patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
        patch("responses_api_agents.gdp_pdf_agent.app.raise_for_status", new=AsyncMock()),
    ):
        result = await agent.run(SimpleNamespace(cookies={}), body)
    assert call.await_count == 5
    assert result.reward == 0
    assert result.response.output == []
    assert result.document_delivery["image_dpi"] == 72
    assert result.document_delivery["image_pages"] == 3
    assert [x["image_dpi"] for x in result.document_delivery["rejected_attempts"]] == [150, 120, 96, 76, 72]


def test_unknown_image_cap_probes_composition_then_coverage(tmp_path: Path) -> None:
    _write_document(tmp_path, 9)
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {"task_prompt": "Do the analysis.", "document_manifest": "document/manifest.json"},
    }
    delivery = DocumentDelivery()
    observed = []
    while True:
        materialize_document(row, tmp_path, _config(), delivery)
        observed.append((delivery.pages_per_image, delivery.image_count, delivery.image_pages))
        if not delivery.adapt("image_count", None):
            break
    assert observed == [(1, 9, 9), (2, 5, 9), (4, 3, 9), (4, 2, 8), (4, 1, 4)]
    assert delivery.image_dpi == 150


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["policy", "judge"])
async def test_unrelated_failures_do_not_retry_policy(tmp_path: Path, failed_stage) -> None:
    _write_document(tmp_path)
    agent = GdpPdfAgent.model_construct(config=_config(documents_base_dir=str(tmp_path)))
    agent.server_client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(cookies={})))
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "seed"},
            "verifier_metadata": {"task_prompt": "Do the analysis.", "document_manifest": "document/manifest.json"},
        }
    )
    failure = _error("rate limit exceeded" if failed_stage == "policy" else "maximum context length")
    with (
        patch.object(
            GdpPdfAgent,
            "_create_episode",
            new=AsyncMock(
                side_effect=failure if failed_stage == "policy" else None,
                return_value=(_response(), None, {}, {}),
            ),
        ) as call,
        patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
        patch(
            "responses_api_agents.gdp_pdf_agent.app.raise_for_status",
            new=AsyncMock(
                side_effect=[None, failure] if failed_stage == "judge" else None,
            ),
        ),
    ):
        with pytest.raises(ClientResponseError):
            await agent.run(SimpleNamespace(cookies={}), body)
    assert call.await_count == 1


def test_resizing_changes_pixels_without_composition(tmp_path: Path) -> None:
    _write_document(tmp_path)
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {"task_prompt": "Do the analysis.", "document_manifest": "document/manifest.json"},
    }
    delivery = DocumentDelivery()
    materialize_document(row, tmp_path, _config(), delivery)
    assert delivery.adapt("context", None)
    enriched = materialize_document(row, tmp_path, _config(), delivery)
    content = enriched["responses_create_params"]["input"][0]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert len(images) == 3
    for item in images:
        with Image.open(io.BytesIO(base64.b64decode(item["image_url"].split(",", 1)[1]))) as resized:
            assert resized.size == (80, 96)
    assert "text 3" in content[-1]["text"]


def test_single_page_remainder_is_labeled(tmp_path: Path) -> None:
    _write_document(tmp_path, 3)
    row = {
        "responses_create_params": {"input": "seed"},
        "verifier_metadata": {"task_prompt": "Analyze.", "document_manifest": "document/manifest.json"},
    }
    with patch("responses_api_agents.gdp_pdf_agent.app.ImageDraw.Draw") as draw:
        materialize_document(row, tmp_path, _config(max_images=2))
    assert [call.args[1] for call in draw.return_value.text.call_args_list] == ["Page 1", "Page 2", "Page 3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("has_usage", [False, True])
async def test_incomplete_response_is_never_resampled(tmp_path: Path, has_usage) -> None:
    _write_document(tmp_path)
    agent = GdpPdfAgent.model_construct(config=_config(documents_base_dir=str(tmp_path), skip_verification=True))
    agent.server_client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(cookies={})))
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "seed", "max_output_tokens": 100},
            "verifier_metadata": {"task_prompt": "Analyze.", "document_manifest": "document/manifest.json"},
        }
    )
    response = _response().model_copy(update={"status": "incomplete", "output": []})
    if has_usage:
        response = NeMoGymResponse.model_validate(
            response.model_dump()
            | {
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "total_tokens": 150,
                    "input_tokens_details": {"cached_tokens": None},
                    "output_tokens_details": {"reasoning_tokens": 100},
                }
            }
        )
    with (
        patch.object(GdpPdfAgent, "_create_episode", new=AsyncMock(return_value=(response, None, {}, {}))) as call,
        patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
        patch("responses_api_agents.gdp_pdf_agent.app.raise_for_status", new=AsyncMock()),
    ):
        if has_usage:
            result = await agent.run(SimpleNamespace(cookies={}), body)
            assert result.response.usage.output_tokens == 100
            assert result.document_delivery["rejected_attempts"] == []
        else:
            with pytest.raises(ValueError, match="cannot classify"):
                await agent.run(SimpleNamespace(cookies={}), body)
    assert call.await_count == 1
    assert call.call_args.args[0].max_output_tokens == 100


@pytest.mark.asyncio
async def test_provider_adapter_preserves_overflow_and_reasoning(tmp_path: Path) -> None:
    _write_document(tmp_path)
    provider = InferenceProvider.model_construct(
        config=InferenceProviderConfig(
            host="localhost",
            port=1234,
            name="policy",
            entrypoint="",
            base_url="http://localhost:1234/v1",
            api_key="unused",
            model="test",
            uses_reasoning_parser=True,
        )
    )
    provider._converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
    provider._semaphore = asyncio.Semaphore(1)
    provider._client = SimpleNamespace(
        create_chat_completion=AsyncMock(
            side_effect=[
                _error("maximum context length exceeded", 400),
                {
                    "id": "chat-1",
                    "created": 0,
                    "model": "test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "Private reasoning",
                                "content": "Final answer",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
            ]
        )
    )
    agent = GdpPdfAgent.model_construct(config=_config(documents_base_dir=str(tmp_path), skip_verification=True))
    agent.server_client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(cookies={})))
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": "seed", "max_output_tokens": 16384},
            "verifier_metadata": {"task_prompt": "Analyze.", "document_manifest": "document/manifest.json"},
        }
    )

    async def episode(params, **kwargs):
        return await provider.responses(SimpleNamespace(), params), None, {}, {}

    with (
        patch.object(GdpPdfAgent, "_create_episode", side_effect=episode),
        patch.object(GdpPdfAgent, "_model_call_capture_enabled", return_value=False),
        patch("responses_api_agents.gdp_pdf_agent.app.raise_for_status", new=AsyncMock()),
    ):
        result = await agent.run(SimpleNamespace(cookies={}), body)
    assert result.document_delivery["image_dpi"] == 120
    assert len(result.document_delivery["rejected_attempts"]) == 1
    assert [item.type for item in result.response.output] == ["reasoning", "message"]
    assert result.response.output[-1].content[0].text == "Final answer"
    assert provider._client.create_chat_completion.await_count == 2
    assert all(call.kwargs["max_tokens"] == 16384 for call in provider._client.create_chat_completion.call_args_list)
