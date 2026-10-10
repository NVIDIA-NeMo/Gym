# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from pydantic import ValidationError

from responses_api_agents.agentic_mme_agent import retrieval
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval, RetrievalConfig, http_payload, public_url
from responses_api_agents.agentic_mme_agent.tools import ImageWorkspace


@pytest.fixture
def workspace():
    workspace = ImageWorkspace()
    workspace.add(Image.new("RGB", (4, 4), "red"))
    return workspace


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/a",
        "http://127.0.0.1",
        "http://169.254.169.254",
        "https://user:pass@example.com",
        "http://10.0.0.1/a",
        "http://[::1]",
        "https://example.com:9000",
        "http://service.internal",
        "http://2130706433",
        "http://127.1",
    ],
)
def test_reject_local_targets(url) -> None:
    with pytest.raises(ValueError):
        public_url(url)


def test_public_urls_and_config() -> None:
    assert public_url("https://example.com/page?q=x") == "https://example.com/page?q=x"
    with pytest.raises(ValidationError):
        RetrievalConfig(mode="live")
    with pytest.raises(ValidationError):
        RetrievalConfig(allow_image_upload=True)


@pytest.mark.asyncio
async def test_replay_matches_normalized_arguments_and_never_uses_network(workspace, monkeypatch) -> None:
    http = AsyncMock()
    monkeypatch.setattr(retrieval, "request", http)
    client = Retrieval(
        RetrievalConfig(mode="replay"),
        [
            {
                "tool_name": "google_search",
                "arguments": {"query": "example", "gl": "us", "hl": "en"},
                "output": {"ok": True, "context": "recorded evidence"},
            }
        ],
    )
    with pytest.raises(ValueError, match="mismatch"):
        await client.call("google_search", {"query": "different"}, workspace)
    assert client.replay_position == 0
    assert (await client.call("google_search", {"query": "example"}, workspace))["context"] == "recorded evidence"
    with pytest.raises(ValueError, match="remains"):
        await client.call("google_search", {"query": "example"}, workspace)
    http.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_and_lens_index(workspace) -> None:
    with pytest.raises(ValueError, match="disabled"):
        await Retrieval(RetrievalConfig()).call("google_search", {"query": "x"}, workspace)
    with pytest.raises(ValueError, match="image_index"):
        await Retrieval(RetrievalConfig(mode="replay")).call("google_lens_search", {"image_index": 8}, workspace)


@pytest.mark.asyncio
async def test_live_search_payload(workspace, monkeypatch) -> None:
    http = AsyncMock(return_value=b'{"organic":[{"title":"Result","link":"https://example.com"}]}')
    monkeypatch.setattr(retrieval, "http_payload", http)
    config = RetrievalConfig(mode="live", serper_api_key="test-serper")
    result = await Retrieval(config).call("google_search", {"query": "q"}, workspace)
    args, kwargs = http.call_args
    assert args[1:] == ("POST", "https://google.serper.dev/search")
    assert kwargs["headers"] == {"X-API-KEY": "test-serper"}
    assert kwargs["json"] == {"q": "q", "gl": "us", "hl": "en", "num": 5}
    assert result["raw"]["organic"][0]["title"] == "Result"
    assert "test-serper" not in json.dumps(result)


@pytest.mark.asyncio
async def test_lens_upload_optin_and_flow(workspace, monkeypatch) -> None:
    http = AsyncMock()
    monkeypatch.setattr(retrieval, "http_payload", http)
    client = Retrieval(RetrievalConfig(mode="live", serper_api_key="test"))
    with pytest.raises(ValueError, match="allow_image_upload"):
        await client.call("google_lens_search", {}, workspace)
    http.assert_not_called()
    client = Retrieval(
        RetrievalConfig(mode="live", serper_api_key="test", imgbb_api_key="upload", allow_image_upload=True)
    )
    http.side_effect = [
        b'{"success":true,"data":{"url":"https://i.ibb.co/test.png"}}',
        b'{"visual_matches":[{"title":"red"}]}',
    ]
    result = await client.call("google_lens_search", {}, workspace)
    assert result["ok"]
    assert http.call_args_list[0].kwargs["data"]["expiration"] == "600"
    assert http.call_args_list[1].kwargs["json"]["url"] == "https://i.ibb.co/test.png"
    assert http.call_args_list[1].args[2] == "https://google.serper.dev/lens"


@pytest.mark.asyncio
async def test_fetch_truncates_and_invalid_providers_fail(workspace, monkeypatch) -> None:
    http = AsyncMock(return_value=b"abcdef")
    monkeypatch.setattr(retrieval, "http_payload", http)
    client = Retrieval(RetrievalConfig(mode="live", serper_api_key="test", jina_api_key="jina"))
    assert (await client.call("fetch_webpage", {"url": "https://example.com", "max_chars": 3}, workspace))[
        "context"
    ] == "abc"
    assert http.call_args.args[2] == "https://r.jina.ai/https://example.com"
    http.return_value = b"[]"
    with pytest.raises(ValueError, match="invalid JSON"):
        await client.call("google_search", {"query": "q"}, workspace)
    http.return_value = b'{"success":false}'
    upload_client = Retrieval(
        RetrievalConfig(mode="live", serper_api_key="test", imgbb_api_key="upload", allow_image_upload=True)
    )
    with pytest.raises(ValueError, match="image upload"):
        await upload_client.call("google_lens_search", {}, workspace)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "chunks", "limit", "error"),
    [
        (200, [b"abc", b"def"], 10, None),
        (200, [b"abc", b"def"], 5, "byte limit"),
        (429, [], 10, "HTTP 429"),
    ],
)
async def test_http_bounds_and_release(monkeypatch, status, chunks, limit, error) -> None:
    async def stream(_):
        for chunk in chunks:
            yield chunk

    response = MagicMock(status=status)
    response.content.iter_chunked = stream
    monkeypatch.setattr(retrieval, "request", AsyncMock(return_value=response))
    config = RetrievalConfig(max_response_bytes=limit)
    if error:
        with pytest.raises(ValueError, match=error):
            await http_payload(config, "GET", "https://example.com")
    else:
        assert await http_payload(config, "GET", "https://example.com") == b"abcdef"
    response.release.assert_called_once()
    assert retrieval.request.call_args.kwargs["allow_redirects"] is False
