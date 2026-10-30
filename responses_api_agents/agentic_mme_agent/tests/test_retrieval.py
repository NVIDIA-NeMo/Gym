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


def _fake_request(payload):
    """aiohttp-like response that streams `payload` and records the call."""
    response = MagicMock()
    response.status = 200
    body = json.dumps(payload).encode()

    async def chunks(_):
        yield body

    response.content.iter_chunked = chunks
    return AsyncMock(return_value=response)


def test_provider_keys_are_required_per_provider() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(mode="live", search_provider="serpapi", serper_api_key="only-serper")
    for provider in ("serper", "serpapi", "tavily", "exa"):
        config = RetrievalConfig(mode="live", search_provider=provider, **{f"{provider}_api_key": "k"})
        assert config.lens_available() == (provider in ("serper", "serpapi"))


@pytest.mark.asyncio
async def test_serpapi_search_maps_to_serper_shape_and_normalises_zh(workspace, monkeypatch) -> None:
    http = _fake_request(
        {
            "organic_results": [{"title": "T", "link": "https://a.example", "snippet": "S"}],
            "knowledge_graph": {"title": "KG", "description": "D"},
        }
    )
    monkeypatch.setattr(retrieval, "request", http)
    client = Retrieval(RetrievalConfig(mode="live", search_provider="serpapi", serpapi_api_key="k"))
    out = await client.call("google_search", {"query": "q", "hl": "zh"}, workspace)
    assert out["raw"]["organic"] == [{"title": "T", "link": "https://a.example", "snippet": "S", "date": None}]
    assert out["raw"]["knowledgeGraph"]["title"] == "KG"
    method, url = http.call_args.args[:2]
    params = http.call_args.kwargs["params"]
    assert (method, url) == ("GET", "https://serpapi.com/search.json")
    assert params["engine"] == "google" and params["hl"] == "zh-cn" and params["api_key"] == "k"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,payload,url",
    [
        (
            "tavily",
            {"results": [{"title": "T", "url": "https://a.example", "content": "C" * 900}]},
            "https://api.tavily.com/search",
        ),
        (
            "exa",
            {"results": [{"title": "T", "url": "https://a.example", "highlights": ["h1", "h2"]}]},
            "https://api.exa.ai/search",
        ),
    ],
)
async def test_tavily_and_exa_search_map_to_serper_shape(workspace, monkeypatch, provider, payload, url) -> None:
    http = _fake_request(payload)
    monkeypatch.setattr(retrieval, "request", http)
    client = Retrieval(RetrievalConfig(mode="live", search_provider=provider, **{f"{provider}_api_key": "k"}))
    out = await client.call("google_search", {"query": "q"}, workspace)
    hit = out["raw"]["organic"][0]
    assert hit["title"] == "T" and hit["link"] == "https://a.example" and 0 < len(hit["snippet"]) <= 600
    assert http.call_args.args[:2] == ("POST", url)
    with pytest.raises(ValueError, match="reverse image search"):
        await client.call("google_lens_search", {"image_index": 0}, workspace)


@pytest.mark.asyncio
async def test_tavily_search_with_serpapi_lens(workspace, monkeypatch) -> None:
    with pytest.raises(ValidationError):  # Lens through SerpAPI needs the SerpAPI key
        RetrievalConfig(mode="live", search_provider="tavily", tavily_api_key="t", lens_provider="serpapi")
    config = RetrievalConfig(
        mode="live",
        search_provider="tavily",
        tavily_api_key="t",
        lens_provider="serpapi",
        serpapi_api_key="s",
        allow_image_upload=True,  # no ImgBB key: SerpAPI hosts the image itself
    )
    assert config.lens_available() and config.image_host_resolved() == "serpapi"
    calls = []

    async def fake_payload(cfg, method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url == "https://serpapi.com/image":
            return json.dumps({"image_id": "img-123"}).encode()
        if "serpapi" in url:
            return json.dumps({"visual_matches": [{"title": "M", "link": "https://m.example"}]}).encode()
        return json.dumps({"results": [{"title": "T", "url": "https://t.example", "content": "c"}]}).encode()

    monkeypatch.setattr(retrieval, "http_payload", fake_payload)
    client = Retrieval(config)
    search = await client.call("google_search", {"query": "q"}, workspace)
    lens = await client.call("google_lens_search", {"image_index": 0}, workspace)
    assert search["raw"]["organic"][0]["link"] == "https://t.example"
    assert lens["raw"]["organic"][0]["title"] == "M"
    assert [c[1] for c in calls] == [
        "https://api.tavily.com/search",
        "https://serpapi.com/image",
        "https://serpapi.com/search.json",
    ]
    assert calls[0][2]["headers"]["Authorization"] == "Bearer t"
    lens_params = calls[2][2]["params"]
    assert lens_params["api_key"] == "s" and lens_params["engine"] == "google_lens"
    assert lens_params["image_id"] == "img-123" and "url" not in lens_params


def test_serpapi_upload_is_jpeg_under_the_size_limit() -> None:
    import random

    rng = random.Random(0)
    noisy = Image.frombytes("RGB", (1600, 1600), bytes(rng.getrandbits(8) for _ in range(1600 * 1600 * 3)))
    data = retrieval.jpeg_under(noisy)
    assert len(data) <= retrieval.SERPAPI_IMAGE_MAX_BYTES and data[:2] == b"\xff\xd8"


def test_image_host_choice_and_keys() -> None:
    # SerpAPI Lens needs no ImgBB key; Serper Lens still uploads to ImgBB for a public URL.
    assert RetrievalConfig(mode="live", serpapi_api_key="s", search_provider="serpapi", allow_image_upload=True)
    with pytest.raises(ValidationError, match="requires imgbb_api_key"):
        RetrievalConfig(mode="live", serper_api_key="s", allow_image_upload=True)
    with pytest.raises(ValidationError, match="image_host=serpapi only serves"):
        RetrievalConfig(mode="live", serper_api_key="s", allow_image_upload=True, image_host="serpapi")
    forced = RetrievalConfig(
        mode="live",
        serpapi_api_key="s",
        search_provider="serpapi",
        allow_image_upload=True,
        image_host="imgbb",
        imgbb_api_key="i",
    )
    assert forced.image_host_resolved() == "imgbb"


@pytest.mark.asyncio
async def test_exa_fetch_provider_uses_contents_endpoint(workspace, monkeypatch) -> None:
    http = AsyncMock(return_value=json.dumps({"results": [{"url": "https://a.example", "text": "x" * 100}]}).encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    with pytest.raises(ValidationError, match="fetch_provider=exa requires exa_api_key"):
        RetrievalConfig(mode="live", serper_api_key="s", fetch_provider="exa")
    client = Retrieval(RetrievalConfig(mode="live", search_provider="exa", exa_api_key="k", fetch_provider="exa"))
    page = await client.call("fetch_webpage", {"url": "https://a.example", "max_chars": 10}, workspace)
    assert page["context"] == "x" * 10 and "full_text" not in page
    args, kwargs = http.call_args
    assert args[1:] == ("POST", "https://api.exa.ai/contents")
    assert kwargs["headers"] == {"x-api-key": "k"}
    assert kwargs["json"] == {"urls": ["https://a.example"], "text": {"maxCharacters": retrieval.MAX_FETCH_CHARS}}
    http.return_value = b'{"results": [], "statuses": [{"id": "https://a.example", "status": "error"}]}'
    with pytest.raises(ValueError, match="no page content"):
        await client.call("fetch_webpage", {"url": "https://a.example"}, workspace)


@pytest.mark.asyncio
async def test_contamination_guard_drops_benchmark_items(workspace, monkeypatch) -> None:
    organic = [
        {"title": "Clean", "link": "https://clean.example", "snippet": "fine"},
        {"title": "Mirror", "link": "https://huggingface.co/datasets/Crystal1047/Agentic-MME", "snippet": "x"},
        {"title": "Paper", "link": "https://p.example", "snippet": "the Agentic MME benchmark answers"},
    ]
    http = AsyncMock(return_value=json.dumps({"organic": organic}).encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    client = Retrieval(RetrievalConfig(mode="live", serper_api_key="s"))
    result = await client.call("google_search", {"query": "q"}, workspace)
    assert [r["title"] for r in result["raw"]["organic"]] == ["Clean"]
    assert result["contaminated"] == 2 and "Mirror" not in result["context"]
    http.return_value = json.dumps({"organic": organic[1:]}).encode()
    result = await client.call("google_search", {"query": "q"}, workspace)
    assert result["context"] == retrieval.CONTAMINATED_MESSAGE and result["raw"] == {}
    # A benchmark URL is refused without a provider call; a contaminated page is withheld.
    n = http.call_count
    page = await client.call("fetch_webpage", {"url": "https://github.com/ChoS3nE11ven/Agentic-MME"}, workspace)
    assert page["context"] == retrieval.CONTAMINATED_MESSAGE and http.call_count == n
    http.return_value = b"intro ... from the agentic_mme dataset ..."
    page = await client.call("fetch_webpage", {"url": "https://blog.example/post"}, workspace)
    assert page["context"] == retrieval.CONTAMINATED_MESSAGE and page["contaminated"] == 1
    # An empty pattern list disables the guard.
    off = Retrieval(RetrievalConfig(mode="live", serper_api_key="s", contamination_patterns=[]))
    assert (
        "agentic_mme" in (await off.call("fetch_webpage", {"url": "https://blog.example/post"}, workspace))["context"]
    )


@pytest.mark.asyncio
async def test_contamination_guard_filters_cache_hits_but_cache_keeps_raw(tmp_path, workspace, monkeypatch) -> None:
    organic = [
        {"title": "Clean", "link": "https://clean.example"},
        {"title": "Mirror", "link": "https://x.example/agentic-mme"},
    ]
    http = AsyncMock(return_value=json.dumps({"organic": organic}).encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    off = Retrieval(
        RetrievalConfig(mode="live", serper_api_key="s", cache_root=str(tmp_path), contamination_patterns=[])
    )
    assert len((await off.call("google_search", {"query": "cached q"}, workspace))["raw"]["organic"]) == 2
    on = Retrieval(RetrievalConfig(mode="live", serper_api_key="s", cache_root=str(tmp_path)))
    hit = await on.call("google_search", {"query": "cached q"}, workspace)
    assert hit["cache"] == "exact" and [r["title"] for r in hit["raw"]["organic"]] == ["Clean"]
    assert http.call_count == 1
