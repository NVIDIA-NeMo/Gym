# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import traceback
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import quote, quote_plus

import pytest
from aiohttp import ClientPayloadError, ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from PIL import Image
from pydantic import ValidationError
from yarl import URL

from responses_api_agents.agentic_mme_agent import retrieval
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval, RetrievalConfig, http_payload, public_url
from responses_api_agents.agentic_mme_agent.tools import ImageWorkspace


def live(**kwargs) -> RetrievalConfig:
    """A live config. Google Lens comes with every text search, so SerpAPI and upload consent are always set."""
    return RetrievalConfig(**({"mode": "live", "serpapi_api_key": "lens", "allow_image_upload": True} | kwargs))


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
        "https://user:pass@example.com",  # pragma: allowlist secret
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
async def test_replay_stays_in_step_after_calls_rejected_before_the_provider(workspace, monkeypatch) -> None:
    # tool_trace records calls that never reached a provider (local URL, contaminated URL, bad
    # image_index, invalid arguments); replay must consume their entries too, or every later call
    # would raise a mismatch.
    http = AsyncMock()
    monkeypatch.setattr(retrieval, "request", http)
    rejected = [
        ("fetch_webpage", {"url": "http://localhost/a"}),
        ("fetch_webpage", {"url": "https://huggingface.co/datasets/agentic-mme"}),
        ("google_lens_search", {"image_index": 8}),
        ("google_search", {"query": ""}),
    ]
    trace = [
        {"tool_name": name, "arguments": args, "output": {"ok": False, "error": "recorded rejection"}}
        for name, args in rejected
    ]
    trace.append(
        {
            "tool_name": "google_search",
            "arguments": {"query": "example"},
            "output": {"ok": True, "context": "evidence"},
        }
    )
    client = Retrieval(RetrievalConfig(mode="replay"), trace)
    for name, args in rejected:
        assert (await client.call(name, args, workspace))["error"] == "recorded rejection"
    assert (await client.call("google_search", {"query": "example"}, workspace))["context"] == "evidence"
    assert client.replay_position == len(trace)
    http.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_and_lens_index(workspace) -> None:
    with pytest.raises(ValueError, match="disabled"):
        await Retrieval(RetrievalConfig()).call("google_search", {"query": "x"}, workspace)
    with pytest.raises(ValueError, match="image_index"):
        config = live(serper_api_key="test")
        await Retrieval(config).call("google_lens_search", {"image_index": 8}, workspace)


@pytest.mark.asyncio
async def test_live_search_payload(workspace, monkeypatch) -> None:
    http = AsyncMock(return_value=b'{"organic":[{"title":"Result","link":"https://example.com"}]}')
    monkeypatch.setattr(retrieval, "http_payload", http)
    config = live(serper_api_key="test-serper")
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
    # Lens comes with every text search, so live retrieval needs upload consent and the SerpAPI key.
    with pytest.raises(ValidationError, match="allow_image_upload"):
        RetrievalConfig(mode="live", search_provider="serpapi", serpapi_api_key="test")
    with pytest.raises(ValidationError, match="serpapi_api_key"):
        RetrievalConfig(mode="live", serper_api_key="test", allow_image_upload=True)
    http.assert_not_called()
    client = Retrieval(live(search_provider="serpapi", serpapi_api_key="test", allow_image_upload=True))
    http.side_effect = [b'{"image_id": "img-1"}', b'{"visual_matches":[{"title":"red"}]}']
    result = await client.call("google_lens_search", {}, workspace)
    assert result["ok"]
    assert [c.args[2] for c in http.call_args_list] == ["https://serpapi.com/image", "https://serpapi.com/search.json"]
    assert http.call_args_list[1].kwargs["params"]["image_id"] == "img-1"


@pytest.mark.asyncio
async def test_fetch_truncates_and_invalid_providers_fail(workspace, monkeypatch) -> None:
    http = AsyncMock(return_value=b"abcdef")
    monkeypatch.setattr(retrieval, "http_payload", http)
    client = Retrieval(live(serper_api_key="test", jina_api_key="jina"))
    assert (await client.call("fetch_webpage", {"url": "https://example.com", "max_chars": 3}, workspace))[
        "context"
    ] == "abc"
    assert http.call_args.args[2] == "https://r.jina.ai/https://example.com"
    http.return_value = b"[]"
    with pytest.raises(ValueError, match="invalid JSON"):
        await client.call("google_search", {"query": "q"}, workspace)
    http.return_value = b'{"error": "no image"}'
    upload_client = Retrieval(live(search_provider="serpapi", serpapi_api_key="test", allow_image_upload=True))
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
    client = MagicMock()
    client.request = AsyncMock(return_value=response)
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: client)
    config = RetrievalConfig(max_response_bytes=limit)
    if error:
        with pytest.raises(ValueError, match=error):
            await http_payload(config, "GET", "https://example.com")
    else:
        assert await http_payload(config, "GET", "https://example.com") == b"abcdef"
    response.release.assert_called_once()
    client.request.assert_awaited_once()
    assert client.request.call_args.kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_body_read_failure_redacts_its_cause_and_releases_response(monkeypatch) -> None:
    canary = "synthetic-stream-canary"
    url = "https://serpapi.com/search.json"
    info = RequestInfo(URL(url).with_query({"api_key": canary}), "GET", CIMultiDictProxy(CIMultiDict()))
    cause = ClientResponseError(info, (), status=503, message=f"upstream closed {canary}")

    async def stream(_):
        yield b"partial"
        raise ClientPayloadError(f"stream ended {canary}") from cause

    response = MagicMock(status=200, headers={})
    response.content.iter_chunked = stream
    client = MagicMock()
    client.request = AsyncMock(return_value=response)
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: client)
    with pytest.raises(ClientPayloadError) as raised:
        await http_payload(RetrievalConfig(serpapi_api_key=canary), "GET", url, params={"api_key": canary})
    assert raised.value.__cause__ is cause and cause.status == 503
    assert "stream ended" in str(raised.value)
    assert canary not in "".join(traceback.format_exception(raised.value)) + repr(cause)
    response.release.assert_called_once()
    client.request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("recovers", [False, True])
async def test_transport_retries_redact_credentials_before_logging(monkeypatch, capsys, debug, recovers) -> None:
    canary = "synthetic-canary+%/not-a-credential"
    encoded = {canary, quote(canary, safe=""), quote_plus(canary)}
    url = "https://serpapi.com/search.json"
    errors = []
    calls = []
    response = MagicMock(status=200, headers={})

    async def chunks(_):
        yield b"recovered"

    response.content.iter_chunked = chunks

    async def send(**kwargs):
        calls.append(kwargs)
        if recovers and len(calls) > 1:
            return response
        headers = CIMultiDictProxy(CIMultiDict({"Authorization": f"Bearer {canary}"}))
        info = RequestInfo(URL(url).with_query({"api_key": canary}), "GET", headers)
        error = ClientResponseError(
            info,
            (MagicMock(name=f"redirect-{canary}"),),
            status=429,
            message=f"upstream rejected {quote_plus(canary)}",
            headers=headers,
        )
        error.add_note(f"request credential {quote(canary, safe='')}")
        errors.append(error)
        try:
            raise OSError(71, f"upstream connection failed {canary}")
        except OSError as cause:
            raise error from cause

    client = MagicMock()
    client.request = AsyncMock(side_effect=send)
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG", debug)
    config = RetrievalConfig(serpapi_api_key=canary)
    if recovers:
        assert await http_payload(config, "GET", url, params={"api_key": canary}) == b"recovered"
        response.release.assert_called_once()
        assert len(calls) == 2
    else:
        with pytest.raises(ClientResponseError) as raised:
            await http_payload(config, "GET", url, params={"api_key": canary})
        assert raised.value is errors[-1]
        assert len(calls) == 3  # Gym's documented external-request retry limit.
    # Authentication and the shared retry policy stay intact; only exception diagnostics change.
    assert all(call["params"]["api_key"] == canary and call["allow_redirects"] is False for call in calls)
    captured = capsys.readouterr()
    diagnostics = captured.out + captured.err
    for error in errors:
        assert error.status == 429
        assert isinstance(error.__cause__, OSError) and error.__cause__.errno == 71
        assert "upstream connection failed" in str(error.__cause__)
        assert error.history == ()
        assert not error.headers and not error.request_info.headers
        diagnostics += repr(error) + repr(error.args) + repr(vars(error))
        diagnostics += "".join(traceback.format_exception(error))
    assert "429" in diagnostics and "upstream rejected" in diagnostics
    assert all(secret not in diagnostics for secret in encoded)


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
    for provider in ("serper", "tavily", "exa"):
        assert live(search_provider=provider, **{f"{provider}_api_key": "k"})
        with pytest.raises(ValidationError, match="serpapi_api_key"):  # Lens needs SerpAPI with any text search
            RetrievalConfig(
                mode="live", search_provider=provider, allow_image_upload=True, **{f"{provider}_api_key": "k"}
            )


@pytest.mark.asyncio
async def test_serpapi_search_maps_to_serper_shape_and_normalises_zh(workspace, monkeypatch) -> None:
    http = _fake_request(
        {
            "organic_results": [{"title": "T", "link": "https://a.example", "snippet": "S"}],
            "knowledge_graph": {"title": "KG", "description": "D"},
        }
    )
    monkeypatch.setattr(retrieval, "request", http)
    client = Retrieval(live(search_provider="serpapi", serpapi_api_key="k"))
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
    client = Retrieval(live(search_provider=provider, **{f"{provider}_api_key": "k"}))
    out = await client.call("google_search", {"query": "q"}, workspace)
    hit = out["raw"]["organic"][0]
    assert hit["title"] == "T" and hit["link"] == "https://a.example" and 0 < len(hit["snippet"]) <= 600
    assert http.call_args.args[:2] == ("POST", url)


@pytest.mark.asyncio
async def test_tavily_search_with_serpapi_lens(workspace, monkeypatch) -> None:
    with pytest.raises(ValidationError):  # Lens through SerpAPI needs the SerpAPI key
        RetrievalConfig(mode="live", search_provider="tavily", tavily_api_key="t", allow_image_upload=True)
    config = RetrievalConfig(
        mode="live",
        search_provider="tavily",
        tavily_api_key="t",
        serpapi_api_key="s",
        allow_image_upload=True,  # SerpAPI hosts the image itself
    )
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


@pytest.mark.asyncio
async def test_exa_fetch_provider_uses_contents_endpoint(workspace, monkeypatch) -> None:
    http = AsyncMock(return_value=json.dumps({"results": [{"url": "https://a.example", "text": "x" * 100}]}).encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    with pytest.raises(ValidationError, match="fetch_provider=exa requires exa_api_key"):
        live(serper_api_key="s", fetch_provider="exa")
    client = Retrieval(live(search_provider="exa", exa_api_key="k", fetch_provider="exa"))
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
    client = Retrieval(live(serper_api_key="s"))
    result = await client.call("google_search", {"query": "q"}, workspace)
    assert [r["title"] for r in result["raw"]["organic"]] == ["Clean"]
    assert result["contaminated"] == 2 and "Mirror" not in result["context"]
    http.return_value = json.dumps({"organic": organic[1:]}).encode()
    result = await client.call("google_search", {"query": "q"}, workspace)
    assert result["context"] == retrieval.NO_RESULTS_MESSAGE and result["raw"] == {"organic": []}
    assert result["contaminated"] == 2
    # A benchmark URL is refused without a provider call; a contaminated page fails like an empty one.
    n = http.call_count
    page = await client.call("fetch_webpage", {"url": "https://github.com/ChoS3nE11ven/Agentic-MME"}, workspace)
    assert page["error"] == retrieval.PAGE_UNAVAILABLE and not page["ok"] and http.call_count == n
    http.return_value = b"intro ... from the agentic_mme dataset ..."
    page = await client.call("fetch_webpage", {"url": "https://blog.example/post"}, workspace)
    assert page["error"] == retrieval.PAGE_UNAVAILABLE and not page["ok"] and page["contaminated"] == 1
    # An empty pattern list disables the guard.
    off = Retrieval(live(serper_api_key="s", contamination_patterns=[]))
    assert (
        "agentic_mme" in (await off.call("fetch_webpage", {"url": "https://blog.example/post"}, workspace))["context"]
    )


@pytest.mark.asyncio
async def test_contamination_guard_scans_the_page_past_max_chars(workspace, monkeypatch) -> None:
    # The answer-bearing prefix is clean; the benchmark attribution only appears after max_chars.
    page_text = "The fossil sold for 44.6 million. " + "filler " * 100 + "Source: the Agentic-MME benchmark."
    http = AsyncMock(return_value=page_text.encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    client = Retrieval(live(serper_api_key="s"))
    page = await client.call("fetch_webpage", {"url": "https://blog.example/post", "max_chars": 40}, workspace)
    assert page["error"] == retrieval.PAGE_UNAVAILABLE and not page["ok"] and page["contaminated"] == 1
    assert "44.6" not in json.dumps(page) and "full_text" not in page


@pytest.mark.asyncio
async def test_contamination_guard_scans_full_cached_pages(tmp_path, workspace, monkeypatch) -> None:
    page_text = "The fossil sold for 44.6 million. " + "filler " * 100 + "Source: the Agentic-MME benchmark."
    http = AsyncMock(return_value=page_text.encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    client = Retrieval(live(serper_api_key="s", cache_root=str(tmp_path)))
    args = {"url": "https://blog.example/post", "max_chars": 40}
    miss = await client.call("fetch_webpage", args, workspace)
    hit = await client.call("fetch_webpage", args, workspace)
    assert (miss["cache"], hit["cache"]) == ("miss", "exact") and http.call_count == 1
    for page in (miss, hit):
        assert page["error"] == retrieval.PAGE_UNAVAILABLE and "44.6" not in json.dumps(page)


@pytest.mark.asyncio
async def test_contamination_guard_filters_cache_hits_but_cache_keeps_raw(tmp_path, workspace, monkeypatch) -> None:
    organic = [
        {"title": "Clean", "link": "https://clean.example"},
        {"title": "Mirror", "link": "https://x.example/agentic-mme"},
    ]
    http = AsyncMock(return_value=json.dumps({"organic": organic}).encode())
    monkeypatch.setattr(retrieval, "http_payload", http)
    off = Retrieval(live(serper_api_key="s", cache_root=str(tmp_path), contamination_patterns=[]))
    assert len((await off.call("google_search", {"query": "cached q"}, workspace))["raw"]["organic"]) == 2
    on = Retrieval(live(serper_api_key="s", cache_root=str(tmp_path)))
    hit = await on.call("google_search", {"query": "cached q"}, workspace)
    assert hit["cache"] == "exact" and [r["title"] for r in hit["raw"]["organic"]] == ["Clean"]
    assert http.call_count == 1


SERPER_ANSWER = {
    "answerBox": {"title": "Eiffel Tower / Height", "answer": "330 m"},
    "knowledgeGraph": {
        "title": "Eiffel Tower",
        "description": "Wrought-iron tower in Paris",
        "attributes": {"Height": "330 m"},
    },
    "organic": [
        {"title": f"Result {i}", "link": f"https://r{i}.example", "snippet": f"snippet {i}", "date": "2024"}
        for i in range(1, 8)
    ],
    "peopleAlsoAsk": [{"question": "How tall is it?", "snippet": "It is tall.", "link": "https://q.example"}],
    "relatedSearches": [{"query": "eiffel tower tickets"}],
}


@pytest.mark.asyncio
async def test_search_is_shown_as_compact_text_without_answer_box(workspace, monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "http_payload", AsyncMock(return_value=json.dumps(SERPER_ANSWER).encode()))
    client = Retrieval(live(serper_api_key="s"))
    result = await client.call("google_search", {"query": "eiffel tower height"}, workspace)
    context = result["context"]
    assert context.startswith("[KnowledgeGraph] Eiffel Tower\n  - Wrought-iron tower in Paris")
    assert "1. Result 1 (2024)\n   https://r1.example\n   snippet 1" in context
    assert "5. Result 5" in context and "Result 6" not in context
    assert "- Q: How tall is it?\n  A: It is tall." in context
    # The answer box, knowledge-graph attributes and related searches never reach the model.
    assert "330" not in context and "tickets" not in context
    assert result["raw"] == SERPER_ANSWER  # the trace keeps the provider answer


def test_serpapi_answer_drops_the_answer_box() -> None:
    data = retrieval.serpapi_to_serper(
        {
            "answer_box": {"answer": "330 m"},
            "organic_results": [{"title": "T", "link": "https://t.example", "snippet": "s"}],
            "related_questions": [{"question": "Q?", "snippet": "A."}],
        },
        lens=False,
    )
    assert "answerBox" not in data and data["peopleAlsoAsk"][0]["question"] == "Q?"
    assert "330" not in retrieval.search_text(data)


def test_compact_text_for_lens_and_empty_answers() -> None:
    lens = {"organic": [{"title": "Red", "source": "Shop", "link": "https://s.example", "thumbnailUrl": "https://t"}]}
    assert (
        retrieval.lens_text(lens) == "[Lens Organic]\n1. Red [Shop]\n   https://s.example\n   thumbnailUrl: https://t"
    )
    assert retrieval.search_text({"organic": []}) == retrieval.NO_RESULTS_MESSAGE
    assert retrieval.lens_text({}) == retrieval.NO_RESULTS_MESSAGE
