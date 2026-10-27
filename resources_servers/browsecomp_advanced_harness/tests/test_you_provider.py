# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the You.com search provider in browsecomp_advanced_harness.

search calls /v1/search with highlights extraction, whose hits nest under results.web.
browse pulls full markdown via /v1/contents, which answers with a bare array.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness import app as app_module


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _config(**overrides) -> app_module.TavilySearchResourcesServerConfig:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        ydc_api_key="test_ydc_key",  # pragma: allowlist secret
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
    )
    kwargs.update(overrides)
    return app_module.TavilySearchResourcesServerConfig(**kwargs)


def _server(**overrides) -> app_module.TavilySearchResourcesServer:
    return app_module.TavilySearchResourcesServer(
        config=_config(**overrides), server_client=MagicMock(spec=ServerClient)
    )


def _req() -> MagicMock:
    m = MagicMock()
    m.session = {SESSION_ID_KEY: "test_session_id"}
    return m


def _http_response(status: int, body) -> MagicMock:
    r = MagicMock()
    r.status = status
    r.ok = status < 400
    r.content.read = AsyncMock(return_value=json.dumps(body).encode())
    r.json = AsyncMock(return_value=body)
    if not r.ok:
        r.raise_for_status = MagicMock(
            side_effect=ClientResponseError(request_info=MagicMock(), history=(), status=status, message="err")
        )
    return r


def _hit(title="T", url="https://x.com", highlights=("hl",), **extra) -> dict:
    return {"title": title, "url": url, "contents": {"highlights": list(highlights)}, **extra}


def _mock_client(web=None, **kwargs) -> MagicMock:
    mock = MagicMock()
    mock.search = AsyncMock(return_value={"results": {"web": web or []}}, **kwargs)
    return mock


class TestYouConfig:
    def test_you_is_default_provider(self) -> None:
        assert _config().search_provider == "ydc"

    def test_requires_you_key(self) -> None:
        with pytest.raises(ValueError, match="ydc_api_key"):
            _config(ydc_api_key=None)

    def test_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="search_provider"):
            _config(search_provider="bing")

    def test_client_per_key_and_json_headers(self) -> None:
        server = _server(ydc_api_key=["k1", "k2"])
        assert [c.headers["X-API-Key"] for c in server._you_clients] == ["k1", "k2"]
        assert server._you_clients[0].headers["Content-Type"] == "application/json"
        assert server._select_you_client() is server._you_clients[0]
        assert server._select_you_client() is server._you_clients[1]


class TestYouClient:
    def _client(self) -> app_module.YouAIOHTTPClient:
        return app_module.YouAIOHTTPClient(headers={}, base_url="https://ydc-index.io")

    async def test_search_payload_uses_highlights(self, monkeypatch) -> None:
        fake_request = AsyncMock(return_value=_http_response(200, {"results": {}}))
        monkeypatch.setattr(app_module, "request", fake_request)

        await self._client().search("q", num_results=5)

        kwargs = fake_request.call_args.kwargs
        assert kwargs["url"] == "https://ydc-index.io/v1/search"
        assert json.loads(kwargs["data"]) == {
            "query": "q",
            "count": 5,
            "extraction": {"extraction_mode": "highlights"},
        }

    async def test_search_caps_exclude_domains(self, monkeypatch) -> None:
        fake_request = AsyncMock(return_value=_http_response(200, {"results": {}}))
        monkeypatch.setattr(app_module, "request", fake_request)
        domains = [f"d{i}.com" for i in range(600)]

        await self._client().search("q", num_results=5, exclude_domains=domains)

        sent = json.loads(fake_request.call_args.kwargs["data"])["exclude_domains"]
        assert sent == domains[:500]

    async def test_get_contents_payload(self, monkeypatch) -> None:
        fake_request = AsyncMock(return_value=_http_response(200, []))
        monkeypatch.setattr(app_module, "request", fake_request)

        await self._client().get_contents(["https://a.com"], crawl_timeout=7)

        kwargs = fake_request.call_args.kwargs
        assert kwargs["url"] == "https://ydc-index.io/v1/contents"
        assert json.loads(kwargs["data"]) == {"urls": ["https://a.com"], "formats": ["markdown"], "crawl_timeout": 7}

    async def test_bad_request_raises_instead_of_returning_error_body(self, monkeypatch) -> None:
        # A 400 must surface, not be parsed as results and silently zero every search.
        fake_request = AsyncMock(return_value=_http_response(400, {"error": "bad query"}))
        monkeypatch.setattr(app_module, "request", fake_request)

        with pytest.raises(ClientResponseError):
            await self._client().search("q", num_results=5)
        fake_request.assert_awaited_once()  # 400 is not retryable

    async def test_invalid_key_aborts_benchmark(self, monkeypatch) -> None:
        fake_request = AsyncMock(return_value=_http_response(401, {"error": "invalid key"}))
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module.os, "_exit", MagicMock(side_effect=SystemExit(1)))

        with pytest.raises(SystemExit):
            await self._client().search("q", num_results=5)
        app_module.os._exit.assert_called_once_with(1)
        fake_request.assert_awaited_once()

    async def test_429_is_retried_and_counted(self, monkeypatch) -> None:
        fake_request = AsyncMock(side_effect=[_http_response(429, {}), _http_response(200, {"results": {}})])
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module, "sleep", AsyncMock())
        app_module._PROVIDER_RETRY_COUNTS.set(None)

        assert await self._client().search("q", num_results=5) == {"results": {}}
        assert fake_request.await_count == 2
        assert app_module._PROVIDER_RETRY_COUNTS.get() == {"num_429_retries": 1, "num_other_retries": 0}

    async def test_5xx_exhausts_retry_budget_and_raises(self, monkeypatch) -> None:
        fake_request = AsyncMock(return_value=_http_response(500, {}))
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module, "sleep", AsyncMock())

        with pytest.raises(ClientResponseError):
            await self._client().search("q", num_results=5)
        assert fake_request.await_count == 3


class TestYouSearch:
    async def test_highlights_rendered_inline(self) -> None:
        server = _server()
        server._you_clients = [_mock_client([_hit("T1", "https://x.com", ["foo", "bar"])])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["who won"]))

        assert "[Search Query]: who won" in resp.results_string
        assert "[Title]: T1" in resp.results_string
        assert "[URL]: https://x.com" in resp.results_string
        assert "[Snippet]: foo ... bar" in resp.results_string

    async def test_falls_back_to_snippets_then_description(self) -> None:
        server = _server()
        server._you_clients = [
            _mock_client(
                [
                    {"title": "A", "url": "https://a.com", "snippets": ["from snippets"]},
                    {"title": "B", "url": "https://b.com", "description": "from description"},
                ]
            )
        ]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"]))

        assert "[Snippet]: from snippets" in resp.results_string
        assert "[Snippet]: from description" in resp.results_string

    async def test_passes_max_results_and_exclude_domains(self) -> None:
        server = _server(max_results=7)
        server._exclude_domains = ["bad.com"]
        mock = _mock_client()
        server._you_clients = [mock]

        await server.search(_req(), app_module.TavilySearchRequest(queries=["q"]))

        mock.search.assert_awaited_once_with("q", num_results=7, exclude_domains=["bad.com"])

    async def test_excluded_domains_filtered_client_side(self) -> None:
        # The API caps exclude_domains, so registries beyond the cap must still apply.
        server = _server()
        server._exclude_domains = ["bad.com"]
        server._you_clients = [
            _mock_client([_hit("Bad", "https://www.bad.com/p"), _hit("Good", "https://good.com/p")])
        ]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"]))

        assert "good.com" in resp.results_string
        assert "bad.com" not in resp.results_string

    async def test_contaminated_hit_is_dropped(self) -> None:
        server = _server()
        server._you_clients = [
            _mock_client([_hit("Mirror", "https://huggingface.co/datasets/x/y"), _hit("Clean", "https://good.com/p")])
        ]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"]))

        assert "Clean" in resp.results_string
        assert "Mirror" not in resp.results_string

    async def test_budget_drops_oversized_entries(self) -> None:
        server = _server()
        server._you_clients = [_mock_client([_hit("Big", highlights=["z" * 5000]), _hit("Small")])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"], max_total_length=200))

        assert "Big" not in resp.results_string

    async def test_oversized_snippet_is_truncated_not_dropped(self) -> None:
        # Regression: one hit whose highlights exceed the query's budget slice used to end
        # the loop on the first hit, returning an empty result although the API offered hits.
        server = _server()
        server._you_clients = [_mock_client([_hit("Huge", highlights=["z" * 20000]), _hit("Next")])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"], max_total_length=8000))

        assert "[Title]: Huge" in resp.results_string
        assert "... [truncated]" in resp.results_string
        assert "z" * 5001 not in resp.results_string
        assert "[Title]: Next" in resp.results_string
        call = server._session_id_to_metrics["test_session_id"].async_tavily_calls[0]
        assert (call.num_results_offered, call.num_results_returned) == (2, 2)

    async def test_snippet_is_cut_to_the_per_query_budget_when_many_queries_share_it(self) -> None:
        # Regression: a fixed 5000-char cap still exceeded a query's slice once enough queries
        # split the budget (6000 // 6 = 1000), so the first hit ended the loop and every query
        # came back empty although You.com offered hits.
        server = _server()
        server._you_clients = [_mock_client([_hit("Huge", highlights=["z" * 20000])])]

        resp = await server.search(
            _req(), app_module.TavilySearchRequest(queries=[f"q{i}" for i in range(6)], max_total_length=6000)
        )

        blocks = resp.results_string.split("\n\n[Search Query]")
        assert len(blocks) == 6
        for block in blocks:
            assert "[Title]: Huge" in block
            assert "... [truncated]" in block
            assert len(block) <= 1000
        calls = server._session_id_to_metrics["test_session_id"].async_tavily_calls
        assert all((c.num_results_offered, c.num_results_returned) == (1, 1) for c in calls)

    async def test_hit_is_not_shown_when_too_little_budget_is_left_to_be_useful(self) -> None:
        server = _server()
        server._you_clients = [_mock_client([_hit("Huge", highlights=["z" * 20000])])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"], max_total_length=150))

        assert "[Title]" not in resp.results_string
        call = server._session_id_to_metrics["test_session_id"].async_tavily_calls[0]
        assert (call.num_results_offered, call.num_results_returned) == (1, 0)

    async def test_terminal_mode_page_keeps_full_highlights_when_inline_is_truncated(self, tmp_path) -> None:
        server = _server(workspace="per_session", workspace_root=str(tmp_path))
        server._you_clients = [_mock_client([_hit("Huge", highlights=["z" * 20000])])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"], max_total_length=8000))

        (page,) = (Path(tmp_path) / "test_session_id" / "pages").iterdir()
        assert "z" * 20000 in page.read_text()
        assert "z" * 5001 not in resp.results_string

    async def test_query_too_long(self) -> None:
        server = _server()
        mock = _mock_client()
        server._you_clients = [mock]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q" * 401]))

        assert "too long" in resp.results_string
        mock.search.assert_not_called()

    async def test_failure_returns_soft_message_and_records_error(self) -> None:
        server = _server()
        mock = MagicMock()
        mock.search = AsyncMock(side_effect=Exception("you boom"))
        server._you_clients = [mock]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"]))

        assert "Search failed: you boom" in resp.results_string
        call = server._session_id_to_metrics["test_session_id"].async_tavily_calls[0]
        assert (call.provider, call.function, call.status) == ("ydc", "search", "error")

    async def test_metering_records_yield(self) -> None:
        server = _server()
        server._you_clients = [_mock_client([_hit(), _hit(url="https://y.com")])]

        await server.search(_req(), app_module.TavilySearchRequest(queries=["a", "b"]))

        calls = server._session_id_to_metrics["test_session_id"].async_tavily_calls
        assert len(calls) == 2
        assert all(c.provider == "ydc" and c.status == "success" for c in calls)
        assert all(c.num_results_offered == 2 and c.num_results_returned == 2 for c in calls)

    async def test_terminal_mode_writes_highlights_to_pages(self, tmp_path) -> None:
        server = _server(workspace="per_session", workspace_root=str(tmp_path))
        server._you_clients = [_mock_client([_hit("T", "https://x.com", ["page highlight"])])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"]))

        pages = list((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 1
        assert "[Saved to]:" in resp.results_string
        assert "page highlight" in pages[0].read_text()

    async def test_terminal_mode_does_not_orphan_page_when_entry_over_budget(self, tmp_path) -> None:
        server = _server(workspace="per_session", workspace_root=str(tmp_path))
        server._you_clients = [_mock_client([_hit("Big", highlights=["z" * 5000])])]

        resp = await server.search(_req(), app_module.TavilySearchRequest(queries=["q"], max_total_length=200))

        assert "[Saved to]" not in resp.results_string
        assert not list((Path(tmp_path) / "test_session_id" / "pages").iterdir())


class TestYouBrowse:
    async def test_inline(self) -> None:
        server = _server()
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=[{"url": "https://x.com", "markdown": "FULL BODY TEXT"}])
        server._you_clients = [mock]

        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://x.com"], goal="g"))

        assert "[URL]: https://x.com" in resp.results_string
        assert "FULL BODY TEXT" in resp.results_string
        assert mock.get_contents.call_args.kwargs == {"urls": ["https://x.com"], "crawl_timeout": 60}

    async def test_plain_string_entries_pair_with_requested_urls(self) -> None:
        server = _server()
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=["PLAIN STRING BODY"])
        server._you_clients = [mock]

        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://x.com"], goal="g"))

        assert "[URL]: https://x.com" in resp.results_string
        assert "PLAIN STRING BODY" in resp.results_string

    async def test_terminal_mode_writes_page(self, tmp_path) -> None:
        server = _server(workspace="per_session", workspace_root=str(tmp_path))
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=[{"url": "https://x.com", "markdown": "PAGE CONTENT"}])
        server._you_clients = [mock]

        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://x.com"], goal="g"))

        pages = list((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 1
        assert "[Saved to]" in resp.results_string
        assert "PAGE CONTENT" in pages[0].read_text()

    async def test_empty_entry_and_no_results(self, tmp_path) -> None:
        server = _server(workspace="per_session", workspace_root=str(tmp_path))
        mock = MagicMock()
        server._you_clients = [mock]

        mock.get_contents = AsyncMock(return_value=[{"url": "https://x.com", "markdown": ""}])
        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://x.com"], goal="g"))
        assert "[Empty content]" in resp.results_string

        mock.get_contents = AsyncMock(return_value=[])
        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://x.com"], goal="g"))
        assert resp.results_string == "No content extracted."

    async def test_failure_returns_soft_message_and_records_error(self) -> None:
        server = _server()
        mock = MagicMock()
        mock.get_contents = AsyncMock(side_effect=Exception("you boom"))
        server._you_clients = [mock]

        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://x.com"], goal="g"))

        assert "Failed to extract content: you boom" in resp.results_string
        call = server._session_id_to_metrics["test_session_id"].async_tavily_calls[0]
        assert (call.provider, call.function, call.status) == ("ydc", "browse", "error")

    async def test_excluded_urls_are_never_fetched(self) -> None:
        server = _server()
        server._exclude_domains = ["bad.com"]
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=[])
        server._you_clients = [mock]

        resp = await server.browse(_req(), app_module.BrowseRequest(urls=["https://bad.com/p"], goal="g"))

        assert resp.results_string == "Error: no URLs provided."
        mock.get_contents.assert_not_called()
