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

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.browsecomp_advanced_harness.app as app_module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BraveAIOHTTPClient,
    BraveSearchConfig,
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_EXCLUDE_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _response(status: int, body: dict, headers: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.ok = 200 <= status < 400
    response.headers = headers or {}
    response.content.read = AsyncMock(return_value=json.dumps(body).encode())
    return response


def _request() -> MagicMock:
    request = MagicMock()
    request.session = {SESSION_ID_KEY: "test_session_id"}
    return request


def _config(**overrides) -> TavilySearchResourcesServerConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "",
        "name": "",
        "search_provider": "brave",
        "brave": BraveSearchConfig(api_key="brave-key"),
        "exclude_domains_file_path": _EXCLUDE_FILE,
        "use_judge": False,
    }
    values.update(overrides)
    return TavilySearchResourcesServerConfig(**values)


@pytest.mark.asyncio
async def test_brave_client_uses_public_llm_context_contract(monkeypatch) -> None:
    fake_request = AsyncMock(
        return_value=_response(
            200,
            {
                "grounding": {
                    "generic": [
                        {
                            "title": "Result",
                            "url": "https://example.com",
                            "snippets": ["first passage", "second passage"],
                        }
                    ]
                },
                "sources": {},
            },
        )
    )
    monkeypatch.setattr(app_module, "request", fake_request)
    client = BraveAIOHTTPClient(
        api_key="brave-key",
        search_url="https://api.search.brave.test/res/v1/llm/context",
    )

    results = await client.search("research query")

    assert results == [
        {
            "title": "Result",
            "url": "https://example.com",
            "content": "first passage\nsecond passage",
        }
    ]
    kwargs = fake_request.await_args.kwargs
    assert kwargs["url"] == "https://api.search.brave.test/res/v1/llm/context"
    assert kwargs["headers"]["X-Subscription-Token"] == "brave-key"
    assert kwargs["json"] == {
        "q": "research query",
        "country": "US",
        "search_lang": "en",
        "count": 20,
        "maximum_number_of_urls": 20,
        "maximum_number_of_tokens": 8192,
        "maximum_number_of_tokens_per_url": 4096,
        "maximum_number_of_snippets": 100,
        "maximum_number_of_snippets_per_url": 100,
        "context_threshold_mode": "lenient",
    }


@pytest.mark.asyncio
async def test_brave_client_uses_source_title_and_normalizes_snippet_shapes(monkeypatch) -> None:
    fake_request = AsyncMock(
        return_value=_response(
            200,
            {
                "grounding": {
                    "generic": [
                        {
                            "url": "https://example.com",
                            "snippets": [
                                {"text": "text passage"},
                                {"content": "content passage"},
                                "plain passage",
                                "",
                            ],
                        },
                        {"title": "Missing URL", "snippets": ["ignored"]},
                    ]
                },
                "sources": {"https://example.com": {"title": "Source title"}},
            },
        )
    )
    monkeypatch.setattr(app_module, "request", fake_request)
    client = BraveAIOHTTPClient(
        api_key="brave-key",
        search_url="https://api.search.brave.test/res/v1/llm/context",
    )

    results = await client.search("research query")

    assert results == [
        {
            "title": "Source title",
            "url": "https://example.com",
            "content": "text passage\ncontent passage\nplain passage",
        }
    ]


@pytest.mark.asyncio
async def test_brave_client_rejects_invalid_public_search_shape(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "request", AsyncMock(return_value=_response(200, {"grounding": {}})))
    client = BraveAIOHTTPClient(
        api_key="brave-key",
        search_url="https://api.search.brave.test/res/v1/llm/context",
    )

    with pytest.raises(ValueError, match="grounding.generic"):
        await client.search("research query")


@pytest.mark.asyncio
async def test_brave_client_uses_urls_with_public_llm_context(monkeypatch) -> None:
    fake_request = AsyncMock(
        return_value=_response(
            200,
            {
                "grounding": {
                    "generic": [
                        {"url": "https://example.com", "snippets": ["page context"]},
                    ]
                }
            },
        )
    )
    monkeypatch.setattr(app_module, "request", fake_request)
    client = BraveAIOHTTPClient(
        api_key="brave-key",
        search_url="https://api.search.brave.test/res/v1/llm/context",
    )

    result = await client.context(
        "find the evidence",
        ["https://example.com", "https://missing.example"],
    )

    assert result == [{"title": "", "url": "https://example.com", "content": "page context"}]
    kwargs = fake_request.await_args.kwargs
    assert kwargs["url"] == "https://api.search.brave.test/res/v1/llm/context"
    assert kwargs["headers"]["X-Subscription-Token"] == "brave-key"
    assert kwargs["json"]["q"] == "find the evidence"
    assert kwargs["json"]["urls"] == ["https://example.com", "https://missing.example"]


@pytest.mark.asyncio
async def test_brave_search_is_formatted_for_the_existing_agent() -> None:
    server = TavilySearchResourcesServer(
        config=_config(),
        server_client=MagicMock(spec=ServerClient),
    )
    server._brave_client = MagicMock()
    server._brave_client.search = AsyncMock(
        return_value=[
            {"title": "Useful", "url": "https://example.com/page", "content": "answer evidence"},
            {"title": "Excluded", "url": "https://blacklisteddomain.com/page", "content": "bad"},
        ]
    )

    response = await server.search(_request(), TavilySearchRequest(queries=["query"]))

    assert "[Search Query]: query" in response.results_string
    assert "[Title]: Useful" in response.results_string
    assert "[URL]: https://example.com/page" in response.results_string
    assert "answer evidence" in response.results_string
    assert "blacklisteddomain.com" not in response.results_string
    metrics = server._session_id_to_metrics["test_session_id"].async_tavily_calls
    assert len(metrics) == 1
    assert metrics[0].provider == "brave"


@pytest.mark.asyncio
async def test_brave_search_uses_tavily_workspace_contract(tmp_path) -> None:
    server = TavilySearchResourcesServer(
        config=_config(workspace="per_session", workspace_root=str(tmp_path)),
        server_client=MagicMock(spec=ServerClient),
    )
    server._brave_client = MagicMock()
    server._brave_client.search = AsyncMock(
        return_value=[{"title": "Useful", "url": "https://example.com/page", "content": "answer evidence"}]
    )

    response = await server.search(_request(), TavilySearchRequest(queries=["query"]))

    assert "[Search Query]: query" in response.results_string
    assert "[Snippet]: answer evidence" in response.results_string
    assert "[Saved to]: pages/" in response.results_string
    pages = list((tmp_path / "test_session_id" / "pages").iterdir())
    assert len(pages) == 1
    assert pages[0].read_text() == (
        "[Query]: query\n[URL]: https://example.com/page\n[Title]: Useful\n\nanswer evidence"
    )


@pytest.mark.asyncio
async def test_brave_browse_uses_context_cached_by_search(tmp_path) -> None:
    server = TavilySearchResourcesServer(
        config=_config(brave=BraveSearchConfig(api_key="brave-key", cache_root=str(tmp_path / "cache-root"))),
        server_client=MagicMock(spec=ServerClient),
    )
    server._brave_client = MagicMock()
    server._brave_client.search = AsyncMock(
        return_value=[{"title": "Example", "url": "https://example.com/", "content": "search context"}]
    )

    await server.search(_request(), TavilySearchRequest(queries=["find the fact"]))

    response = await server.browse(
        _request(),
        BrowseRequest(
            urls=["https://example.com", "https://missing.example"],
            goal="find the fact",
        ),
    )

    assert "[URL]: https://example.com" in response.results_string
    assert "search context" in response.results_string
    assert "missing.example" not in response.results_string
    server._brave_client.context.assert_not_called()
    cached_files = list((tmp_path / "cache-root" / "test_session_id").glob("*.json"))
    assert len(cached_files) == 1
    assert json.loads(cached_files[0].read_text())["content"] == "search context"


@pytest.mark.asyncio
async def test_brave_browse_all_failed_returns_no_content() -> None:
    server = TavilySearchResourcesServer(
        config=_config(),
        server_client=MagicMock(spec=ServerClient),
    )
    server._brave_client = MagicMock()
    server._brave_client.context = AsyncMock()

    response = await server.browse(
        _request(),
        BrowseRequest(urls=["https://missing.example"], goal="find the fact"),
    )

    assert response.results_string == "No content extracted."
    server._brave_client.context.assert_not_called()


def test_brave_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match="brave configuration"):
        _config(brave=None)
