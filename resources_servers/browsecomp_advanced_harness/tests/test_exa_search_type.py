# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the configurable Exa /search ``type`` (exa_search_type).

Default stays "auto" (the reference recipe — byte-identical request body
to what the harness sent before this flag). The deep variants ("deep-lite"/"deep"/"deep-reasoning")
additionally ask for a per-result ``summary`` and an ``outputSchema`` text synthesis,
mirroring the exa-py reference call:

    exa.search(q, num_results=10, output_schema={"type": "text"},
               type="deep", contents={"highlights": True, "summary": True})

The synthesized answer renders as a [Deep Answer] block ahead of the per-URL entries,
capped at a fraction of the per-query budget so it cannot starve them.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.browsecomp_advanced_harness.app as app_module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    ExaAIOHTTPClient,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _exa_config(**overrides) -> TavilySearchResourcesServerConfig:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider="exa",
        exa_api_key="test_exa_key",  # pragma: allowlist secret
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
    )
    kwargs.update(overrides)
    return TavilySearchResourcesServerConfig(**kwargs)


def _exa_server(**overrides) -> TavilySearchResourcesServer:
    return TavilySearchResourcesServer(config=_exa_config(**overrides), server_client=MagicMock(spec=ServerClient))


def _req() -> MagicMock:
    m = MagicMock()
    m.session = {SESSION_ID_KEY: "test_session_id"}
    return m


class TestExaSearchTypeConfig:
    def test_default_is_auto(self) -> None:
        assert _exa_config().exa_search_type == "auto"

    @pytest.mark.parametrize("search_type", ["instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"])
    def test_accepts_every_exa_type(self, search_type: str) -> None:
        assert _exa_config(exa_search_type=search_type).exa_search_type == search_type

    def test_rejects_unknown_type(self) -> None:
        # a typo must fail at config load, not silently 400 on every live query
        with pytest.raises(ValueError):
            _exa_config(exa_search_type="deepest")


class TestExaSearchTypeRequestBody:
    """The exact JSON posted to Exa's /search for each type."""

    async def test_auto_body_unchanged(self, monkeypatch) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        post = AsyncMock(return_value={"results": []})
        monkeypatch.setattr(client, "_post", post)

        await client.search("q", num_results=10, search_type="auto")
        endpoint, body = post.call_args[0]
        assert endpoint == "/search"
        assert body == {"query": "q", "numResults": 10, "type": "auto", "contents": {"highlights": True}}

    async def test_default_search_type_is_auto(self, monkeypatch) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        post = AsyncMock(return_value={"results": []})
        monkeypatch.setattr(client, "_post", post)

        await client.search("q", num_results=5)
        assert post.call_args[0][1]["type"] == "auto"

    async def test_deep_body_adds_summary_and_output_schema(self, monkeypatch) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        post = AsyncMock(return_value={"results": []})
        monkeypatch.setattr(client, "_post", post)

        await client.search("q", num_results=10, search_type="deep")
        body = post.call_args[0][1]
        assert body["type"] == "deep"
        assert body["contents"] == {"highlights": True, "summary": True}
        assert body["outputSchema"] == {"type": "text"}

    @pytest.mark.parametrize("search_type", ["deep-lite", "deep", "deep-reasoning"])
    def test_deep_types_recognized(self, search_type: str) -> None:
        assert search_type in app_module._EXA_DEEP_TYPES

    @pytest.mark.parametrize("search_type", ["instant", "fast", "auto"])
    async def test_shallow_types_have_no_output_schema(self, monkeypatch, search_type: str) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        post = AsyncMock(return_value={"results": []})
        monkeypatch.setattr(client, "_post", post)

        await client.search("q", num_results=5, search_type=search_type)
        body = post.call_args[0][1]
        assert "outputSchema" not in body
        assert body["contents"] == {"highlights": True}

    async def test_exclude_domains_still_sent_on_deep(self, monkeypatch) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        post = AsyncMock(return_value={"results": []})
        monkeypatch.setattr(client, "_post", post)

        await client.search("q", num_results=5, search_type="deep", exclude_domains=["bad.com"])
        assert post.call_args[0][1]["excludeDomains"] == ["bad.com"]


class TestExaSearchTypePlumbing:
    """config.exa_search_type reaches the client on every query."""

    async def test_server_passes_configured_type(self) -> None:
        server = _exa_server(exa_search_type="deep")
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert mock.search.call_args.kwargs.get("search_type") == "deep"

    async def test_server_defaults_to_auto(self) -> None:
        server = _exa_server()
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert mock.search.call_args.kwargs.get("search_type") == "auto"

    async def test_max_results_5_still_honored_on_deep(self) -> None:
        # the "5 searches from Exa" arm: SEARCH_MAX_RESULTS=5 -> num_results=5
        server = _exa_server(max_results=5)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert mock.search.call_args.kwargs.get("num_results") == 5


class TestExaDeepRendering:
    async def test_summary_rendered_when_present(self) -> None:
        server = _exa_server(exa_search_type="deep")
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": [
                    {"title": "T1", "url": "https://x.com", "highlights": ["hl"], "summary": "SUMMARY BODY"},
                ]
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert "[Snippet]: hl" in resp.results_string
        assert "[Summary]: SUMMARY BODY" in resp.results_string

    async def test_no_summary_key_when_absent(self) -> None:
        # auto-type responses carry no summary -> format must stay byte-identical
        server = _exa_server()
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [{"title": "T1", "url": "https://x.com", "highlights": ["hl"]}]}
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert resp.results_string == "[Search Query]: q\n[Title]: T1\n[URL]: https://x.com\n[Snippet]: hl\n"

    async def test_deep_answer_rendered_before_results(self) -> None:
        server = _exa_server(exa_search_type="deep")
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "output": {"content": "THE SYNTHESIZED ANSWER"},
                "results": [{"title": "T1", "url": "https://x.com", "highlights": ["hl"]}],
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert "[Deep Answer]: THE SYNTHESIZED ANSWER" in resp.results_string
        assert resp.results_string.index("[Deep Answer]") < resp.results_string.index("[Title]: T1")

    async def test_non_string_output_content_is_json_encoded(self) -> None:
        server = _exa_server(exa_search_type="deep")
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"output": {"content": {"answer": "42"}}, "results": []})
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert json.dumps({"answer": "42"}, ensure_ascii=False) in resp.results_string

    async def test_missing_output_renders_nothing(self) -> None:
        server = _exa_server(exa_search_type="deep")
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [{"title": "T", "url": "https://x.com", "highlights": []}]})
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))
        assert "[Deep Answer]" not in resp.results_string

    async def test_oversized_deep_answer_capped_so_results_survive(self) -> None:
        # a runaway synthesis must not eat the whole per-query budget: the URLs are
        # what the model needs for browse/bash follow-up.
        server = _exa_server(exa_search_type="deep")
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "output": {"content": "A" * 100_000},
                "results": [{"title": "T1", "url": "https://x.com", "highlights": ["hl"]}],
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"], max_total_length=2000))
        assert "[Deep Answer]" in resp.results_string
        assert "[URL]: https://x.com" in resp.results_string  # result survived the cap
        assert len(resp.results_string) <= 2000
