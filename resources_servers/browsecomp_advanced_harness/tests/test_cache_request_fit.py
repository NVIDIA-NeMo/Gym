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
"""A cache hit answers the current request as a live call would, and a bad entry or setting is refused."""

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness import app as APP
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)
from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache, normalize_query
from resources_servers.browsecomp_advanced_harness.sqlite_fuzzy_index import SqliteFuzzyIndex


_EXCLUDE = str(Path(__file__).with_name("dummy_exclude_domains_file.json"))


def _config(cache_dir: str | None, *, provider: str = "tavily", **overrides) -> TavilySearchResourcesServerConfig:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider=provider,
        exclude_domains_file_path=_EXCLUDE,
        search_cache_dir=cache_dir,
    )
    kwargs["exa_api_key" if provider == "exa" else "tavily_api_key"] = "test_key"  # pragma: allowlist secret
    return TavilySearchResourcesServerConfig(**(kwargs | overrides))


def _server(cache_dir: str | None, *, provider: str = "tavily", **overrides) -> TavilySearchResourcesServer:
    return TavilySearchResourcesServer(
        config=_config(cache_dir, provider=provider, **overrides), server_client=MagicMock(spec=ServerClient)
    )


def _req() -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: "s"}, "headers": []})


def _result(i: int, host: str = "a.example") -> dict:
    return {"title": f"T{i}", "url": f"https://{host}/{i}", "content": f"c{i}", "raw_content": f"BODY{i}"}


def _entry_path(root: str, kind: str, norm: str) -> Path:
    h = hashlib.sha256(norm.encode()).hexdigest()
    return Path(root, kind, h[:2], h[2:4], f"{h}.json")


class TestSearchHitsFollowTheCurrentRequest:
    async def test_hit_drops_domains_this_run_excludes(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        SearchCache(root).put_search("q", [_result(1), _result(2, "blacklisteddomain.com")], provider="tavily")
        server = _server(root)
        server._async_tavily_clients = [MagicMock(search=AsyncMock())]

        out = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        assert "BODY1" in out
        assert "blacklisteddomain.com" not in out and "BODY2" not in out

    async def test_hit_keeps_at_most_max_results(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        SearchCache(root).put_search("q", [_result(i) for i in range(6)], provider="tavily")
        server = _server(root, max_results=2)
        server._async_tavily_clients = [MagicMock(search=AsyncMock())]

        out = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        assert "BODY0" in out and "BODY1" in out and "BODY2" not in out

    async def test_exa_deep_answer_survives_a_hit(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        response = {
            "results": [{"title": "T", "url": "https://a.example/1", "highlights": ["h"]}],
            "output": {"content": "SYNTHESIS"},
        }
        miss_server = _server(root, provider="exa", exa_search_type="deep")
        client = MagicMock(search=AsyncMock(return_value=response))
        miss_server._exa_clients = [client]
        miss = (await miss_server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        hit_server = _server(root, provider="exa", exa_search_type="deep")
        unused = MagicMock(search=AsyncMock())
        hit_server._exa_clients = [unused]
        hit = (await hit_server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        unused.search.assert_not_awaited()
        assert "[Deep Answer]: SYNTHESIS" in miss
        assert hit == miss

    async def test_exa_entry_of_another_search_type_is_a_miss(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        auto = {"results": [{"title": "T", "url": "https://a.example/1", "highlights": ["auto"]}]}
        writer = _server(root, provider="exa")
        writer._exa_clients = [MagicMock(search=AsyncMock(return_value=auto))]
        await writer.search(_req(), TavilySearchRequest(queries=["q"]))

        deep_server = _server(root, provider="exa", exa_search_type="deep")
        deep = {"results": [{"title": "T", "url": "https://a.example/1", "highlights": ["deep"]}]}
        client = MagicMock(search=AsyncMock(return_value=deep))
        deep_server._exa_clients = [client]
        out = (await deep_server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        client.search.assert_awaited_once()
        assert "deep" in out and "auto" not in out


class TestBrowseHitsFollowTheExaBudget:
    async def test_a_page_cut_under_a_smaller_budget_is_refetched(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        SearchCache(root).put_extract("https://a.example/p", "x" * 100, provider="exa", max_characters=100)
        server = _server(root, provider="exa")
        client = MagicMock(
            get_contents=AsyncMock(return_value={"results": [{"url": "https://a.example/p", "text": "FULL"}]})
        )
        server._exa_clients = [client]

        out = (await server.browse(_req(), BrowseRequest(urls=["https://a.example/p"]))).results_string

        client.get_contents.assert_awaited_once()
        assert "FULL" in out

    async def test_an_exa_run_cuts_a_longer_hit_to_its_own_budget(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        SearchCache(root).put_extract("https://a.example/p", "y" * 5000, provider="tavily")
        server = _server(root, provider="exa")
        client = MagicMock(get_contents=AsyncMock())
        server._exa_clients = [client]

        out = (
            await server.browse(_req(), BrowseRequest(urls=["https://a.example/p"], max_total_length=1000))
        ).results_string

        client.get_contents.assert_not_awaited()
        assert "y" * 1000 in out and "y" * 1001 not in out
        assert "[truncated]" not in out  # a live exa page of this budget would not be cut again

    def test_a_tavily_page_replaces_a_capped_exa_page_but_not_a_complete_one(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_extract("https://a.example/capped", "x" * 10, provider="exa", max_characters=10)
        cache.put_extract("https://a.example/whole", "short", provider="exa", max_characters=10)
        cache.put_extract("https://a.example/capped", "FULL PAGE", provider="tavily")
        cache.put_extract("https://a.example/whole", "OTHER", provider="tavily")
        assert cache.get_extract("https://a.example/capped") == ("tavily", "FULL PAGE")
        assert cache.get_extract("https://a.example/whole") == ("exa", "short")


class TestBadEntriesAreMisses:
    @pytest.mark.parametrize(
        "raw",
        [b"\xff\xfe not utf-8", b"[1, 2]", b'{"provider": "tavily"}', b'{"results": "not a list"}'],
        ids=["invalid-utf8", "not-a-dict", "no-results", "results-not-a-list"],
    )
    def test_malformed_search_entry_is_a_miss(self, tmp_path, raw: bytes) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_search("q", [_result(1)], provider="tavily")
        _entry_path(str(tmp_path), "search", normalize_query("q")).write_bytes(raw)
        assert cache.get_search("q") is None

    def test_browse_entry_without_content_is_a_miss(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_extract("https://a.example/p", "BODY", provider="tavily")
        _entry_path(str(tmp_path), "browse", "https://a.example/p").write_text(json.dumps({"provider": "exa"}))
        assert cache.get_extract("https://a.example/p") is None


class TestWritesAreBestEffort:
    async def test_a_failed_publish_returns_the_provider_result_and_is_counted(self, tmp_path, caplog) -> None:
        caplog.set_level("WARNING")
        server = _server(str(tmp_path / "c"))
        server._async_tavily_clients = [MagicMock(search=AsyncMock(return_value={"results": [_result(1)]}))]
        full = OSError(28, "full")
        # A new entry publishes with os.link and falls back to os.replace; both fail on a full disk.
        with (
            patch("resources_servers.browsecomp_advanced_harness.search_cache.os.link", side_effect=full),
            patch("resources_servers.browsecomp_advanced_harness.search_cache.os.replace", side_effect=full),
        ):
            out = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        assert "BODY1" in out
        assert server._search_cache.stats()["put_errors"] == 1
        assert any("[cache][put_error]" in r.message for r in caplog.records)
        assert not list((tmp_path / "c").rglob("*.tmp"))
        assert "put errors 1" in server._search_cache.summary_str()


class TestSettingsAreCheckedAtStartup:
    @pytest.mark.parametrize("threshold", [0.0, -0.5, 1.5, 85.0, float("nan")])
    def test_fuzzy_threshold_outside_zero_one(self, tmp_path, threshold: float) -> None:
        with pytest.raises(ValidationError, match="search_cache_fuzzy_threshold"):
            _config(str(tmp_path), search_cache_fuzzy_threshold=threshold)

    @pytest.mark.parametrize("setting", [{"search_cache_fuzzy_threshold": 0.9}, {"search_cache_read_only": True}])
    def test_cache_settings_without_a_cache_dir(self, setting: dict) -> None:
        with pytest.raises(ValidationError, match="require search_cache_dir"):
            _config(None, **setting)

    def test_read_only_root_that_does_not_exist(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            _server(str(tmp_path / "missing"), search_cache_read_only=True)

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
    def test_writable_cache_on_a_root_with_unwritable_entry_dirs(self, tmp_path) -> None:
        SearchCache(str(tmp_path))
        (tmp_path / "search").chmod(0o555)
        try:
            with pytest.raises(ValueError, match="search_cache_read_only"):
                _server(str(tmp_path))
        finally:
            (tmp_path / "search").chmod(0o755)


class TestFuzzyCandidatesAreDeterministic:
    def test_equal_length_terms_break_ties_alphabetically(self, tmp_path) -> None:
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
        assert idx._terms("delta alpha gamma beta") == ["alpha", "delta", "gamma", "beta"]

    def test_a_failing_stage_falls_through_to_the_next(self, tmp_path) -> None:
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
        conn = MagicMock()
        conn.execute.side_effect = [
            sqlite3.OperationalError("unterminated string"),
            MagicMock(fetchall=MagicMock(return_value=[("matched norm",)])),
        ]
        assert idx._candidates(conn, "alpha beta gamma") == ["matched norm"]


@pytest.mark.parametrize(("level_env", "info_enabled"), [(None, True), ("warning", False)])
def test_configure_logging_shows_diagnostics_unless_silenced(monkeypatch, level_env, info_enabled: bool) -> None:
    names = (APP.__name__, "resources_servers.browsecomp_advanced_harness")
    saved = {name: logging.getLogger(name).level for name in names}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)  # leave the root handlers to pytest
    if level_env is None:
        monkeypatch.delenv("NEMO_GYM_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("NEMO_GYM_LOG_LEVEL", level_env)
    try:
        APP._configure_logging()
        assert (
            logging.getLogger("resources_servers.browsecomp_advanced_harness.search_cache").isEnabledFor(logging.INFO)
            is info_enabled
        )
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)
