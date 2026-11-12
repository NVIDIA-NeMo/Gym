# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test: the cache phase lines and totals follow the rollout id.

The unit tests in test_rollout_context.py cover the helpers in isolation. This
covers the cache side: per-call cache lines are off by default, the per-rollout
cache totals accrue, and the totals are released. The source repo's tests for the
`[browsecomp][provider]` line and the search_live/search_hit phases are not
ported, because this branch has no such telemetry.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import fixture

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness import rollout_context as RC
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import (
    _close,
    _seed,
    _verify_body,
)
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import (
    _server as session_server,
)


_DUMMY_EXCLUDE_DOMAINS_FILE = str(Path(__file__).with_name("dummy_exclude_domains_file.json"))

_RESULTS = {"results": [{"title": "T1", "url": "https://x.com/a", "content": "s", "raw_content": "RAW"}]}


@fixture(autouse=True)
def _clean():
    RC.reset_all()
    yield
    RC.reset_all()


def _server(cache_dir=None, verbose=False) -> TavilySearchResourcesServer:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider="tavily",
        tavily_api_key="test_tavily_key",  # pragma: allowlist secret
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        search_cache_dir=cache_dir,
        verbose_phase_log=verbose,
    )
    return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _req(rollout_id=None, session_id="test_session_id") -> MagicMock:
    m = MagicMock()
    m.session = {SESSION_ID_KEY: session_id}
    m.headers = {"x-rollout-id": rollout_id} if rollout_id else {}
    return m


async def _search(server, request, query="q1"):
    server._select_tavily_client = MagicMock(
        return_value=MagicMock(search=AsyncMock(return_value=_RESULTS)),
    )
    return await server.search(request, TavilySearchRequest(queries=[query]))


class TestPerRolloutBudget:
    async def test_per_call_cache_lines_are_off_by_default(self, tmp_path, caplog) -> None:
        """Cache gets run several times per turn; a leg already writes ~600k
        driver-log lines, so these must not be on unless asked for."""
        await _search(_server(cache_dir=str(tmp_path / "c")), _req("rV"))
        out = caplog.text
        assert "[cache][get]" not in out and "[cache][put]" not in out

    async def test_verbose_phase_log_enables_the_per_call_cache_lines(self, tmp_path, caplog) -> None:
        await _search(_server(cache_dir=str(tmp_path / "c"), verbose=True), _req("rW"))
        out = caplog.text
        assert "[cache][get] rid=rW" in out
        assert "[cache][put] rid=rW" in out

    async def test_cache_totals_accrue_even_with_per_call_logging_off(self, tmp_path) -> None:
        """The per-rollout budget is the always-on product; the per-call lines
        are optional detail on top of it."""
        await _search(_server(cache_dir=str(tmp_path / "c")), _req("rX"))
        totals = RC.peek_phase_totals("rX")
        assert totals["cache_get_search_n"] >= 1
        assert totals["cache_put_search_n"] >= 1

    def test_totals_are_released_so_a_4h_leg_does_not_leak(self) -> None:
        RC.record_phase("search_live", 1.0, rid="rG")
        assert RC.pop_phase_totals("rG")["search_live_s"] == 1.0
        assert RC.peek_phase_totals("rG") == {}

    def test_format_renders_a_readable_budget(self) -> None:
        RC.record_phase("search_live", 2.0, rid="rH")
        RC.record_phase("bash_exec", 0.5, rid="rH")
        assert RC.format_phase_totals(RC.pop_phase_totals("rH")) == "bash_exec=0.50s/1 search_live=2.00s/1"


@pytest.fixture(autouse=True)
def _log_level(caplog) -> None:
    caplog.set_level("INFO")


@pytest.mark.parametrize("verify", [False, True])
def test_episode_cache_totals_survive_reverify_and_release_on_close(tmp_path: Path, caplog, verify: bool) -> None:
    server = session_server(tmp_path / "workspace", workspace="none", search_cache_dir=str(tmp_path / "cache"))
    provider = MagicMock(search=AsyncMock(return_value=_RESULTS))
    server._select_tavily_client = MagicMock(return_value=provider)
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed()).is_success
        headers = {"x-rollout-id": "training-rollout"}
        first = client.post("/search", json={"queries": ["q1"]}, headers=headers)
        second = client.post("/search", json={"queries": ["q1"]}, headers=headers)
        assert first.is_success and second.json() == first.json()
        provider.search.assert_awaited_once()
        totals = RC.peek_phase_totals("training-rollout")
        assert totals["cache_get_search_n"] == 2
        assert totals["cache_put_search_n"] == 1
        if verify:
            # Env Server's verify does not carry the tool caller's custom header.
            first_verify = client.post("/verify", json=_verify_body("Paris"))
            second_verify = client.post("/verify", json=_verify_body("Paris"))
            assert first_verify.is_success and first_verify.json() == second_verify.json()
            assert RC.peek_phase_totals("training-rollout") == totals
            phase_lines = [r.message for r in caplog.records if "[browsecomp][phases]" in r.message]
            assert len(phase_lines) == 2
            assert all("rid=training-rollout" in line and "cache_get_search=" in line for line in phase_lines)
            assert any("search 1/2 (50%)" in r.message for r in caplog.records)
        assert client.post("/close_session", json=_close()).is_success
        assert RC.peek_phase_totals("training-rollout") == {}
        assert server._session_rollout_ids == {}
        late = client.post("/verify", json=_verify_body("Paris"))
        assert late.json()["mask_sample"] is True
        assert server._session_rollout_ids == {}


async def test_search_restores_the_callers_rollout_context(tmp_path: Path) -> None:
    RC.set_rollout_id("caller")
    await _search(_server(cache_dir=str(tmp_path / "cache")), _req("episode"))
    assert RC.current_rollout_id() == "caller"
