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
"""Tests for per-query search-yield metrics.

Before this, a search call recorded provider/function/status/latency and nothing about
what it actually returned. Three things were therefore indistinguishable in the data:
a query the provider had no results for, a query whose results were dropped by the
character budget, and a query that returned everything asked for.

That gap blocks the search_max_total_length ablation: without a truncation count there
is no way to confirm a bigger budget actually delivered more evidence, only to observe
downstream accuracy. These counters are METRICS-ONLY and never reach the model, so they
do not change the conditions of the arm they measure.
"""

import os
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _result(title: str, highlight: str) -> dict:
    return {"title": title, "url": f"https://{title}.com", "highlights": [highlight]}


class TestSearchYieldMetrics:
    def _server(self, **overrides) -> TavilySearchResourcesServer:
        config = TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="exa",
            exa_api_key="test_exa_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            **overrides,
        )
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @fixture
    def req(self) -> MagicMock:
        m = MagicMock()
        m.session = {SESSION_ID_KEY: "test_session_id"}
        return m

    def _calls(self, server: TavilySearchResourcesServer):
        return server._session_id_to_metrics["test_session_id"].async_tavily_calls

    # ---- nothing dropped ----

    async def test_records_offered_and_returned_when_all_fit(self, req: MagicMock) -> None:
        server = self._server(max_results=3)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", "x"), _result("b", "y")]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        call = self._calls(server)[0]
        assert call.num_results_offered == 2
        assert call.num_results_returned == 2
        assert call.num_results_truncated == 0
        assert call.chars_returned > 0

    # ---- provider genuinely had nothing ----

    async def test_zero_results_is_distinguishable_from_truncation(self, req: MagicMock) -> None:
        server = self._server()
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        call = self._calls(server)[0]
        assert call.num_results_offered == 0
        assert call.num_results_returned == 0
        # zero returned because there was nothing, NOT because the budget ran out
        assert call.num_results_truncated == 0

    # ---- budget dropped results ----

    async def test_budget_truncation_is_counted(self, req: MagicMock) -> None:
        server = self._server(max_results=4)
        big = "z" * 400
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [_result("a", big), _result("b", big), _result("c", big), _result("d", big)]}
        )
        server._exa_clients = [mock]

        # budget fits roughly one entry, so the rest must be reported as truncated
        await server.search(req, TavilySearchRequest(queries=["q"], max_total_length=520))

        call = self._calls(server)[0]
        assert call.num_results_offered == 4
        assert call.num_results_returned < 4
        assert call.num_results_truncated == call.num_results_offered - call.num_results_returned
        assert call.num_results_truncated > 0

    # ---- per-query granularity ----

    async def test_one_record_per_query_not_per_call(self, req: MagicMock) -> None:
        server = self._server()
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", "x")]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q1", "q2", "q3"]))

        search_calls = [c for c in self._calls(server) if c.function == "search"]
        assert len(search_calls) == 3
        assert all(c.num_results_returned == 1 for c in search_calls)

    # ---- failures leave the counters at their neutral default ----

    async def test_failed_search_leaves_counters_none(self, req: MagicMock) -> None:
        server = self._server()
        mock = MagicMock()
        mock.search = AsyncMock(side_effect=RuntimeError("provider down"))
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        call = self._calls(server)[0]
        assert call.status == "error"
        # a call that never got results must not report 0 offered, which would read
        # as "the provider had nothing" in the aggregate
        assert call.num_results_offered is None
        assert call.num_results_returned is None
        assert call.num_results_truncated is None

    # ---- the counters never reach the model ----

    async def test_counters_are_not_visible_in_the_tool_response(self, req: MagicMock) -> None:
        server = self._server(max_results=4)
        big = "z" * 400
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", big), _result("b", big)]})
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"], max_total_length=520))

        assert "truncated" not in resp.results_string.lower()
        assert "omitted" not in resp.results_string.lower()
