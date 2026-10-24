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
"""Tests for the configurable per-search-call character budget.

`search` splits a total character budget across the queries in one call
(`max_total_length // len(queries)`). That total was hardcoded to 30000 on the
request model and is NOT exposed in the tool schema, so it could never be varied.
`search_max_total_length` makes it a server config field so an ablation can move it.

Measured motivation (a control run): search output is 92.5% of all tool-output
characters, and this character budget — not `max_results` — is what binds. 64% of query
blocks already return fewer than 5 results because the per-query slice runs out.
"""

import os
from typing import List
from unittest.mock import MagicMock

from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


class TestSearchBudget:
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

    @staticmethod
    def _capture_budgets(server: TavilySearchResourcesServer) -> List[int]:
        """Record the per-query max_length each _exa_search_one call receives."""
        seen: List[int] = []

        async def fake_search_one(query, max_length, metrics, page_writer=None):
            seen.append(max_length)
            return f"[Search Query]: {query}"

        server._exa_search_one = fake_search_one
        return seen

    # ---- default preserved ----

    async def test_default_total_budget_is_30000(self, req: MagicMock) -> None:
        """Unset config must behave exactly like the pre-change hardcoded 30000."""
        server = self._server()
        assert server.config.search_max_total_length == 30000
        seen = self._capture_budgets(server)

        await server.search(req, TavilySearchRequest(queries=["a", "b", "c"]))

        assert seen == [10000, 10000, 10000]

    async def test_default_single_query_gets_whole_budget(self, req: MagicMock) -> None:
        server = self._server()
        seen = self._capture_budgets(server)

        await server.search(req, TavilySearchRequest(queries=["only"]))

        assert seen == [30000]

    # ---- config override ----

    async def test_config_can_shrink_the_budget(self, req: MagicMock) -> None:
        server = self._server(search_max_total_length=15000)
        seen = self._capture_budgets(server)

        await server.search(req, TavilySearchRequest(queries=["a", "b", "c"]))

        assert seen == [5000, 5000, 5000]

    async def test_config_can_grow_the_budget(self, req: MagicMock) -> None:
        server = self._server(search_max_total_length=45000)
        seen = self._capture_budgets(server)

        await server.search(req, TavilySearchRequest(queries=["a", "b", "c"]))

        assert seen == [15000, 15000, 15000]

    # ---- explicit request value still wins ----

    async def test_explicit_request_value_overrides_config(self, req: MagicMock) -> None:
        """A caller that passes max_total_length explicitly keeps control.

        The tool schema does not expose this field, so in a real run the config value
        always applies — but the request field stays authoritative when set, so tests
        and any programmatic caller are unaffected by the config default.
        """
        server = self._server(search_max_total_length=45000)
        seen = self._capture_budgets(server)

        await server.search(req, TavilySearchRequest(queries=["a", "b", "c"], max_total_length=9000))

        assert seen == [3000, 3000, 3000]

    # ---- empty query list is still rejected before any budget maths ----

    async def test_empty_queries_short_circuits(self, req: MagicMock) -> None:
        """len(queries) == 0 must not reach the division (ZeroDivisionError)."""
        server = self._server(search_max_total_length=45000)
        self._capture_budgets(server)

        resp = await server.search(req, TavilySearchRequest(queries=[]))

        assert "none or empty" in resp.results_string
