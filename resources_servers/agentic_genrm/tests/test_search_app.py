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
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch, fixture

import resources_servers.browsecomp_advanced_harness.app as harness
from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, harvest_tools, maybe_auto_expose
from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_genrm.search_app import (
    TOOL_SPECS,
    GenRMSearchResourcesServer,
    GenRMSearchResourcesServerConfig,
)


EXCLUSIONS = Path(__file__).parents[1] / "data" / "search_exclusions.json"


@fixture
def server(monkeypatch: MonkeyPatch) -> GenRMSearchResourcesServer:
    # The server extends the harness's module-level filter lists; restore them after each test.
    monkeypatch.setattr(harness, "CONTAMINATION_PATTERNS", list(harness.CONTAMINATION_PATTERNS))
    monkeypatch.setattr(harness, "CONTAMINATED_URL_SUBSTRINGS", list(harness.CONTAMINATED_URL_SUBSTRINGS))
    config = GenRMSearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="search_app.py",
        name="genrm_search",
        expose_tools_over_mcp=True,
        search_provider="exa",
        exa_api_key=["k1", "k2"],
        exclude_domains_file_path=str(EXCLUSIONS),
        extra_exclude_domains=["example-leak.org", "lmarena.ai"],
        extra_contaminated_url_substrings=["Kaggle.com/datasets"],
        extra_contamination_patterns=["HelpSteer"],
        max_tracked_sessions=3,
    )
    return GenRMSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_extra_exclusions_extend_the_harness_lists(server) -> None:
    assert server._exclude_domains == ["lmarena.ai", "lmsys.org", "example-leak.org"]
    assert server._is_url_excluded("https://blog.lmarena.ai/post")
    assert not server._is_url_excluded("https://docs.python.org/3/")
    assert "kaggle.com/datasets" in harness.CONTAMINATED_URL_SUBSTRINGS
    assert "huggingface.co/datasets" in harness.CONTAMINATED_URL_SUBSTRINGS  # the harness's own entry stays
    assert "helpsteer" in harness.CONTAMINATION_PATTERNS

    results = [
        {"url": "https://www.kaggle.com/datasets/someone/arena-prefs", "title": "prefs", "text": "..."},
        {"url": "https://example.com/a", "title": "HelpSteer3 preference data", "text": "..."},
        {"url": "https://example.com/b", "title": "BenQ XL2586X+ specs", "highlights": ["DyAc 2, 540 Hz"]},
    ]
    kept, dropped = harness._drop_contaminated(results)
    assert dropped == 2
    assert kept == [results[2]]


def test_only_search_and_browse_are_exposed_with_descriptions(server) -> None:
    tools = harvest_tools(server.setup_webserver(), server)
    assert sorted(tools) == ["browse", "search"]
    for name, tool in tools.items():
        assert tool.tool == TOOL_SPECS[name]
        assert tool.tool.description
        assert tool.binding is not None


def test_session_metrics_are_bounded(server) -> None:
    metrics = [server._session_id_to_metrics[sid] for sid in ["a", "b", "c", "d", "e"]]
    assert list(server._session_id_to_metrics) == ["c", "d", "e"]
    # A tracked session keeps its object; only the oldest are dropped.
    assert server._session_id_to_metrics["e"] is metrics[-1]
    assert isinstance(server._session_id_to_metrics["e"], harness.TavilySearchMetrics)


class _FakeExa:
    def __init__(self) -> None:
        self.calls = []

    async def search(self, query, num_results, exclude_domains, include_text, search_type):
        self.calls.append({"query": query, "exclude_domains": list(exclude_domains), "include_text": include_text})
        return {
            "results": [
                {"url": "https://example.com/specs", "title": "BenQ XL2586X+ specs", "highlights": ["DyAc 2"]},
                {"url": "https://example.com/hs3", "title": "HelpSteer3 preference labels", "highlights": ["x"]},
                {"url": "https://huggingface.co/datasets/someone/prefs", "title": "prefs", "highlights": ["y"]},
            ]
        }


def test_search_over_mcp_round_trip(server) -> None:
    """Seed a session, then list and call the tools through the real /mcp endpoint."""
    fake = _FakeExa()
    server._select_exa_client = lambda: fake
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def rpc(client, method, params, token, rid):
        body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        return client.post("/mcp", headers=headers | {TOKEN_HEADER: token}, json=body).json()

    with TestClient(app) as client:
        token = client.post("/seed_session", json={}).json()["mcp"]["headers"][TOKEN_HEADER]
        init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}
        rpc(client, "initialize", init, token, 1)
        client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools = rpc(client, "tools/list", {}, token, 2)["result"]["tools"]
        assert sorted(t["name"] for t in tools) == ["browse", "search"]

        result = rpc(client, "tools/call", {"name": "search", "arguments": {"queries": ["XL2586X+ DyAc"]}}, token, 3)
        assert result["result"].get("isError") is not True
        text = json.loads(result["result"]["content"][0]["text"])["results_string"]

    assert "BenQ XL2586X+ specs" in text
    assert "HelpSteer3" not in text and "huggingface.co/datasets" not in text
    assert fake.calls == [
        {
            "query": "XL2586X+ DyAc",
            "exclude_domains": ["lmarena.ai", "lmsys.org", "example-leak.org"],
            "include_text": False,
        }
    ]


def test_gym_root_relative_exclusions_resolve_from_the_server_dir(monkeypatch: MonkeyPatch) -> None:
    # Gym starts each server from its own directory, where the config's root-relative path does not exist.
    monkeypatch.setattr(harness, "CONTAMINATION_PATTERNS", list(harness.CONTAMINATION_PATTERNS))
    monkeypatch.setattr(harness, "CONTAMINATED_URL_SUBSTRINGS", list(harness.CONTAMINATED_URL_SUBSTRINGS))
    monkeypatch.chdir(EXCLUSIONS.parents[1])
    config = GenRMSearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="search_app.py",
        name="genrm_search",
        search_provider="exa",
        exa_api_key="k",
        exclude_domains_file_path="resources_servers/agentic_genrm/data/search_exclusions.json",
    )
    server = GenRMSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert server._exclude_domains == ["lmarena.ai", "lmsys.org"]
