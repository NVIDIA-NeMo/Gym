# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nemo_gym.mcp_auto_exposure import harvest_tools
from nemo_gym.server_utils import ServerClient
from resources_servers.gdpval import app
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig


# The tool JSON the certified runs sent to the model.
_CERTIFIED_TOOL_DEFINITIONS = Path(__file__).parent / "data" / "tool_definitions.json"


def _server(**extra) -> GDPValResourcesServer:
    config = GDPValResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="gdpval_resources_server",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        preconvert_office_to_pdf=False,
        sandbox_provider="test",
        sandbox_config={},
        persist_deliverables_dir="unused",
        **{"tavily_api_key": "test-key", **extra},
    )
    return GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def server() -> GDPValResourcesServer:
    return _server()


def _tavily_upstream(monkeypatch, respond) -> list[httpx.Request]:
    """Route the server's outgoing web calls to ``respond`` and return the requests it received."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    monkeypatch.setattr(app, "web_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return requests


def test_tool_definitions_match_the_certified_runs(server):
    tools = harvest_tools(server.setup_webserver(), server)

    definitions = [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.tool.description, "parameters": t.tool.inputSchema},
        }
        for t in tools.values()
    ]

    assert definitions == json.loads(_CERTIFIED_TOOL_DEFINITIONS.read_text())
    assert all(t.binding is not None for t in tools.values()), "every tool must be callable over MCP"


def test_web_search_rotates_through_the_configured_keys_and_returns_the_formatted_reply(monkeypatch):
    reply = {"answer": "A & B", "results": [{"title": "<T>", "url": "https://x.test/?a=1&b=2", "content": "c"}]}
    requests = _tavily_upstream(
        monkeypatch,
        lambda request: httpx.Response(401 if request.headers["Authorization"] == "Bearer k1" else 200, json=reply),
    )
    server = _server(tavily_api_key="[k1, k2]")

    response = TestClient(server.setup_webserver()).post("/web_search", json={"query": "what is <x>?"})

    assert response.json() == (
        "<answer>A &amp; B</answer>\n<results>\n<result>\n<title>&lt;T&gt;</title>\n"
        "<url>https://x.test/?a=1&amp;b=2</url>\n<content>c</content>\n</result>\n</results>"
    )
    assert [r.headers["Authorization"] for r in requests] == ["Bearer k1", "Bearer k2"]
    assert json.loads(requests[-1].content) == {"query": "what is <x>?", "max_results": 5, "include_answer": True}


def test_web_search_gives_up_after_the_configured_sweeps(monkeypatch):
    requests = _tavily_upstream(monkeypatch, lambda request: httpx.Response(429))
    server = _server(tavily_api_key="k1,k2", tavily_max_sweeps=2)

    response = TestClient(server.setup_webserver()).post("/web_search", json={"query": "q"})

    assert [r.headers["Authorization"] for r in requests] == ["Bearer k1", "Bearer k2"] * 2
    assert response.json() == (
        "<error>Tavily exhausted 4 attempt(s) (2 key(s) × 2 sweep(s)) on retryable errors "
        "(last status=429). Refresh keys or check upstream.</error>"
    )
