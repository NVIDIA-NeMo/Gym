# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.data import load_prompt_config
from nemo_gym.server_runtime.mcp_auto_exposure import harvest_tools
from resources_servers.gdpval import app
from resources_servers.gdpval.app import GDPValResourcesServer, GDPValResourcesServerConfig


_BENCHMARKS = Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval"
_BENCHMARK_TOOLS = _BENCHMARKS / "tools.json"
_PROMPT = _BENCHMARKS / "prompts" / "default-stirrup.yaml"


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


def test_benchmark_tools_are_the_tools_the_server_serves(server):
    tools = harvest_tools(server.setup_webserver(), server)

    served = [
        {
            "type": "function",
            "name": t.name,
            "description": t.tool.description,
            "parameters": t.tool.model_dump(by_alias=True)["inputSchema"],
            "strict": None,
        }
        for t in tools.values()
    ]

    assert json.loads(_BENCHMARK_TOOLS.read_text()) == served
    assert all(t.binding is not None for t in tools.values()), "every tool must be callable over MCP"


def test_the_prompt_advertises_the_served_tools_plus_the_agents_exec_tool():
    served = {t["name"] for t in json.loads(_BENCHMARK_TOOLS.read_text())}
    prompt = load_prompt_config(str(_PROMPT)).user

    advertised = set(re.findall(r"`(\w+)` tool", prompt))

    assert {"finish", "abandon_task_finish"} <= served
    assert advertised - served == {"code_exec"}
    assert advertised & served == {"finish", "abandon_task_finish"}


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


def test_fetch_web_page_returns_the_extracted_page_to_the_model(monkeypatch):
    html = "<html><body><main><h1>Tiers</h1><p>Gold guarantees a two-hour infusion slot.</p></main></body></html>"
    requests = _tavily_upstream(monkeypatch, lambda request: httpx.Response(200, text=html))

    response = TestClient(_server().setup_webserver()).post("/fetch_web_page", json={"url": "https://x.test/p"})

    assert [str(r.url) for r in requests] == ["https://x.test/p"]
    assert response.json().startswith("<web_fetch><url>https://x.test/p</url><body>")
    assert "Gold guarantees a two-hour infusion slot." in response.json()


def test_fetch_web_page_reports_an_upstream_error_as_text(monkeypatch):
    _tavily_upstream(monkeypatch, lambda request: httpx.Response(404))

    response = TestClient(_server().setup_webserver()).post("/fetch_web_page", json={"url": "https://x.test/p"})

    assert response.status_code == 200
    assert "<error>Client error" in response.json()


@pytest.mark.parametrize(
    "path, body",
    [
        ("/web_search", {}),
        ("/web_search", {"query": 5}),
        ("/fetch_web_page", {}),
        ("/fetch_web_page", {"url": ["https://x.test"]}),
    ],
)
def test_malformed_tool_arguments_are_rejected_before_any_web_call(monkeypatch, path, body):
    requests = _tavily_upstream(monkeypatch, lambda request: httpx.Response(200, json={}))

    response = TestClient(_server().setup_webserver()).post(path, json=body)

    assert response.status_code == 422
    assert requests == []
