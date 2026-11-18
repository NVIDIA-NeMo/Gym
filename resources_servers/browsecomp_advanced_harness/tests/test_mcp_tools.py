# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tools a harness gets over MCP match the ones the rows give browsecomp_agent."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_runtime import harvest_tools, maybe_auto_expose
from resources_servers.browsecomp_advanced_harness.app import TavilySearchMetrics, TavilySearchResourcesServer
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import _seed, _server
from resources_servers.browsecomp_advanced_harness.tool_definitions import INLINE_TOOLS, WORKSPACE_TOOLS


TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
EXAMPLE_ROWS = Path(__file__).resolve().parents[1] / "data" / "example.jsonl"


def _rpc(client: TestClient, method: str, params: dict | None = None, token: str | None = None, rid: int = 1) -> dict:
    headers = dict(RPC_HEADERS)
    if token:
        headers[TOKEN_HEADER] = token
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", headers=headers, json=body, follow_redirects=False).json()


def _mcp_client(server: TavilySearchResourcesServer) -> TestClient:
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _seeded_token(client: TestClient) -> str:
    seed = client.post("/seed_session", json=_seed())
    assert seed.status_code == 200, seed.text
    token = seed.json()["resources_tools"]["headers"][TOKEN_HEADER]
    _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        token=token,
    )
    client.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return token


def test_inline_definitions_are_the_example_rows_tools() -> None:
    rows = [json.loads(line) for line in EXAMPLE_ROWS.read_text().splitlines()]
    assert all(row["responses_create_params"]["tools"] == INLINE_TOOLS for row in rows)


@pytest.mark.parametrize(
    ("workspace", "definitions"),
    [("per_session", WORKSPACE_TOOLS), ("none", INLINE_TOOLS)],
    ids=["workspace", "inline"],
)
def test_mcp_lists_the_row_tools(tmp_path: Path, workspace: str, definitions: list[dict]) -> None:
    pytest.importorskip("mcp")
    server = _server(tmp_path, workspace=workspace, expose_tools_over_mcp=True)
    with _mcp_client(server) as client:
        token = _seeded_token(client)
        tools = _rpc(client, "tools/list", {}, token=token, rid=2)["result"]["tools"]

    # Not the request models' schemas: those also take max_total_length, which the rows do not offer the model.
    # Without a workspace there are no pages for bash_command to read, as in the example rows.
    assert [(t["name"], t["description"], t["inputSchema"]) for t in tools] == [
        (d["name"], d["description"], d["parameters"]) for d in definitions
    ]


class _FakeTavily:
    """Stands in for the provider client, below the search handler and its page writer."""

    async def search(self, query: str, **kwargs) -> dict:
        return {
            "results": [
                {
                    "title": f"Page about {query}",
                    "url": f"https://example.org/{query}",
                    "content": "snippet",
                    "raw_content": f"needle-{query} full page body",
                }
            ]
        }


def _text(result: dict) -> str:
    assert result["result"].get("isError") is not True, result
    return json.loads(result["result"]["content"][0]["text"])["results_string"]


def test_pages_saved_by_mcp_search_are_read_by_mcp_bash_only_in_their_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, workspace="per_session", expose_tools_over_mcp=True)
    monkeypatch.setattr(server, "_select_tavily_client", lambda: _FakeTavily())
    with _mcp_client(server) as client:
        token = _seeded_token(client)
        search = _text(
            _rpc(client, "tools/call", {"name": "search", "arguments": {"queries": ["alpha"]}}, token=token, rid=2)
        )
        found = _text(
            _rpc(
                client,
                "tools/call",
                {
                    "name": "bash_command",
                    "arguments": {"keystrokes": "grep -l needle-alpha pages/*.txt", "duration": 5},
                },
                token=token,
                rid=3,
            )
        )
        other = client.post("/seed_session", json=_seed("other-session"))
        assert other.status_code == 200, other.text
        other_token = other.json()["resources_tools"]["headers"]["X-NeMo-Gym-Session-Token"]
        listing = _text(
            _rpc(
                client,
                "tools/call",
                {"name": "bash_command", "arguments": {"keystrokes": "ls pages/ | wc -l", "duration": 5}},
                token=other_token,
                rid=4,
            )
        )

    saved = [line.split("[Saved to]: ", 1)[1].split(" ", 1)[0] for line in search.splitlines() if "[Saved to]" in line]
    assert len(saved) == 1, search
    # The page landed in the seeded session's workspace, and bash_command over MCP reads it there.
    assert (tmp_path / "resources-session" / saved[0]).read_text().endswith("needle-alpha full page body")
    assert saved[0] in found, found
    # Another session's shell sees none of it.
    assert listing.splitlines()[0].strip() == "0", listing


def test_mcp_search_binds_the_seeded_sessions_metrics_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity, not equality: two empty TavilySearchMetrics compare equal, so == cannot tell sessions apart."""
    server = _server(tmp_path, workspace="none", expose_tools_over_mcp=True, search_max_total_length=1234)
    calls = []

    async def search_one(query: str, max_length: int, metrics: TavilySearchMetrics) -> str:
        calls.append((query, max_length, metrics))
        return f"results for {query}"

    monkeypatch.setattr(server, "_search_one", search_one)
    with _mcp_client(server) as client:
        token = _seeded_token(client)
        result = _rpc(
            client, "tools/call", {"name": "search", "arguments": {"queries": ["q1", "q2"]}}, token=token, rid=2
        )

    assert _text(result) == "results for q1\n\nresults for q2"
    assert list(server._session_id_to_metrics) == ["resources-session"]
    seeded = server._session_id_to_metrics["resources-session"]
    # The configured (non-default) budget is split across the call's queries.
    assert [(q, n) for q, n, _ in calls] == [("q1", 617), ("q2", 617)]
    assert all(metrics is seeded for _, _, metrics in calls)


@pytest.mark.parametrize(("workspace", "definitions"), [("per_session", WORKSPACE_TOOLS), ("none", INLINE_TOOLS)])
def test_advertised_parameters_are_fields_of_the_handlers_request_models(
    tmp_path: Path, workspace: str, definitions: list[dict]
) -> None:
    """The MCP schema is now hand-written; each advertised parameter must reach the handler's request model."""
    server = _server(tmp_path, workspace=workspace, expose_tools_over_mcp=True)
    tools = harvest_tools(server.setup_webserver(), server)
    assert sorted(tools) == sorted(d["name"] for d in definitions)
    for definition in definitions:
        body_model = tools[definition["name"]].binding.body_model
        params = definition["parameters"]
        assert set(params["properties"]) <= set(body_model.model_fields), definition["name"]
        assert set(params.get("required", [])) <= set(params["properties"]), definition["name"]
        required_fields = {name for name, field in body_model.model_fields.items() if field.is_required()}
        assert required_fields <= set(params.get("required", [])), definition["name"]
        # Arguments the model builds from the advertised schema reach the handler unchanged.
        sample = {
            name: (["x"] if spec["type"] == "array" else 7 if spec["type"] == "number" else "x")
            for name, spec in params["properties"].items()
        }
        parsed = body_model.model_validate(sample)
        assert {name: getattr(parsed, name) for name in sample} == sample, definition["name"]
