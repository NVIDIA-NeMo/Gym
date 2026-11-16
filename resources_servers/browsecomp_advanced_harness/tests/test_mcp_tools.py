# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tools a harness gets over MCP match the ones the rows give browsecomp_agent."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_runtime import maybe_auto_expose
from resources_servers.browsecomp_advanced_harness.app import TavilySearchResourcesServer
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


def test_mcp_search_runs_in_the_seeded_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mcp")
    server = _server(tmp_path, workspace="none", expose_tools_over_mcp=True)
    calls = []

    async def search_one(query: str, max_length: int, metrics) -> str:
        calls.append((query, max_length, metrics))
        return f"results for {query}"

    monkeypatch.setattr(server, "_search_one", search_one)
    with _mcp_client(server) as client:
        token = _seeded_token(client)
        result = _rpc(client, "tools/call", {"name": "search", "arguments": {"queries": ["q1"]}}, token=token, rid=2)

    assert result["result"].get("isError") is not True, result
    assert json.loads(result["result"]["content"][0]["text"]) == {"results_string": "results for q1"}
    # The configured budget applies, and the call counts toward the seeded session's metrics.
    assert calls == [("q1", server.config.search_max_total_length, server._session_id_to_metrics["resources-session"])]
