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
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_gym.mcp_test_utils import TOKEN_HEADER, assert_transport_parity, mcp_call, mcp_list_tools, seed_token
from nemo_gym.server_utils import ServerClient
from resources_servers.math_with_code.app import (
    PythonExecutorResourcesServer,
    PythonExecutorResourcesServerConfig,
)


class TestApp:
    """Tests for the Python Executor server."""

    SERVER_NAME = "math_with_code"

    def test_sanity(self) -> None:
        """Basic instantiation test - always runs."""
        config = PythonExecutorResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        PythonExecutorResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


# ================================================================================
# Dual-registration wire-format tests (HTTP replay, MCP round-trip, transport parity)
# ================================================================================

EXPECTED_TOOLS = {"execute_python", "end_session"}


def _json_bytes(payload: Any) -> bytes:
    """Starlette JSONResponse byte encoding (compact separators, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


@contextmanager
def _wire_client():
    """In-memory TestClient over the full app (no fixed ports)."""
    pytest.importorskip("mcp")
    from fastapi.testclient import TestClient

    server = PythonExecutorResourcesServer(
        config=PythonExecutorResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="math_with_code"),
        server_client=MagicMock(spec=ServerClient),
    )
    app = server.setup_webserver()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            yield server, app, client
    finally:
        # Reap any worker subprocesses left behind by the test.
        for sid in list(server._sessions):
            server._cleanup_session(sid)


class TestHTTPReplay:
    """Byte-equal replay of the pre-migration HTTP wire format."""

    def test_execute_python_success_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/execute_python", json={"code": "x = 5\ny = 10\nx + y"})
            assert resp.status_code == 200
            assert resp.content == _json_bytes(
                {"success": True, "stdout": "", "stderr": "", "error_message": None, "result": "15"}
            )

    def test_execute_python_stdout_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/execute_python", json={"code": 'print("hi")'})
            assert resp.status_code == 200
            assert resp.content == _json_bytes(
                {"success": True, "stdout": "hi\n", "stderr": "", "error_message": None, "result": None}
            )

    def test_execute_python_error_keeps_soft_200_contract(self) -> None:
        """Runtime errors keep the historic 200-with-error_message contract, byte for byte."""
        with _wire_client() as (_server, _app, client):
            resp = client.post("/execute_python", json={"code": "1/0"})
            assert resp.status_code == 200
            assert resp.content == _json_bytes(
                {"success": False, "stdout": "", "stderr": "", "error_message": "division by zero", "result": None}
            )

    def test_execute_python_missing_code_is_422(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/execute_python", json={})
            assert resp.status_code == 422
            assert any(error["loc"] == ["body", "code"] for error in resp.json()["detail"])

    def test_state_persists_across_calls_in_one_session(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/execute_python", json={"code": "a = 21"})
            assert resp.status_code == 200
            resp = client.post("/execute_python", json={"code": "a * 2"})
            assert resp.status_code == 200
            assert resp.json()["result"] == "42"

    def test_end_session_bytes_and_state_cleanup(self) -> None:
        with _wire_client() as (server, _app, client):
            resp = client.post("/execute_python", json={"code": "a = 1"})
            assert resp.status_code == 200
            assert len(server._sessions) == 1
            resp = client.post("/end_session", json={})
            assert resp.status_code == 200
            assert resp.content == _json_bytes(
                {"success": True, "stdout": "", "stderr": "", "error_message": None, "result": None}
            )
            assert server._sessions == {}

    def test_end_session_without_live_session_is_still_200(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/end_session", json={})
            assert resp.status_code == 200
            assert resp.json()["success"] is True

    def test_seed_session_now_carries_mcp_metadata(self) -> None:
        with _wire_client() as (_server, _app, client):
            body = client.post("/seed_session", json={}).json()
            assert body["mcp"]["url_path"] == "/mcp"
            assert TOKEN_HEADER in body["mcp"]["headers"]


class TestMCPRoundTrip:
    def test_tools_list_names_and_call(self) -> None:
        with _wire_client() as (server, _app, client):
            token = seed_token(client)
            tools = mcp_list_tools(client, token=token)
            assert set(tools) == EXPECTED_TOOLS
            # session_id is never model-visible in the advertised schemas.
            for tool in tools.values():
                assert "session_id" not in tool["inputSchema"].get("properties", {})

            result = mcp_call(client, "execute_python", {"code": "20 + 22"}, token=token)
            assert result.get("isError") is not True
            assert result["structuredContent"] == {
                "success": True,
                "stdout": "",
                "stderr": "",
                "error_message": None,
                "result": "42",
            }

            end_result = mcp_call(client, "end_session", {}, token=token)
            assert end_result.get("isError") is not True
            assert server._sessions == {}

    def test_call_without_token_is_clean_tool_error(self) -> None:
        with _wire_client() as (_server, _app, client):
            result = mcp_call(client, "execute_python", {"code": "1 + 1"}, token=None)
            assert result["isError"] is True
            assert TOKEN_HEADER in result["content"][0]["text"]


class TestTransportParity:
    def test_tool_sets_identical_across_transports(self) -> None:
        with _wire_client() as (_server, app, client):
            assert_transport_parity(app, client, EXPECTED_TOOLS)
