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
import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app import (
    NSToolsConfig,
    NSToolsResourcesServer,
    NSToolsVerifyRequest,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.mcp_test_utils import TOKEN_HEADER, assert_transport_parity, mcp_call, mcp_list_tools, seed_token
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


try:
    from nemo_skills.mcp.tool_manager import ToolManager
except ImportError:  # nemo_skills is a per-server dependency; most tests here stub it out
    ToolManager = None

requires_nemo_skills = pytest.mark.skipif(ToolManager is None, reason="nemo_skills is not installed in this venv")


class TestApp:
    def test_sanity(self) -> None:
        """Test that the server can be instantiated with minimal config."""
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
        )
        NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_with_verifiers(self) -> None:
        """Test configuration with verifiers."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert len(server.config.verifiers) == 1
        assert "math_with_judge" in server.config.verifiers
        assert server.config.default_verifier == "math_with_judge"

    def test_tool_sidecar_detection(self) -> None:
        """Legacy HTTP PythonTool variants require a sidecar; direct tools do not."""

        class PythonTool:
            def default_config(self):
                return {"client_params": {"base_url": "http://127.0.0.1:8765/mcp"}}

        class DirectPythonTool:
            def default_config(self):
                return {"sandbox": {}}

        server = NSToolsResourcesServer(
            config=NSToolsConfig(host="0.0.0.0", port=8080, entrypoint="", name="ns_tools"),
            server_client=MagicMock(spec=ServerClient),
        )

        with patch("app.locate", return_value=PythonTool):
            assert server._tool_uses_python_tool_sidecar("legacy.python_tool.PythonTool") is True

        with patch("app.locate", return_value=DirectPythonTool):
            assert (
                server._tool_uses_python_tool_sidecar("nemo_skills.mcp.servers.python_tool::DirectPythonTool") is False
            )

    @requires_nemo_skills
    async def test_verify_delegates_to_math_with_judge(self) -> None:
        """Test that verification is delegated to math_with_judge verifier."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server.tool_manager = MagicMock(spec_set=ToolManager)
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rollout-123"}

        # Mock the server_client.post to return a successful verification
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 1.0, "extracted_answer": "4"})
        server.server_client.post = AsyncMock(return_value=mock_response)

        # Build a NeMoGymResponse with a valid output
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The answer is \\boxed{4}.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = NSToolsVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "system", "content": "You are a helpful math assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
            },
            response=response,
            question="What is 2 + 2?",
            expected_answer="4",
        )

        result = await server.verify(request, verify_request)

        assert result.reward == 1.0
        assert result.delegated_response is not None
        assert result.delegated_response["reward"] == 1.0

        # Verify the server_client.post was called with correct args
        server.server_client.post.assert_called_once()
        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "math_with_judge"
        assert call_args.kwargs["url_path"] == "/verify"
        server.tool_manager.cleanup_request.assert_awaited_once_with("rollout-123")

    @requires_nemo_skills
    async def test_verify_cleans_up_when_verifier_fails(self) -> None:
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server.tool_manager = MagicMock(spec_set=ToolManager)
        server.server_client.post = AsyncMock(side_effect=RuntimeError("verifier failed"))
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rollout-123"}
        verify_request = MagicMock(spec=NSToolsVerifyRequest)
        verify_request.verifier_type = None

        with pytest.raises(RuntimeError, match="verifier failed"):
            await server.verify(request, verify_request)

        server.tool_manager.cleanup_request.assert_awaited_once_with("rollout-123")

    async def test_verify_uses_default_verifier(self) -> None:
        """Test that default verifier is used when verifier_type not specified."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 0.0})
        server.server_client.post = AsyncMock(return_value=mock_response)

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The answer is \\boxed{5}.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        # No verifier_type specified - should use default
        verify_request = NSToolsVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "What is 2 + 2?"}],
            },
            response=response,
            question="What is 2 + 2?",
            expected_answer="4",
        )

        result = await server.verify(None, verify_request)

        assert result.reward == 0.0
        call_args = server.server_client.post.call_args
        assert call_args.kwargs["server_name"] == "math_with_judge"

    async def test_verify_passes_through_fields(self) -> None:
        """Test that all sample fields are passed through to the delegated verifier."""
        verifiers = {
            "math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge"),
        }
        config = NSToolsConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="ns_tools",
            verifiers=verifiers,
            default_verifier="math_with_judge",
        )
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=mock_response)

        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_test",
                    "content": [{"annotations": [], "text": "\\boxed{4}", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = NSToolsVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "What is 2 + 2?"}],
            },
            response=response,
            question="What is 2 + 2?",
            expected_answer="4",
        )

        await server.verify(None, verify_request)

        call_args = server.server_client.post.call_args
        json_data = call_args.kwargs["json"]

        # Verify fields are passed through
        assert "question" in json_data
        assert "expected_answer" in json_data
        assert "responses_create_params" in json_data
        assert "response" in json_data


class TestPythonToolShutdownReap:
    """Shutdown must reap the python_tool subprocess on every exit path."""

    def _make_server(self) -> NSToolsResourcesServer:
        config = NSToolsConfig(host="0.0.0.0", port=8080, entrypoint="", name="ns_tools")
        return NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_kill_is_followed_by_reap_wait(self) -> None:
        server = self._make_server()
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="python_tool", timeout=5), 0]
        server._python_tool_process = proc

        await server.shutdown()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2
        assert server._python_tool_process is None

    async def test_unreaped_child_after_sigkill_is_logged(self) -> None:
        server = self._make_server()
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="python_tool", timeout=5)
        server._python_tool_process = proc

        with patch("app.logger") as mock_logger:
            await server.shutdown()

        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2
        mock_logger.error.assert_called_once()
        assert server._python_tool_process is None

    async def test_graceful_termination_does_not_kill(self) -> None:
        server = self._make_server()
        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.return_value = 0
        server._python_tool_process = proc

        await server.shutdown()

        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        proc.wait.assert_called_once_with(timeout=5)
        assert server._python_tool_process is None


class TestSidecarTeardownWiredToLifespan:
    """shutdown() must be connected to the server lifetime, not just defined.

    The base runner only starts Uvicorn; nothing calls shutdown() on its own.
    setup_webserver() must register it so a normal server stop reaps the
    python_tool sidecar instead of leaving it bound to its fixed port.
    """

    def _make_server(self) -> NSToolsResourcesServer:
        config = NSToolsConfig(host="0.0.0.0", port=8080, entrypoint="", name="ns_tools")
        return NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_lifespan_context_invokes_shutdown(self) -> None:
        server = self._make_server()
        app = server.setup_webserver()

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.return_value = 0
        server._python_tool_process = proc

        # Exiting the app lifespan context must reap the sidecar.
        async with app.router.lifespan_context(app):
            pass

        proc.terminate.assert_called_once()
        assert server._python_tool_process is None

    def test_lifespan_shutdown_reaps_sidecar_via_testclient(self) -> None:
        server = self._make_server()
        app = server.setup_webserver()

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.return_value = 0
        server._python_tool_process = proc

        # Entering/exiting TestClient drives the app startup/shutdown lifespan.
        with TestClient(app):
            pass

        proc.terminate.assert_called_once()
        assert server._python_tool_process is None

    async def test_shutdown_runs_when_lifespan_body_raises(self) -> None:
        """The try/finally must reap the sidecar even if the running app errors/cancels.

        An exception raised inside the lifespan body is thrown into the wrapper at
        the `yield`. Without the finally, shutdown() is skipped and the sidecar leaks.
        """
        server = self._make_server()
        app = server.setup_webserver()

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.return_value = 0
        server._python_tool_process = proc

        with pytest.raises(RuntimeError, match="boom"):
            async with app.router.lifespan_context(app):
                raise RuntimeError("boom")

        proc.terminate.assert_called_once()
        assert server._python_tool_process is None

    async def test_shutdown_runs_when_inner_lifespan_teardown_raises(self) -> None:
        """The try/finally must reap the sidecar even if the inner lifespan teardown raises.

        The wrapper wraps the base app's lifespan; if that inner teardown raises on a
        clean exit, the finally still runs shutdown() before the error propagates.
        """

        @asynccontextmanager
        async def raising_inner_lifespan(app):
            yield None
            raise RuntimeError("inner lifespan teardown failed")

        base_app = FastAPI()
        base_app.router.lifespan_context = raising_inner_lifespan

        server = self._make_server()
        with patch.object(SimpleResourcesServer, "setup_webserver", return_value=base_app):
            app = server.setup_webserver()

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.wait.return_value = 0
        server._python_tool_process = proc

        with pytest.raises(RuntimeError, match="inner lifespan teardown failed"):
            async with app.router.lifespan_context(app):
                pass

        proc.terminate.assert_called_once()
        assert server._python_tool_process is None


# ============================================================
# Dual-transport migration: HTTP wire-format replay + MCP round-trip
# ============================================================

PYTHON_TOOL_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "description": "Python code to execute"}},
    "required": ["code"],
}

DICT_RESULT = {"process_status": "completed", "stdout": "4\n", "stderr": ""}
STR_RESULT = '{"process_status": "completed", "stdout": "raw string result"}'


class _StubToolManager:
    """Stands in for the nemo_skills ToolManager: fixed discovery + scripted execute results."""

    def __init__(self, module_specs: List[str], overrides: Any = None, context: Any = None):
        self.calls: List[Any] = []

    async def list_all_tools(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        return [
            {
                "name": "execute_python",
                "description": "Execute python code.",
                "input_schema": dict(PYTHON_TOOL_SCHEMA),
                "server": "DirectPythonTool",
            }
        ]

    async def execute_tool(self, raw_name: str, args: Dict[str, Any], extra_args: Any = None) -> Any:
        self.calls.append((raw_name, dict(args), dict(extra_args or {})))
        code = args.get("code")
        if code == "boom":
            raise RuntimeError("sandbox exploded")
        if code == "request-timeout":
            raise TimeoutError("request timed out")
        if code == "internal-timeout":
            return {"process_status": "timeout", "stdout": ""}
        if code == "str-result":
            return STR_RESULT
        return dict(DICT_RESULT)

    async def cleanup_request(self, request_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _FakeDirectPythonTool:
    """Sidecar detection double: not named PythonTool, so no sidecar subprocess is spawned."""

    def default_config(self) -> Dict[str, Any]:
        return {"sandbox": {}}


def _make_tools_server() -> NSToolsResourcesServer:
    pytest.importorskip("mcp")
    config = NSToolsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="ns_tools",
        nemo_skills_tools=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
    )
    # model_post_init discovers tools with run_until_complete: give it a scratch loop.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with (
            patch("app.ToolManager", _StubToolManager),
            patch("app.locate", return_value=_FakeDirectPythonTool),
        ):
            return NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    finally:
        asyncio.set_event_loop(None)
        loop.close()


class TestHTTPWireContractReplay:
    """The migrated gym_tool routes must serve byte-identical bodies to the old catch-all route."""

    def test_dict_result_is_json_dumped_text_plain(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/execute_python", json={"code": "print(2+2)"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/plain")
            # Python's default json.dumps separators, exactly as the old PlainTextResponse body.
            assert resp.text == json.dumps(DICT_RESULT)
            assert resp.text == '{"process_status": "completed", "stdout": "4\\n", "stderr": ""}'

            # The rollout session id is wired through as the nemo_skills request_id...
            raw_name, args, extra = server.tool_manager.calls[0]
            assert raw_name == "execute_python"
            assert args == {"code": "print(2+2)"}
            session_id = extra["request_id"]
            assert session_id
            # ...and the timing bookkeeping is recorded under that session.
            records = server._timing_by_session[session_id]
            assert len(records) == 1
            assert records[0]["tool_name"] == "execute_python"
            assert records[0]["is_internal_timeout"] is False
            assert records[0]["is_request_timeout"] is False

    def test_str_result_passes_through_verbatim(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/execute_python", json={"code": "str-result"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/plain")
            assert resp.text == STR_RESULT  # no double JSON serialization

    def test_unknown_tool_keeps_the_200_soft_error_bytes(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/nonexistent_tool", json={"code": "x"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/plain")
            assert resp.text == '{"error": "Unknown tool: nonexistent_tool"}'

    def test_tool_exception_becomes_200_error_string(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/execute_python", json={"code": "boom"})
            assert resp.status_code == 200
            assert resp.text == '{"error": "sandbox exploded"}'

    def test_request_timeout_becomes_200_timeout_body_and_is_counted(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/execute_python", json={"code": "request-timeout"})
            assert resp.status_code == 200
            assert resp.text == '{"error": "Request timeout", "process_status": "timeout"}'

            (records,) = server._timing_by_session.values()
            assert records[0]["is_request_timeout"] is True
            assert records[0]["is_internal_timeout"] is False

    def test_internal_sandbox_timeout_is_flagged_and_body_preserved(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/execute_python", json={"code": "internal-timeout"})
            assert resp.status_code == 200
            assert resp.text == '{"process_status": "timeout", "stdout": ""}'

            (records,) = server._timing_by_session.values()
            assert records[0]["is_internal_timeout"] is True
            assert records[0]["is_request_timeout"] is False

    def test_tool_less_server_keeps_the_stock_404(self) -> None:
        config = NSToolsConfig(host="0.0.0.0", port=8080, entrypoint="", name="ns_tools")
        server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            resp = client.post("/anything", json={})
            assert resp.status_code == 404
            assert resp.json() == {"detail": "Not Found"}  # no catch-all when no tools configured


class TestMCPRoundTrip:
    """tools/list advertises the nemo_skills schema verbatim; tools/call shares the HTTP session."""

    def test_list_and_call_over_mcp(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = seed_token(client)

            tools = mcp_list_tools(client, token=token)
            assert set(tools) == {"execute_python"}
            assert tools["execute_python"]["inputSchema"] == PYTHON_TOOL_SCHEMA
            assert tools["execute_python"]["description"] == "Execute python code."

            result = mcp_call(client, "execute_python", {"code": "print(2+2)"}, token=token)
            assert result.get("isError") is not True
            assert result["content"][0]["text"] == json.dumps(DICT_RESULT)

            # The MCP call resolved the same session id the HTTP cookie transport would use.
            _, _, extra = server.tool_manager.calls[0]
            assert extra["request_id"]
            assert extra["request_id"] in server._timing_by_session

    def test_call_without_session_token_is_a_tool_error(self) -> None:
        server = _make_tools_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            result = mcp_call(client, "execute_python", {"code": "print(2+2)"}, token=None)
            assert result["isError"] is True
            assert TOKEN_HEADER in result["content"][0]["text"]


class TestTransportParity:
    """MCP tools/list names == HTTP POST tool routes == the discovered nemo_skills inventory."""

    def test_tool_sets_identical_across_transports(self) -> None:
        server = _make_tools_server()
        app = server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            assert_transport_parity(app, client, {"execute_python"})
