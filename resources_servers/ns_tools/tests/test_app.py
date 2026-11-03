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
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app import (
    NSToolsConfig,
    NSToolsResourcesServer,
    NSToolsVerifyRequest,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_skills.mcp.tool_manager import ToolManager

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.config import ResourcesServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.testing.session_conformance import check_resources_session_contract


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
        from fastapi.testclient import TestClient

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


class FakePythonToolManager:
    """Stands in for the NeMo Skills ToolManager: like DirectPythonTool, one python session per request_id."""

    def __init__(self) -> None:
        self.python_sessions: Dict[str, list] = {}
        self.cleaned_up: list = []
        self.cleanup_error: Optional[Exception] = None
        # Runs inside execute_tool while the sandbox runs the code, before the call records its python session.
        self.during_execute: Optional[Callable[[str], Awaitable[None]]] = None

    async def execute_tool(self, raw_name: str, args: Dict[str, Any], extra_args: Dict[str, Any]) -> str:
        request_id = extra_args["request_id"]
        if self.during_execute is not None:
            await self.during_execute(request_id)
        # Like DirectPythonTool, map the request to its python session once the sandbox call returns.
        self.python_sessions.setdefault(request_id, []).append(args["code"])
        return "4"

    async def cleanup_request(self, request_id: str) -> None:
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.cleaned_up.append(request_id)
        self.python_sessions.pop(request_id, None)

    async def shutdown(self) -> None:
        pass


def _session_server(tool_manager: FakePythonToolManager) -> tuple[NSToolsResourcesServer, FastAPI]:
    config = NSToolsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="ns_tools",
        verifiers={"math_with_judge": ResourcesServerRef(type="resources_servers", name="math_with_judge")},
        nemo_skills_tools=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
    )
    server = NSToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def install_tools(self: NSToolsResourcesServer) -> None:
        self.tool_manager = tool_manager
        self._tool_name_map["stateful_python_code_exec"] = "stateful_python_code_exec"

    # Registers the catch-all tool route as the configured tools do, so the checks below also cover its routing.
    with patch.object(NSToolsResourcesServer, "_initialize_nemo_skills_tools", install_tools):
        app = server.setup_webserver()
    return server, app


EPISODE_ID = EpisodeId(rollout_id="rollout", attempt=0)


def _seed(resources_session_id: str) -> Dict[str, Any]:
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EPISODE_ID,
        task_id=TaskId(taskset="ns_tools", task_id="0"),
        task_data={"question": "What is 2 + 2?", "expected_answer": "4"},
    ).model_dump(mode="json")


def _close(resources_session_id: str, episode_id: EpisodeId = EPISODE_ID) -> Dict[str, Any]:
    return ResourcesCloseSessionRequest(resources_session_id=resources_session_id, episode_id=episode_id).model_dump(
        mode="json"
    )


VERIFY_BODY = {
    "responses_create_params": {"input": [{"role": "user", "content": "What is 2 + 2?"}]},
    "response": {
        "id": "resp_test",
        "created_at": 0.0,
        "model": "dummy",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    },
    "question": "What is 2 + 2?",
    "expected_answer": "4",
}


class TestResourcesSessions:
    def test_session_contract(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)

        check_resources_session_contract(
            app,
            ResourcesSeedSessionRequest.model_validate(_seed("contract-session")),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._session_episodes,
            verify_body=VERIFY_BODY,
        )
        # The verify of the closed session was masked without reaching the delegated verifier.
        server.server_client.post.assert_not_called()

    def test_close_without_verify_releases_python_session(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app)

        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        tool_call = client.post("/stateful_python_code_exec", json={"code": "print(2 + 2)"})
        assert tool_call.status_code == 200 and tool_call.text == "4"
        assert "rs-1" in tool_manager.python_sessions
        assert len(server._timing_by_session["rs-1"]) == 1

        # An Environment Server whose seed reply was lost closes without the cookie.
        client.cookies.clear()
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        assert "rs-1" not in tool_manager.python_sessions
        assert "rs-1" not in server._timing_by_session
        assert "rs-1" not in server._session_episodes

    def test_failed_cleanup_leaves_session_open_for_a_retry(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "x = 1"}).status_code == 200

        tool_manager.cleanup_error = RuntimeError("sandbox unreachable")
        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        assert server._fence.closed_episode("rs-1") is None
        assert "rs-1" in tool_manager.python_sessions

        tool_manager.cleanup_error = None
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert "rs-1" not in tool_manager.python_sessions
        assert server._fence.closed_episode("rs-1") == EPISODE_ID

    def test_tool_call_after_close_is_a_lost_session(self) -> None:
        tool_manager = FakePythonToolManager()
        _, app = _session_server(tool_manager)
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        response = client.post("/stateful_python_code_exec", json={"code": "print(1)"})

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"
        # The refused call opened no python session that nothing would release.
        assert tool_manager.python_sessions == {}

    def test_tool_call_racing_close_releases_its_python_session(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200

        async def close_mid_call(request_id: str) -> None:
            tool_manager.during_execute = None
            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close(request_id)))

        tool_manager.during_execute = close_mid_call
        response = client.post("/stateful_python_code_exec", json={"code": "print(1)"})

        assert response.status_code == 404
        # The close released the session before the call recorded it; the call released it after.
        assert tool_manager.cleaned_up == ["rs-1", "rs-1"]
        assert tool_manager.python_sessions == {}
        assert "rs-1" not in server._timing_by_session

    def test_verify_then_close_releases_once_more_harmlessly(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(2 + 2)"}).status_code == 200
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)

        verified = client.post("/verify", json=VERIFY_BODY)
        assert verified.status_code == 200
        assert verified.json()["reward"] == 1.0 and verified.json()["num_tool_calls"] == 1
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        assert tool_manager.cleaned_up == ["rs-1", "rs-1"]
        assert tool_manager.python_sessions == {}

    def test_legacy_seed_keeps_working(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app)

        # An Agent's /run seeds with the row itself.
        seeded = client.post("/seed_session", json=VERIFY_BODY | {"verifier_type": "math_with_judge"})
        assert seeded.status_code == 200 and seeded.json() == {}
        assert server._session_episodes == {}
        assert client.post("/stateful_python_code_exec", json={"code": "print(1)"}).status_code == 200
        assert len(tool_manager.python_sessions) == 1
