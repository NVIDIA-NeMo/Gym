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
import subprocess
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientOSError, ClientResponseError, web
from anyio import CancelScope, get_cancelled_exc_class
from app import (
    NSToolsConfig,
    NSToolsResourcesServer,
    NSToolsToolManager,
    NSToolsVerifyRequest,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_skills.mcp.servers.python_tool import DirectPythonTool
from nemo_skills.mcp.tool_manager import ToolManager

from nemo_gym.client import (
    SESSION_ID_KEY,
    GlobalAIOHTTPAsyncClientConfig,
    ServerClient,
    set_global_aiohttp_client,
)
from nemo_gym.client import request as http_request
from nemo_gym.config import ResourcesServerRef
from nemo_gym.core import JUDGE_FAILED, JUDGE_UNPARSEABLE, VERIFIER_ERROR, EpisodeId, NeMoGymResponse, TaskId
from nemo_gym.server_runtime import (
    BaseSeedSessionRequest,
    ResourcesCloseSessionRequest,
    ResourcesParticipant,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
    check_resources_session_contract,
)


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
        assert result.mask_sample is False
        assert result.failure_kind is None
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

    @pytest.mark.parametrize(
        ("delegated", "expected"),
        [
            pytest.param({"reward": 1.0}, (1.0, False, None, None), id="scored"),
            # The row math_with_judge's judge_failsafe returns when its judge call fails: an outage, not a wrong answer.
            pytest.param(
                {
                    "reward": 0.0,
                    "mask_sample": True,
                    "failure_kind": JUDGE_FAILED,
                    "failure_reason": "judge request timed out",
                    "_ng_failure_class": "judge_failed",
                },
                (0.0, True, JUDGE_FAILED, "judge request timed out"),
                id="masked_judge_failure",
            ),
            # Naming a failure kind does not mask the sample.
            pytest.param(
                {
                    "reward": 0.0,
                    "mask_sample": False,
                    "failure_kind": JUDGE_UNPARSEABLE,
                    "failure_reason": "judge verdict had no grade",
                },
                (0.0, False, JUDGE_UNPARSEABLE, "judge verdict had no grade"),
                id="unmasked_diagnostic",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "request_fields",
        [
            pytest.param({}, id="no_conflict"),
            pytest.param({"mask_sample": True}, id="request_masked"),
            pytest.param({"mask_sample": False}, id="request_unmasked"),
            pytest.param({"failure_kind": VERIFIER_ERROR}, id="request_failure_kind"),
            pytest.param({"failure_reason": "reason the request carried"}, id="request_failure_reason"),
        ],
    )
    async def test_verify_takes_verdict_fields_from_delegated_result(
        self, delegated: dict, expected: tuple, request_fields: dict
    ) -> None:
        """reward, mask_sample, failure_kind and failure_reason come from the delegated result, not the request."""
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
        mock_response.json = AsyncMock(return_value=delegated)
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
            **request_fields,
        )

        result = await server.verify(None, verify_request)

        assert (result.reward, result.mask_sample, result.failure_kind, result.failure_reason) == expected
        assert result.delegated_response == delegated
        # Request fields the verdict does not name stay on the response.
        assert result.model_dump()["question"] == "What is 2 + 2?"

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
        # An ordinary wrong answer is a valid measurement, so it stays unmasked.
        assert result.mask_sample is False
        assert result.failure_kind is None
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


def _session_server(tool_manager: FakePythonToolManager | ToolManager) -> tuple[NSToolsResourcesServer, FastAPI]:
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


class FakeSandboxEndpoint:
    """The NeMo Skills sandbox's execute and session DELETE routes. A DELETE answers queued statuses first."""

    def __init__(self, delete_statuses: list[int]) -> None:
        self.sessions: set[str] = set()
        self.deletes: list[str] = []
        self.delete_statuses = delete_statuses

    async def execute(self, request: web.Request) -> web.Response:
        await request.json()
        session_id = request.headers["X-Session-ID"]
        created = session_id not in self.sessions
        self.sessions.add(session_id)
        return web.json_response(
            {"process_status": "completed", "stdout": "4", "stderr": "", "new_session_created": created}
        )

    async def delete(self, request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        self.deletes.append(session_id)
        if self.delete_statuses:
            return web.Response(status=self.delete_statuses.pop(0))
        if session_id not in self.sessions:
            return web.Response(status=404)
        self.sessions.remove(session_id)
        return web.Response(status=200)


@asynccontextmanager
async def _real_python_tool_server(
    port: int, sandbox: FakeSandboxEndpoint, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[NSToolsResourcesServer, DirectPythonTool, MagicMock]]:
    """Yield a server whose session ``rs-1`` holds one Python session in ``sandbox``, through the real Skills tool."""
    backend = web.Application()
    backend.router.add_post("/execute", sandbox.execute)
    backend.router.add_delete("/sessions/{session_id}", sandbox.delete)
    runner = web.AppRunner(backend)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    http_client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
    tool_manager = NSToolsToolManager(
        module_specs=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
        context={"sandbox": {"host": "127.0.0.1", "port": str(port)}},
    )
    tool = tool_manager._tools["DirectPythonTool"]
    try:
        await tool_manager.list_all_tools()
        server, _ = _session_server(tool_manager)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        request.json = AsyncMock(return_value={"code": "print(4)"})
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        assert (await server.execute_tool("stateful_python_code_exec", request)).body == b"4"
        yield server, tool, request
    finally:
        await tool._sandbox.close()
        await http_client.close()
        await runner.cleanup()


class TestResourcesSessions:
    async def test_cookie_reseed_waits_for_delegated_verify_and_keeps_replacement_state(self) -> None:
        manager = FakePythonToolManager()
        server, _ = _session_server(manager)
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "cookie"}
        request.json = AsyncMock(return_value={"code": "print(4)"})
        seed_body = BaseSeedSessionRequest.model_validate(VERIFY_BODY)
        await server.seed_session(seed_body, request=request)
        await server.execute_tool("stateful_python_code_exec", request)
        verifying = asyncio.Event()
        finish_verification = asyncio.Event()
        reseeding = asyncio.Event()
        reply = AsyncMock()
        reply.json = AsyncMock(return_value={"reward": 1.0})

        async def delegate(**kwargs):
            verifying.set()
            await finish_verification.wait()
            return reply

        async def reseed_and_execute() -> None:
            reseeding.set()
            await server.seed_session(seed_body, request=request)
            await server.execute_tool("stateful_python_code_exec", request)

        server.server_client.post = AsyncMock(side_effect=delegate)
        verify = asyncio.create_task(server.verify(request, NSToolsVerifyRequest.model_validate(VERIFY_BODY)))
        reseed = None
        try:
            await asyncio.wait_for(verifying.wait(), timeout=5)
            reseed = asyncio.create_task(reseed_and_execute())
            await asyncio.wait_for(reseeding.wait(), timeout=5)
            await asyncio.sleep(0)
            assert not reseed.done()
        finally:
            finish_verification.set()
            await asyncio.wait_for(asyncio.gather(verify, *([reseed] if reseed else [])), timeout=5)
        assert verify.result().num_tool_calls == 1
        assert manager.python_sessions == {"cookie": ["print(4)"]}
        assert len(server._timing_by_session["cookie"]) == 1
        assert not server.session_is_verified("cookie")
        assert (await server.verify(request, NSToolsVerifyRequest.model_validate(VERIFY_BODY))).num_tool_calls == 1
        assert manager.python_sessions == {} and server._timing_by_session == {}

    def test_checkpoint_failed_verify_cleanup_stays_owned_until_retry(self) -> None:
        manager = FakePythonToolManager()
        server, app = _session_server(manager)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        server.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test-only"}}
        server.setup_resources_checkpoint(app)
        client = TestClient(app, raise_server_exceptions=False)
        prefix = "/ng-rollout/rollout"
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post(prefix + "/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        manager.cleanup_error = RuntimeError("cleanup failed")
        first = client.post(prefix + "/verify", json=VERIFY_BODY, headers={"x-ng-checkpoint-close-follows": "1"})
        assert first.status_code == 200 and first.json()["reward"] == 1.0
        assert server.checkpoint_participant.is_open("rs-1")
        assert "rs-1" in server._session_episodes
        assert "rs-1" in manager.python_sessions
        manager.cleanup_error = None
        second = client.post(prefix + "/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["reward"] == 1.0
        assert not server.checkpoint_participant.is_open("rs-1")
        assert server._session_episodes == {}
        assert manager.python_sessions == {}

    def test_checkpoint_verify_releases_active_identity_but_allows_score_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_gym.server_runtime.base_resources_server as base_resources

        server, app = _session_server(FakePythonToolManager())
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        server.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test-only"}}
        server.setup_resources_checkpoint(app)
        client = TestClient(app)
        prefix = "/ng-rollout/rollout"
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post(prefix + "/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        assert server.checkpoint_participant.is_open("rs-1")

        first = client.post(prefix + "/verify", json=VERIFY_BODY, headers={"x-ng-checkpoint-close-follows": "1"})
        assert first.status_code == 200 and first.json()["reward"] == 1.0
        # Default wait verification unbinds before the HTTP reply. A replacement may never send close.
        assert not server.checkpoint_participant.is_open("rs-1")
        assert server._session_episodes == {}
        assert server.session_identity("rs-1") == (
            EPISODE_ID,
            TaskId.model_validate(_seed("rs-1")["task_id"]),
        )
        second = client.post(prefix + "/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["reward"] == 1.0
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 409
        assert client.post(prefix + "/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 404
        wrong_close = _close("rs-1") | {"episode_id": {"rollout_id": "different", "attempt": 0}}
        assert client.post(prefix + "/close_session", json=wrong_close).status_code == 409

        now = base_resources.monotonic()
        monkeypatch.setattr(
            base_resources, "monotonic", lambda: now + server.config.session_close_retry_window_seconds + 1
        )
        assert not server.session_is_verified("rs-1")

    async def test_checkpoint_retire_releases_owned_python_state_and_retries_failure(self) -> None:
        tool_manager = FakePythonToolManager()
        server, _ = _session_server(tool_manager)
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        request.json = AsyncMock(return_value={"code": "print(4)"})
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        await server.execute_tool("stateful_python_code_exec", request)
        participant = ResourcesParticipant(server, "restart_only")
        participant.seeded("rs-1", EPISODE_ID)
        tool_manager.cleanup_error = RuntimeError("sandbox unavailable")
        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            await participant.retire(EPISODE_ID)
        assert "rs-1" in participant._sessions and "rs-1" in tool_manager.python_sessions
        tool_manager.cleanup_error = None
        await participant.retire(EPISODE_ID)
        assert tool_manager.python_sessions == {} and server._timing_by_session == {}
        assert server._session_episodes == {} and participant._sessions == {}

    @pytest.mark.parametrize("cancel_mode", ["anyio", "asyncio", "asyncio_close_retry"])
    async def test_cancelled_python_call_keeps_ownership_until_close(
        self, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch, cancel_mode: str
    ) -> None:
        received, finish = asyncio.Event(), asyncio.Event()
        sessions, scopes, cancellations = set(), [], []

        async def execute(request: web.Request) -> web.Response:
            await request.json()
            received.set()
            await finish.wait()
            sessions.add(request.headers["X-Session-ID"])
            return web.json_response(
                {"process_status": "completed", "stdout": "4", "stderr": "", "new_session_created": True}
            )

        async def delete(request: web.Request) -> web.Response:
            session_id = request.match_info["session_id"]
            if session_id not in sessions:
                return web.Response(status=404)
            sessions.remove(session_id)
            return web.Response(status=200)

        backend = web.Application()
        backend.router.add_post("/execute", execute)
        backend.router.add_delete("/sessions/{session_id}", delete)
        runner = web.AppRunner(backend)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        http_client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        manager = NSToolsToolManager(
            module_specs=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
            context={"sandbox": {"host": "127.0.0.1", "port": str(unused_tcp_port)}},
        )
        await manager.list_all_tools()
        tool = manager._tools["DirectPythonTool"]
        server, _ = _session_server(manager)
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        request.json = AsyncMock(return_value={"code": "print(4)"})
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)

        async def run_tool() -> None:
            with CancelScope() as scope:
                scopes.append(scope)
                try:
                    await server.execute_tool("stateful_python_code_exec", request)
                except get_cancelled_exc_class():
                    cancellations.append(True)
                    raise

        call = asyncio.create_task(run_tool())
        close = None
        try:
            await asyncio.wait_for(received.wait(), timeout=5)
            if cancel_mode == "anyio":
                scopes[0].cancel()
            else:
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
            # A close that races the cancelled call must not DELETE before the POST creates the python session.
            close = asyncio.create_task(
                server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            )
            await asyncio.sleep(0.05)
            assert not close.done()
            if cancel_mode == "anyio":
                assert not call.done()
            if cancel_mode == "asyncio_close_retry":
                close.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await close
                assert not server.session_is_closed("rs-1")
                close = asyncio.create_task(
                    server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
                )
            finish.set()
            if cancel_mode == "anyio":
                await asyncio.wait_for(call, timeout=5)
            await asyncio.wait_for(close, timeout=5)
            assert cancellations == [True]
            assert manager._python_requests == {}
            assert sessions == set()
            assert "rs-1" not in tool.requests_to_sessions
        finally:
            finish.set()
            await asyncio.gather(call, *([close] if close else []), return_exceptions=True)
            await tool._sandbox.close()
            await http_client.close()
            await runner.cleanup()

    async def test_python_transport_error_keeps_generated_session_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = []

        async def failed_send(sandbox, body, timeout):
            sent.append(body["session_id"])
            raise RuntimeError("reply could not be decoded")

        monkeypatch.setattr("nemo_skills.code_execution.sandbox.LocalSandbox._send_request", failed_send)
        manager = NSToolsToolManager(module_specs=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"])
        await manager.list_all_tools()
        tool = manager._tools["DirectPythonTool"]
        try:
            await manager.execute_tool("stateful_python_code_exec", {"code": "print(4)"}, {"request_id": "rs-1"})
            assert len(sent) == 1
            assert tool.requests_to_sessions["rs-1"] == sent[0]
        finally:
            await tool._sandbox.close()

    @pytest.mark.parametrize("deleted_status", [200, 404])
    async def test_real_python_tool_retains_failed_deletion_for_retry(
        self, unused_tcp_port: int, deleted_status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app._DELETE_RETRY_DELAY_SECONDS", 0)
        status = 503
        seen = []

        async def delete_session(request: web.Request) -> web.Response:
            seen.append((request.match_info["session_id"], request.headers.get("X-Session-ID")))
            return web.Response(status=status)

        backend = web.Application()
        backend.router.add_delete("/sessions/{session_id}", delete_session)
        runner = web.AppRunner(backend)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        http_client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        tool_manager = NSToolsToolManager(
            module_specs=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
            context={"sandbox": {"host": "127.0.0.1", "port": str(unused_tcp_port)}},
        )
        tool = tool_manager._tools["DirectPythonTool"]
        tool.requests_to_sessions["rs-1"] = "python-session"
        tool._sandbox.session_histories["python-session"] = ["x = 1"]
        server, _ = _session_server(tool_manager)
        server.restore_session_identity("rs-1", episode_id=EPISODE_ID, task_id=TaskId(taskset="ns_tools", task_id="0"))
        try:
            with pytest.raises(ClientResponseError, match="503"):
                await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            assert tool.requests_to_sessions["rs-1"] == "python-session"
            assert tool._sandbox.session_histories["python-session"] == ["x = 1"]
            assert not server.session_is_closed("rs-1")

            status = deleted_status
            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            assert "rs-1" not in tool.requests_to_sessions
            assert "python-session" not in tool._sandbox.session_histories
            assert server.session_is_closed("rs-1")
            # Three attempts for the failed close, one for the close that succeeds.
            assert seen == [("python-session", "python-session")] * 4
        finally:
            await tool._sandbox.close()
            await http_client.close()
            await runner.cleanup()

    async def test_delete_to_an_unreachable_sandbox_gives_up(
        self, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing listens on the port, so every DELETE is refused.
        monkeypatch.setattr("app._DELETE_RETRY_DELAY_SECONDS", 0)
        tries_per_call = []

        async def counted_request(*args: Any, **kwargs: Any) -> Any:
            tries_per_call.append(kwargs["_max_num_tries"])
            return await http_request(*args, **kwargs)

        monkeypatch.setattr("app.http_request", counted_request)
        monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
        http_client = set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        tool_manager = NSToolsToolManager(
            module_specs=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"],
            context={"sandbox": {"host": "127.0.0.1", "port": str(unused_tcp_port)}},
        )
        tool = tool_manager._tools["DirectPythonTool"]
        tool.requests_to_sessions["rs-1"] = "python-session"
        server, _ = _session_server(tool_manager)
        try:
            with pytest.raises(ClientOSError):
                await asyncio.wait_for(
                    server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1"))),
                    timeout=10,
                )
            # Refused connections and 5xx replies share one budget of three single-try attempts.
            assert tries_per_call == [1, 1, 1]
            assert tool.requests_to_sessions["rs-1"] == "python-session"
            assert not server.session_is_closed("rs-1")
        finally:
            await tool._sandbox.close()
            await http_client.close()

    async def test_python_session_release_retries_a_5xx_reply(
        self, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app._DELETE_RETRY_DELAY_SECONDS", 0)
        sandbox = FakeSandboxEndpoint(delete_statuses=[502])
        async with _real_python_tool_server(unused_tcp_port, sandbox, monkeypatch) as (server, tool, request):
            [python_session] = sandbox.sessions

            # A proxy's 502 while a sandbox worker restarts must not turn the scored verify into a 500.
            verified = await server.verify(request, NSToolsVerifyRequest.model_validate(VERIFY_BODY))

            assert verified.reward == 1.0 and verified.num_tool_calls == 1
            assert sandbox.deletes == [python_session] * 2
            assert sandbox.sessions == set() and "rs-1" not in tool.requests_to_sessions
            assert server._session_episodes == {}

    async def test_python_session_release_gives_up_after_three_5xx_replies(
        self, unused_tcp_port: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app._DELETE_RETRY_DELAY_SECONDS", 0)
        sandbox = FakeSandboxEndpoint(delete_statuses=[503] * 3)
        async with _real_python_tool_server(unused_tcp_port, sandbox, monkeypatch) as (server, tool, request):
            [python_session] = sandbox.sessions

            verified = await server.verify(request, NSToolsVerifyRequest.model_validate(VERIFY_BODY))
            assert verified.reward == 1.0 and verified.num_tool_calls == 1

            assert sandbox.deletes == [python_session] * 3
            # Cleanup failed, so the Python session and the active identity stay owned for the close.
            assert str(tool.requests_to_sessions["rs-1"]) == python_session
            assert sandbox.sessions == {python_session}
            assert server._session_episodes["rs-1"] == (EPISODE_ID, TaskId(taskset="ns_tools", task_id="0"))

            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))

            assert sandbox.deletes == [python_session] * 4
            assert sandbox.sessions == set() and "rs-1" not in tool.requests_to_sessions
            assert server.session_is_closed("rs-1")

    def test_failed_cleanup_at_verify_preserves_score_and_keeps_cleanup_owned(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "x = 1"}).status_code == 200
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        tool_manager.cleanup_error = RuntimeError("sandbox unreachable")

        verified = client.post("/verify", json=VERIFY_BODY)

        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        # The python session stays owned, so the close retries the release and reports its failure.
        assert "rs-1" in tool_manager.python_sessions
        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        tool_manager.cleanup_error = None
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert tool_manager.python_sessions == {}

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
        assert not server.session_is_closed("rs-1")
        assert "rs-1" in tool_manager.python_sessions

        tool_manager.cleanup_error = None
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert "rs-1" not in tool_manager.python_sessions
        assert server.session_is_closed("rs-1")

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

    async def test_close_waits_for_tool_ownership_and_retries_cleanup(self) -> None:
        tool_manager = FakePythonToolManager()
        server, _ = _session_server(tool_manager)
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        request.json = AsyncMock(return_value={"code": "print(1)"})
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        started, finish = asyncio.Event(), asyncio.Event()

        async def pause_execution(request_id: str) -> None:
            started.set()
            await finish.wait()

        tool_manager.during_execute = pause_execution
        tool_call = asyncio.create_task(server.execute_tool("stateful_python_code_exec", request))
        await started.wait()
        tool_manager.cleanup_error = RuntimeError("sandbox unreachable")
        close = asyncio.create_task(
            server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        )
        await asyncio.sleep(0)
        closed_before_publish = close.done()
        finish.set()
        try:
            assert not closed_before_publish
            assert (await tool_call).body == b"4"
            with pytest.raises(RuntimeError, match="sandbox unreachable"):
                await close
            assert "rs-1" in tool_manager.python_sessions
            assert not server.session_is_closed("rs-1")
            tool_manager.cleanup_error = None
            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            assert tool_manager.python_sessions == {}
            assert "rs-1" not in server._timing_by_session
        finally:
            await asyncio.gather(tool_call, close, return_exceptions=True)

    @pytest.mark.parametrize("first_call_fails", [False, True])
    def test_verify_retry_keeps_metrics_after_detailed_timings_are_released(self, first_call_fails: bool) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        recorded = server._timing_by_session["rs-1"][0]
        expected_seconds = recorded["execution_time_seconds"]
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(
            side_effect=[RuntimeError("verifier reply lost") if first_call_fails else verifier_reply, verifier_reply]
        )

        first = client.post("/verify", json=VERIFY_BODY)
        assert first.status_code == (500 if first_call_fails else 200)
        # Either the delegated reply or this first HTTP response can be lost. The caller retries the same body.
        second = client.post("/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["reward"] == 1.0
        assert second.json()["num_tool_calls"] == 1
        assert second.json()["total_tool_execution_time_seconds"] == expected_seconds
        assert server._timing_by_session == {} and tool_manager.python_sessions == {}
        assert server.server_client.post.await_count == 2

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

    @pytest.mark.parametrize("verified", [False, True])
    def test_typed_seed_cannot_take_over_existing_cookie_state(self, verified: bool) -> None:
        manager = FakePythonToolManager()
        server, app = _session_server(manager)
        client = TestClient(app)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        [session_id] = manager.python_sessions
        python_session = manager.python_sessions[session_id]
        timings = server._timing_by_session[session_id]
        if verified:
            assert client.post("/verify", json=VERIFY_BODY).json()["num_tool_calls"] == 1

        assert client.post("/seed_session", json=_seed(session_id)).status_code == 409
        assert server._session_episodes == {}
        if verified:
            assert server.session_is_verified(session_id)
            assert manager.python_sessions == {}
        else:
            assert manager.python_sessions[session_id] is python_session
            assert server._timing_by_session[session_id] is timings
            assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        assert client.post("/verify", json=VERIFY_BODY).json()["num_tool_calls"] == (1 if verified else 2)

    def test_cookie_reseed_keeps_id_and_replaces_verification_metrics(self) -> None:
        manager = FakePythonToolManager()
        server, app = _session_server(manager)
        client = TestClient(app)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)

        # SimpleAgent /run forwards its caller's cookies to seed, then uses the returned cookies for tools/verify.
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        [first_session] = manager.python_sessions
        first = client.post("/verify", json=VERIFY_BODY)
        assert first.status_code == 200 and first.json()["num_tool_calls"] == 1
        assert manager.python_sessions == {} and server._timing_by_session == {}
        assert client.post("/verify", json=VERIFY_BODY).json() == first.json()

        # Deprecated cookie seeds reuse the ID but start a new run and discard the old retry record.
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert not server.session_is_verified(first_session)
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        [second_session] = manager.python_sessions
        assert second_session == first_session
        second = client.post("/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["num_tool_calls"] == 2
        assert manager.python_sessions == {} and server._timing_by_session == {}

    def test_failed_cookie_reseed_keeps_verification_metrics_and_python_ownership(self) -> None:
        manager = FakePythonToolManager()
        server, app = _session_server(manager)
        client = TestClient(app, raise_server_exceptions=False)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        [session_id] = manager.python_sessions
        python_session = manager.python_sessions[session_id]
        manager.cleanup_error = RuntimeError("sandbox release unavailable")
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0

        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 500
        assert server.session_is_verified(session_id)
        assert manager.python_sessions[session_id] is python_session
        manager.cleanup_error = None
        assert client.post("/verify", json=VERIFY_BODY).json()["num_tool_calls"] == 1
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert not server.session_is_verified(session_id)
        assert manager.python_sessions == {}

    def test_failed_python_release_keeps_metrics_owned_past_retry_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_gym.server_runtime.base_resources_server as base_resources

        manager = FakePythonToolManager()
        server, app = _session_server(manager)
        client = TestClient(app, raise_server_exceptions=False)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/stateful_python_code_exec", json={"code": "print(4)"}).status_code == 200
        [session_id] = manager.python_sessions
        manager.cleanup_error = RuntimeError("sandbox release unavailable")
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
        now = base_resources.monotonic()
        monkeypatch.setattr(
            base_resources, "monotonic", lambda: now + server.config.session_close_retry_window_seconds + 1
        )
        assert server.session_is_verified(session_id)

        assert client.post("/seed_session", json=_seed(session_id)).status_code == 409
        assert server._session_episodes == {}
        assert session_id in manager.python_sessions
        assert server._timing_by_session[session_id] == []
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 500
        assert server._timing_by_session[session_id] == []
        manager.cleanup_error = None
        assert client.post("/verify", json=VERIFY_BODY).json()["num_tool_calls"] == 1
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert manager.python_sessions == {} and server._timing_by_session == {}

    @pytest.mark.parametrize("state", ["active", "verified", "closed"])
    def test_cookie_seed_does_not_take_over_a_typed_cookie(self, state: str) -> None:
        manager = FakePythonToolManager()
        server, app = _session_server(manager)
        client = TestClient(app)
        verifier_reply = AsyncMock()
        verifier_reply.json = AsyncMock(return_value={"reward": 1.0})
        server.server_client.post = AsyncMock(return_value=verifier_reply)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        if state == "verified":
            assert client.post("/verify", json=VERIFY_BODY).status_code == 200
        elif state == "closed":
            assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        cookies = dict(client.cookies)

        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 409
        # SessionMiddleware may refresh a cookie's signature timestamp, but its encoded session is unchanged.
        assert {name: value.split(".")[0] for name, value in client.cookies.items()} == {
            name: value.split(".")[0] for name, value in cookies.items()
        }
        assert manager.python_sessions == {}

    def test_cookie_seed_keeps_working(self) -> None:
        tool_manager = FakePythonToolManager()
        server, app = _session_server(tool_manager)
        client = TestClient(app)

        # An Agent's /run seeds with the row itself.
        seeded = client.post("/seed_session", json=VERIFY_BODY | {"verifier_type": "math_with_judge"})
        assert seeded.status_code == 200 and seeded.json() == {}
        assert server._session_episodes == {}
        assert client.post("/stateful_python_code_exec", json={"code": "print(1)"}).status_code == 200
        assert len(tool_manager.python_sessions) == 1
