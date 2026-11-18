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
import multiprocessing
import signal
import socket
import threading
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from app import (
    ExecutePythonRequest,
    PythonExecutorResourcesServer,
    PythonExecutorResourcesServerConfig,
    PythonMathVerifyRequest,
    _answers_match,
    _extract_boxed_answer,
    _normalize_answer,
    _SessionHandle,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesParticipant,
    ResourcesSeedSessionRequest,
    ResourcesSessionLostError,
    check_resources_session_contract,
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
        PythonExecutorResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"The answer is \boxed{42}.", "42"),
            (r"First \boxed{41}, then \boxed{42}.", "42"),
            (r"The answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
            (r"The answer is \boxed{}.", None),
            (r"The answer is \boxed{42", None),
            ("The answer is 42.", None),
        ],
        ids=[
            "single-boxed-answer",
            "last-boxed-answer",
            "nested-braces",
            "empty-box",
            "unclosed-box",
            "missing-box",
        ],
    )
    def test_extract_boxed_answer(
        self,
        text: str,
        expected: str | None,
    ) -> None:
        assert _extract_boxed_answer(text) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("42", "42"),
            ("  42  ", "42"),
            (r"\(42\)", "42"),
            ("$42$", "42"),
            (r"\text{42}", "42"),
            ("1   2", "1 2"),
        ],
        ids=[
            "plain",
            "whitespace",
            "latex-parentheses",
            "dollar-delimiters",
            "text-wrapper",
            "internal-whitespace",
        ],
    )
    def test_normalize_answer(
        self,
        answer: str,
        expected: str,
    ) -> None:
        assert _normalize_answer(answer) == expected

    @pytest.mark.parametrize(
        ("actual", "expected", "matches"),
        [
            ("42", "42", True),
            (r"\(42\)", "42", True),
            ("42", "42.0", True),
            ("0.5", ".5", True),
            ("43", "42", False),
            ("not-a-number", "42", False),
            (None, "42", False),
        ],
        ids=[
            "exact-match",
            "normalized-exact-match",
            "numeric-equivalent",
            "decimal-equivalent",
            "numeric-mismatch",
            "non-numeric-mismatch",
            "missing-answer",
        ],
    )
    def test_answers_match(
        self,
        actual: str | None,
        expected: str,
        matches: bool,
    ) -> None:
        assert _answers_match(actual, expected) is matches


def _session_server() -> tuple[PythonExecutorResourcesServer, FastAPI]:
    config = PythonExecutorResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="math_with_code")
    server = PythonExecutorResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server, server.setup_webserver()


EPISODE_ID = EpisodeId(rollout_id="rollout", attempt=0)


def _seed(resources_session_id: str) -> Dict[str, Any]:
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EPISODE_ID,
        task_id=TaskId(taskset="math_with_code", task_id="0"),
        task_data={"expected_result": "4"},
    ).model_dump(mode="json")


def _close(resources_session_id: str) -> Dict[str, Any]:
    return ResourcesCloseSessionRequest(resources_session_id=resources_session_id, episode_id=EPISODE_ID).model_dump(
        mode="json"
    )


VERIFY_BODY = {
    "responses_create_params": {"input": [{"role": "user", "content": "What is 2 + 2?"}]},
    "response": {
        "id": "resp_test",
        "created_at": 0.0,
        "model": "dummy",
        "object": "response",
        "output": [
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": "\\boxed{4}", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    },
    "expected_result": "4",
}


class _UnstoppableHandle:
    """A registered worker whose close fails the way a process that outlives SIGKILL does."""

    def __init__(self) -> None:
        self.stoppable = False
        self.close_attempts = 0

    async def close(self) -> None:
        self.close_attempts += 1
        if not self.stoppable:
            raise RuntimeError("Python worker process 1234 is still alive after SIGKILL")


class TestResourcesSessions:
    async def test_request_scored_verify_retries_after_scoring_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        handle = _UnstoppableHandle()
        handle.stoppable = True
        server._sessions["rs-1"] = handle
        calls = 0

        def fail_once_then_score(actual, expected):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("scoring unavailable")
            return _answers_match(actual, expected)

        monkeypatch.setattr("app._answers_match", fail_once_then_score)
        body = PythonMathVerifyRequest.model_validate(VERIFY_BODY)
        with pytest.raises(ValueError, match="scoring unavailable"):
            await server.verify(request, body)
        assert server._sessions == {} and handle.close_attempts == 1
        retried = await server.verify(request, body)
        assert retried.reward == 1.0 and not retried.mask_sample
        assert calls == 2
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))

    async def test_verify_cleanup_prevents_late_worker_allocation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        stopped, release = asyncio.Event(), asyncio.Event()
        original_cleanup = PythonExecutorResourcesServer._cleanup_session

        async def delayed_cleanup(server, session_id):
            await original_cleanup(server, session_id)
            stopped.set()
            await asyncio.wait_for(release.wait(), timeout=5)

        monkeypatch.setattr(PythonExecutorResourcesServer, "_cleanup_session", delayed_cleanup)
        verification = asyncio.create_task(server.verify(request, PythonMathVerifyRequest.model_validate(VERIFY_BODY)))
        await asyncio.wait_for(stopped.wait(), timeout=5)
        tool = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
        try:
            await asyncio.sleep(0.05)
            assert not tool.done() and server._sessions == {}
            release.set()
            assert (await verification).reward == 1.0
            with pytest.raises(ResourcesSessionLostError):
                await tool
            assert server._sessions == {}
        finally:
            release.set()
            await asyncio.gather(verification, tool, return_exceptions=True)

    def test_checkpoint_failed_verify_cleanup_stays_owned_until_retry(self) -> None:
        server, app = _session_server()
        server.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test-only"}}
        server.setup_resources_checkpoint(app)
        client = TestClient(app, raise_server_exceptions=False)
        prefix = "/ng-rollout/rollout"
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 200
        handle = _UnstoppableHandle()
        server._sessions["rs-1"] = handle
        first = client.post(prefix + "/verify", json=VERIFY_BODY, headers={"x-ng-checkpoint-close-follows": "1"})
        assert first.status_code == 200 and first.json()["reward"] == 1.0
        assert server.checkpoint_participant.is_open("rs-1")
        assert "rs-1" in server._session_episodes
        assert server._sessions["rs-1"] is handle
        handle.stoppable = True
        second = client.post(prefix + "/verify", json=VERIFY_BODY)
        assert second.status_code == 200 and second.json()["reward"] == 1.0
        assert not server.checkpoint_participant.is_open("rs-1")
        assert server._session_episodes == {}
        assert server._sessions == {}

    def test_checkpoint_verify_releases_active_identity_but_allows_score_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_gym.server_runtime.base_resources_server as base_resources

        server, app = _session_server()
        server.server_client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "test-only"}}
        server.setup_resources_checkpoint(app)
        client = TestClient(app)
        prefix = "/ng-rollout/rollout"
        assert client.post(prefix + "/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post(prefix + "/execute_python", json={"code": "print(4)"}).status_code == 200
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
        assert client.post(prefix + "/execute_python", json={"code": "print(4)"}).status_code == 404
        wrong_close = _close("rs-1") | {"episode_id": {"rollout_id": "different", "attempt": 0}}
        assert client.post(prefix + "/close_session", json=wrong_close).status_code == 409

        now = base_resources.monotonic()
        monkeypatch.setattr(
            base_resources, "monotonic", lambda: now + server.config.session_close_retry_window_seconds + 1
        )
        assert not server.session_is_verified("rs-1")

    async def test_checkpoint_retire_releases_owned_worker_and_retries_failure(self) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        handle = _UnstoppableHandle()
        server._sessions["rs-1"] = handle
        participant = ResourcesParticipant(server, "restart_only")
        participant.seeded("rs-1", EPISODE_ID)
        with pytest.raises(RuntimeError):
            await participant.retire(EPISODE_ID)
        assert server._sessions["rs-1"] is handle and "rs-1" in participant._sessions
        handle.stoppable = True
        await participant.retire(EPISODE_ID)
        assert server._sessions == {} and server._session_episodes == {} and participant._sessions == {}

    async def test_tool_waiting_for_close_does_not_create_a_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        stopped, finish_close = asyncio.Event(), asyncio.Event()
        real_cleanup = PythonExecutorResourcesServer._cleanup_session

        async def delayed_cleanup(server, session_id):
            await real_cleanup(server, session_id)
            stopped.set()
            await asyncio.wait_for(finish_close.wait(), timeout=5)

        worker_factory = MagicMock()
        worker_factory.return_value.exec.return_value = ("", "", "1")
        monkeypatch.setattr(PythonExecutorResourcesServer, "_cleanup_session", delayed_cleanup)
        monkeypatch.setattr("app._SessionHandle", worker_factory)
        close = asyncio.create_task(
            server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
        )
        await asyncio.wait_for(stopped.wait(), timeout=5)
        call = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
        try:
            await asyncio.sleep(0)
            worker_factory.assert_not_called()
            finish_close.set()
            await close
            with pytest.raises(ResourcesSessionLostError):
                await call
            assert server._sessions == {}
        finally:
            finish_close.set()
            await asyncio.gather(close, call, return_exceptions=True)

    async def test_worker_starts_off_the_event_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        worker = MagicMock()
        worker.exec.return_value = ("", "", "1")

        def start_worker(max_execution_time: int) -> MagicMock:
            # The first worker waits while the forkserver starts and imports the server module.
            time.sleep(1)
            return worker

        monkeypatch.setattr("app._SessionHandle", start_worker)
        call = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
        started = time.monotonic()
        await asyncio.sleep(0.1)

        assert time.monotonic() - started < 0.5
        assert (await call).result == "1"
        assert server._sessions == {"rs-1": worker}

    async def test_cancelled_worker_start_remains_owned_until_close(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        started, finish_start, constructed = threading.Event(), threading.Event(), threading.Event()
        handles = []

        def start_worker(max_execution_time: int) -> _SessionHandle:
            started.set()
            assert finish_start.wait(15)
            handle = _SessionHandle(max_execution_time)
            handles.append(handle)
            constructed.set()
            return handle

        monkeypatch.setattr("app._SessionHandle", start_worker)
        call = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1")))
        close = None
        try:
            assert await asyncio.to_thread(started.wait, 5)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            # The constructor thread still runs after cancellation, so its future must remain discoverable.
            assert server.has_session_state("rs-1")
            close = asyncio.create_task(
                server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            )
            await asyncio.sleep(0)
            assert not close.done()
            finish_start.set()
            await asyncio.wait_for(close, 15)
            assert len(handles) == 1 and not handles[0]._proc.is_alive()
            assert not server.has_session_state("rs-1")
            assert server.session_is_closed("rs-1")
        finally:
            finish_start.set()
            await asyncio.gather(call, *([close] if close else []), return_exceptions=True)
            assert await asyncio.to_thread(constructed.wait, 15)
            for handle in handles:
                await handle.close()

    def test_session_contract(self) -> None:
        server, app = _session_server()

        check_resources_session_contract(
            app,
            ResourcesSeedSessionRequest.model_validate(_seed("contract-session")),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._session_episodes,
            verify_body=VERIFY_BODY,
        )

    def test_close_without_verify_stops_the_worker(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        executed = client.post("/execute_python", json={"code": "x = 2 + 2\nx"})
        assert executed.status_code == 200 and executed.json()["result"] == "4"
        worker = server._sessions["rs-1"]._proc
        assert worker.is_alive()

        # An Environment Server whose seed reply was lost closes without the cookie.
        client.cookies.clear()
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        assert not worker.is_alive()
        assert server._sessions == {}
        assert "rs-1" not in server._session_episodes

    @pytest.mark.parametrize("cancel_first", [False, True])
    async def test_parallel_calls_keep_their_own_pipe_replies(self, monkeypatch, cancel_first: bool) -> None:
        handle = _SessionHandle(max_execution_time=10)
        first_receiving, finish_first = threading.Event(), threading.Event()
        second_started, first_done = threading.Event(), threading.Event()
        real_recv = handle._conn.recv
        calls = 0

        def delayed_recv():
            nonlocal calls
            calls += 1
            if calls == 1:
                first_receiving.set()
                assert finish_first.wait(10)
            return real_recv()

        def first_exec():
            try:
                return handle.exec("1")
            finally:
                first_done.set()

        def second_exec():
            second_started.set()
            return handle.exec("2")

        monkeypatch.setattr(handle._conn, "recv", delayed_recv)
        first = asyncio.create_task(asyncio.to_thread(first_exec))
        second = None
        try:
            assert await asyncio.to_thread(first_receiving.wait, 10)
            if cancel_first:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            second = asyncio.create_task(asyncio.to_thread(second_exec))
            assert await asyncio.to_thread(second_started.wait, 10)
            # The second call cannot read the first response while its original caller is paused or cancelled.
            await asyncio.sleep(0.05)
            assert not second.done()
            finish_first.set()
            assert (await asyncio.wait_for(second, 10))[2] == "2"
            if not cancel_first:
                assert (await first)[2] == "1"
        finally:
            finish_first.set()
            assert await asyncio.to_thread(first_done.wait, 10)
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
            await handle.close()

    async def test_worker_busy_in_user_code_is_terminated(self) -> None:
        handle = _SessionHandle(max_execution_time=60)
        assert handle.exec("2 + 2")[2] == "4"
        # Send the cell without waiting for its reply, so the worker is busy when the close arrives.
        handle._conn.send({"cmd": "exec", "code": "import time\ntime.sleep(60)"})
        await asyncio.sleep(0.5)

        started = time.monotonic()
        await handle.close()

        assert not handle._proc.is_alive()
        assert time.monotonic() - started < 10

    async def test_forked_worker_does_not_keep_the_servers_sigterm_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Under uvicorn the server has a Python SIGTERM handler, which a forked worker would inherit.
        fork_context = multiprocessing.get_context("fork")
        monkeypatch.setattr("app.multiprocessing.get_context", lambda method: fork_context)
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: None)
        try:
            handle = _SessionHandle(max_execution_time=60)
        finally:
            signal.signal(signal.SIGTERM, previous)
        assert handle.exec("2 + 2")[2] == "4"
        handle._conn.send({"cmd": "exec", "code": "import time\ntime.sleep(60)"})
        await asyncio.sleep(0.5)

        started = time.monotonic()
        await handle.close()

        assert not handle._proc.is_alive()
        # 1 s for the close message, then terminate() ends it at once; only SIGKILL would take 5 s more.
        assert time.monotonic() - started < 3
        assert handle._proc.exitcode == -signal.SIGTERM

    async def test_worker_does_not_retain_server_sockets(self) -> None:
        # The socket is open when the worker starts, as the server's sockets are; a worker forked from this process
        # would keep its own copy open.
        server_socket, peer = socket.socketpair()
        peer.settimeout(1)
        handle = _SessionHandle(max_execution_time=10)

        def execute(code: str) -> str:
            handle._conn.send({"cmd": "exec", "code": code})
            assert handle._conn.poll(60), "Python worker did not reply"
            reply = handle._conn.recv()
            assert reply["ok"], reply
            return reply["res"]

        try:
            assert execute("value = 40\nvalue") == "40"
            server_socket.close()
            # The live worker must not keep the server's connection open after the server closes its copy.
            assert handle._proc.is_alive()
            assert peer.recv(1) == b""
            assert execute("value += 2\nvalue") == "42"
        finally:
            server_socket.close()
            peer.close()
            await handle.close()

    async def test_worker_ignoring_sigterm_is_killed_within_bound(self) -> None:
        handle = _SessionHandle(max_execution_time=60)
        assert handle.exec("1")[2] == "1"
        code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(120)"
        handle._conn.send({"cmd": "exec", "code": code})
        await asyncio.sleep(0.5)

        started = time.monotonic()
        await handle.close()
        elapsed = time.monotonic() - started

        assert not handle._proc.is_alive()
        assert handle._proc.exitcode == -signal.SIGKILL
        # 1 s for the close message and 5 s for terminate(), then kill() ends it at once.
        assert 5.5 < elapsed < 8, elapsed

    async def test_release_does_not_queue_behind_blocked_execs(self) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        assert (await server.execute_python(request, ExecutePythonRequest(code="1"))).result == "1"
        worker = server._sessions["rs-1"]._proc
        # Model code can catch the worker's alarm, so its exec holds a default-executor thread until the worker stops.
        code = "import time\nwhile True:\n    try:\n        time.sleep(60)\n    except:\n        pass"
        stuck = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code=code)))
        await asyncio.sleep(0.5)
        # Stand-ins for other sessions' blocked execs take the rest of the default executor's threads.
        unblock = threading.Event()
        blocked = [asyncio.get_running_loop().run_in_executor(None, unblock.wait) for _ in range(64)]
        try:
            started = time.monotonic()
            await asyncio.wait_for(server.release_session_state("rs-1"), timeout=11)

            # The busy worker could not read the close message, so terminate() stopped it after the first wait.
            assert time.monotonic() - started < 6
            assert worker.exitcode == -signal.SIGTERM and server._sessions == {}
            assert not (await stuck).success
        finally:
            unblock.set()
            if worker.is_alive():
                worker.kill()
            await asyncio.gather(stuck, *blocked, return_exceptions=True)

    async def test_worker_that_outlives_sigkill_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handle = _SessionHandle(max_execution_time=10)
        real_proc = handle._proc
        handle._proc = MagicMock(spec=multiprocessing.Process, pid=real_proc.pid)
        handle._proc.is_alive.return_value = True
        # Skip the waits, which would only time out against the mock.
        monkeypatch.setattr(handle, "_wait_for_exit", AsyncMock())
        try:
            with pytest.raises(RuntimeError, match="still alive after SIGKILL"):
                await handle.close()
            handle._proc.terminate.assert_called_once()
            handle._proc.kill.assert_called_once()
        finally:
            monkeypatch.undo()
            handle._proc = real_proc
            await handle.close()

    def test_failed_stop_leaves_session_open_for_a_retry(self) -> None:
        server, app = _session_server()
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        handle = _UnstoppableHandle()
        server._sessions["rs-1"] = handle

        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        assert not server.session_is_closed("rs-1")
        assert server._sessions["rs-1"] is handle

        handle.stoppable = True
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert server._sessions == {}
        assert server.session_is_closed("rs-1")

    def test_failed_stop_at_verify_preserves_score_and_keeps_cleanup_owned(self) -> None:
        server, app = _session_server()
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        handle = _UnstoppableHandle()
        server._sessions["rs-1"] = handle

        verified = client.post("/verify", json=VERIFY_BODY)

        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert handle.close_attempts == 1
        # The worker stays registered, so the close retries the stop and reports a failure.
        assert server._sessions["rs-1"] is handle
        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        handle.stoppable = True
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert server._sessions == {}

    def test_end_session_stops_the_worker_without_fencing_the_session(self) -> None:
        server, app = _session_server()
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/execute_python", json={"code": "1 + 1"}).status_code == 200
        worker = server._sessions["rs-1"]._proc
        assert worker.is_alive()

        assert client.post("/end_session").status_code == 200
        assert not worker.is_alive()
        assert server._sessions == {}
        assert client.post("/end_session").status_code == 200

        # end_session does not fence: the next tool call starts a fresh worker.
        assert client.post("/execute_python", json={"code": "2 + 2\n2 + 2"}).json()["result"] == "4"
        assert server._sessions["rs-1"]._proc is not worker
        handle = _UnstoppableHandle()
        server._sessions["rs-1"], previous = handle, server._sessions["rs-1"]
        assert client.post("/end_session").status_code == 500
        assert server._sessions["rs-1"] is handle
        handle.stoppable = True
        assert client.post("/end_session").status_code == 200
        assert server._sessions == {}
        previous._proc.kill()

    def test_execute_after_close_is_a_lost_session(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        response = client.post("/execute_python", json={"code": "1 + 1"})

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"
        assert server._sessions == {}

    async def test_execute_racing_close_stops_its_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        executed, closed = threading.Event(), threading.Event()
        started_workers = []
        real_exec = _SessionHandle.exec

        def exec_then_wait_for_close(handle: _SessionHandle, code: str):
            started_workers.append(handle._proc)
            result = real_exec(handle, code)
            executed.set()
            assert closed.wait(15)
            return result

        monkeypatch.setattr(_SessionHandle, "exec", exec_then_wait_for_close)
        call = asyncio.create_task(server.execute_python(request, ExecutePythonRequest(code="1 + 1")))
        try:
            assert await asyncio.to_thread(executed.wait, 15)
            await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))
            closed.set()
            with pytest.raises(ResourcesSessionLostError):
                await call
            assert server._sessions == {}
            assert not started_workers[0].is_alive()
        finally:
            closed.set()
            await asyncio.gather(call, return_exceptions=True)
            await server.release_session_state("rs-1")

    def test_verify_then_close(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/execute_python", json={"code": "2 + 2"}).status_code == 200
        worker = server._sessions["rs-1"]._proc

        verified = client.post("/verify", json=VERIFY_BODY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert not worker.is_alive()
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

    def test_typed_seed_cannot_take_over_existing_cookie_worker(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/execute_python", json={"code": "value = 42"}).json()["success"]
        [session_id] = server._sessions
        worker = server._sessions[session_id]
        try:
            assert client.post("/seed_session", json=_seed(session_id)).status_code == 409
            assert server._session_episodes == {}
            assert server._sessions[session_id] is worker and worker._proc.is_alive()
            assert client.post("/execute_python", json={"code": "value"}).json()["result"] == "42"
        finally:
            assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
        assert server._sessions == {} and not worker._proc.is_alive()

    def test_cookie_seed_keeps_working(self) -> None:
        server, app = _session_server()
        client = TestClient(app)

        # An Agent's /run seeds with the row itself.
        seeded = client.post("/seed_session", json=VERIFY_BODY)
        assert seeded.status_code == 200 and seeded.json() == {}
        assert server._session_episodes == {}
        assert client.post("/execute_python", json={"code": "2 + 2"}).json()["result"] == "4"
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
        assert server._sessions == {}

    def test_cookie_reseed_retries_failed_worker_release_before_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, app = _session_server()
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert client.post("/execute_python", json={"code": "value = 42"}).json()["success"]
        [session_id] = server._sessions
        worker = server._sessions[session_id]
        close = worker.close

        async def fail_close() -> None:
            raise RuntimeError("worker still alive")

        monkeypatch.setattr(worker, "close", fail_close)
        try:
            assert client.post("/seed_session", json=VERIFY_BODY).status_code == 500
            assert server._sessions[session_id] is worker and worker._proc.is_alive()
            assert client.post("/execute_python", json={"code": "value"}).json()["result"] == "42"
        finally:
            monkeypatch.setattr(worker, "close", close)
            assert client.post("/seed_session", json=VERIFY_BODY).status_code == 200
        assert server._sessions == {} and not worker._proc.is_alive()
        try:
            response = client.post("/execute_python", json={"code": "value"}).json()
            assert not response["success"] and "value" in response["error_message"]
            assert list(server._sessions) == [session_id]
        finally:
            assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
