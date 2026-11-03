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
import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from app import (
    ExecutePythonRequest,
    PythonExecutorResourcesServer,
    PythonExecutorResourcesServerConfig,
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
    ResourcesSeedSessionRequest,
    ResourcesSessionLostError,
)
from nemo_gym.testing.session_conformance import check_resources_session_contract


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

    def close(self) -> None:
        self.close_attempts += 1
        if not self.stoppable:
            raise RuntimeError("Python worker process 1234 is still alive after SIGKILL")


class TestResourcesSessions:
    async def test_tool_waiting_for_close_does_not_create_a_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(request, ResourcesSeedSessionRequest.model_validate(_seed("rs-1")))
        stopped, finish_close = asyncio.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        real_cleanup = PythonExecutorResourcesServer._cleanup_session

        def delayed_cleanup(server, session_id):
            real_cleanup(server, session_id)
            loop.call_soon_threadsafe(stopped.set)
            assert finish_close.wait(timeout=5)

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

    def test_worker_busy_in_user_code_is_terminated(self) -> None:
        handle = _SessionHandle(max_execution_time=60)
        # Send the cell without waiting for its reply, so the worker is busy when the close arrives.
        handle._conn.send({"cmd": "exec", "code": "import time\ntime.sleep(60)"})
        time.sleep(0.5)

        started = time.monotonic()
        handle.close()

        assert not handle._proc.is_alive()
        assert time.monotonic() - started < 10

    def test_worker_that_outlives_sigkill_raises(self) -> None:
        handle = _SessionHandle(max_execution_time=10)
        real_proc = handle._proc
        handle._proc = MagicMock(spec=multiprocessing.Process, pid=real_proc.pid)
        handle._proc.is_alive.return_value = True
        try:
            with pytest.raises(RuntimeError, match="still alive after SIGKILL"):
                handle.close()
            handle._proc.terminate.assert_called_once()
            handle._proc.kill.assert_called_once()
        finally:
            handle._proc = real_proc
            handle.close()

    def test_failed_stop_leaves_session_open_for_a_retry(self) -> None:
        server, app = _session_server()
        client = TestClient(app, raise_server_exceptions=False)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        handle = _UnstoppableHandle()
        server._sessions["rs-1"] = handle

        assert client.post("/close_session", json=_close("rs-1")).status_code == 500
        assert server._fence.closed_episode("rs-1") is None
        assert server._sessions["rs-1"] is handle

        handle.stoppable = True
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert server._sessions == {}
        assert server._fence.closed_episode("rs-1") == EPISODE_ID

    def test_failed_stop_at_verify_keeps_the_score(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        handle = _UnstoppableHandle()
        server._sessions["rs-1"] = handle

        verified = client.post("/verify", json=VERIFY_BODY)

        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert handle.close_attempts == 1
        # The worker stays registered, so the close retries the stop and reports a failure.
        assert server._sessions["rs-1"] is handle

    def test_execute_after_close_is_a_lost_session(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

        response = client.post("/execute_python", json={"code": "1 + 1"})

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"
        assert server._sessions == {}

    def test_execute_racing_close_stops_its_worker(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200
        started_workers = []
        real_exec = _SessionHandle.exec

        def exec_then_close(handle: _SessionHandle, code: str):
            started_workers.append(handle._proc)
            result = real_exec(handle, code)
            # The close lands while the call is still running, after it started the session's first worker.
            server._fence.mark_closed("rs-1", EPISODE_ID)
            return result

        _SessionHandle.exec = exec_then_close
        try:
            response = client.post("/execute_python", json={"code": "1 + 1"})
        finally:
            _SessionHandle.exec = real_exec

        assert response.status_code == 404
        assert server._sessions == {}
        assert not started_workers[0].is_alive()

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

    def test_legacy_seed_keeps_working(self) -> None:
        server, app = _session_server()
        client = TestClient(app)

        # An Agent's /run seeds with the row itself.
        seeded = client.post("/seed_session", json=VERIFY_BODY)
        assert seeded.status_code == 200 and seeded.json() == {}
        assert server._session_episodes == {}
        assert client.post("/execute_python", json={"code": "2 + 2"}).json()["result"] == "4"
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
        assert server._sessions == {}
