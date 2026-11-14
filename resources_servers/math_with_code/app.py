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
import logging
import multiprocessing
import threading
import time
from typing import Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.math_with_code.session_worker import _session_worker


logger = logging.getLogger(__name__)


class PythonExecutorResourcesServerConfig(BaseResourcesServerConfig):
    max_execution_time: int = 10


class ExecutePythonRequest(BaseModel):
    code: str


class ExecutePythonResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str
    error_message: Optional[str] = None
    result: Optional[str] = None


class _SessionHandle:
    """Light wrapper around one long-lived worker process."""

    def __init__(self, max_execution_time: int):
        """Start the worker; the first one blocks for seconds while the forkserver starts, so call this in a thread."""
        # A worker forked from the server would retain its listening and accepted sockets and keep idle connections
        # open. The forkserver is a fresh interpreter without them that imports the worker's module once, so each
        # worker forks in milliseconds. "__main__" preloads a server started as a script, as `__mp_main__`, so a
        # child need not run it again. This module is not preloaded by name: that import would run its entrypoint
        # branch and build a second server in the forkserver when the server runs with several workers.
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload(["__main__", _session_worker.__module__])
        parent_conn, child_conn = context.Pipe()
        self._conn = parent_conn
        self._exec_lock = threading.Lock()
        self._proc = context.Process(
            target=_session_worker,
            args=(child_conn, max_execution_time),
            daemon=True,
        )
        self._proc.start()
        self.last_used = time.time()

    def exec(self, code: str):
        # A Connection has one reply stream: parallel tool calls must not consume each other's replies.
        # Keep the lock in the executor thread so cancelling its HTTP waiter cannot release it mid-execution.
        with self._exec_lock:
            if not self._proc.is_alive():
                raise RuntimeError("Python worker has already stopped")
            self._conn.send({"cmd": "exec", "code": code})
            reply = self._conn.recv()
            self.last_used = time.time()
            if reply["ok"]:
                return reply["out"], reply["err"], reply["res"]
            raise RuntimeError(reply["error"])

    async def close(self) -> None:
        """Stop the worker, escalating to terminate and then kill; raise if it is still alive.

        It waits on the event loop rather than in an executor thread: execs hold executor threads for as long as the
        model's code runs, and a close queued behind them could never stop the workers that would free them.
        """
        # Do not write into a concurrent exec's pipe frame or wait behind its blocking recv.
        # A busy worker is stopped below even if its executor thread cannot make progress.
        if self._exec_lock.acquire(blocking=False):
            try:
                self._conn.send({"cmd": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                self._exec_lock.release()
        await self._wait_for_exit(1)
        if self._proc.is_alive():
            # Still running user code, which the alarm stops only after max_execution_time.
            self._proc.terminate()
            await self._wait_for_exit(5)
        if self._proc.is_alive():
            self._proc.kill()
            await self._wait_for_exit(5)
        if self._proc.is_alive():
            raise RuntimeError(f"Python worker process {self._proc.pid} is still alive after SIGKILL")

    async def _wait_for_exit(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while self._proc.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)


# ------------------------------


class PythonMathRunRequest(BaseRunRequest):
    expected_result: str  # Add the expected result field
    expected_code_contains: str = ""  # Optional validation


class PythonMathVerifyRequest(PythonMathRunRequest, BaseVerifyRequest):
    pass


class PythonMathVerifyResponse(BaseVerifyResponse):
    extracted_answer: Optional[str] = None
    accuracy: bool = False


class PythonExecutorResourcesServer(SimpleResourcesServer):
    # new: create the pool once

    config: PythonExecutorResourcesServerConfig

    _sessions: Dict[str, _SessionHandle] = PrivateAttr(default_factory=dict)
    _starting_sessions: Dict[str, asyncio.Task[_SessionHandle]] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/execute_python")(self.execute_python)
        app.post("/end_session")(self.end_session)
        return app

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # _sessions dict already initialised by default_factory

    async def execute_python(self, request: Request, body: ExecutePythonRequest) -> ExecutePythonResponse:
        sid = request.session[SESSION_ID_KEY]
        loop = asyncio.get_running_loop()
        # Close awaits the worker's stop before marking the fence. Serialize publication with that whole operation,
        # so a tool cannot create an untracked replacement in the gap, even if its request is cancelled.
        async with self.session_lock(sid):
            if self.session_is_closed(sid) or self.session_is_verified(sid):
                raise ResourcesSessionLostError(f"math_with_code session {sid} is closed or already verified")
            try:
                if sid not in self._sessions:
                    if sid not in self._starting_sessions:
                        self._starting_sessions[sid] = asyncio.create_task(
                            asyncio.to_thread(_SessionHandle, self.config.max_execution_time)
                        )
                    starting = self._starting_sessions[sid]
                    # Cancelling the HTTP request cannot stop its constructor thread. Keep that task owned
                    # until a later call or cleanup collects the worker it creates.
                    try:
                        self._sessions[sid] = await asyncio.shield(starting)
                    except Exception:
                        self._starting_sessions.pop(sid, None)
                        raise
                    self._starting_sessions.pop(sid, None)
                handle = self._sessions[sid]
            except Exception as e:
                logger.exception("Python worker failed to start for resources session %s", sid)
                return ExecutePythonResponse(success=False, stdout="", stderr="", error_message=str(e))
        try:
            stdout, stderr, result = await loop.run_in_executor(
                None,
                handle.exec,
                body.code,
            )
            response = ExecutePythonResponse(
                success=True,
                stdout=stdout,
                stderr=stderr,
                result=result,
            )
        except Exception as e:
            response = ExecutePythonResponse(
                success=False,
                stdout="",
                stderr="",
                error_message=str(e),
            )
        if self.session_is_closed(sid):
            # The session closed while this call ran, so its result is discarded; close already stopped the worker.
            raise ResourcesSessionLostError(f"math_with_code session {sid} closed during the call")
        return response

    async def end_session(self, request: Request) -> ExecutePythonResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.session_lock(session_id):
            await self._cleanup_session(session_id)
        return ExecutePythonResponse(success=True, stdout="", stderr="")

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._sessions or session_id in self._starting_sessions

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """The worker starts on the first tool call; the base releases a prior cookie run."""

    async def release_session_state(self, session_id: str) -> None:
        """Release the worker, retaining its reference if cleanup fails."""
        await self._cleanup_session(session_id)

    async def verify(self, request: Request, body: PythonMathVerifyRequest) -> PythonMathVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        async with self.verifying(session_id, allow_missing=True) as verification:
            if verification.is_lost:
                # An Environment Server verifies before it closes, so this verify is out of order; mask it.
                return PythonMathVerifyResponse(
                    **(body.model_dump() | lost_session_verification(f"math_with_code session {session_id} is closed"))
                )

            # The retry request carries the answer and grading data; no Python worker state is required.
            verification.retain(None)
            expected = body.expected_result

            # Extract actual answer from final assistant message
            actual = None
            for output in reversed(body.response.output):
                if output.type == "message" and output.role == "assistant":
                    text_content = ""
                    for content in output.content:
                        if content.type == "output_text":
                            text_content += content.text

                    actual = _extract_boxed_answer(text_content)
                    if actual is not None:
                        break

            # Fallback: the model may print the boxed answer inside executed code,
            # so search tool output (stdout) as well.
            if actual is None:
                for output in reversed(body.response.output):
                    if output.type == "function_call_output":
                        try:
                            tool_resp = json.loads(output.output)
                            stdout_text = tool_resp.get("stdout", "")
                        except (json.JSONDecodeError, AttributeError):
                            stdout_text = output.output
                        actual = _extract_boxed_answer(stdout_text)
                        if actual is not None:
                            break

            accuracy = _answers_match(actual, expected)
            reward = 1.0 if accuracy else 0.0

            return PythonMathVerifyResponse(
                **body.model_dump(),
                reward=reward,
                extracted_answer=actual,
                accuracy=accuracy,
            )

    async def _cleanup_session(self, session_id: str) -> None:
        """Stop the session's worker process; one that survives stays registered, so a later cleanup retries it."""
        starting = self._starting_sessions.get(session_id)
        if starting is not None:
            try:
                self._sessions[session_id] = await asyncio.shield(starting)
            except Exception:
                # A failed constructor returned no worker. Retrieving its exception also avoids leaving an
                # unobserved task when its requesting HTTP call was cancelled.
                logger.warning("Python worker startup failed for resources session %s", session_id, exc_info=True)
                self._starting_sessions.pop(session_id, None)
            else:
                self._starting_sessions.pop(session_id, None)
        handle = self._sessions.get(session_id)
        if handle is not None:
            await handle.close()
            # A concurrent cleanup may already have dropped it, and a later call may have started another.
            if self._sessions.get(session_id) is handle:
                del self._sessions[session_id]


def _normalize_answer(s: str) -> str:
    """Strip common math delimiters and whitespace for fair comparison.

    Many expected_result values are wrapped in \\(...\\) or $...$,
    but model answers inside \\boxed{} never include those wrappers.
    Without this normalization, ~67% of answers can never match.
    """
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1].strip()
    if s.startswith("\\text{") and s.endswith("}"):
        s = s[6:-1].strip()
    s = " ".join(s.split())
    return s


def _answers_match(actual: Optional[str], expected: str) -> bool:
    """Compare answers after normalization, with numeric fallback."""
    if actual is None:
        return False

    norm_actual = _normalize_answer(actual)
    norm_expected = _normalize_answer(expected)

    # Exact match after normalization
    if norm_actual == norm_expected:
        return True

    # Numeric fallback (handles "42" vs "42.0", "0.5" vs ".5", etc.)
    try:
        fa = float(norm_actual)
        fe = float(norm_expected)
        if abs(fa - fe) < 1e-6 * max(1.0, abs(fe)):
            return True
    except (ValueError, OverflowError):
        pass

    return False


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content inside the last \\boxed{...} in text.

    Uses brace-depth tracking so nested braces are handled correctly
    (e.g. \\boxed{\\frac{1}{2}} correctly returns '\\frac{1}{2}').
    """
    marker = "\\boxed{"
    # Use the last occurrence so chain-of-thought intermediate boxes are skipped.
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    result = text[start : i - 1].strip()
    return result if result else None


# -----------

if __name__ == "__main__":
    PythonExecutorResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = PythonExecutorResourcesServer.run_webserver()  # noqa: F401
