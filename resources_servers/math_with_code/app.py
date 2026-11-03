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
import io
import json
import logging
import multiprocessing
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy
from fastapi import FastAPI, Request
from pydantic import BaseModel, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)


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


def _session_worker(child_conn, max_execution_time: int):
    """Runs forever in its own process, keeping globals between calls."""
    exec_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "__import__": __import__,
        },
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "pd": pd,
        "pandas": pd,
    }
    exec_locals = {}
    while True:
        msg = child_conn.recv()
        if msg["cmd"] == "exec":
            code = msg["code"]
            try:
                out, err, res = _run_code_in_existing_env(code, exec_globals, exec_locals, max_execution_time)
                child_conn.send({"ok": True, "out": out, "err": err, "res": res})
            except Exception as e:
                child_conn.send({"ok": False, "error": str(e)})
        elif msg["cmd"] == "close":
            break


def _run_code_in_existing_env(code, globals_d, locals_d, timeout_s):
    """Re-uses the same globals/locals dictionary between calls."""

    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()

    def _handle_timeout(signum, frame):
        raise TimeoutError("code timed-out")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_s)
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_d, locals_d)
            result = _get_last_expr_value(code, globals_d, locals_d)
    finally:
        signal.alarm(0)
    return stdout_capture.getvalue(), stderr_capture.getvalue(), result


class _SessionHandle:
    """Light wrapper around one long-lived worker process."""

    def __init__(self, max_execution_time: int):
        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn
        self._proc = multiprocessing.Process(
            target=_session_worker,
            args=(child_conn, max_execution_time),
            daemon=True,
        )
        self._proc.start()
        self.last_used = time.time()

    def exec(self, code: str):
        self._conn.send({"cmd": "exec", "code": code})
        reply = self._conn.recv()
        self.last_used = time.time()
        if reply["ok"]:
            return reply["out"], reply["err"], reply["res"]
        raise RuntimeError(reply["error"])

    def close(self):
        """Stop the worker, escalating to terminate and then kill; raise if it is still alive."""
        try:
            self._conn.send({"cmd": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        self._proc.join(timeout=1)
        if self._proc.is_alive():
            # Still running user code, which the alarm stops only after max_execution_time.
            self._proc.terminate()
            self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=5)
        if self._proc.is_alive():
            raise RuntimeError(f"Python worker process {self._proc.pid} is still alive after SIGKILL")


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
    # The episode and task each Environment Server seed bound to its resources_session_id.
    _session_episodes: Dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _fence: ResourcesSessionFence = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

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
        if self._fence.closed_episode(sid) is not None:
            raise ResourcesSessionLostError(f"math_with_code session {sid} is closed")
        loop = asyncio.get_running_loop()
        try:
            if sid not in self._sessions:
                self._sessions[sid] = _SessionHandle(self.config.max_execution_time)
            handle = self._sessions[sid]

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
        if self._fence.closed_episode(sid) is not None:
            # The session closed while this call ran, so stop the worker the call may have started.
            await asyncio.to_thread(self._cleanup_session, sid)
            raise ResourcesSessionLostError(f"math_with_code session {sid} closed during the call")
        return response

    async def end_session(self, request: Request) -> ExecutePythonResponse:
        session_id = request.session[SESSION_ID_KEY]
        await asyncio.to_thread(self._cleanup_session, session_id)
        return ExecutePythonResponse(success=True, stdout="", stderr="")

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest | BaseSeedSessionRequest,
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        """Bind an Environment Server's resources session to the request's cookie.

        The worker process starts lazily on the first execute_python call. An Agent's ``/run`` seeds with its
        legacy body, which keeps no state here; verify stops that rollout's worker.
        """
        if not isinstance(body, ResourcesSeedSessionRequest):
            return BaseSeedSessionResponse()

        resources_session_id = body.resources_session_id
        async with self._fence.locked(resources_session_id):
            self._fence.raise_if_closed(resources_session_id)
            identity = self._session_episodes.setdefault(resources_session_id, (body.episode_id, body.task_id))
            if identity != (body.episode_id, body.task_id):
                raise ResourcesSessionConflictError("resources_session_id is already bound to another episode or task")
        # execute_python and verify find the worker by the cookie, so it must name this resources session.
        request.session[SESSION_ID_KEY] = resources_session_id
        return ResourcesSeedSessionResponse(resources_session_id=resources_session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Stop the session's worker process, even when verify never ran.

        Releases by ``resources_session_id``, since a close whose seed reply was lost carries no cookie. A worker
        that outlives SIGKILL raises, so the session is not recorded as closed and the caller can retry.
        """
        resources_session_id = body.resources_session_id
        async with self._fence.locked(resources_session_id):
            closed_episode = self._fence.closed_episode(resources_session_id)
            if closed_episode is not None:
                if closed_episode != body.episode_id:
                    raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
                return ResourcesCloseSessionResponse(resources_session_id=resources_session_id)
            identity = self._session_episodes.get(resources_session_id)
            if identity is not None and identity[0] != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
            # Verify has already stopped a verified episode's worker, so this finds none.
            await asyncio.to_thread(self._cleanup_session, resources_session_id)
            self._session_episodes.pop(resources_session_id, None)
            self._fence.mark_closed(resources_session_id, body.episode_id)
        return ResourcesCloseSessionResponse(resources_session_id=resources_session_id)

    async def verify(self, request: Request, body: PythonMathVerifyRequest) -> PythonMathVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        if self._fence.closed_episode(session_id) is not None:
            # An Environment Server verifies before it closes, so this verify is out of order; mask it.
            return PythonMathVerifyResponse(
                **(body.model_dump() | lost_session_verification(f"math_with_code session {session_id} is closed"))
            )

        try:
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
        finally:
            try:
                await asyncio.to_thread(self._cleanup_session, session_id)
            except RuntimeError:
                # The score stands; the worker stays registered, so /close_session retries and reports it.
                logger.exception("Could not stop the Python worker of session %s", session_id)

    def _cleanup_session(self, session_id: str) -> None:
        """Stop the session's worker process; one that survives stays registered, so a later cleanup retries it."""
        handle = self._sessions.get(session_id)
        if handle is not None:
            handle.close()
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


def _get_last_expr_value(code: str, globals_dict: dict, locals_dict: dict):
    """
    Replicates the behaviour that used to live inside execute_python:
    try to evaluate the last line of the submitted code and return its
    string representation.  If the last line is not a bare expression
    (or evaluation fails), return None.
    """
    lines = code.strip().split("\n")
    if not lines:
        return None

    last_line = lines[-1].strip()

    # Ignore lines that are obviously not bare expressions
    if last_line.startswith(("print", "import", "from", "def", "class", "if", "for", "while", "try", "with")):
        return None

    try:
        return str(eval(last_line, globals_dict, locals_dict))
    except Exception:
        return None


# -----------

if __name__ == "__main__":
    PythonExecutorResourcesServer.run_webserver()
