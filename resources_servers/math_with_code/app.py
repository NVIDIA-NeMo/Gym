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
import multiprocessing
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy
from fastapi import FastAPI, Request
from pydantic import BaseModel, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


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
        try:
            self._conn.send({"cmd": "close"})
        except (BrokenPipeError, EOFError):
            pass
        self._proc.join(timeout=1)


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

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/execute_python")(self.execute_python)
        app.post("/end_session")(self.end_session)
        return app

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # _sessions dict already initialised by default_factory

    async def execute_python(self, request: Request, body: ExecutePythonRequest) -> ExecutePythonResponse:
        loop = asyncio.get_running_loop()
        try:
            sid = request.session[SESSION_ID_KEY]
            if sid not in self._sessions:
                self._sessions[sid] = _SessionHandle(self.config.max_execution_time)
            handle = self._sessions[sid]

            stdout, stderr, result = await loop.run_in_executor(
                None,
                handle.exec,
                body.code,
            )
            return ExecutePythonResponse(
                success=True,
                stdout=stdout,
                stderr=stderr,
                result=result,
            )
        except Exception as e:
            return ExecutePythonResponse(
                success=False,
                stdout="",
                stderr="",
                error_message=str(e),
            )

    async def end_session(self, request: Request) -> ExecutePythonResponse:
        sid = request.session[SESSION_ID_KEY]
        if sid in self._sessions:
            self._sessions[sid].close()
            del self._sessions[sid]
        return ExecutePythonResponse(success=True, stdout="", stderr="")

    async def verify(self, body: PythonMathVerifyRequest) -> PythonMathVerifyResponse:
        expected = body.expected_result

        # Search assistant messages (last-to-first) for \boxed{}.
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
