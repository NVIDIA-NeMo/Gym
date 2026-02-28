# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import io
import logging
import math
import multiprocessing
import re
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy


def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Python code for safety and correctness.

    Args:
        code: Python code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)

        dangerous_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"import\s+subprocess",
            r"__import__",
            r"eval\(",
            r"exec\(",
            r"open\(",
            r"file\(",
            r"input\(",
            r"raw_input\(",
            r"compile\(",
            r"globals\(",
            r"locals\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Code contains potentially dangerous operation: {pattern}"

        return True, None

    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


class SessionHandle:
    """Light wrapper around one long-lived worker process."""

    def __init__(self, max_execution_time: int):
        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn
        self._max_execution_time = max_execution_time
        self._proc = multiprocessing.Process(
            target=_session_worker,
            args=(child_conn, max_execution_time),
            daemon=True,
        )
        self._proc.start()
        self.is_closed = False

    def exec(self, code: str):
        try:
            self._conn.send({"cmd": "exec", "code": code})
        except (BrokenPipeError, EOFError, ConnectionError):
            self.close()
            raise

        if self._conn.poll(self._max_execution_time + 5):
            try:
                reply = self._conn.recv()
            except (BrokenPipeError, EOFError, ConnectionError):
                self.close()
                raise

            if reply["ok"]:
                return reply["out"], reply["err"], reply["res"]

            error_msg = reply["error"]
            if "traceback" in reply:
                error_msg += f"\nTraceback:\n{reply['traceback']}"
            raise RuntimeError(error_msg)

        self.close()
        raise TimeoutError("Execution timed out (worker unresponsive)")

    def close(self):
        if self.is_closed:
            return
        self.is_closed = True

        try:
            self._conn.send({"cmd": "close"})
        except Exception:
            logging.debug("Failed to send close command to worker")

        try:
            self._conn.close()
        except Exception:
            logging.exception("Error while closing session pipe")

        self._proc.join(timeout=1)
        if self._proc.is_alive():
            logging.warning(f"Session process {self._proc.pid} still alive after close; escalating to terminate.")
            try:
                self._proc.terminate()
                self._proc.join(timeout=1)
                if self._proc.is_alive():
                    logging.error(f"Session process {self._proc.pid} resisted terminate; resorting to kill.")
                    self._proc.kill()
                    self._proc.join()
            except Exception:
                logging.exception(f"Error while force-terminating process {self._proc.pid}")

        try:
            self._proc.close()
        except Exception:
            logging.exception("Error during final Process object cleanup")


def _get_last_expr_value(code: str, globals_dict: dict, locals_dict: dict):
    """
    Try to evaluate the last line of the submitted code and return its
    string representation. If the last line is not a bare expression
    (or evaluation fails), return None.
    """
    lines = code.strip().split("\n")
    if not lines:
        return None

    last_line = lines[-1].strip()

    if last_line.startswith(("print", "import", "from", "def", "class", "if", "for", "while", "try", "with")):
        return None

    try:
        return str(eval(last_line, globals_dict, locals_dict))
    except Exception:
        return None


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


def _session_worker(child_conn, max_execution_time: int):
    """Runs forever in its own process, keeping globals between calls."""

    safe_builtins_list = [
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "Exception",
        "ArithmeticError",
        "AssertionError",
        "AttributeError",
        "BufferError",
        "EOFError",
        "ImportError",
        "IndexError",
        "KeyError",
        "MemoryError",
        "NameError",
        "NotImplementedError",
        "OSError",
        "OverflowError",
        "ReferenceError",
        "RuntimeError",
        "StopIteration",
        "SyntaxError",
        "SystemError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
        "__import__",
    ]

    exec_globals = {
        "__builtins__": {
            name: __builtins__.get(name) if isinstance(__builtins__, dict) else getattr(__builtins__, name)
            for name in safe_builtins_list
            if (isinstance(__builtins__, dict) and name in __builtins__) or hasattr(__builtins__, name)
        },
        "np": np,
        "numpy": np,
        "scipy": scipy,
        "pd": pd,
        "pandas": pd,
        "math": math,
    }

    exec_locals = {}
    while True:
        try:
            msg = child_conn.recv()
        except EOFError:
            # Parent closed the connection without sending "close" command.
            # Exit gracefully instead of crashing with a traceback.
            break

        if msg["cmd"] == "exec":
            code = msg["code"]
            try:
                out, err, res = _run_code_in_existing_env(code, exec_globals, exec_locals, max_execution_time)
                child_conn.send({"ok": True, "out": out, "err": err, "res": res})
            except Exception as e:
                child_conn.send({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
        elif msg["cmd"] == "close":
            break
