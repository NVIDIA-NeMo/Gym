#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# ruff: noqa
# fmt: off
# Vendored sandbox code: kept as run by the original harness; not linted or reformatted.
"""Fork-per-request Python sandbox server for the ARC code environment.

Single-threaded line protocol on stdin/stdout (one JSON object per line):

  request : {"id": ..., "header": "<python defining examples/test_inputs>", "codes": ["<earlier code block>", ...],
             "code": "<code to run now>", "timeout_s": 10, "out_chars": 1000, "mem_mb": 6144}
  response: {"id": ..., "output": "<captured stdout, truncated>", "timed_out": bool, "error": "<short error or empty>"}

At start the server execs the helper preload (arc_solve_tools.py: check, objects, diff, crop, predict) once.
Every request forks a child that
  1. applies resource limits (address space, CPU seconds, file size, open files),
  2. replaces the namespace builtins with a restricted set (no open/input/exit, allowlisted imports),
  3. execs `header`, then replays `codes` silently (output discarded, errors ignored), then runs `code` capturing stdout,
     echoing the value of a final bare expression like a notebook cell,
and the parent kills the child at the deadline. Episode state is rebuilt by replay, so no process outlives a request.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import resource
import select
import shutil
import signal
import sys
import tempfile
import time
import traceback
import types

ALLOWED_IMPORTS = {
    "numpy", "scipy", "networkx", "collections", "itertools", "functools", "math", "heapq", "bisect", "copy", "json", "re", "string",
    "statistics", "fractions", "operator", "random", "typing", "dataclasses", "enum", "decimal", "numbers", "array", "abc",
}
DENIED_BUILTINS = {"open", "input", "breakpoint", "exit", "quit", "help", "license", "credits", "copyright"}


# Per-request extras (request field "extra_imports"): more standard-library modules the model may import. "sys" is never the real module:
# it is a stub with only the recursion-limit functions and a few constants.
SYS_STUB = types.SimpleNamespace(setrecursionlimit=sys.setrecursionlimit, getrecursionlimit=sys.getrecursionlimit, maxsize=sys.maxsize,
                                 version_info=sys.version_info, version=sys.version, float_info=sys.float_info, byteorder=sys.byteorder)


def restricted_builtins(extra_imports=()) -> dict:
    safe = {k: v for k, v in vars(builtins).items() if k not in DENIED_BUILTINS}
    allowed = ALLOWED_IMPORTS | set(extra_imports)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name.split(".")[0] not in allowed:
            raise ImportError(f"import of '{name}' is not allowed in this sandbox")
        if name == "sys":
            return SYS_STUB
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe["__import__"] = safe_import
    return safe


def run_cell(code: str, ns: dict) -> None:
    """Exec `code`; if its last statement is a bare expression, print its repr when it is not None."""
    tree = ast.parse(code)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), ns)
    if last is not None:
        value = eval(compile(last, "<cell>", "eval"), ns)
        if value is not None:
            print(repr(value))


def short_error(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    cell = [f for f in tb if f.filename == "<cell>"]
    where = f" (line {cell[-1].lineno})" if cell else ""
    return f"{type(exc).__name__}: {exc}{where}"


def child_main(base_ns: dict, req: dict, write_fd: int, workdir: str) -> None:
    try:
        os.chdir(workdir)  # per-run scratch directory (removed by the parent)
    except OSError:
        pass
    timeout = float(req.get("timeout_s", 10))
    mem = int(req.get("mem_mb", 6144)) * 1024 * 1024
    for limit, value in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, int(timeout) + 2),
                         (resource.RLIMIT_FSIZE, 1_000_000), (resource.RLIMIT_NOFILE, 64)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass
    if req.get("recursion_limit"):
        sys.setrecursionlimit(int(req["recursion_limit"]))
    ns = base_ns
    ns["__builtins__"] = restricted_builtins(req.get("extra_imports") or ())
    if req.get("allow_trace"):   # cov_check: the harness header traces the model's functions; model code still sees the sys stub
        ns["_harness_settrace"] = sys.settrace
    result = {"output": "", "error": ""}
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        exec(compile(req.get("header", ""), "<header>", "exec"), ns)
        for earlier in req.get("codes", []):  # silent replay: output discarded, errors ignored as they were when first seen
            try:
                run_cell(earlier, ns)
            except BaseException:  # noqa: BLE001
                pass
            sys.stdout = io.StringIO()
        buf = sys.stdout = io.StringIO()
        ns["_harness_extra"] = None   # data a header may leave for the harness (not shown to the model); only this cell's counts
        try:
            run_cell(req["code"], ns)
        except BaseException as exc:  # noqa: BLE001 - everything is reported to the model
            result["error"] = short_error(exc)
        result["output"] = buf.getvalue()
        extra = ns.get("_harness_extra")
        if extra is not None:
            try:
                json.dumps(extra)
                result["extra"] = extra
            except (TypeError, ValueError):
                pass
    except BaseException as exc:  # noqa: BLE001
        result["error"] = f"sandbox error: {short_error(exc)}"
    cap = int(req.get("out_chars", 1000))
    out = result["output"]
    if len(out) > cap:
        result["output"] = out[:cap] + f"\n... [truncated {len(out) - cap} chars]"
    payload = json.dumps(result).encode()
    view = memoryview(payload)
    while view:
        view = view[os.write(write_fd, view):]
    os._exit(0)


def handle(base_ns: dict, req: dict) -> dict:
    timeout = float(req.get("timeout_s", 10))
    read_fd, write_fd = os.pipe()
    workdir = tempfile.mkdtemp(prefix="arc_sandbox_")
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            os.setsid()
            child_main(base_ns, req, write_fd, workdir)
        finally:
            os._exit(1)
    os.close(write_fd)
    deadline = time.monotonic() + timeout
    chunks, timed_out = [], False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        ready, _, _ = select.select([read_fd], [], [], remaining)
        if not ready:
            timed_out = True
            break
        block = os.read(read_fd, 1 << 20)
        if not block:
            break
        chunks.append(block)
    if timed_out:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    shutil.rmtree(workdir, ignore_errors=True)
    if timed_out:
        return {"id": req.get("id"), "output": "", "timed_out": True, "error": f"Tool timed out after {int(timeout)}s"}
    try:
        result = json.loads(b"".join(chunks))
    except ValueError:
        reason = f"signal {os.WTERMSIG(status)}" if os.WIFSIGNALED(status) else "no result"
        return {"id": req.get("id"), "output": "", "timed_out": False, "error": f"sandbox process failed ({reason}; likely memory or CPU limit)"}
    reply = {"id": req.get("id"), "output": result["output"], "timed_out": False, "error": result["error"]}
    if "extra" in result:
        reply["extra"] = result["extra"]
    return reply


def load_base_namespace(tools_dir: str) -> dict:
    os.environ["ARC_TOOLS_DIR"] = tools_dir
    ns = {"__name__": "__main__"}
    with open(os.path.join(tools_dir, "arc_solve_tools.py"), encoding="utf-8") as handle_:
        exec(compile(handle_.read(), "arc_solve_tools.py", "exec"), ns)
    return ns


def main() -> None:
    tools_dir = sys.argv[1]
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    base_ns = load_base_namespace(tools_dir)
    try:   # loaded once here so every forked run gets it for free (a cold scipy import per run can exceed the run timeout); names unchanged
        import scipy.ndimage  # noqa: F401
    except ImportError:
        pass
    out = sys.stdout
    sys.stdout = sys.stderr  # protocol replies use the saved handle only; stray prints go to stderr
    print(json.dumps({"ready": True}), file=out, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        sys.stdout = sys.stderr
        try:
            reply = handle(base_ns, json.loads(line))
        except Exception as exc:  # noqa: BLE001
            reply = {"id": None, "output": "", "timed_out": False, "error": f"server error: {exc}"}
        print(json.dumps(reply), file=out, flush=True)


if __name__ == "__main__":
    main()
