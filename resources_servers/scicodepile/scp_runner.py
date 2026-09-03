# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Out-of-process runner for one SciCodePile task.

Reads a JSON request on stdin and writes a JSON result on stdout:

    stdin:  {"setup_code", "code", "test", "entry_point", "timeout", "max_as_limit"}
    stdout: {"status": "pass"|"fail"|"entry_point_missing"|"error"|"timeout", "details": {...}}

The task's own ``test`` field defines ``check(candidate)``; the runner executes
``setup_code + code + test`` in one namespace and then calls ``check`` with the
function named by ``entry_point``. This mirrors SciCodePile's own harness and was
validated against all 200 canonical solutions before it was written.

The runner lives in its own process so that model code cannot corrupt the
resources server, and so a hang is bounded by the parent's timeout. Every task in
this benchmark carries the ``env_sensitive`` audit flag and 117 of 200 carry
``globals_patch``, so tests mutate global state freely — a fresh process per task
is what keeps them independent.

``code`` is unreviewed model output, and this runner is **not** a security
sandbox: task code runs with the privileges and environment of the resources
server, and can shell out, open sockets, or write outside its CWD. Containment is
limited to process isolation, an address-space cap, a throwaway CWD, and the
parent's timeout. Do not run untrusted rollouts on shared nodes without a real
sandbox (see ``nemo_gym/sandbox/``). The result channel is kept off fd 1 so that
stdout writes from task code can neither forge nor corrupt the verdict; see
``main``.
"""

import contextlib
import faulthandler
import io
import json
import os
import platform
import sys
import tempfile


def _apply_limits(max_as_limit_mb: int) -> None:
    """Bound address space so a runaway allocation fails instead of taking down the node."""
    if platform.system() == "Linux" and max_as_limit_mb > 0:
        try:
            import resource

            nbytes = max_as_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except Exception:
            # Best effort: an unsettable rlimit must not fail the task.
            pass


def run_task(req: dict) -> dict:
    setup_code = req.get("setup_code") or ""
    code = req.get("code") or ""
    test = req.get("test") or ""
    entry_point = req.get("entry_point") or ""

    source = ""
    if setup_code.strip():
        source += setup_code + "\n"
    source += code + "\n" + test + "\n"

    # `__name__` is deliberately not "__main__": some harvested sources guard
    # side effects behind a __main__ check and must not run them here.
    namespace: dict = {"__name__": "__scicodepile__"}

    try:
        compiled = compile(source, "<scicodepile_task>", "exec")
    except SyntaxError as exc:
        return {"status": "error", "details": {"reason": "syntax_error", "message": str(exc)[:500]}}

    try:
        exec(compiled, namespace)
    except BaseException as exc:
        return {
            "status": "error",
            "details": {"reason": "exec_failed", "type": type(exc).__name__, "message": str(exc)[:500]},
        }

    candidate = namespace.get(entry_point)
    if candidate is None:
        return {"status": "entry_point_missing", "details": {"entry_point": entry_point}}

    check = namespace.get("check")
    if check is None:
        return {"status": "error", "details": {"reason": "test_defines_no_check"}}

    try:
        check(candidate)
    except BaseException as exc:
        return {
            "status": "fail",
            "details": {"type": type(exc).__name__, "message": str(exc)[:500]},
        }
    return {"status": "pass", "details": {}}


def main() -> None:
    req = json.loads(sys.stdin.read())
    _apply_limits(int(req.get("max_as_limit", 0)))
    faulthandler.disable()

    # Move the result channel off fd 1 before any task code runs, then point fd 1
    # at /dev/null. Task code owns fd 1 too, and `redirect_stdout` below only
    # rebinds `sys.stdout` — it does not protect the descriptor. Without this, a
    # completion doing `os.write(1, b'{"status": "pass"}')` forges a reward-1.0
    # verdict without `check()` ever running, and incidental C-level writes to
    # fd 1 corrupt the JSON of an honest one.
    result_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)

    # Run inside a throwaway working directory. Several tasks write files relative
    # to the CWD (observed: bioinformatics tasks emitting .fasta/.a3m/.pdb), which
    # would otherwise litter the repository, let concurrent tasks collide on
    # identical filenames, and leak state into later runs so a task could pass only
    # because an earlier one left a file behind.
    origin = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="scicodepile_") as workdir:
            os.chdir(workdir)
            # Task code prints freely; stdout is the result channel, so capture both streams.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_task(req)
    except BaseException as exc:  # pragma: no cover - defensive
        result = {"status": "error", "details": {"reason": "runner_crashed", "message": str(exc)[:500]}}
    finally:
        try:
            os.chdir(origin)
        except OSError:
            pass

    # Written to the private duplicate, not fd 1. A task that calls `os._exit`
    # skips this entirely, leaving the parent an empty read that it reports as
    # `unparseable_runner_output` — the runner fails closed, never to `pass`.
    os.write(result_fd, json.dumps(result).encode())
    os.close(result_fd)


if __name__ == "__main__":
    main()
