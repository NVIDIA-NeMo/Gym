# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Out-of-process runner for one SciCodePile task.

Reads a JSON request on stdin and writes a JSON result on stdout:

    stdin:  {"setup_code", "code", "test", "entry_point", "max_as_limit", "workdir"}
    stdout: {"status": "pass"|"fail"|"entry_point_missing"|"error"|"timeout", "details": {...}}

The task's own ``test`` field defines ``check(candidate)``. ``setup_code``, ``code``
and ``test`` are compiled and executed as **three separate units sharing one
namespace**, then ``check`` is called with the function named by ``entry_point``.
The separation is load-bearing, not stylistic: concatenating them into one unit let
a trailing decorator in the model's code bind to the test's own ``def check`` and
replace the assertions with a no-op, so a solution returning ``None`` passed all 200
tasks. It also pushed ``from __future__`` imports off the top of the file, scoring
correct solutions as ``syntax_error`` on the 105 tasks with non-empty ``setup_code``.
Validated against all 200 canonical solutions.

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
sandbox (see ``nemo_gym/sandbox/``).

**The verdict is not tamper-proof.** The result channel is kept off fd 1 so that
incidental stdout writes from an honest task cannot corrupt its own verdict, but
that is a robustness property only: task code can still reach the channel through
another descriptor, or replace ``json.dumps`` before the runner serialises. Treat a
verdict as trustworthy only to the extent the executed code is.
"""

import contextlib
import faulthandler
import io
import json
import os
import platform
import sys
import tempfile
import traceback


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


@contextlib.contextmanager
def _working_directory(workdir):
    """Run the task in a throwaway CWD, restoring the original on the way out.

    Several tasks write files relative to the CWD (observed: bioinformatics tasks
    emitting .fasta/.a3m/.pdb), which would otherwise litter the repository, let
    concurrent tasks collide on identical filenames, and leak state into later runs
    so a task could pass only because an earlier one left a file behind.

    When the parent supplies ``workdir`` it also owns the cleanup, because a timed-out
    task is SIGKILLed and a task can call ``os._exit`` — neither runs anything here.
    The server always supplies one. Standalone invocations fall back to a self-managed
    temp directory, which is best-effort for that same reason: cleanup is skipped if the
    task exits the process abruptly.
    """
    origin = os.getcwd()
    try:
        if workdir:
            os.chdir(workdir)
            yield
        else:
            with tempfile.TemporaryDirectory(prefix="scicodepile_") as owned:
                os.chdir(owned)
                yield
    finally:
        try:
            os.chdir(origin)
        except OSError:
            pass


def _phase_error(reason: str, exc: BaseException, phase: str) -> dict:
    """Build an error result, recording which compile unit raised.

    ``phase`` is ``setup``, ``model`` or ``test``. Only ``model`` is the model's own
    code; the other two are dataset-owned, and the parent uses ``harness_fault`` to
    keep them out of an accuracy figure instead of scoring them as wrong answers.
    """
    details = {"reason": reason, "type": type(exc).__name__, "message": str(exc)[:500], "phase": phase}
    if phase != "model":
        details["harness_fault"] = True
    return {"status": "error", "details": details}


def run_task(req: dict) -> dict:
    setup_code = req.get("setup_code") or ""
    code = req.get("code") or ""
    test = req.get("test") or ""
    entry_point = req.get("entry_point") or ""

    # `__name__` is deliberately not "__main__": some harvested sources guard
    # side effects behind a __main__ check and must not run them here.
    namespace: dict = {"__name__": "__scicodepile__"}

    # Three separate compile units sharing one namespace, never one concatenated
    # source. Concatenating lets a trailing decorator in the model's code bind to
    # the test's own `def check`, replacing the assertions with a no-op — a
    # solution returning None then passed all 200 tasks. It also pushed
    # `from __future__` imports off the top of the file, so a correct solution
    # using them was scored `syntax_error` on every task with setup_code.
    # Distinct filenames keep tracebacks and line numbers pointing at the right
    # source instead of an offset into the concatenation.
    for phase, src, filename in (
        ("setup", setup_code, "<scicodepile_setup>"),
        ("model", code, "<scicodepile_solution>"),
    ):
        if phase == "setup" and not src.strip():
            continue
        try:
            compiled = compile(src, filename, "exec")
        except SyntaxError as exc:
            return _phase_error("syntax_error", exc, phase)
        try:
            exec(compiled, namespace)
        except BaseException as exc:
            return _phase_error("exec_failed", exc, phase)

    # Everything from here on runs *after* the model's module body. Model code can
    # mutate interpreter-wide state — rebind a builtin, lower the recursion limit,
    # install a trace hook — so a failure in the runner's own machinery below is no
    # longer evidence that the runner is broken. Attribute it to the model rather
    # than letting it escape to `main` and be reported as a harness fault.
    try:
        return _run_after_model(namespace, test, entry_point)
    except BaseException as exc:
        return _phase_error("runner_crashed", exc, "model")


def _run_after_model(namespace: dict, test: str, entry_point: str) -> dict:
    candidate = namespace.get(entry_point)
    if candidate is None:
        return {"status": "entry_point_missing", "details": {"entry_point": entry_point}}

    # The test is its own compile unit, but shares the namespace: many tests reach
    # into the solution's globals (`candidate.__globals__[...] = stub`) or define
    # helpers the solution calls, and 13 of the 200 tasks fail if the test is given
    # a copy. Sharing is safe here because a separate unit is already enough — a
    # dangling decorator cannot bind across a compile boundary — and the test's own
    # `def check` executes after the model's code, so it rebinds any `check` the
    # model may have defined.
    try:
        compiled_test = compile(test, "<scicodepile_test>", "exec")
    except SyntaxError as exc:
        return _phase_error("syntax_error", exc, "test")
    try:
        exec(compiled_test, namespace)
    except BaseException as exc:
        return _phase_error("exec_failed", exc, "test")

    check = namespace.get("check")
    if check is None or not callable(check):
        return {
            "status": "error",
            "details": {"reason": "test_defines_no_check", "phase": "test", "harness_fault": True},
        }

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
    # rebinds `sys.stdout` — it does not protect the descriptor, so incidental
    # C-level writes to fd 1 would otherwise corrupt an honest task's JSON.
    # This does not make the verdict unforgeable; see the module docstring.
    #
    # fd 2 goes to /dev/null as well, and for a different reason: the parent waits
    # for EOF on both pipes. Anything the task spawns inherits these descriptors, so
    # a task that leaves a child running would hold the stderr pipe open and turn a
    # written `pass` into a `timeout`. The real stderr is kept on a private duplicate
    # so a runner-internal crash is still reportable.
    result_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    try:
        with _working_directory(req.get("workdir")):
            # Task code prints freely; stdout is the result channel, so capture both streams.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = run_task(req)
    except BaseException as exc:  # pragma: no cover - defensive
        with contextlib.suppress(BaseException):
            os.write(stderr_fd, traceback.format_exc().encode("utf-8", "replace"))
        # Reached only for failures outside `run_task` — setting up the working
        # directory or the stream redirection — i.e. before any model code runs.
        # A crash *after* model code runs is attributed to the model inside
        # `run_task`, because model code can corrupt the runner's own machinery.
        result = {
            "status": "error",
            "details": {
                "reason": "runner_crashed",
                "message": str(exc)[:500],
                "phase": "runner",
                "harness_fault": True,
            },
        }

    # Written to the private duplicate, not fd 1. A task that calls `os._exit`
    # skips this entirely; if it wrote nothing to the channel first, the parent
    # gets an empty read and reports `unparseable_runner_output`. That is not a
    # guarantee — a task that writes to the channel before exiting decides the
    # verdict. See the module docstring.
    os.write(result_fd, json.dumps(result).encode())
    os.close(result_fd)
    os.close(stderr_fd)

    # Hard exit, not a return: the verdict is written and nothing left to do here is
    # worth waiting on. A normal interpreter shutdown joins non-daemon threads and
    # runs `atexit` hooks, both of which task code can leave behind, so a task whose
    # `check` passed would sit until the parent's timeout fired and be scored
    # `timeout` instead of `pass`. Everything this process owns is either already
    # closed or owned by the parent (the working directory in particular).
    os._exit(0)


if __name__ == "__main__":
    main()
