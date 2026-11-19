# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import signal
import sys
import types

from sandbox_limits import address_space_bytes, limit_address_space


# Result codes shared with the stdin/stdout harness.
_COMPILE_ERROR, _ASSERTION_FAILED, _TIME_LIMIT, _RUNTIME_ERROR = -1, -2, -3, -4
_FAILURE_CODES = frozenset({_COMPILE_ERROR, _ASSERTION_FAILED, _TIME_LIMIT, _RUNTIME_ERROR})
_MAX_ERROR_CHARS = 300


class _AssertionTimeout(BaseException):
    """A BaseException, so candidate code that catches Exception cannot swallow the time limit."""


def _error_metadata(code: int, message: str, error: BaseException | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {"error_code": code, "error_message": message}
    if error is not None:
        metadata["error"] = repr(error)[:_MAX_ERROR_CHARS]
    return metadata


def _timeout_handler(signum, frame) -> None:
    raise _AssertionTimeout


def _run_assertions(generation, tests, time_limit_s, memory_limit_mb, result, metadata_list) -> None:
    memory = limit_address_space(memory_limit_mb)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    # As when run as a script: the candidate and its tests share one __main__ module.
    module = types.ModuleType("__main__")
    sys.modules["__main__"] = module
    results: list[bool | int] = []
    metadata: dict[str, object] = {}
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, time_limit_s)
    try:
        try:
            candidate = compile(generation, "<candidate>", "exec")
        except (SyntaxError, ValueError) as error:
            results.append(_COMPILE_ERROR)
            metadata = _error_metadata(_COMPILE_ERROR, "Compilation Error", error)
        else:
            exec(candidate, module.__dict__)
            for index, test in enumerate(tests):
                exec(compile(test, f"<test {index}>", "exec"), module.__dict__)
                results.append(True)
    except _AssertionTimeout:
        results.append(_TIME_LIMIT)
        metadata = _error_metadata(_TIME_LIMIT, "Time Limit Exceeded")
    except AssertionError as error:
        results.append(_ASSERTION_FAILED)
        metadata = _error_metadata(_ASSERTION_FAILED, "Assertion Failed", error)
    except BaseException as error:
        # Includes SystemExit: exiting early never passes.
        results.append(_RUNTIME_ERROR)
        metadata = _error_metadata(_RUNTIME_ERROR, "Runtime Error", error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result.append(results)
    metadata_list.append({**metadata, "memory": memory})


def _checked(results: object, metadata: object, num_tests: int) -> tuple[list[bool | int], dict[str, object]]:
    """Accept only what the child produces: every test passed, or passes up to one failure code."""
    if isinstance(results, list) and isinstance(metadata, dict):
        passed = 0
        while passed < len(results) and results[passed] is True:
            passed += 1
        if passed == num_tests == len(results):
            return results, metadata
        if passed < num_tests and len(results) == passed + 1:
            last = results[-1]
            if type(last) is int and last in _FAILURE_CODES:
                return results, metadata
    return [_RUNTIME_ERROR], _error_metadata(_RUNTIME_ERROR, "Malformed assertion results")


def check_assertions(
    generation: str, tests: list[str], *, time_limit_s: float, memory_limit_mb: int, deadline_s: float
) -> tuple[list[bool | int], dict[str, object]]:
    """Run *generation*, then the assertion *tests* in order, in a child process; stop at the first failure.

    *time_limit_s* covers the candidate and all its tests; *deadline_s* bounds the child process.
    """
    # Fails here, in the runner, so a sandbox that cannot measure memory masks the sample.
    address_space_bytes()
    manager = multiprocessing.Manager()
    p: multiprocessing.Process | None = None
    try:
        result = manager.list()
        metadata_list = manager.list()
        p = multiprocessing.Process(
            target=_run_assertions,
            args=(generation, tests, time_limit_s, memory_limit_mb, result, metadata_list),
        )
        p.start()
        p.join(timeout=deadline_s)
        timed_out = p.is_alive()
        if timed_out:
            p.kill()
            p.join(timeout=5)
        try:
            results, metadatas = list(result), list(metadata_list)
        except (OSError, EOFError):
            # The candidate shares the child's process and can break its connection to the Manager.
            results, metadatas = [], []
        if results:
            return _checked(results[0], metadatas[0] if metadatas else None, len(tests))
        if timed_out:
            return [_TIME_LIMIT], _error_metadata(_TIME_LIMIT, "Time Limit Exceeded")
        return [_RUNTIME_ERROR], _error_metadata(_RUNTIME_ERROR, "Exited before its assertions completed")
    finally:
        if p is not None and p.is_alive():
            p.kill()
            p.join(timeout=5)
        manager.shutdown()
