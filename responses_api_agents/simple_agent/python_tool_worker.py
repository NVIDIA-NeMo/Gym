# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interpreter of one Python tool, run inside the agent's sandbox by the sandboxed agent loop.

``python -I python_tool_worker.py``, started by ``python_tool_client.PythonTools`` with the tool's own Python runtime. It
uses the standard library only. It reads its settings, then one message per call, and keeps one namespace for the
life of the process, so names a call defines stay defined for the next one.

Messages are JSON, each prefixed with its length as a 4-byte big-endian integer, on private copies of the
process's stdin and stdout. The code's own file descriptors 0, 1 and 2 point to ``/dev/null``: what it prints
through ``sys.stdout`` and ``sys.stderr`` comes back in the reply, and what it or its subprocesses write to those
descriptors directly is dropped, so neither can corrupt a reply.
"""

import builtins
import importlib
import io
import json
import os
import signal
import struct
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, BinaryIO


_LENGTH = struct.Struct(">I")


def read_message(stream: BinaryIO) -> Any:
    """The next message, or None when the other end closed the stream."""
    header = stream.read(_LENGTH.size)
    if len(header) < _LENGTH.size:
        return None
    (length,) = _LENGTH.unpack(header)
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body)


def write_message(stream: BinaryIO, message: Any) -> None:
    # ASCII escapes keep a lone surrogate the code printed encodable.
    body = json.dumps(message).encode()
    stream.write(_LENGTH.pack(len(body)) + body)
    stream.flush()


def limit_memory(memory_mib: int | None) -> None:
    """Cap what this process may allocate, so code that asks for more gets a MemoryError.

    The hard limit is set too, so the code cannot raise it again unless it holds CAP_SYS_RESOURCE, as root does when
    the sandbox keeps that capability (Docker drops it by default). The cap guards the agent loop against code that
    asks for too much by mistake; the sandbox's own memory limit is the boundary.
    """
    if memory_mib is None:
        return
    import resource

    limit = memory_mib * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))


def namespace(preload: dict[str, str], allowed_builtins: list[str] | None) -> dict[str, Any]:
    """The globals every call runs in: the allowed builtins, or all of them, and the preloaded modules by name."""
    allowed = (
        builtins.__dict__ if allowed_builtins is None else {name: getattr(builtins, name) for name in allowed_builtins}
    )
    names: dict[str, Any] = {"__builtins__": allowed}
    for name, module in preload.items():
        names[name] = importlib.import_module(module)
    return names


def run_code(code: str, globals_d: dict[str, Any], locals_d: dict[str, Any], timeout_s: int) -> tuple[str, str, Any]:
    """Run ``code`` in the call's namespace, and return what it printed and its last line's value.

    Top-level names land in ``locals_d``, which every call shares, and functions look names up in ``globals_d``.
    ``SIGALRM`` stops code that runs past ``timeout_s`` seconds with ``TimeoutError("code timed-out")``; code that
    blocks the signal is stopped by the caller, which kills this process.
    """
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("code timed-out")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(timeout_s)
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_d, locals_d)
            result = last_expression_value(code, globals_d, locals_d)
    finally:
        signal.alarm(0)
    return stdout_capture.getvalue(), stderr_capture.getvalue(), result


def last_expression_value(code: str, globals_d: dict[str, Any], locals_d: dict[str, Any]) -> str | None:
    """The ``str`` of the code's last line evaluated again, or None when it is not a bare expression or fails."""
    lines = code.strip().split("\n")
    last_line = lines[-1].strip()
    if last_line.startswith(("print", "import", "from", "def", "class", "if", "for", "while", "try", "with")):
        return None
    try:
        return str(eval(last_line, globals_d, locals_d))
    except Exception:
        return None


def main() -> int:
    # Buffered, so a read returns the whole message however many pipe writes it took, and a write sends it all.
    requests = os.fdopen(os.dup(0), "rb")
    replies = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(null, descriptor)
    os.close(null)
    # A worker the agent loop stops gets the default handlers, so terminate() is not swallowed.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    settings = read_message(requests)
    if settings is None:
        return 1
    try:
        # Before the preload, whose modules count toward the limit too.
        limit_memory(settings.get("memory_mib"))
        globals_d = namespace(settings["preload"], settings["builtins"])
    except Exception as error:
        write_message(replies, {"ready": False, "error": f"{type(error).__name__}: {error}"})
        return 1
    locals_d: dict[str, Any] = {}
    write_message(replies, {"ready": True})
    while (request := read_message(requests)) is not None:
        try:
            out, err, res = run_code(request["code"], globals_d, locals_d, settings["timeout_seconds"])
            reply = {"ok": True, "out": out, "err": err, "res": res}
        except Exception as error:
            # Code that raises SystemExit, such as sys.exit(), ends the worker; the next call reports it stopped.
            reply = {"ok": False, "error": str(error)}
        write_message(replies, reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
