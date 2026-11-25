# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandboxed agent loop's client of its Python tool workers.

The runner (``sandbox_runner.py``) builds ``PythonTools`` from the specs the agent server staged. Each tool's worker
(``python_tool_worker.py``) starts on the tool's first call and runs each call's code; messages are JSON, each prefixed
with its length as a 4-byte big-endian integer. The tool takes ``{"code": "<source>"}`` and returns the JSON a
Resources Server's Python tool returned before it moved here: ``{"success", "stdout", "stderr", "error_message",
"result"}``.
"""

import asyncio
import json
import os
import signal
import struct
from typing import Any

from responses_api_agents.simple_agent.python_tool import PythonToolSpec


_LENGTH = struct.Struct(">I")
# How long past a call's timeout the loop waits for a worker that has not answered before it kills the worker. The
# worker's SIGALRM stops ordinary code at the timeout, but only between bytecodes: a long call into C, such as
# sum(range(10**10)), answers "code timed-out" once it returns. Only code that runs past this grace is killed.
KILL_GRACE_SECONDS = 20.0
# How long a worker may take to import its preloaded modules before its first call fails.
START_TIMEOUT_SECONDS = 120.0
# How long a killed worker's process group may take to exit before the loop stops waiting for it.
REAP_TIMEOUT_SECONDS = 5.0
RESTARTS = "the next call starts a new Python worker, without the names earlier calls defined"
# What a Resources Server answers a call to a tool it does not serve.
NOT_FOUND = json.dumps({"detail": "Not Found"}, separators=(",", ":"))


def tool_output(
    *, success: bool, stdout: str = "", stderr: str = "", error_message: str | None = None, result: str | None = None
) -> str:
    """The tool's output, as the model reads it."""
    fields = {"success": success, "stdout": stdout, "stderr": stderr, "error_message": error_message, "result": result}
    # A lone surrogate the code printed cannot reach the model server as UTF-8: it becomes "?".
    clean = {
        key: value.encode("utf-8", "replace").decode("utf-8") if isinstance(value, str) else value
        for key, value in fields.items()
    }
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


async def _send(process: asyncio.subprocess.Process, message: Any) -> None:
    assert process.stdin is not None
    body = json.dumps(message).encode()
    process.stdin.write(_LENGTH.pack(len(body)) + body)
    await process.stdin.drain()


async def _receive(process: asyncio.subprocess.Process) -> dict[str, Any] | None:
    """The worker's next message, or None when it exited first or sent something that is not a message."""
    assert process.stdout is not None
    try:
        (length,) = _LENGTH.unpack(await process.stdout.readexactly(_LENGTH.size))
        message = json.loads(await process.stdout.readexactly(length))
    except (asyncio.IncompleteReadError, ValueError):
        return None
    return message if isinstance(message, dict) else None


def _reply_output(reply: dict[str, Any]) -> str | None:
    """The tool output a worker's reply to a call gives, or None for a reply no worker sends."""
    if reply.get("ok") is True and all(isinstance(reply.get(key), str) for key in ("out", "err")):
        result = reply.get("res")
        if result is None or isinstance(result, str):
            return tool_output(success=True, stdout=reply["out"], stderr=reply["err"], result=result)
    elif reply.get("ok") is False and isinstance(reply.get("error"), str):
        return tool_output(success=False, error_message=reply["error"])
    return None


async def _kill(process: asyncio.subprocess.Process) -> None:
    """Kill the worker and everything the code started in its process group, and reap it."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        # A process the code started outside the group may hold the pipes open, and wait() waits for them too. The
        # activation's supervisor stops what is left.
        await asyncio.wait_for(process.wait(), timeout=REAP_TIMEOUT_SECONDS)
    except TimeoutError:
        pass


class PythonTools:
    """The Python tools a task opted into, as the sandboxed agent loop calls them."""

    def __init__(self, specs: dict[str, PythonToolSpec]) -> None:
        self._specs = specs
        self._workers: dict[str, asyncio.subprocess.Process] = {}
        self._locks = {name: asyncio.Lock() for name in specs}

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    async def answer(self, name: str, arguments: Any, *, granted: bool) -> tuple[str, str | None] | None:
        """Answer a call the loop's Python tools own: one of them, or any name when the session has no tool grant.

        Returns the output and the error type to record, or None for a call to pass to the tool grant. With no grant,
        every tool of the task runs here, so another name is the model's mistake: it gets the reply a Resources Server
        gives a call to a tool it does not serve, and the turn goes on.
        """
        if name in self._specs:
            return await self.call(name, arguments)
        return None if granted else (NOT_FOUND, "unknown_tool")

    async def call(self, name: str, arguments: Any) -> tuple[str, str | None]:
        """Run one call's code. Return the tool's output and, for a call without code, the error type to record.

        Errors in the code, and a worker that stops, come back in the output, for the model to read.
        """
        if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
            output = tool_output(success=False, error_message=f'{name} takes {{"code": "<Python source>"}}')
            return output, "invalid_arguments"
        spec = self._specs[name]
        # One call at a time per worker, so each gets its own reply.
        async with self._locks[name]:
            try:
                process = self._workers.get(name) or await self._start(name)
            except Exception as error:
                # A worker that failed to start is started again on the next call.
                return tool_output(success=False, error_message=str(error)), None
            try:
                await _send(process, {"code": arguments["code"]})
                reply = await asyncio.wait_for(
                    _receive(process), timeout=spec.config.timeout_seconds + KILL_GRACE_SECONDS
                )
            except TimeoutError:
                await self._stop(name)
                return tool_output(success=False, error_message=f"code timed-out; {RESTARTS}"), None
            except (BrokenPipeError, ConnectionResetError):
                reply = None
            output = _reply_output(reply) if reply is not None else None
            if output is None:
                # The worker exited, or the code wrote into the reply channel: nothing it sends can be trusted now.
                await self._stop(name)
                return tool_output(success=False, error_message=f"Python worker stopped; {RESTARTS}"), None
            return output, None

    async def _start(self, name: str) -> asyncio.subprocess.Process:
        spec = self._specs[name]
        # -I: the task's Python settings, user site-packages and the working directory never shape the interpreter.
        # Its own session, so a kill reaches every process the code starts in its group.
        process = await asyncio.create_subprocess_exec(
            spec.python,
            "-I",
            spec.worker,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=spec.workdir,
            start_new_session=True,
        )
        try:
            settings = {
                "timeout_seconds": spec.config.timeout_seconds,
                "preload": spec.config.preload,
                "builtins": spec.config.builtins,
                "memory_mib": spec.config.memory_mib,
            }
            await _send(process, settings)
            ready = await asyncio.wait_for(_receive(process), timeout=START_TIMEOUT_SECONDS)
        except TimeoutError:
            await _kill(process)
            raise RuntimeError(f"Python worker did not start within {START_TIMEOUT_SECONDS:.0f} s") from None
        except BaseException:
            await _kill(process)
            raise
        if ready is None or ready.get("ready") is not True:
            await _kill(process)
            detail = ready.get("error") if ready else f"it exited with status {process.returncode}"
            raise RuntimeError(f"Python worker failed to start: {detail}")
        self._workers[name] = process
        return process

    async def _stop(self, name: str) -> None:
        if (process := self._workers.pop(name, None)) is not None:
            await _kill(process)

    async def close(self) -> None:
        """Stop every worker and what its code started. The activation's supervisor would stop them too."""
        for name in list(self._workers):
            await self._stop(name)
