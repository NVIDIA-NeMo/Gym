# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent's Python tools: the model's code runs in a Python interpreter beside the agent loop, in its sandbox.

A Python tool is a harness tool. The agent server's config defines it (``python_tools``) and a task opts into it by
name in its ``agent_context`` (``python_tools``), as a Harbor task declares the tools it needs. Only a sandboxed loop
runs one: each tool gets one worker process (``python_tool_worker.py``) in the sandbox the environment lends, started
on the tool's first call and kept for the loop's activation, so names a call defines stay defined for the next call.

The tool takes ``{"code": "<source>"}`` and returns the JSON a Resources Server's Python tool returned before it moved
here: ``{"success", "stdout", "stderr", "error_message", "result"}``.
"""

import asyncio
import builtins
import json
import struct
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


_LENGTH = struct.Struct(">I")
# How long past a call's timeout the loop waits for a worker that has not answered before it kills the worker. The
# worker's own SIGALRM stops ordinary code at the timeout; only code that blocks the signal reaches this.
KILL_GRACE_SECONDS = 5.0
# How long a worker may take to import its preloaded modules before its first call fails.
START_TIMEOUT_SECONDS = 120.0
STOPPED = "Python worker has already stopped"


class PythonToolConfig(BaseModel):
    """How one Python tool runs. Every task that opts into the tool by name gets these settings."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(
        default=10,
        ge=1,
        description="How long one call's code may run before it fails with 'code timed-out'.",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Packages for the tool's own Python runtime, which this server builds and uploads into the sandbox at "
            "seed. Pin each one, for example numpy==2.5.5. Empty, the tool runs on the agent loop's runtime."
        ),
    )
    preload: dict[str, str] = Field(
        default_factory=dict,
        description="Global names bound to modules before the first call, for example {np: numpy}.",
    )
    builtins: list[str] | None = Field(
        default=None,
        description="The builtins the code may use. Unset, all of them.",
    )

    @field_validator("preload")
    @classmethod
    def require_names_and_modules(cls, preload: dict[str, str]) -> dict[str, str]:
        for name, module in preload.items():
            if not name.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
                raise ValueError(
                    f"preload maps a Python name to a module name, such as np: numpy; got {name}: {module}"
                )
        return preload

    @field_validator("builtins")
    @classmethod
    def require_known_builtins(cls, names: list[str] | None) -> list[str] | None:
        if names is not None and (unknown := sorted(name for name in names if not hasattr(builtins, name))):
            raise ValueError(f"builtins names no builtin {unknown}")
        return names


class PythonToolSpec(BaseModel):
    """What the sandboxed loop needs to start a tool's worker: its config and where its files are in the sandbox."""

    model_config = ConfigDict(extra="forbid")

    config: PythonToolConfig
    python: str
    worker: str
    workdir: str | None = None


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


async def _receive(process: asyncio.subprocess.Process) -> Any:
    """The worker's next message, or None when it exited before sending one."""
    assert process.stdout is not None
    try:
        (length,) = _LENGTH.unpack(await process.stdout.readexactly(_LENGTH.size))
        return json.loads(await process.stdout.readexactly(length))
    except asyncio.IncompleteReadError:
        return None


async def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()


class PythonTools:
    """The Python tools a task opted into, as the sandboxed agent loop calls them."""

    def __init__(self, specs: dict[str, PythonToolSpec]) -> None:
        self._specs = specs
        self._workers: dict[str, asyncio.subprocess.Process] = {}
        # Tools whose worker died or was killed: like the Resources Server's worker before, it is not started again.
        self._stopped: set[str] = set()
        self._locks = {name: asyncio.Lock() for name in specs}

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    async def call(self, name: str, arguments: Any) -> str:
        """Run one call's code and return the tool's output. Errors in the code or the worker come back in it."""
        if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
            return tool_output(success=False, error_message=f'{name} takes {{"code": "<Python source>"}}')
        spec = self._specs[name]
        # One call at a time per worker, so each gets its own reply.
        async with self._locks[name]:
            if name in self._stopped:
                return tool_output(success=False, error_message=STOPPED)
            try:
                process = self._workers.get(name) or await self._start(name)
            except Exception as error:
                # A worker that failed to start is started again on the next call.
                return tool_output(success=False, error_message=str(error))
            try:
                await _send(process, {"code": arguments["code"]})
                reply = await asyncio.wait_for(
                    _receive(process), timeout=spec.config.timeout_seconds + KILL_GRACE_SECONDS
                )
            except TimeoutError:
                # The code blocked the worker's alarm: stop it, as a timed-out call.
                await self._stop(name)
                return tool_output(success=False, error_message="code timed-out")
            except (BrokenPipeError, ConnectionResetError):
                reply = None
            if reply is None:
                await self._stop(name)
                return tool_output(success=False, error_message="Python worker stopped while running the code")
            if reply["ok"]:
                return tool_output(success=True, stdout=reply["out"], stderr=reply["err"], result=reply["res"])
            return tool_output(success=False, error_message=reply["error"])

    async def _start(self, name: str) -> asyncio.subprocess.Process:
        spec = self._specs[name]
        # -I: the task's Python settings, user site-packages and the working directory never shape the interpreter.
        process = await asyncio.create_subprocess_exec(
            spec.python,
            "-I",
            spec.worker,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=spec.workdir,
        )
        try:
            settings = {
                "timeout_seconds": spec.config.timeout_seconds,
                "preload": spec.config.preload,
                "builtins": spec.config.builtins,
            }
            await _send(process, settings)
            ready = await asyncio.wait_for(_receive(process), timeout=START_TIMEOUT_SECONDS)
        except BaseException:
            await _kill(process)
            raise
        if ready is None or not ready.get("ready"):
            await _kill(process)
            detail = ready.get("error") if ready else f"it exited with status {process.returncode}"
            raise RuntimeError(f"Python worker failed to start: {detail}")
        self._workers[name] = process
        return process

    async def _stop(self, name: str) -> None:
        self._stopped.add(name)
        if (process := self._workers.pop(name, None)) is not None:
            await _kill(process)

    async def close(self) -> None:
        """Stop every worker. The activation's supervisor would stop them too; this does not wait for it."""
        for name in list(self._workers):
            process = self._workers.pop(name)
            await _kill(process)
