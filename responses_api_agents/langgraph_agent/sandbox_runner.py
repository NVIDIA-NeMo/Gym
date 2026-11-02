# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a complete LangGraph workflow in the sandbox supervised by Gym."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4


if TYPE_CHECKING:
    from pydantic import JsonValue


_CANCELLATION_GRACE_SECONDS = 5.0


def _runtime() -> dict[str, str | int]:
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.executable}


def _write_atomic(path: Path, payload: dict[str, JsonValue]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


async def _run(payload: object) -> dict[str, JsonValue]:
    # Imports occur inside the worker failure boundary, so missing runtime dependencies also leave a receipt.
    from pydantic import BaseModel, ConfigDict, Field
    from pydantic import JsonValue as RuntimeJsonValue

    from nemo_gym.client import get_global_aiohttp_client, is_global_aiohttp_client_setup
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming

    if __package__:
        from .workflows import run_workflow
    else:
        from workflows import run_workflow

    class RunnerInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        workflow: str
        config: dict[str, RuntimeJsonValue] = Field(default_factory=dict)
        request: NeMoGymResponseCreateParamsNonStreaming
        model_url: str

    RunnerInput.model_rebuild(
        _types_namespace={
            "RuntimeJsonValue": RuntimeJsonValue,
            "NeMoGymResponseCreateParamsNonStreaming": NeMoGymResponseCreateParamsNonStreaming,
        }
    )
    parsed = RunnerInput.model_validate(payload)
    url = urlsplit(parsed.model_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username is not None or url.password is not None:
        raise ValueError("A credential-free HTTP Model Server URL is required")
    try:
        response = await run_workflow(
            workflow=parsed.workflow,
            config=parsed.config,
            request_body=parsed.request,
            model_url=parsed.model_url,
        )
        return {"response": response.model_dump(mode="json"), "runtime": _runtime()}
    finally:
        if is_global_aiohttp_client_setup():
            async with asyncio.timeout(_CANCELLATION_GRACE_SECONDS):
                await get_global_aiohttp_client().close()


async def _run_with_signals(payload: object, output_path: Path, exit_timers: list[Timer]) -> dict[str, JsonValue]:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupt() -> None:
        if exit_timers:
            return
        # Keep the deadline alive through asyncio.run's residual-task shutdown too.
        timer = Timer(_CANCELLATION_GRACE_SECONDS, os._exit, args=(143,))
        timer.daemon = True
        exit_timers.append(timer)
        timer.start()
        try:
            # Write before cancellation: even a task that fails to unwind leaves a nonsecret failure receipt.
            _write_atomic(
                output_path,
                {"error": "LangGraph workflow cancelled.", "error_type": "CancelledError", "runtime": _runtime()},
            )
        finally:
            if task is not None:
                task.cancel()

    for sig in previous_handlers:
        loop.add_signal_handler(sig, interrupt)
    try:
        return await _run(payload)
    finally:
        for sig, previous in previous_handlers.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous)


def _run_worker(input_path: Path, output_path: Path) -> int:
    exit_timers: list[Timer] = []
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        output = asyncio.run(_run_with_signals(payload, output_path, exit_timers))
    except asyncio.CancelledError:
        _write_atomic(
            output_path,
            {"error": "LangGraph workflow cancelled.", "error_type": "CancelledError", "runtime": _runtime()},
        )
        return 143
    except Exception as error:
        # Error strings, HTTP bodies and tracebacks can carry endpoint credentials or user input.
        _write_atomic(
            output_path,
            {"error": "LangGraph workflow failed.", "error_type": type(error).__name__, "runtime": _runtime()},
        )
        return 1
    finally:
        for timer in exit_timers:
            timer.cancel()
    _write_atomic(output_path, output)
    return 0


def main() -> int:
    """Run one input/output exchange; the uploaded supervisor controls the process lifetime."""
    if len(sys.argv) != 3:
        print("usage: sandbox_runner.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    return _run_worker(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    raise SystemExit(main())
