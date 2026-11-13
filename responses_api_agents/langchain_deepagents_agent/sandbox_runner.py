# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one deepagents graph inside the task sandbox.

The agent server stages this file and responses_chat_model.py under the deepagents runtime it installs in the
sandbox, and starts it under Gym's process supervisor with that runtime's Python. The graph's file tools and its
`execute` tool work on the sandbox's own filesystem, in the task's working directory, and every model call goes from
here to the Gym Model Server at the rollout's URL. On SIGTERM, which the supervisor sends at the deadline and on close,
the runner stops the graph and reports the messages finished so far.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


# A SIGTERM that arrives while the runner still imports deepagents (seconds in a cold sandbox) is noted here instead of
# killing the runner without a report; the event loop's handler takes over once the graph can be cancelled.
_terminated_before_start = False


def _note_termination(signum: int, frame: object) -> None:
    global _terminated_before_start
    _terminated_before_start = True


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _note_termination)

# This file is staged below a private package root, so no task file is imported as code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp  # noqa: E402
from deepagents import create_deep_agent  # noqa: E402
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend  # noqa: E402
from deepagents.backends.protocol import ExecuteResponse  # noqa: E402
from langchain_core.messages import messages_from_dict, messages_to_dict  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from pydantic import Field  # noqa: E402

from responses_api_agents.langchain_deepagents_agent.responses_chat_model import (  # noqa: E402
    ResponsesChatModel,
    is_context_limit,
)


# deepagents' own files (offloaded conversation history, evicted large tool results) stay in graph state under this
# path, so the harness writes nothing of its own into the task sandbox.
HARNESS_FILES_ROOT = "/.deepagents/"
# Statuses after which the same model request can still succeed, so it is sent again a few times: these and every 5xx.
# Any other 4xx fails the same way each time. Three attempts wait at most 2 x 30 s in all.
_RETRY_STATUSES = frozenset({408, 425, 429})
_MODEL_ATTEMPTS = 3
_MAX_RETRY_DELAY_S = 30.0
# A failed runner's error text and traceback are capped, since they end up in the activation's error.
_MAX_ERROR_CHARS = 4000


class ModelRequestError(RuntimeError):
    """The Model Server answered a model request with an HTTP error, after any retries."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Model Server HTTP {status}: {body[:2000]}")
        self.status = status
        self.context_limit = is_context_limit(body)


class ModelTransportError(RuntimeError):
    """A model request never got an HTTP answer from the Model Server, after any retries."""


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    """The server's ``Retry-After`` in seconds when it gives one, else an exponential backoff; capped either way."""
    try:
        delay = float(retry_after) if retry_after else 2.0**attempt
    except ValueError:
        delay = 2.0**attempt
    return min(max(delay, 0.0), _MAX_RETRY_DELAY_S)


class SandboxResponsesChatModel(ResponsesChatModel):
    """Send each model request from the sandbox to the Model Server, at the rollout's URL."""

    http: Any = Field(exclude=True)
    url: str
    # The activation request's sampling controls, sent with every model call.
    request_defaults: dict[str, Any] = Field(default_factory=dict)
    # Model calls the graph made, and the usage of each that returned (None where a response had no usage).
    calls: int = 0
    usage_incomplete: bool = False
    usages: list[dict[str, Any] | None] = Field(default_factory=list)

    async def create_response(self, request_body: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        request = {**self.request_defaults, **request_body}
        for attempt in range(1, _MODEL_ATTEMPTS + 1):
            try:
                async with self.http.post(self.url, json=request) as response:
                    if response.status < 400:
                        try:
                            body = await response.json(content_type=None)
                        except (aiohttp.ClientError, TimeoutError):
                            # Successful headers may precede a generation whose usage we never receive.
                            self.usage_incomplete = True
                            raise
                        self.usages.append(body.get("usage"))
                        return body
                    failure: Exception = ModelRequestError(response.status, await response.text())
                    retryable = response.status in _RETRY_STATUSES or 500 <= response.status < 600
                    retry_after = response.headers.get("Retry-After")
            except (aiohttp.ClientError, TimeoutError) as error:
                failure, retryable, retry_after = ModelTransportError(f"{type(error).__name__}: {error}"), True, None
            if not retryable or attempt == _MODEL_ATTEMPTS:
                raise failure
            await asyncio.sleep(_retry_delay(retry_after, attempt))
        raise AssertionError("unreachable")


class TaskShellBackend(LocalShellBackend):
    """deepagents' local shell, run so that a command can neither outlive its timeout nor hold the runner up.

    Output goes to files, not pipes: a background process that a command starts would keep a pipe open after the
    command returns. Each command runs in a session of its own, and a timeout or the runner's stop kills that
    process group; a process the command started in another session survives as a task service. The output reads
    as LocalShellBackend's.
    """

    def __init__(self, *, scratch: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._scratch = scratch

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1, truncated=False)
        limit = timeout if timeout is not None else self._default_timeout
        if limit <= 0:
            raise ValueError(f"timeout must be positive, got {limit}")
        # Commands may remove the scratch directory; recreate it as the task's user before the next call.
        self._scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=self._scratch) as stdout, tempfile.TemporaryFile(dir=self._scratch) as stderr:
            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=str(self.cwd),
                    env=self._env,
                    start_new_session=True,
                )
            except OSError as error:
                return ExecuteResponse(
                    output=f"Error executing command ({type(error).__name__}): {error}", exit_code=1, truncated=False
                )
            try:
                await asyncio.wait_for(process.wait(), limit)
            except TimeoutError:
                await _kill(process)
                if timeout is not None:
                    message = (
                        f"Error: Command timed out after {limit} seconds (custom timeout). "
                        "The command may be stuck or require more time."
                    )
                else:
                    message = (
                        f"Error: Command timed out after {limit} seconds. "
                        "For long-running commands, re-run using the timeout parameter."
                    )
                return ExecuteResponse(output=message, exit_code=124, truncated=False)
            except asyncio.CancelledError:
                await _kill(process)
                raise
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read(self._max_output_bytes + 1).decode("utf-8", errors="replace")
            err = stderr.read(self._max_output_bytes + 1).decode("utf-8", errors="replace")
        parts = [out] if out else []
        if err:
            parts.extend(f"[stderr] {line}" for line in err.strip().split("\n"))
        output = "\n".join(parts) if parts else "<no output>"
        truncated = len(output) > self._max_output_bytes
        if truncated:
            output = f"{output[: self._max_output_bytes]}\n\n... Output truncated at {self._max_output_bytes} bytes."
        if process.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {process.returncode}"
        return ExecuteResponse(output=output, exit_code=process.returncode, truncated=truncated)


async def _kill(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


async def run(payload: dict[str, Any], scratch: Path) -> dict[str, Any]:
    """Run the graph on the payload's messages and report how it ended, with the messages it added."""
    # Relative paths in the file tools resolve against the task's working directory, as the shell's do.
    os.chdir(payload["workdir"])
    input_messages = messages_from_dict(payload["messages"])
    messages = input_messages
    report: dict[str, Any] = {"status": "completed", "stop_reason": None, "model_error": None}
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout, cookie_jar=aiohttp.CookieJar(unsafe=True)) as http:
        model = SandboxResponsesChatModel(
            http=http, url=payload["model_url"], request_defaults=payload["request_defaults"]
        )
        # deepagents reads the summarization threshold from the model's profile.
        model.profile = {"max_input_tokens": payload["max_input_tokens"]}
        shell = TaskShellBackend(
            root_dir=payload["workdir"],
            virtual_mode=False,
            inherit_env=True,
            env=payload["task_env"],
            timeout=payload["shell_timeout_seconds"],
            scratch=scratch,
        )
        backend = CompositeBackend(
            default=shell, routes={HARNESS_FILES_ROOT: StateBackend()}, artifacts_root=HARNESS_FILES_ROOT
        )
        graph = create_deep_agent(model=model, backend=backend, system_prompt=payload["system_prompt"])
        # The rollout's ID names the file that summarization offloads history to, as in the in-process graph.
        config = {"configurable": {"thread_id": payload["rollout_id"]}}
        try:
            async for state in graph.astream({"messages": input_messages}, config=config, stream_mode="values"):
                messages = state["messages"]
        except asyncio.CancelledError:
            # SIGTERM, from the supervisor at the deadline or on close: report the messages finished so far.
            asyncio.current_task().uncancel()
            report.update(status="incomplete", stop_reason=_stop_reason(payload))
        except GraphRecursionError:
            report.update(status="incomplete", stop_reason="max_iterations")
        except ModelRequestError as error:
            report.update(
                status="incomplete" if error.context_limit else "failed",
                stop_reason="context_limit" if error.context_limit else None,
                model_error=None if error.context_limit else str(error),
                model_error_status=error.status,
                model_context_limit=error.context_limit,
            )
        except ModelTransportError as error:
            report.update(status="failed", model_error=str(error), model_transport_error=True)
    return {
        **report,
        "messages": messages_to_dict(messages[len(input_messages) :]),
        "model_calls": model.calls,
        "usages": model.usages,
        "usage_incomplete": model.usage_incomplete,
    }


def _stop_reason(payload: dict[str, Any]) -> str:
    """Why a SIGTERM stopped the run: close writes its stop request first; otherwise the runner takes it for the
    deadline, which the agent server checks against the supervisor's receipt."""
    return "cancelled" if Path(payload["stop_request_path"]).exists() else "wall_time"


def _stopped_before_start(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "stop_reason": _stop_reason(payload),
        "model_error": None,
        "messages": [],
        "model_calls": 0,
        "usages": [],
        "usage_incomplete": False,
    }


async def _run_until_terminated(payload: dict[str, Any], scratch: Path) -> dict[str, Any]:
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    if _terminated_before_start:
        return _stopped_before_start(payload)
    try:
        return await run(payload, scratch)
    except asyncio.CancelledError:
        # A SIGTERM before the graph started.
        asyncio.current_task().uncancel()
        return _stopped_before_start(payload)


def _write_output(path: Path, output: dict[str, Any]) -> None:
    """Write the report whole, so a reader never sees part of it even if the supervisor kills the runner meanwhile."""
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(output))
    os.replace(partial, path)


def main(input_path: str, output_path: str) -> None:
    payload = json.loads(Path(input_path).read_text())
    loop = asyncio.new_event_loop()
    try:
        output = loop.run_until_complete(_run_until_terminated(payload, Path(input_path).parent))
    except BaseException as error:
        output = {
            "status": "failed",
            "error": repr(error)[:_MAX_ERROR_CHARS],
            "traceback": traceback.format_exc()[-_MAX_ERROR_CHARS:],
        }
    # Written before the loop closes: asyncio.run() would first wait for deepagents' file-tool threads, which its own
    # timeouts do not stop, and the supervisor kills a runner that outlasts its stop grace period.
    _write_output(Path(output_path), output)
    loop.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
    # Exit without joining those threads either.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
