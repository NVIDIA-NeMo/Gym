# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one CrewAI crew turn inside the task sandbox.

The Agent Server stages this file, ``crew.py`` and ``mcp_client.py`` under the CrewAI runtime it installs in the
sandbox, and starts it under Gym's process supervisor with that runtime's Python. The crew's shell tool runs bash in
the task's working directory, its grant tools call the episode's MCP grants, and every model call goes from here to
the Gym Model Server at the rollout's Chat Completions URL. On SIGTERM, which the supervisor sends at the deadline and
on close, the runner stops the crew and reports what it did until then.

The report holds the transcript as Responses API items and the records the Agent Server turns into its observations.
This file needs nothing from Gym, which the sandbox lacks.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


# This file is staged below a private package root, so no task file is imported as code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp  # noqa: E402
import httpx  # noqa: E402


SHELL_TOOL_NAME = "shell"
# CrewAI adds keys of its own to messages, such as cache_breakpoint; the Model Server validates messages strictly.
_CHAT_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id")
# Statuses after which the same model request can still succeed: 408, 425, 429 and every 5xx.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 425, 429})


class ConfigurationError(Exception):
    """The activation asks for tools the episode's grants cannot provide."""


class InfrastructureError(Exception):
    """A tool grant or the sandbox failed underneath the crew; the episode cannot be graded."""

    def __init__(self, message: str, *, operation: str, grant_name: str | None = None) -> None:
        super().__init__(message)
        self.operation = operation
        self.grant_name = grant_name

    def to_report(self) -> dict[str, Any]:
        """Selected process/MCP facts, without serializing requests, URLs, headers or response bodies."""
        causes = []
        error = self.__cause__
        seen: set[int] = set()
        while error is not None and id(error) not in seen and len(causes) < 8:
            seen.add(id(error))
            kind = "agent_run_error"
            status = None
            message = "Operation raised an unexpected exception"
            if isinstance(error, httpx.HTTPStatusError):
                status = error.response.status_code
                message, kind = f"MCP server answered HTTP {status}", "agent_request_failed"
            elif isinstance(error, (httpx.TimeoutException, TimeoutError)):
                message, kind = "MCP operation timed out", "transport_timeout"
            elif isinstance(error, httpx.ConnectError):
                message, kind = "Connection to MCP server failed", "transport_unreachable"
            elif isinstance(error, OSError):
                message = f"OS error {error.errno}: {error.strerror or 'operation failed'}"
            elif isinstance(error, BaseExceptionGroup):
                message = f"Exception group with {len(error.exceptions)} nested exceptions"
            causes.append(
                {
                    "failure_kind": kind,
                    "exception_type": f"{type(error).__module__}.{type(error).__qualname__}"[:256],
                    "message": message[:2000],
                    "http_status": status,
                }
            )
            # MCP task groups wrap one originating error. Do not select an arbitrary cause from a multi-error group.
            error = (
                error.__cause__
                if error.__cause__ is not None
                else error.exceptions[0]
                if isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1
                else None
            )
        return {
            "message": str(self)[:2000],
            "operation": self.operation,
            "grant_name": self.grant_name,
            "causes": causes,
        }


def _retry_wait_seconds(retry: int, retry_after: str | None) -> float:
    """Full-jitter backoff or a provider Retry-After, with Gym's shared 1s base and 60s cap."""
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                seconds = (
                    retry_at.replace(tzinfo=retry_at.tzinfo or timezone.utc) - datetime.now(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = 0.0
        if math.isfinite(seconds) and seconds > 0:
            return min(seconds, 60.0)
    return random.uniform(0, min(60.0, 2.0 ** min(retry, 6)))


class ModelRequestError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        # The upstream body stays in the message: CrewAI reads a context-limit error from it and summarizes.
        super().__init__(f"Model HTTP {status}: {detail}")
        self.status = status
        try:
            document = json.loads(detail)
        except (TypeError, ValueError):
            document = None
        error = document.get("error") if isinstance(document, dict) else None
        # HTTP status alone cannot prove a context limit; use the provider's explicit error code.
        self.context_limit = isinstance(error, dict) and error.get("code") == "context_length_exceeded"


def _retryable(status: int) -> bool:
    return status in _RETRYABLE_CLIENT_STATUSES or status >= 500


def configure_crewai(home: Path) -> dict[str, str]:
    """Point CrewAI's own files at ``home`` before it is imported; return the task environment for tools.

    A Gym run sends no telemetry or traces to CrewAI and does not check PyPI for a newer release. CrewAI keeps its
    replay database and credential key under ``HOME`` and its data directory, so the runner gets a private ``HOME``
    and the task's own ``HOME`` and ``XDG_*`` directories stay untouched; tool commands still run with them.
    """
    task_environment = dict(os.environ)
    home.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CREWAI_DISABLE_TELEMETRY="true",
        CREWAI_DISABLE_TRACKING="true",
        CREWAI_TRACING_ENABLED="false",
        CREWAI_DISABLE_VERSION_CHECK="true",
        # An absolute path: CrewAI joins it under its data directory, which keeps an absolute path as is.
        CREWAI_STORAGE_DIR=str(home / "storage"),
        HOME=str(home),
        XDG_DATA_HOME=str(home / "data"),
        XDG_CONFIG_HOME=str(home / "config"),
        XDG_CACHE_HOME=str(home / "cache"),
    )
    return task_environment


class Shell:
    """The shell tool's commands: a fresh bash in the task directory for each, in a process group of its own.

    Output goes to a file, not a pipe, so a background process that a command leaves (a task service) can neither
    hold the call open nor die with its command. A command's deadline, and the runner's stop, kill its group.
    """

    def __init__(
        self, *, workdir: str, env: dict[str, str], timeout_seconds: float, max_output_bytes: int, scratch: Path
    ) -> None:
        self.workdir = workdir
        self.env = env
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.scratch = scratch

    @property
    def description(self) -> str:
        return (
            f"Run a bash command in the task's environment. Each call starts a fresh bash in {self.workdir}, so cd "
            "and variables do not carry over. Returns the combined stdout and stderr and the exit code. A command is "
            f"killed after {self.timeout_seconds:g} seconds."
        )

    async def run(self, command: str) -> tuple[str, str, str | None]:
        """Run ``command``; return what the model sees, the call's status and its error type."""
        # An earlier command may have deleted the session directory; the next one still runs.
        self.scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=self.scratch) as output:
            try:
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=output,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.workdir,
                    env=self.env,
                    start_new_session=True,
                )
            except OSError as error:
                raise InfrastructureError("could not start bash in the sandbox", operation="shell_start") from error
            timed_out = False
            try:
                await asyncio.wait_for(process.wait(), self.timeout_seconds)
            except TimeoutError:
                timed_out = True
                await _kill_group(process)
            except asyncio.CancelledError:
                await _kill_group(process)
                raise
            text = _bounded_text(output, self.max_output_bytes)
        if timed_out:
            ending, status = f"[command timed out after {self.timeout_seconds:g} s and was killed]", "timeout"
        else:
            ending = f"[exit code: {process.returncode}]"
            status = "failed" if process.returncode else "completed"
        error_type = {"timeout": "timeout", "failed": f"exit_code_{process.returncode}"}.get(status)
        return (f"{text}\n{ending}" if text else ending), status, error_type


async def _kill_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


def _bounded_text(output: Any, limit: int) -> str:
    """A command's output, keeping its head and tail when it is longer than ``limit`` bytes."""
    size = output.seek(0, os.SEEK_END)
    output.seek(0)
    if size <= limit:
        data = output.read()
    else:
        head = output.read(limit // 2)
        output.seek(size - limit // 2)
        data = head + f"\n[... {size - limit} bytes of output omitted ...]\n".encode() + output.read()
    return data.decode(errors="replace")


class Transcript:
    """What the crew did, in the order it did it, recorded on the runner's event loop."""

    def __init__(self, invocation_id: str) -> None:
        self.invocation_id = invocation_id
        self.started = time.monotonic()
        # CrewAI's name for each tool the crew has, to the tool's own name, which the transcript records.
        self.tool_names: dict[str, str] = {}
        self.output: list[dict[str, Any]] = []
        self.response_ids: list[str] = []
        self.model_calls = 0
        self.tool_runs: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.usage: dict[str, int] | None = {"input": 0, "output": 0, "total": 0, "cached": 0, "reasoning": 0}
        self.cached_known = True
        self.reasoning_known = True
        self.finish_reason: str | None = None
        # Whether a model request is out: one the stop interrupts may have used tokens that no response reports.
        self.model_call_pending = False
        # The failure behind a crew run that raised: the last model request that failed, if one did.
        self.model_failure: dict[str, Any] | None = None
        # Set when a grant or the sandbox failed underneath the crew; the crew is stopped at its next call.
        self.infrastructure_error: dict[str, Any] | None = None
        self.on_infrastructure_error: Callable[[], None] = lambda: None
        self._answered_calls: set[str] = set()

    def gap(self, code: str, detail: str | None = None) -> None:
        self.gaps.append({"code": code, "invocation_id": self.invocation_id, "detail": detail})

    def record_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """Tool results the model is about to see, as CrewAI formatted them, in the order it sends them."""
        for message in messages:
            call_id = message.get("tool_call_id")
            if message.get("role") != "tool" or not call_id or call_id in self._answered_calls:
                continue
            self._answered_calls.add(call_id)
            content = message.get("content")
            self.output.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )

    def record_model_response(self, data: dict[str, Any], tool_name: Callable[[str], str]) -> dict[str, Any]:
        """Add a Chat Completions response to the transcript; return its assistant message for CrewAI."""
        choice = (data.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        self.finish_reason = choice.get("finish_reason")
        self.model_calls += 1
        response_id = data.get("id")
        if response_id:
            self.response_ids.append(response_id)
        else:
            self.gap("model_call_reference_unavailable", f"turn:{self.model_calls}")
        self._add_usage(data.get("usage"))

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            self.output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{uuid4().hex}",
                    "summary": [{"type": "summary_text", "text": reasoning}],
                }
            )
        content = message.get("content")
        if isinstance(content, str) and content:
            self.output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            )
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            # Some servers omit a call's ID; CrewAI and the transcript need the same one.
            call = {
                "id": call.get("id") or f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                },
            }
            tool_calls.append(call)
            name = call["function"]["name"]
            self.output.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    # Verifiers compare the tool's own name; CrewAI matched the call by its sanitized form.
                    "name": self.tool_names.get(tool_name(name), name),
                    "arguments": call["function"]["arguments"],
                    "status": "completed",
                }
            )
        message["tool_calls"] = tool_calls
        return message

    def _add_usage(self, usage: Any) -> None:
        if self.usage is None:
            return
        if not isinstance(usage, Mapping) or not all(
            isinstance(usage.get(key), int) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            # Without every call's usage the total is unknown, not a subtotal.
            self.usage = None
            self.gap("model_call_usage_unavailable")
            return
        self.usage["input"] += usage["prompt_tokens"]
        self.usage["output"] += usage["completion_tokens"]
        self.usage["total"] += usage["total_tokens"]
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if isinstance(cached, int):
            self.usage["cached"] += cached
        else:
            self.cached_known = False
        if isinstance(reasoning, int):
            self.usage["reasoning"] += reasoning
        else:
            self.reasoning_known = False

    async def run_tool(
        self, name: str, arguments: dict[str, Any], execute: Callable[[], Awaitable[tuple[Any, str, str | None]]]
    ) -> Any:
        """Run one tool call and record its timing and outcome. ``execute`` returns (result, status, error type)."""
        run: dict[str, Any] = {"name": name, "arguments": arguments, "started_at": time.time()}
        self.tool_runs.append(run)
        started = time.perf_counter()
        try:
            result, run["status"], run["error_type"] = await execute()
            # A ToolFailure carries the text the model sees in its message.
            run["output"] = result if isinstance(result, str) else str(getattr(result, "message", result))
            return result
        except asyncio.CancelledError:
            run["status"], run["error_type"] = "cancelled", "cancelled"
            raise
        except InfrastructureError as error:
            run["status"], run["error_type"] = "failed", "infrastructure_error"
            # The crew would show the model this error and carry on; stop it at its next call instead, so the
            # episode is not graded. This call still returns its error to CrewAI.
            self.infrastructure_error = error.to_report()
            self.on_infrastructure_error()
            raise
        finally:
            run["completed_at"] = max(run["started_at"], time.time())
            run["duration_ms"] = (time.perf_counter() - started) * 1000

    def tool_calls(self) -> list[dict[str, Any]]:
        """Join each executed tool run to the model's call: CrewAI runs a call without passing its ID to the tool."""
        unmatched = list(self.tool_runs)
        calls = [item for item in self.output if item["type"] == "function_call"]
        joined: dict[str, dict[str, Any]] = {}
        # Exact matches across all calls first, so a call CrewAI refused (invalid arguments) cannot take the run of a
        # later call by its name alone.
        for call in calls:
            try:
                arguments = json.loads(call["arguments"])
            except (json.JSONDecodeError, TypeError):
                continue
            run = next(
                (run for run in unmatched if run["name"] == call["name"] and run["arguments"] == arguments), None
            )
            if run is not None:
                unmatched.remove(run)
                joined[call["call_id"]] = run
        for call in calls:
            if call["call_id"] not in joined:
                run = next((run for run in unmatched if run["name"] == call["name"]), None)
                if run is not None:
                    unmatched.remove(run)
                    joined[call["call_id"]] = run
        records = []
        for call in calls:
            run = joined.get(call["call_id"])
            if run is None:
                # CrewAI answered the call itself (an unknown tool or invalid arguments) without running a tool.
                continue
            if "completed_at" not in run:
                # Still running when the run ended; its cancellation reaches it on a later turn of the event loop.
                run["status"], run["error_type"] = "cancelled", "cancelled"
                run["completed_at"] = max(run["started_at"], time.time())
                run["duration_ms"] = (run["completed_at"] - run["started_at"]) * 1000
            records.append(
                {
                    "tool_call_id": call["call_id"],
                    "tool_name": run["name"],
                    "started_at": run["started_at"],
                    "completed_at": run.get("completed_at", run["started_at"]),
                    "duration_ms": run.get("duration_ms", 0.0),
                    "status": run.get("status", "unknown"),
                    "error_type": run.get("error_type"),
                    "output": run.get("output"),
                }
            )
        for run in unmatched:
            self.gap("tool_call_identity_unavailable", f"tool:{run['name']}")
        return records

    def report(self) -> dict[str, Any]:
        tool_calls = self.tool_calls()
        if self.model_call_pending and self.usage is not None:
            self.usage = None
            self.gap("model_call_usage_unavailable", "interrupted")
        if self.usage is not None and not self.cached_known:
            self.gap("cached_token_usage_unavailable")
        if self.usage is not None and not self.reasoning_known:
            self.gap("reasoning_token_usage_unavailable")
        return {
            "invocation_id": self.invocation_id,
            "output": self.output,
            "response_ids": self.response_ids,
            "model_calls": self.model_calls,
            "tool_runs": len(self.tool_runs),
            "tool_calls": tool_calls,
            "usage": None
            if self.usage is None
            else {
                **self.usage,
                "cached": self.usage["cached"] if self.cached_known else None,
                "reasoning": self.usage["reasoning"] if self.reasoning_known else None,
            },
            "finish_reason": self.finish_reason,
            "model_failure": self.model_failure,
            "infrastructure_error": self.infrastructure_error,
            "gaps": self.gaps,
            "duration_ms": (time.monotonic() - self.started) * 1000,
        }


def _chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [{key: value for key, value in dict(message).items() if key in _CHAT_MESSAGE_KEYS} for message in messages]


async def run(payload: dict[str, Any], *, scratch: Path, task_environment: dict[str, str]) -> dict[str, Any]:
    """Run the crew for the payload's instruction and report how it ended, with what it did."""
    from mcp import McpError

    from responses_api_agents.crewai_agent import mcp_client
    from responses_api_agents.crewai_agent.crew import (
        CrewRun,
        GrantTool,
        GymChatLLM,
        ShellTool,
        build_crew,
        grant_tool_args_schema,
        grant_tool_result,
        tool_name,
    )

    crew_run = CrewRun(asyncio.get_running_loop())
    transcript = Transcript(payload["invocation_id"])
    transcript.on_infrastructure_error = crew_run.refuse_new_calls
    report: dict[str, Any] = {"status": "completed", "stop_reason": None, "error_type": None, "error_message": None}
    mcp_timeout = payload["mcp_timeout_seconds"]

    def grant_tool(grant: dict[str, Any], tool: Any) -> GrantTool:
        async def call(arguments: dict[str, Any]) -> Any:
            async def execute() -> tuple[Any, str, str | None]:
                try:
                    result = await asyncio.wait_for(
                        mcp_client.call_tool(grant, tool.name, arguments, timeout_seconds=mcp_timeout), mcp_timeout
                    )
                except TimeoutError:
                    message = f"The tool call timed out after {mcp_timeout:g} seconds"
                    return grant_tool_result(message, failed=True), "timeout", "timeout"
                except McpError as error:
                    # A JSON-RPC error from the server, such as invalid arguments: the model sees it.
                    return grant_tool_result(str(error), failed=True), "failed", "mcp_error"
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise InfrastructureError(
                        f"MCP grant {grant['name']} failed",
                        operation="mcp_call_tool",
                        grant_name=grant["name"],
                    ) from error
                if result.isError:
                    return grant_tool_result(result.content, failed=True), "failed", "tool_error"
                return grant_tool_result(result.content, failed=False), "completed", None

            return await transcript.run_tool(tool.name, arguments, execute)

        name = tool_name(tool.name)
        return GrantTool(
            crew_run=crew_run,
            execute=call,
            name=name,
            description=tool.description or f"Tool {tool.name} from {grant['name']}",
            args_schema=grant_tool_args_schema(name, tool.inputSchema),
        )

    async def crew_tools() -> list[Any]:
        shell = Shell(
            workdir=payload["workdir"],
            env={**task_environment, **payload["task_env"]},
            timeout_seconds=payload["shell_timeout_seconds"],
            max_output_bytes=payload["shell_max_output_bytes"],
            scratch=scratch,
        )

        async def run_command(command: str) -> str:
            return await transcript.run_tool(SHELL_TOOL_NAME, {"command": command}, lambda: shell.run(command))

        tools: list[Any] = [ShellTool(crew_run=crew_run, execute=run_command, description=shell.description)]
        transcript.tool_names[SHELL_TOOL_NAME] = SHELL_TOOL_NAME
        offered = None if payload["tools"] is None else set(payload["tools"])
        served: set[str] = set()
        for grant in payload["grants"]:
            try:
                listed = await asyncio.wait_for(mcp_client.list_tools(grant, timeout_seconds=mcp_timeout), mcp_timeout)
            except Exception as error:
                if grant["required"]:
                    raise InfrastructureError(
                        f"required MCP grant {grant['name']} did not list its tools",
                        operation="mcp_list_tools",
                        grant_name=grant["name"],
                    ) from error
                transcript.gap("mcp_server_skipped", f"{grant['name']}: {type(error).__name__}")
                continue
            for tool in listed:
                served.add(tool.name)
                if offered is None or tool.name in offered:
                    tools.append(grant_tool(grant, tool))
                    transcript.tool_names[tool_name(tool.name)] = tool.name
        if offered is not None and offered - served:
            raise ConfigurationError(
                "CrewAI runs request tools only through an MCP grant; none serves "
                + ", ".join(sorted(offered - served))
            )
        # Grant tools keep their own names, which verifiers compare, rather than CrewAI's server-prefixed MCP names.
        names = [tool.name for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ConfigurationError(f"CrewAI tool names collide after sanitizing: {', '.join(duplicates)}")
        return tools

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout, cookie_jar=aiohttp.CookieJar(unsafe=True)) as http:

        async def complete(messages: list[Any], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
            """One model call of the crew, with bounded retries of transient HTTP statuses."""
            question = _chat_messages(messages)
            transcript.record_tool_results(question)
            body: dict[str, Any] = {"messages": question, **payload["sampling"]}
            if payload["model"]:
                body["model"] = payload["model"]
            if tools:
                body["tools"] = tools
                if not payload["parallel_tool_calls"]:
                    body["parallel_tool_calls"] = False
            transcript.model_call_pending = True
            retry = 0
            while True:
                try:
                    # Rollout collection assigns each captured call, failed ones included, to this invocation.
                    async with http.post(
                        payload["model_url"], json=body, headers={"x-session-id": transcript.invocation_id}
                    ) as response:
                        if response.status < 400:
                            data = await response.json(content_type=None)
                            break
                        status, detail = response.status, await response.text(errors="replace")
                        retry_after = response.headers.get("Retry-After")
                except (aiohttp.ClientError, TimeoutError):
                    transcript.model_call_pending = False
                    transcript.model_failure = {"transport": True}
                    raise
                if retry < payload["max_model_retries"] and _retryable(status):
                    await asyncio.sleep(_retry_wait_seconds(retry, retry_after))
                    retry += 1
                    continue
                transcript.model_call_pending = False
                failure = ModelRequestError(status, detail)
                transcript.model_failure = {"status": status, "context_limit": failure.context_limit}
                raise failure
            # A model call that went through clears an earlier failure that CrewAI recovered from.
            transcript.model_call_pending = False
            transcript.model_failure = None
            return transcript.record_model_response(data, tool_name)

        try:
            tools = await crew_tools()
            llm = GymChatLLM(
                crew_run=crew_run,
                complete=complete,
                model=payload["llm_model"],
                context_window_size=payload["context_window_size"],
            )
            crew = build_crew(
                llm=llm,
                tools=tools,
                role=payload["role"],
                goal=payload["goal"],
                backstory=payload["backstory"],
                description=payload["description"],
                expected_output=payload["expected_output"],
                max_iter=payload["max_iter"],
            )
            await crew.akickoff()
        except asyncio.CancelledError:
            # SIGTERM from the supervisor, at the deadline or on close: report what the crew did so far.
            asyncio.current_task().uncancel()
            if Path(payload["stop_request_path"]).exists():
                report.update(status="incomplete", stop_reason="cancelled")
                transcript.gap("agent_activation_interrupted")
            else:
                report.update(status="incomplete", stop_reason="wall_time")
                transcript.gap("agent_run_timeout")
        except ConfigurationError as error:
            report.update(status="failed", configuration_error=str(error))
        except InfrastructureError as error:
            transcript.infrastructure_error = error.to_report()
        except Exception as raised:
            # Including a model request that timed out, which CrewAI raises as it is.
            report.update(
                status="failed",
                error_type=type(raised).__name__,
                error_message=f"{type(raised).__name__}: {raised}"[:2000],
            )
        finally:
            crew_run.cancel()
    report.update(transcript.report())
    if report["infrastructure_error"] is not None:
        report["status"] = "failed"
    elif report["status"] == "failed" and (report.get("model_failure") or {}).get("context_limit"):
        report.update(status="incomplete", stop_reason="context_limit", error_message=None)
    return report


async def _run_until_terminated(payload: dict[str, Any], output_path: Path, task_environment: dict[str, str]) -> None:
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    # A command may delete the session directory; the scratch files and the report then recreate it.
    with tempfile.TemporaryDirectory(dir=output_path.parent, ignore_cleanup_errors=True) as scratch:
        try:
            output = await run(payload, scratch=Path(scratch), task_environment=task_environment)
        except BaseException as error:
            output = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
    # Written before the event loop shuts down: CrewAI's worker threads may still be unwinding.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output))


def main(input_path: str, output_path: str) -> None:
    payload = json.loads(Path(input_path).read_text())
    task_environment = configure_crewai(Path(input_path).parent / "crewai-home")
    try:
        asyncio.run(_run_until_terminated(payload, Path(output_path), task_environment))
    except BaseException as error:
        if not Path(output_path).exists():
            output = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
            Path(output_path).write_text(json.dumps(output))


if __name__ == "__main__":
    main(*sys.argv[1:])
