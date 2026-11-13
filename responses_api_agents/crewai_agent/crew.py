# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CrewAI's model and tools for one activation, served from the sandbox runner's event loop.

CrewAI 1.15 runs its agent loop as a Flow whose steps call the model and the tools synchronously from worker threads,
even under ``Crew.akickoff()``. ``CrewRun`` hands each of those calls to the event loop and waits for it, so model
requests and commands run as coroutines the runner can cancel. Cancelling the run cancels every pending call and fails
every later one, which stops the crew at its next step.

The runner imports this module only after ``configure_crewai()`` has set CrewAI's environment.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.mcp.content import content_blocks_to_text
from crewai.tools import BaseTool
from crewai.tools.tool_failure import ToolFailure, ToolFailureReason
from crewai.utilities.pydantic_schema_utils import create_model_from_schema
from crewai.utilities.string_utils import sanitize_tool_name
from pydantic import BaseModel, Field, PrivateAttr


_T = TypeVar("_T")


def tool_name(name: str) -> str:
    """A tool's name as CrewAI's function calling matches it: lowercase letters, digits and underscores."""
    return sanitize_tool_name(name)


def grant_tool_args_schema(name: str, input_schema: dict[str, Any] | None) -> type[BaseModel]:
    """The argument model CrewAI's own MCP tools build from an MCP tool's input schema."""
    schema = input_schema or {"type": "object", "properties": {}}
    return create_model_from_schema(schema, model_name=f"{name}_arguments", enrich_descriptions=True)


def grant_tool_result(content: Any, *, failed: bool) -> str | ToolFailure:
    """An MCP result as CrewAI's own MCP tools give it to the agent: its text, or a ToolFailure for an error."""
    text = content if isinstance(content, str) else content_blocks_to_text(content)
    return ToolFailure(message=text, reason=ToolFailureReason.MCP_ERROR) if failed else text


class CrewRunCancelled(Exception):
    """The activation ended (close or deadline) while the crew still ran."""


class CrewRun:
    """Runs CrewAI's blocking model and tool calls on the event loop, until the activation ends."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._cancelled = threading.Event()
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def call(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        """Run ``coroutine`` on the event loop from a CrewAI worker thread and return its result."""
        if self._cancelled.is_set():
            coroutine.close()
            raise CrewRunCancelled("the activation has ended")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            coroutine.close()
            raise RuntimeError("CrewAI made a blocking model or tool call on the runner's event loop")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        with self._lock:
            self._pending.add(future)
        try:
            # A cancel that came after the first check but before the future was registered.
            if self._cancelled.is_set():
                future.cancel()
            return future.result()
        except concurrent.futures.CancelledError as error:
            raise CrewRunCancelled("the activation has ended") from error
        finally:
            with self._lock:
                self._pending.discard(future)

    async def acall(self, coroutine: Coroutine[Any, Any, _T]) -> _T:
        """Run ``coroutine`` for a CrewAI step that is already on the event loop."""
        if self._cancelled.is_set():
            coroutine.close()
            raise CrewRunCancelled("the activation has ended")
        return await coroutine

    def refuse_new_calls(self) -> None:
        """Fail every later call with ``CrewRunCancelled``; the calls in flight finish."""
        self._cancelled.set()

    def cancel(self) -> None:
        """Cancel the calls in flight; later calls raise ``CrewRunCancelled``."""
        self.refuse_new_calls()
        with self._lock:
            pending = list(self._pending)
        for future in pending:
            future.cancel()


ChatCompletion = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[dict[str, Any]]]


class GymChatLLM(BaseLLM):
    """CrewAI's model: each call is one Chat Completions request to the Gym Model Server.

    ``complete`` sends CrewAI's messages and tool schemas and returns the response's assistant message. Tool calls go
    back to CrewAI in the OpenAI shape its native function calling reads.
    """

    llm_type: str = "gym"
    _crew_run: CrewRun = PrivateAttr()
    _complete: ChatCompletion = PrivateAttr()

    def __init__(self, *, crew_run: CrewRun, complete: ChatCompletion, **data: Any) -> None:
        super().__init__(**data)
        self._crew_run = crew_run
        self._complete = complete

    @classmethod
    def _wrap_call_method(cls, method_name: str) -> None:
        """Keep ``call`` and ``acall`` without CrewAI's retry wrapper.

        CrewAI's wrapper sends a failed call up to three times when the error's text reads like a rate limit, a 4xx
        included. ``complete`` is the only retry layer: it retries HTTP 408, 425, 429 and 5xx within a bounded wait.
        """

    def call(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | list[dict[str, Any]]:
        return self._crew_run.call(self._answer(messages, tools, available_functions, response_model))

    async def acall(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | list[dict[str, Any]]:
        return await self._crew_run.acall(self._answer(messages, tools, available_functions, response_model))

    async def _answer(
        self,
        messages: Any,
        tools: list[dict[str, Any]] | None,
        available_functions: dict[str, Any] | None,
        response_model: type[BaseModel] | None,
    ) -> str | list[dict[str, Any]]:
        # The crew is built without structured outputs or a function map, so CrewAI never asks for either.
        if available_functions or response_model is not None:
            raise ValueError("The Gym model for CrewAI supports neither available_functions nor response_model")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        message = await self._complete(messages, tools)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            return [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
                }
                for call in tool_calls
            ]
        return message.get("content") or ""

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        # Native function calling ends each turn without stop sequences, so none are sent.
        return False


class ShellArguments(BaseModel):
    command: str = Field(description="The bash command to run.")


class ShellTool(BaseTool):
    """CrewAI's shell tool: a bash command in the task's working directory."""

    name: str = "shell"
    args_schema: type[BaseModel] = ShellArguments
    _crew_run: CrewRun = PrivateAttr()
    _execute: Callable[[str], Awaitable[str]] = PrivateAttr()

    def __init__(self, *, crew_run: CrewRun, execute: Callable[[str], Awaitable[str]], **data: Any) -> None:
        super().__init__(**data)
        self._crew_run = crew_run
        self._execute = execute

    def _run(self, command: str) -> str:
        return self._crew_run.call(self._execute(command))

    async def _arun(self, command: str) -> str:
        return await self._crew_run.acall(self._execute(command))


class GrantTool(BaseTool):
    """One tool of an MCP grant, with the argument model CrewAI builds from its input schema."""

    _crew_run: CrewRun = PrivateAttr()
    _execute: Callable[[dict[str, Any]], Awaitable[Any]] = PrivateAttr()

    def __init__(self, *, crew_run: CrewRun, execute: Callable[[dict[str, Any]], Awaitable[Any]], **data: Any) -> None:
        super().__init__(**data)
        self._crew_run = crew_run
        self._execute = execute

    def _run(self, **arguments: Any) -> Any:
        return self._crew_run.call(self._execute(arguments))

    async def _arun(self, **arguments: Any) -> Any:
        return await self._crew_run.acall(self._execute(arguments))


def _never_cache(arguments: Any, result: Any) -> bool:
    # A command or tool call can return something new each time, so no result is reused.
    return False


def build_crew(
    *,
    llm: GymChatLLM,
    tools: list[BaseTool],
    role: str,
    goal: str,
    backstory: str,
    description: str,
    expected_output: str,
    max_iter: int,
) -> Crew:
    """One agent with one task: a single CrewAI turn for the activation's instruction."""
    for tool in tools:
        tool.cache_function = _never_cache
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        function_calling_llm=llm,
        tools=tools,
        max_iter=max_iter,
        allow_delegation=False,
        cache=False,
        memory=False,
        verbose=False,
        # CrewAI would run a failed task again from the start, in the sandbox the first attempt already changed. A
        # failure ends the run instead, and the Environment Server's outcome policy decides what follows.
        max_retry_limit=0,
        # Oversized tool results would move to temporary files that the model reads through an extra CrewAI tool,
        # outside the shell and the grants the transcript records.
        use_better_compaction=False,
    )
    # The instruction is the task's text as is. The crew is started without inputs, so CrewAI does not read the
    # braces in it as template variables.
    task = Task(description=description, expected_output=expected_output, agent=agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        memory=False,
        planning=False,
        tracing=False,
        verbose=False,
    )
