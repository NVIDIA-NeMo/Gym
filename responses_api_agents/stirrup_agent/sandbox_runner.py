# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one Stirrup conversation inside the task sandbox.

The agent server stages this file and the Stirrup modules it imports in the sandbox and launches it with the
sandbox's Stirrup runtime. Shell commands run locally in the working directory; task tools are the resources
server's routes, called over HTTP with the episode's cookies.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import tempfile
import time
import traceback
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402
from openai.types.responses import FunctionToolParam  # noqa: E402
from pydantic import BaseModel, ConfigDict, create_model  # noqa: E402
from stirrup.core.models import (  # noqa: E402
    ImageContentBlock,
    Tool,
    ToolMessage,
    ToolProvider,
    ToolResult,
    ToolUseCountMetadata,
)
from stirrup.tools.code_backends.base import CodeExecToolProvider, CommandResult  # noqa: E402


if TYPE_CHECKING:
    from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent
    from responses_api_agents.stirrup_agent.nemo_client import DynamicMaxTokensChatCompletionsClient


# Web fetches can take minutes; the runner's own deadline still bounds a hung call.
_TOOL_TIMEOUT_S = 900


class LocalShell(CodeExecToolProvider):
    """Mimic original ``code_exec`` behavior from the Apptainer backend."""

    def __init__(self, workdir: str, scratch: Path) -> None:
        super().__init__()
        self._workdir = workdir
        self._scratch = scratch
        # How the last command ended, for its tool-call observation.
        self.last_exit_code: int | None = None
        self.last_timed_out = False

    async def __aenter__(self) -> Tool:
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        timeout = timeout or self._shell_timeout
        script = f"cd {shlex.quote(self._workdir)} && ( timeout -k 10 {timeout} bash -c {shlex.quote(cmd)} )"
        # Files, not pipes: a background process keeps a pipe open after the command returns.
        with tempfile.TemporaryFile(dir=self._scratch) as out, tempfile.TemporaryFile(dir=self._scratch) as err:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                script,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout + 30)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            out.seek(0)
            err.seek(0)
            stdout = out.read().decode("utf-8", errors="replace")
            stderr = err.read().decode("utf-8", errors="replace")
        # The Apptainer backend's end marker left every output ending in a newline.
        if not stdout.endswith("\n"):
            stdout += "\n"
        self.last_exit_code = process.returncode
        self.last_timed_out = process.returncode in (124, 137, -signal.SIGKILL)
        if self.last_timed_out:
            message = f"Command timed out after {timeout} seconds"
            return CommandResult(exit_code=1, stdout=stdout, stderr=f"{stderr}\n{message}" if stderr else message)
        return CommandResult(exit_code=process.returncode, stdout=stdout, stderr=stderr)

    def _path(self, path: str) -> Path:
        return Path(path if path.startswith("/") else f"{self._workdir}/{path}")

    async def read_file_bytes(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    async def file_exists(self, path: str) -> bool:
        return self._path(path).is_file()

    async def is_directory(self, path: str) -> bool:
        return self._path(path).is_dir()

    async def list_files(self, path: str) -> list[str]:
        root = self._path(path)
        return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]

    async def view_image(self, path: str) -> ImageContentBlock:
        return ImageContentBlock(data=await self.read_file_bytes(path))


class InvalidArguments(Exception):
    """The route rejected the arguments; ``detail`` is FastAPI's validation error list."""

    def __init__(self, detail: list[dict[str, Any]]) -> None:
        super().__init__(detail)
        self.detail = detail


class _RemoteParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_: ClassVar[dict[str, Any]]

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return cls.schema_


def remote_tool(spec: FunctionToolParam, client: httpx.AsyncClient) -> Tool:
    """Expose one resources-server route as a Stirrup tool with the schema the row declares.

    Arguments are validated by the route, so any JSON object is accepted here. The parameter model takes the
    schema's title, which Pydantic names in the error for a non-object argument.
    """
    schema = spec.get("parameters") or {"type": "object", "properties": {}}
    params = create_model(schema.get("title", "Params"), __base__=_RemoteParams)
    params.schema_ = schema
    name = spec["name"]

    async def executor(arguments: _RemoteParams) -> ToolResult:
        response = await client.post(f"/{name}", json=arguments.model_dump())
        # A 422 (FastAPI's validation error) and a 400 (a call the route refused, e.g. finish without the files)
        # are mistakes the model can fix, so it sees them and the episode goes on.
        if response.status_code == 422:
            # Answered in run_tool, which has the raw arguments the message quotes.
            raise InvalidArguments(response.json()["detail"])
        if response.status_code == 400:
            return ToolResult(content=response.json()["detail"], success=False, metadata=ToolUseCountMetadata())
        # Anything else is an infrastructure failure: the run fails and the environment server retries a 5xx.
        response.raise_for_status()
        return ToolResult(content=response.json(), metadata=ToolUseCountMetadata())

    return Tool(name=name, description=spec.get("description") or "", parameters=params, executor=executor)


class TaskTools(ToolProvider):
    """The non-finish task tools, entered after the shell like the Tavily provider was, which fixes their order."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools

    async def __aenter__(self) -> list[Tool]:
        return self._tools

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


def _report_invalid_arguments() -> None:
    """Answer a rejected call as Stirrup answers arguments that fail local validation (see ``nemo_client``)."""
    import stirrup.core.agent as stirrup_agent

    run_tool = stirrup_agent.Agent.run_tool

    async def run_tool_reporting_invalid_arguments(self: Any, tool_call: Any, run_metadata: Any) -> ToolMessage:
        started = time.perf_counter()
        try:
            return await run_tool(self, tool_call, run_metadata)
        except InvalidArguments as error:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'][1:]) or '<root>'}: {e['msg']} (type={e.get('type', '?')})"
                for e in error.detail
            )
            return ToolMessage(
                content=(
                    f"Tool arguments are not valid: {errors}. "
                    f"Submitted arguments (first 500 chars): {(tool_call.arguments or '')[:500]!r}"
                ),
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                args_was_valid=False,
                success=False,
                tool_start_time=started,
                tool_end_time=time.perf_counter(),
            )

    stirrup_agent.Agent.run_tool = run_tool_reporting_invalid_arguments


# The episode's one agent invocation. Gym's agents name their top-level invocation "root"; the model-call capture
# assigns a call to the invocation whose ID the call sends as x-session-id.
INVOCATION_ID = "root"


# Values of ``AgentInvocation.status`` and ``ToolCallObservation.status`` in ``nemo_gym.rollout_observability``,
# which the sandbox cannot import.
class InvocationStatus(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class ToolStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Observer:
    """Record the model responses and tool executions the agent server reports as rollout evidence."""

    def __init__(self) -> None:
        self.response_ids: list[str] = []
        self.usages: list[dict[str, Any]] = []
        self.shell: LocalShell | None = None
        # Keyed by the identity of the message Stirrup puts in the history for the call.
        self._tools: dict[int, dict[str, Any]] = {}
        self._clock_offset = time.time() - time.perf_counter()

    def watch_model_calls(self, client: DynamicMaxTokensChatCompletionsClient) -> None:
        # ``_client`` is the ``openai.AsyncOpenAI`` client Stirrup's ChatCompletionsClient sends every call with.
        # A call that fails has no response ID; the header still assigns it to this invocation in the capture.
        client._client = client._client.with_options(default_headers={"x-session-id": INVOCATION_ID})
        completions = client._client.chat.completions
        create = completions.create

        async def observed_create(**kwargs: Any) -> Any:
            response = await create(**kwargs)
            if response.id:
                self.response_ids.append(response.id)
            if response.usage is not None:
                self.usages.append(response.usage.model_dump())
            return response

        completions.create = observed_create

    def watch_tool_calls(self, agent_class: type[NeMoAgent]) -> None:
        run_tool = agent_class.run_tool

        async def observed_run_tool(agent: Any, tool_call: Any, run_metadata: Any) -> Any:
            shell = self.shell
            shell.last_exit_code, shell.last_timed_out = None, False
            message = await run_tool(agent, tool_call, run_metadata)
            if shell.last_timed_out:
                status = ToolStatus.TIMEOUT
            elif not message.success or shell.last_exit_code not in (None, 0):
                status = ToolStatus.FAILED
            else:
                status = ToolStatus.COMPLETED
            started, completed = message.tool_start_time, message.tool_end_time
            self._tools[id(message)] = {
                "kind": "tool_call",
                "invocation_id": INVOCATION_ID,
                "tool_call_id": tool_call.tool_call_id,
                "tool_name": tool_call.name,
                "started_at": started + self._clock_offset,
                "completed_at": completed + self._clock_offset,
                "duration_ms": (completed - started) * 1000,
                "timing_source": "harness",
                "status": status,
            }
            return message

        agent_class.run_tool = observed_run_tool

    def tool_calls(self, history: list[list[Any]] | None, output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The tool observations, under the call IDs the conversation uses for them."""
        if history is not None:
            from stirrup.core.models import ToolMessage

            from responses_api_agents.stirrup_agent.nemo_agent import NeMoUserMessage

            # The same messages, in the same order, that become function_call_output items.
            results = [
                msg
                for turn in history
                for msg in turn
                if (isinstance(msg, NeMoUserMessage) and msg.tool_call_id) or isinstance(msg, ToolMessage)
            ]
            outputs = [item for item in output_items if item["type"] == "function_call_output"]
            for message, item in zip(results, outputs):
                if id(message) in self._tools:
                    self._tools[id(message)]["tool_call_id"] = item["call_id"]
        return list(self._tools.values())

    def report(
        self,
        status: InvocationStatus,
        history: list[list[Any]] | None = None,
        output_items: list[dict[str, Any]] = (),
    ) -> dict:
        return {
            "observations": {
                "invocation_id": INVOCATION_ID,
                "status": status,
                "model_response_ids": self.response_ids,
                "tool_calls": self.tool_calls(history, list(output_items)),
            },
            "usages": self.usages,
        }


async def run(payload: dict[str, Any], scratch: Path, observer: Observer) -> dict[str, Any]:
    from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent
    from responses_api_agents.stirrup_agent.nemo_client import DynamicMaxTokensChatCompletionsClient
    from responses_api_agents.stirrup_agent.stirrup_utils import (
        convert_stirrup_history_to_output_items,
        messages_from_input,
    )

    _report_invalid_arguments()
    system_prompt, messages = messages_from_input(payload["input"], payload["instructions"])
    access = payload["tool_access"]
    async with httpx.AsyncClient(
        base_url=access["base_url"].rstrip("/"),
        cookies=access["cookies"],
        headers=access["headers"],
        timeout=_TOOL_TIMEOUT_S,
    ) as http:
        finish_names = payload["finish_tool_names"]
        task_tools = [remote_tool(spec, http) for spec in payload["tools"]]
        missing = set(finish_names or []) - {tool.name for tool in task_tools}
        if missing:
            raise ValueError(f"Finish tools {sorted(missing)} are not among the request's tools")
        client = DynamicMaxTokensChatCompletionsClient(api_key="gym", **payload["client"])
        shell = LocalShell(payload["workdir"], scratch)
        observer.watch_model_calls(client)
        observer.watch_tool_calls(NeMoAgent)
        observer.shell = shell
        agent = NeMoAgent(
            client=client,
            name="stirrup_agent",
            max_turns=payload["max_turns"],
            tools=[
                shell,
                TaskTools([tool for tool in task_tools if tool.name not in (finish_names or [])]),
            ],
            finish_tool=[tool for tool in task_tools if tool.name in finish_names] if finish_names else None,
            tool_response_as_user=True,
            skip_input_file_listing=True,
            min_compaction_summary_words=payload["min_compaction_summary_words"],
            **({"system_prompt": system_prompt} if system_prompt else {}),
        )
        started = time.time()
        async with agent.session(cache_on_interrupt=False) as session:
            finish_params, history, _ = await session.run(messages)
        status = InvocationStatus.COMPLETED if finish_params is not None else InvocationStatus.INCOMPLETE
        input_items, output_items = convert_stirrup_history_to_output_items(history)
        return {
            "input_items": input_items,
            "output_items": output_items,
            "elapsed_seconds": time.time() - started,
            "resources_cookies": {cookie.name: cookie.value for cookie in http.cookies.jar},
            **observer.report(status, history, output_items),
        }


def main(input_path: str, output_path: str) -> None:
    payload = json.loads(Path(input_path).read_text())
    observer = Observer()
    try:
        output = asyncio.run(run(payload, Path(input_path).parent, observer))
    except BaseException as error:
        output = {
            "error": repr(error),
            "traceback": traceback.format_exc(),
            **observer.report(InvocationStatus.FAILED),
        }
    Path(output_path).write_text(json.dumps(output))


if __name__ == "__main__":
    main(*sys.argv[1:])
