# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native implementation of the Legal Agent Bench tool loop."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import Body, Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


INITIAL_EMPTY_RESPONSE_NUDGE = (
    "Your last response was empty and did not call any tools. Continue the task. "
    "Use the available tools to inspect the documents and write the required deliverables."
)
CONTAINER_TOOL_RUNNER = Path("/opt/legal-agent-bench/container_tool_runner.py")
SUPPORTED_TOOLS = frozenset({"bash", "read", "write", "write_docx", "edit", "glob", "grep"})


class LegalAgentBenchNativeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_turns: int = Field(default=60, ge=1)
    shell_timeout: int = Field(default=60, ge=1)
    model_timeout_seconds: int = Field(default=1800, ge=1)
    max_output_chars: int = Field(default=16_384, ge=1)


class LegalAgentBenchNativeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class LabToolExecutor:
    """Execute the upstream LAB tools in the current task sandbox."""

    def __init__(self, *, timeout_seconds: int, max_output_chars: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    async def preflight(self) -> None:
        result = await self.execute("preflight", {})
        if result.startswith("Error:"):
            raise RuntimeError(result)

    async def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        if name != "preflight" and name not in SUPPORTED_TOOLS:
            return f"Error: unknown tool: {name}"
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return f"Error: invalid JSON arguments for {name}: {exc}"
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            return f"Error: arguments for {name} must be a JSON object"

        if name == "bash":
            command = parsed.get("command")
            if not isinstance(command, str) or not command:
                return "Error: command is required"
            result = await self._run(["/bin/bash", "-lc", command])
        else:
            result = await self._run(["/usr/local/bin/python", str(CONTAINER_TOOL_RUNNER), name, json.dumps(parsed)])
        full_read = name == "read" and "limit" in parsed and parsed.get("limit") in {0, None}
        return result if full_read else self._truncate(result)

    async def _run(self, command: list[str]) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd="/workspace/output",
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.communicate()
            return f"Error: tool timed out after {self.timeout_seconds}s"

        output = stdout.decode(errors="replace")
        error = stderr.decode(errors="replace")
        command_name = Path(command[1]).name if len(command) > 1 else "tool"
        if command[0] == "/usr/local/bin/python":
            try:
                payload = json.loads(next(line for line in reversed(output.splitlines()) if line.strip()))
            except (StopIteration, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and "result" in payload:
                output = str(payload["result"])
        if process.returncode:
            detail = error or output or f"{command_name} exited with code {process.returncode}"
            return f"Error: {detail.strip()}"
        if error:
            output = f"{output}\nSTDERR:\n{error}" if output else error
        return output or "(no output)"

    def _truncate(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars] + "\n[output truncated]"


def _output_text(messages: list[NeMoGymResponseOutputMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        for content in message.content:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _merge_usage(total: Any, current: Any) -> Any:
    if current is None:
        return total
    if total is None:
        return current.model_copy(deep=True)
    total.input_tokens += current.input_tokens
    total.output_tokens += current.output_tokens
    total.total_tokens += current.total_tokens
    total.input_tokens_details.cached_tokens += current.input_tokens_details.cached_tokens
    total.output_tokens_details.reasoning_tokens += current.output_tokens_details.reasoning_tokens
    return total


def _failed_response(
    *,
    body: NeMoGymResponseCreateParamsNonStreaming,
    response: Optional[NeMoGymResponse],
    output: list[Any],
    usage: Any,
    message: str,
) -> NeMoGymResponse:
    base = response or NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=0,
        model=body.model or "policy_model",
        object="response",
        output=[],
        parallel_tool_calls=body.parallel_tool_calls,
        tool_choice=body.tool_choice,
        tools=body.tools,
    )
    return NeMoGymResponse.model_validate(
        base.model_dump(mode="json")
        | {
            "status": "failed",
            "error": {"code": "server_error", "message": message[-2000:]},
            "output": output,
            "usage": usage,
        }
    )


def _limit_response(
    *,
    body: NeMoGymResponseCreateParamsNonStreaming,
    response: Optional[NeMoGymResponse],
    output: list[Any],
    usage: Any,
    stop_reason: str,
) -> NeMoGymResponse:
    """Return a scoreable incomplete response for a normal agent-loop limit."""
    base = response or NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=0,
        model=body.model or "policy_model",
        object="response",
        output=[],
        parallel_tool_calls=body.parallel_tool_calls,
        tool_choice=body.tool_choice,
        tools=body.tools,
    )
    metadata = dict(base.metadata or {})
    metadata["nemo_gym_stop_reason"] = stop_reason
    return NeMoGymResponse.model_validate(
        base.model_dump(mode="json")
        | {
            "status": "incomplete",
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"},
            "metadata": metadata,
            "output": output,
            "usage": usage,
        }
    )


class LegalAgentBenchNativeAgent(SimpleResponsesAPIAgent):
    """Run LAB's canonical tool loop through Gym's Responses API model server."""

    config: LegalAgentBenchNativeAgentConfig

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        executor = LabToolExecutor(
            timeout_seconds=self.config.shell_timeout,
            max_output_chars=self.config.max_output_chars,
        )
        try:
            await executor.preflight()
        except Exception as exc:
            return _failed_response(
                body=body,
                response=None,
                output=[],
                usage=None,
                message=f"LAB tool preflight failed: {type(exc).__name__}: {exc}",
            )

        trajectory: list[Any] = []
        usage = None
        last_response: Optional[NeMoGymResponse] = None
        model_cookies = None
        empty_responses = 0

        for _turn in range(self.config.max_turns):
            model_input = body.model_copy(update={"input": list(body.input) + trajectory})
            try:
                raw_response = await asyncio.wait_for(
                    self.server_client.post(
                        server_name=self.config.model_server.name,
                        url_path=self.url_path_for_request("/v1/responses", request),
                        json=model_input,
                        cookies=model_cookies,
                    ),
                    timeout=self.config.model_timeout_seconds,
                )
                await raise_for_status(raw_response)
                model_response = NeMoGymResponse.model_validate(await get_response_json(raw_response))
                model_cookies = raw_response.cookies
            except Exception as exc:
                return _failed_response(
                    body=body,
                    response=last_response,
                    output=trajectory,
                    usage=usage,
                    message=f"LAB model call failed: {type(exc).__name__}: {exc}",
                )

            last_response = model_response
            usage = _merge_usage(usage, model_response.usage)
            trajectory.extend(model_response.output)
            if model_response.error is not None or model_response.incomplete_details is not None:
                return model_response.model_copy(update={"output": trajectory, "usage": usage})

            function_calls = [
                item for item in model_response.output if isinstance(item, NeMoGymResponseFunctionToolCall)
            ]
            assistant_messages = [
                item for item in model_response.output if isinstance(item, NeMoGymResponseOutputMessage)
            ]
            if not function_calls and assistant_messages:
                if _output_text(assistant_messages):
                    return model_response.model_copy(
                        update={"status": "completed", "output": trajectory, "usage": usage}
                    )
                empty_responses += 1
                if empty_responses > 2:
                    return _failed_response(
                        body=body,
                        response=model_response,
                        output=trajectory,
                        usage=usage,
                        message="LAB model repeatedly returned empty responses",
                    )
                trajectory.append(NeMoGymEasyInputMessage(role="user", content=INITIAL_EMPTY_RESPONSE_NUDGE))
                continue

            if not function_calls and not model_response.output:
                empty_responses += 1
                if empty_responses > 2:
                    return _failed_response(
                        body=body,
                        response=model_response,
                        output=trajectory,
                        usage=usage,
                        message="LAB model repeatedly returned empty responses",
                    )
                trajectory.append(NeMoGymEasyInputMessage(role="user", content=INITIAL_EMPTY_RESPONSE_NUDGE))
                continue

            empty_responses = 0
            for call in function_calls:
                result = await executor.execute(call.name, call.arguments)
                trajectory.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call.call_id,
                        output=result,
                    )
                )

        return _limit_response(
            body=body,
            response=last_response,
            output=trajectory,
            usage=usage,
            stop_reason="max_turns",
        )

    async def run(self, request: Request, body: LegalAgentBenchNativeRunRequest):
        raise NotImplementedError("The LAB-native agent runs inside the task-driven LAB sandbox")


if __name__ == "__main__":
    LegalAgentBenchNativeAgent.run_webserver()
