# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes execution with injected model I/O and a caller-owned sandbox."""

import asyncio
import json
import os
from concurrent.futures import Future
from functools import cache
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from threading import Lock
from time import time
from typing import Literal
from uuid import uuid4

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from nemo_gym.openai_utils import NeMoGymFunctionCallOutput, NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ModelCallRef,
    TrajectoryRecord,
    TrajectoryToolCall,
    TrajectoryTurn,
)


HERMES_REVISION = "26bb847a88493342ca1b194e0455b479073ae21d"
TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a bash command in the task sandbox. Each call starts in the task working directory.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


@cache
def hermes_runtime():
    directory = TemporaryDirectory(prefix="gym-hermes-")
    Path(directory.name, "config.yaml").write_text(
        "model:\n  context_length: 128000\ncompression:\n  enabled: false\n"
        "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n"
        "honcho:\n  enabled: false\n"
    )
    return directory


class HermesConfig(BaseModel):
    name: Literal["hermes"]
    max_turns: int = Field(default=90, gt=0)
    step_timeout_sec: int = Field(default=600, gt=0)


class HarnessContext(BaseModel):
    session_id: str
    instruction: str
    user: str | int | None = None
    workdir: str | None = None
    setup_timeout_sec: float = Field(default=360, gt=0)
    mcp_servers: list[dict] = Field(default_factory=list)
    skills_dir: str | None = None


class HarnessOutcome(BaseModel):
    reason: str
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class WorkerBridge:
    """Join Hermes' synchronous loop and the asynchronous operations it started."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.pending = set()
        self.tasks = set()
        self.closed = False
        self.lock = Lock()

    def start(self, factory, future):
        with self.lock:
            if self.closed:
                future.cancel()
                return
            task = self.loop.create_task(factory())
            self.tasks.add(task)

        def completed(task):
            self.tasks.discard(task)
            if future.cancelled():
                return
            if task.cancelled():
                future.cancel()
            elif error := task.exception():
                future.set_exception(error)
            else:
                future.set_result(task.result())

        task.add_done_callback(completed)

    def call(self, factory):
        with self.lock:
            if self.closed:
                raise RuntimeError("Episode is closed")
            future = Future()
            self.pending.add(future)
            self.loop.call_soon_threadsafe(self.start, factory, future)
        try:
            return future.result()
        finally:
            with self.lock:
                self.pending.discard(future)

    async def close(self):
        with self.lock:
            self.closed = True
            for future in self.pending:
                future.cancel()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class HermesHarness:
    """Execute only; provisioning, verification and cleanup belong to the caller."""

    def __init__(self, *, sandbox, context, config, params, query, model_name, directory: Path):
        self.sandbox = sandbox
        self.context = context
        self.config = config
        self.params = params
        self.query = query
        self.model_name = model_name
        self.directory = directory

    async def setup(self):
        if self.context.mcp_servers:
            raise ValueError("The Hermes terminal profile does not support task MCP servers")
        self.directory.mkdir(parents=True, exist_ok=True)
        os.environ["HERMES_HOME"] = hermes_runtime().name
        result = await self.sandbox.exec("command -v setsid", user=self.context.user, cwd=self.context.workdir)
        if result.return_code:
            raise RuntimeError("Hermes requires setsid for process cleanup")

    async def execute(self, budget):
        from run_agent import AIAgent

        bridge = WorkerBridge()
        responses, output, messages = [], [], []
        tool_error = None
        recorded_tools = set()
        tool_observations = []
        turns = []
        invocation_id = self.context.session_id
        trajectory = self.directory / "trajectory.json"
        converter = ResponsesConverter(return_token_id_information=True)

        def persist(native_messages):
            messages[:] = native_messages
            accepted_calls = {call["id"] for message in messages for call in message.get("tool_calls") or []}
            rejected_items = {
                id(item)
                for response in responses
                if any(item.type == "function_call" and item.call_id not in accepted_calls for item in response.output)
                for item in response.output
            }
            output[:] = [item for item in output if id(item) not in rejected_items]
            for message in messages:
                if message["role"] == "tool" and message["tool_call_id"] not in recorded_tools:
                    output.append(
                        NeMoGymFunctionCallOutput(
                            type="function_call_output",
                            call_id=message["tool_call_id"],
                            output=message["content"],
                        )
                    )
                    recorded_tools.add(message["tool_call_id"])
            trajectory.write_text(json.dumps({"messages": messages, "harness_revision": HERMES_REVISION}, indent=2))

        async def query(kwargs):
            persist(kwargs["messages"])
            params = self.params.model_dump(exclude_none=True)
            params["input"] = [
                item.model_dump(exclude_none=True)
                for item in converter.chat_completions_messages_to_responses_items(kwargs["messages"])
            ]
            params["tools"] = [{"type": "function", **TERMINAL_TOOL["function"], "strict": False}]
            turn_started = time()
            response = await self.query(params)
            responses.append(response)
            turns.append(
                TrajectoryTurn(
                    invocation_id=invocation_id,
                    task_id="unscoped",
                    rollout_id="unscoped",
                    turn_no=len(responses),
                    timestamp=turn_started,
                    question=params["input"],
                    answer=[item for item in response.output if item.type != "reasoning"],
                    reasoning_content=[item for item in response.output if item.type == "reasoning"] or None,
                    step_count=len(tool_observations),
                    model_calls=[
                        ModelCallRef(
                            model_ref={"type": "responses_api_models", "name": self.model_name},
                            response_id=response.id,
                        )
                    ],
                )
            )
            output.extend(response.output)
            if response.error:
                raise RuntimeError(response.error.message)
            content, calls = [], []
            for item in response.output:
                if item.type == "reasoning":
                    content.append("<think>" + "\n".join(part.text for part in item.summary) + "</think>")
                elif item.type == "message":
                    content.extend(part.text for part in item.content if part.type == "output_text")
                elif item.type == "function_call":
                    calls.append(
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                    )
            usage = response.usage
            return ChatCompletion.model_validate(
                {
                    "id": response.id,
                    "created": response.created_at,
                    "model": self.model_name,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length"
                            if response.status == "incomplete"
                            else "tool_calls"
                            if calls
                            else "stop",
                            "message": {
                                "role": "assistant",
                                "content": "\n".join(content),
                                "tool_calls": calls or None,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": usage.input_tokens,
                        "completion_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }
                    if usage
                    else None,
                }
            )

        async def command(call):
            try:
                args = json.loads(call.function.arguments)
                if call.function.name != "terminal" or not isinstance(args, dict) or set(args) != {"command"}:
                    raise ValueError("Use terminal with a single command argument")
                if not isinstance(args["command"], str):
                    raise ValueError("command must be a string")
            except (ValueError, TypeError) as exc:
                return json.dumps({"error": str(exc)})
            pidfile = quote(f"/tmp/{self.context.session_id}.pids")
            result = await self.sandbox.exec(
                "setsid --wait bash -c " + quote(f"echo $$ >> {pidfile}; " + args["command"]),
                user=self.context.user,
                cwd=self.context.workdir,
                timeout_s=min(budget, self.config.step_timeout_sec),
            )
            if result.error_type and result.error_type != "timeout":
                raise RuntimeError(f"Sandbox execution failed: {result.error_type}")
            return json.dumps(
                {
                    "output": (result.stdout or "") + (result.stderr or ""),
                    "exit_code": result.return_code,
                    "error": result.error_type,
                }
            )

        class SandboxedAgent(AIAgent):
            def _handle_max_iterations(self, messages, api_call_count):
                # Native Hermes makes an extra direct model call here, beyond the turn budget.
                return None

            def _interruptible_api_call(self, api_kwargs):
                return bridge.call(lambda: query(api_kwargs))

            def _execute_tool_calls(self, assistant_message, native_messages, effective_task_id, api_call_count=0):
                nonlocal tool_error
                for call in assistant_message.tool_calls:
                    started_at = time()
                    try:
                        if tool_error:
                            raise RuntimeError("Tool not executed because the episode stopped")
                        result = bridge.call(lambda: command(call))
                    except Exception as exc:
                        tool_error = f"{type(exc).__name__}: {exc}"
                        self.interrupt(tool_error)
                        result = json.dumps({"error": tool_error})
                    native_messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                    completed_at = time()
                    tool_observations.append(
                        TrajectoryToolCall(
                            invocation_id=invocation_id,
                            tool_call_id=call.id,
                            tool_name=call.function.name,
                            output=result,
                            started_at=started_at,
                            completed_at=completed_at,
                            duration_ms=max(0, (completed_at - started_at) * 1000),
                            timing_source="executor",
                            status="failed" if json.loads(result).get("error") else "completed",
                        )
                    )
                    turns[-1].step_count = len(tool_observations)
                    persist(native_messages)

        agent = SandboxedAgent(
            base_url="http://gym.invalid/v1",
            api_key="dummy-key",
            model=self.model_name,
            api_mode="chat_completions",
            use_streaming=False,
            insert_reasoning=False,
            max_iterations=self.config.max_turns,
            max_tokens=self.params.max_output_tokens,
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            persist_session=False,
            save_trajectories=False,
            checkpoints_enabled=False,
        )
        agent.tools = [TERMINAL_TOOL]
        agent.valid_tool_names = {"terminal"}
        agent.compression_enabled = False
        instruction = self.context.instruction
        if self.context.skills_dir:
            instruction += f"\nTask skills are in {self.context.skills_dir}. Read the relevant SKILL.md files."
        worker = asyncio.create_task(asyncio.to_thread(agent.run_conversation, instruction))
        outcome = HarnessOutcome(reason="completed")
        result = {}
        try:
            result = await asyncio.wait_for(asyncio.shield(worker), budget)
            if tool_error:
                outcome = HarnessOutcome(reason="infrastructure_error", detail=tool_error)
            elif result.get("partial"):
                outcome = HarnessOutcome(
                    reason="nonzero_exit", detail=result.get("error") or "Model output was truncated"
                )
            elif result.get("error") or result.get("failed"):
                outcome = HarnessOutcome(reason="infrastructure_error", detail=str(result.get("error")))
            elif not result.get("completed"):
                outcome = HarnessOutcome(
                    reason="nonzero_exit", detail="Hermes did not complete within its turn budget"
                )
        except asyncio.CancelledError:
            outcome = HarnessOutcome(reason="cancelled")
        except TimeoutError:
            outcome = HarnessOutcome(reason="timeout")
        except Exception as exc:
            outcome = HarnessOutcome(reason="infrastructure_error", detail=f"{type(exc).__name__}: {exc}")
        finally:
            agent.interrupt("Episode closed")
            await bridge.close()
            await asyncio.gather(worker, return_exceptions=True)
            agent.client.close()
        persist(result.get("messages") or messages)
        outcome.artifacts = [str(trajectory)]
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.model_name,
            object="response",
            status="completed" if outcome.reason == "completed" else "incomplete",
            output=output,
            tool_choice=self.params.tool_choice,
            tools=self.params.tools,
            parallel_tool_calls=self.params.parallel_tool_calls,
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in responses if r.usage]),
        )
        trajectory_record = TrajectoryRecord(
            task_id="unscoped",
            rollout_id="unscoped",
            turns=turns,
            tool_calls=tool_observations,
            invocations=[
                AgentInvocation(
                    invocation_id=invocation_id,
                    status="completed"
                    if outcome.reason == "completed"
                    else "failed"
                    if outcome.reason == "infrastructure_error"
                    else "incomplete",
                    conversation=[
                        *(turns[0].question if turns else [{"role": "user", "content": instruction}]),
                        *output,
                    ],
                    model_calls=[
                        ModelCallRef(
                            model_ref={"type": "responses_api_models", "name": self.model_name},
                            response_id=r.id,
                        )
                        for r in responses
                    ],
                )
            ],
        )
        return (
            response,
            outcome,
            {
                "hermes_trajectory": result,
                "harness_revision": HERMES_REVISION,
                "ng_trajectory": trajectory_record.model_dump(mode="json"),
            },
        )
