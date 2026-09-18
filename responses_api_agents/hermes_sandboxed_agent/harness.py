# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes execution with injected model I/O and a caller-owned sandbox."""

import asyncio
import json
import os
from functools import cache
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from time import time
from typing import Literal
from uuid import uuid4

import yaml
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
from nemo_gym.sandbox.harness import HarnessContext, HarnessOutcome, WorkerBridge


HERMES_CONFIG = yaml.safe_load((Path(__file__).parent / "configs" / "hermes.yaml").read_text())
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
    Path(directory.name, "config.yaml").write_text(yaml.safe_dump(HERMES_CONFIG["runtime"]))
    return directory


class HermesConfig(BaseModel):
    name: Literal["hermes"]
    max_turns: int = Field(default=90, gt=0)
    step_timeout_sec: int = Field(default=600, gt=0)


class TrajectoryRecorder:
    def __init__(self, context, directory):
        self.invocation_id = context.session_id
        self.trajectory = directory / "trajectory.json"
        self.responses = []
        self.output = []
        self.messages = []
        self.recorded_tools = set()
        self.tool_observations = []
        self.turns = []

    def persist(self, native_messages):
        self.messages[:] = native_messages
        accepted_calls = {call["id"] for message in self.messages for call in message.get("tool_calls") or []}
        rejected_items = {
            id(item)
            for response in self.responses
            if any(item.type == "function_call" and item.call_id not in accepted_calls for item in response.output)
            for item in response.output
        }
        self.output[:] = [item for item in self.output if id(item) not in rejected_items]
        for message in self.messages:
            if message["role"] == "tool" and message["tool_call_id"] not in self.recorded_tools:
                self.output.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=message["tool_call_id"],
                        output=message["content"],
                    )
                )
                self.recorded_tools.add(message["tool_call_id"])
        self.trajectory.write_text(
            json.dumps({"messages": self.messages, "harness_revision": HERMES_REVISION}, indent=2)
        )


class GymModel:
    def __init__(self, bridge, query, params, model_name, recorder):
        self.bridge = bridge
        self.callback = query
        self.params = params
        self.model_name = model_name
        self.recorder = recorder
        self.converter = ResponsesConverter(return_token_id_information=True)

    def query(self, kwargs):
        return self.bridge.call(lambda: self._query(kwargs))

    async def _query(self, kwargs):
        self.recorder.persist(kwargs["messages"])
        params = self.params.model_dump(exclude_none=True)
        params["input"] = [
            item.model_dump(exclude_none=True)
            for item in self.converter.chat_completions_messages_to_responses_items(kwargs["messages"])
        ]
        params["tools"] = [{"type": "function", **TERMINAL_TOOL["function"], "strict": False}]
        turn_started = time()
        response = await self.callback(params)
        self.recorder.responses.append(response)
        self.recorder.turns.append(
            TrajectoryTurn(
                invocation_id=self.recorder.invocation_id,
                task_id="unscoped",
                rollout_id="unscoped",
                turn_no=len(self.recorder.responses),
                timestamp=turn_started,
                question=params["input"],
                answer=[item for item in response.output if item.type != "reasoning"],
                reasoning_content=[item for item in response.output if item.type == "reasoning"] or None,
                step_count=len(self.recorder.tool_observations),
                model_calls=[
                    ModelCallRef(
                        model_ref={"type": "responses_api_models", "name": self.model_name},
                        response_id=response.id,
                    )
                ],
            )
        )
        self.recorder.output.extend(response.output)
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


class SandboxEnvironment:
    def __init__(self, bridge, sandbox, context, timeout, recorder):
        self.bridge = bridge
        self.sandbox = sandbox
        self.context = context
        self.timeout = timeout
        self.recorder = recorder
        self.error = None

    async def _execute(self, call):
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
            timeout_s=self.timeout,
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

    def execute(self, assistant_message, native_messages):
        for call in assistant_message.tool_calls:
            started_at = time()
            try:
                if self.error:
                    raise RuntimeError("Tool not executed because the episode stopped")
                result = self.bridge.call(lambda: self._execute(call))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                result = json.dumps({"error": self.error})
            native_messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            completed_at = time()
            self.recorder.tool_observations.append(
                TrajectoryToolCall(
                    invocation_id=self.recorder.invocation_id,
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
            self.recorder.turns[-1].step_count = len(self.recorder.tool_observations)
            self.recorder.persist(native_messages)


class HermesHarness:
    """Execute only; provisioning, verification and cleanup belong to the caller."""

    def __init__(self, *, sandbox, context: HarnessContext, config, params, query, model_name, directory: Path):
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
        recorder = TrajectoryRecorder(self.context, self.directory)
        model = GymModel(bridge, self.query, self.params, self.model_name, recorder)
        environment = SandboxEnvironment(
            bridge, self.sandbox, self.context, min(budget, self.config.step_timeout_sec), recorder
        )

        class SandboxedAgent(AIAgent):
            def _handle_max_iterations(self, messages, api_call_count):
                # Native Hermes makes an extra direct model call here, beyond the turn budget.
                return None

            def _interruptible_api_call(self, api_kwargs):
                return model.query(api_kwargs)

            def _execute_tool_calls(self, assistant_message, native_messages, effective_task_id, api_call_count=0):
                environment.execute(assistant_message, native_messages)
                if environment.error:
                    self.interrupt(environment.error)

        agent = SandboxedAgent(
            **HERMES_CONFIG["agent"],
            base_url="http://gym.invalid/v1",
            api_key="dummy-key",
            model=self.model_name,
            max_iterations=self.config.max_turns,
            max_tokens=self.params.max_output_tokens,
        )
        agent.tools = [TERMINAL_TOOL]
        agent.valid_tool_names = {"terminal"}
        instruction = self.context.instruction
        if self.context.skills_dir:
            instruction += f"\nTask skills are in {self.context.skills_dir}. Read the relevant SKILL.md files."
        worker = asyncio.create_task(asyncio.to_thread(agent.run_conversation, instruction))
        outcome = HarnessOutcome(reason="completed")
        result = {}
        try:
            result = await asyncio.wait_for(asyncio.shield(worker), budget)
            if environment.error:
                outcome = HarnessOutcome(reason="infrastructure_error", detail=environment.error)
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
            await bridge.aclose()
            await asyncio.gather(worker, return_exceptions=True)
            agent.client.close()
        recorder.persist(result.get("messages") or recorder.messages)
        outcome.artifacts = [str(recorder.trajectory)]
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.model_name,
            object="response",
            status="completed" if outcome.reason == "completed" else "incomplete",
            output=recorder.output,
            tool_choice=self.params.tool_choice,
            tools=self.params.tools,
            parallel_tool_calls=self.params.parallel_tool_calls,
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in recorder.responses if r.usage]),
        )
        trajectory_record = TrajectoryRecord(
            task_id="unscoped",
            rollout_id="unscoped",
            turns=recorder.turns,
            tool_calls=recorder.tool_observations,
            invocations=[
                AgentInvocation(
                    invocation_id=recorder.invocation_id,
                    status="completed"
                    if outcome.reason == "completed"
                    else "failed"
                    if outcome.reason == "infrastructure_error"
                    else "incomplete",
                    conversation=[
                        *(
                            recorder.turns[0].question
                            if recorder.turns
                            else [{"role": "user", "content": instruction}]
                        ),
                        *recorder.output,
                    ],
                    model_calls=[
                        ModelCallRef(
                            model_ref={"type": "responses_api_models", "name": self.model_name},
                            response_id=r.id,
                        )
                        for r in recorder.responses
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
