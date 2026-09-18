# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes execution with injected model I/O and a caller-owned sandbox."""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from shlex import quote
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
from nemo_gym.sandbox.harness import HarnessContext, HarnessOutcome


HERMES_CONFIG = yaml.safe_load((Path(__file__).parent / "configs" / "hermes.yaml").read_text())
HERMES_REVISION = "26bb847a88493342ca1b194e0455b479073ae21d"


class HermesConfig(BaseModel):
    name: Literal["hermes"]
    max_turns: int = Field(default=90, gt=0)
    step_timeout_sec: int = Field(default=600, gt=0)


class TrajectoryRecorder:
    def __init__(self, context, directory):
        self.task_id = context.task_id or context.session_id
        self.rollout_id = context.rollout_id or context.session_id
        self.invocation_id = context.session_id
        self.trajectory = directory / "trajectory.json"
        self.responses = []
        self.output = []
        self.messages = []
        self.recorded_tools = set()
        self.tool_observations = []
        self.turns = []
        self.tool_started = {}

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
        native_results = {
            message["tool_call_id"]: message["content"] for message in self.messages if message["role"] == "tool"
        }
        for item in self.output:
            if item.type == "function_call_output" and item.call_id in native_results:
                item.output = native_results[item.call_id]
        for observation in self.tool_observations:
            if observation.tool_call_id in native_results:
                observation.output = native_results[observation.tool_call_id]
        self.trajectory.write_text(
            json.dumps({"messages": self.messages, "harness_revision": HERMES_REVISION}, indent=2)
        )

    def tool_start(self, call_id, name, args):
        self.tool_started[call_id] = time()

    def tool_complete(self, call_id, name, args, result, failed):
        completed = time()
        started = self.tool_started.pop(call_id)
        self.tool_observations.append(
            TrajectoryToolCall(
                invocation_id=self.invocation_id,
                tool_call_id=call_id,
                tool_name=name,
                output=result,
                started_at=started,
                completed_at=completed,
                duration_ms=max(0, (completed - started) * 1000),
                timing_source="harness",
                status="failed" if failed else "completed",
            )
        )
        self.turns[-1].step_count = len(self.tool_observations)
        self.persist([*self.messages, {"role": "tool", "tool_call_id": call_id, "content": result}])


class GymModel:
    def __init__(self, query, params, model_name, recorder):
        self.callback = query
        self.params = params
        self.model_name = model_name
        self.recorder = recorder
        self.converter = ResponsesConverter(return_token_id_information=True)

    async def query(self, kwargs):
        self.recorder.persist(kwargs["messages"])
        params = self.params.model_dump(exclude_none=True)
        params["input"] = [
            item.model_dump(exclude_none=True)
            for item in self.converter.chat_completions_messages_to_responses_items(kwargs["messages"])
        ]
        params["tools"] = [
            {"type": "function", **tool["function"], "strict": False} for tool in kwargs.get("tools", [])
        ]
        turn_started = time()
        response = await self.callback(params)
        self.recorder.responses.append(response)
        self.recorder.turns.append(
            TrajectoryTurn(
                invocation_id=self.recorder.invocation_id,
                task_id=self.recorder.task_id,
                rollout_id=self.recorder.rollout_id,
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
    def __init__(self, sandbox, context):
        self.sandbox = sandbox
        self.context = context
        self.error = None

    async def execute(self, command, cwd, timeout, stdin_data):
        if stdin_data is not None:
            encoded = base64.b64encode(stdin_data.encode()).decode()
            command = f"printf %s {quote(encoded)} | base64 -d | bash -c {quote(command)}"
        pidfile = quote(f"/tmp/{self.context.session_id}.pids")
        try:
            result = await self.sandbox.exec(
                "setsid --wait bash -c " + quote(f"echo $$ >> {pidfile}; " + command),
                user=self.context.user,
                cwd=cwd,
                timeout_s=timeout,
            )
            if result.error_type and result.error_type != "timeout":
                raise RuntimeError(f"Sandbox execution failed: {result.error_type}")
            return {
                "output": (result.stdout or "") + (result.stderr or ""),
                "returncode": 124 if result.error_type == "timeout" else result.return_code,
            }
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return {"output": self.error, "returncode": 1}


class HermesHarness:
    """Execute only; provisioning, verification and cleanup belong to the caller."""

    def __init__(
        self,
        *,
        sandbox,
        context: HarnessContext,
        config,
        params,
        query,
        model_name,
        directory: Path,
        observability_enabled: bool = False,
    ):
        self.sandbox = sandbox
        self.context = context
        self.observability_enabled = observability_enabled
        self.config = config
        self.params = params
        self.query = query
        self.model_name = model_name
        self.directory = directory

    async def setup(self):
        if self.context.mcp_servers:
            raise ValueError("The Hermes terminal profile does not support task MCP servers")
        self.directory.mkdir(parents=True, exist_ok=True)
        result = await self.sandbox.exec("command -v setsid", user=self.context.user, cwd=self.context.workdir)
        if result.return_code:
            raise RuntimeError("Hermes requires setsid for process cleanup")

    async def execute(self, budget):
        recorder = TrajectoryRecorder(self.context, self.directory)
        model = GymModel(self.query, self.params, self.model_name, recorder)
        environment = SandboxEnvironment(self.sandbox, self.context)
        home = self.directory / "home"
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump(HERMES_CONFIG["runtime"]))
        instruction = self.context.instruction
        if self.context.skills_dir:
            instruction += f"\nTask skills are in {self.context.skills_dir}. Read the relevant SKILL.md files."
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "HERMES_HOME": str(home),
            "TERMINAL_ENV": "gym",
            "TERMINAL_LIFETIME_SECONDS": str(int(budget) + 60),
            "PYTHONUNBUFFERED": "1",
        }
        payload = {
            "context": self.context.model_dump(),
            "instruction": instruction,
            "agent_config": HERMES_CONFIG["agent"],
            "toolsets": HERMES_CONFIG["toolsets"],
            "max_turns": self.config.max_turns,
            "max_tokens": self.params.max_output_tokens,
            "step_timeout": min(budget, self.config.step_timeout_sec),
            "model_name": self.model_name,
            "trajectory": str(recorder.trajectory),
        }
        outcome = HarnessOutcome(reason="completed")
        result = {}
        with (self.directory / "worker.log").open("w") as log:
            worker = await asyncio.create_subprocess_exec(
                sys.executable,
                str(Path(__file__).with_name("worker.py")),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                env=env,
                limit=64 * 1024 * 1024,
            )
            pending = []

            async def dispatch(message):
                nonlocal result
                request_id = message.pop("id")
                operation = message.pop("operation")
                try:
                    if operation == "model":
                        response = await model.query(message["kwargs"])
                        value = response.model_dump(mode="json")
                    elif operation == "sandbox":
                        value = await environment.execute(**message)
                    elif operation == "tool_start":
                        value = recorder.tool_start(**message)
                    elif operation == "tool_complete":
                        value = recorder.tool_complete(**message)
                    elif operation == "messages":
                        recorder.persist(message["messages"])
                        value = environment.error
                    elif operation == "result":
                        result = message["result"]
                        value = None
                    else:
                        raise RuntimeError(f"Unknown Hermes worker operation: {operation}")
                    reply = {"id": request_id, "result": value}
                except Exception as exc:
                    reply = {"id": request_id, "error": f"{type(exc).__name__}: {exc}"}
                worker.stdin.write((json.dumps(reply) + "\n").encode())
                await worker.stdin.drain()

            try:
                async with asyncio.timeout(budget):
                    worker.stdin.write((json.dumps(payload) + "\n").encode())
                    await worker.stdin.drain()
                    while line := await worker.stdout.readline():
                        task = asyncio.create_task(dispatch(json.loads(line)))
                        pending.append(task)
                    code = await worker.wait()
                    if code:
                        raise RuntimeError(f"Hermes worker exited with {code}; see worker.log")
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
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if worker.returncode is None:
                    worker.kill()
                await worker.wait()
        # On interruption the latest model request contains native tool observations.
        if recorder.trajectory.exists():
            result.setdefault("messages", json.loads(recorder.trajectory.read_text())["messages"])
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
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in recorder.responses])
            if recorder.responses and all(r.usage is not None for r in recorder.responses)
            else None,
        )
        trajectory_record = TrajectoryRecord(
            task_id=recorder.task_id,
            rollout_id=recorder.rollout_id,
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
                **({"ng_trajectory": trajectory_record.model_dump(mode="json")} if self.observability_enabled else {}),
            },
        )
