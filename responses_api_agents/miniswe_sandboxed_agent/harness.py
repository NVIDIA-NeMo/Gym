# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE execution on a caller-owned sandbox with an injected model callback."""

import asyncio
import json
from pathlib import Path
from shlex import quote
from time import monotonic, time
from typing import Any
from uuid import uuid4

import yaml
from minisweagent import __version__ as mini_swe_version
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from pydantic import BaseModel, Field, TypeAdapter

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionMessageToolCall,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseInputItem,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    TrajectoryTurn,
)
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.harness import HarnessContext, HarnessOutcome, WorkerBridge


MINI_CONFIG = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text())


def responses_input(messages):
    """Replay native Responses items and associate observations with their calls."""
    items = []
    for message in messages:
        if message["role"] == "tool":
            items.append(
                {"type": "function_call_output", "call_id": message["tool_call_id"], "output": message["content"]}
            )
        elif "response_output" in message.get("extra", {}):
            items.extend(message["extra"]["response_output"])
        else:
            items.append({"role": message["role"], "content": message.get("content", "")})
    return items


class MiniSWEConfig(BaseModel):
    step_limit: int = Field(default=0, ge=0)
    step_timeout_sec: int = Field(default=600, gt=0)


class GymModel:
    def __init__(self, bridge, query):
        self.bridge, self._query = bridge, query

    def query(self, messages):
        return self.bridge.call(lambda: self._query(messages))

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        messages = format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            # Intentionally retain full observations instead of mini.yaml's head/tail truncation.
            observation_template="<returncode>{{output.returncode}}</returncode>\n{{output.output}}",
        )
        for observation, output in zip(messages, outputs):
            if output.get("images"):
                observation["content"] = [{"type": "input_text", "text": observation["content"]}] + [
                    {"type": "input_image", "image_url": uri} for uri in output["images"]
                ]
        return messages

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {"info": {"model_transport": "nemo_gym_responses"}}


class SandboxEnvironment:
    def __init__(self, bridge, execute, system_info):
        self.bridge, self._execute = bridge, execute
        self.system_info = system_info

    def execute(self, action):
        output = self.bridge.call(lambda: self._execute(action))
        LocalEnvironment._check_finished(self, output)
        return output

    def get_template_vars(self):
        return self.system_info

    def serialize(self):
        return {"info": {"environment_type": "gym_sandbox"}}


class MiniSWEHarness:
    """Execute only: the caller provisions, grades, and destroys the sandbox."""

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox,
        context: HarnessContext,
        config: MiniSWEConfig,
        params,
        query,
        model_name: str,
        directory: Path,
        observability_enabled: bool = False,
    ):
        self.sandbox = sandbox
        self.context = context
        self.config = config
        self.params = params
        self.query = query
        self.model_name = model_name
        self.directory = directory
        self.observability_enabled = observability_enabled
        self.extra_instruction = ""
        self.system_info = {}
        self.result = None

    async def setup(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        result = await self.sandbox.exec("command -v setsid", user=self.context.user, cwd=self.context.workdir)
        if result.return_code:
            raise RuntimeError("mini-SWE requires setsid for process cleanup")
        result = await self.sandbox.exec(
            "uname -s; uname -r; uname -v; uname -m", user=self.context.user, cwd=self.context.workdir
        )
        if result.return_code or len(result.stdout.splitlines()) != 4:
            raise RuntimeError("Could not read task environment system information")
        self.system_info.update(
            zip(("system", "release", "version", "machine"), result.stdout.splitlines(), strict=True)
        )
        if self.context.skills_dir:
            self.extra_instruction += (
                f"\nTask skills are in {self.context.skills_dir}. Read the relevant SKILL.md files.\n"
            )
        if self.context.mcp_servers:
            (self.directory / "mcp.json").write_text(json.dumps(self.context.mcp_servers))
            remote = f"/tmp/{self.context.session_id}-mcp"
            command = f"python3 -m venv {remote} && {remote}/bin/pip -q install mcp==1.29.0 httpx-aiohttp==0.2.0"
            result = await self.sandbox.exec(
                command, user=self.context.user, cwd=self.context.workdir, timeout_s=self.context.setup_timeout_sec
            )
            if result.return_code:
                raise RuntimeError(f"Task MCP client setup failed: {result.stderr}")
            await self.sandbox.upload(Path(__file__).with_name("mcp_client.py"), remote + "/client.py")
            await self.sandbox.upload(self.directory / "mcp.json", remote + "/servers.json")
            cli = f"{remote}/bin/python {remote}/client.py"
            daemon = f"echo $$ >> /tmp/{self.context.session_id}.pids; exec {cli} serve"
            started = await self.sandbox.exec(
                "bash -c "
                + quote(
                    f"setsid --fork bash -c {quote(daemon)} > {remote}/server.log 2>&1 < /dev/null; "
                    f"for i in $(seq 1 60); do [ -S {remote}/server.sock ] && exit 0; sleep 1; done; "
                    f"cat {remote}/server.log; exit 1"
                ),
                user=self.context.user,
                cwd=self.context.workdir,
                timeout_s=65,
            )
            if started.return_code:
                raise RuntimeError(f"Task MCP session setup failed: {started.stdout}")
            listed = await self.sandbox.exec(
                cli + " list", user=self.context.user, cwd=self.context.workdir, timeout_s=60
            )
            if listed.return_code:
                raise RuntimeError(f"Task MCP discovery failed: {listed.stderr}")
            self.extra_instruction += (
                f"\nTask MCP tools (JSON schemas): {listed.stdout}\n"
                f"Call with: {cli} call SERVER TOOL 'JSON_ARGUMENTS'.\n"
            )

    async def execute(self, budget):
        bridge = WorkerBridge()
        responses = []
        output_items = []
        invocation = observations = trajectory = conversation_adapter = None
        if self.observability_enabled:
            invocation = AgentInvocation(invocation_id=self.context.session_id)
            observations = AgentObservationBundle(source="miniswe", records=[invocation])
            trajectory = TrajectoryRecord(
                task_id=self.context.task_id or self.context.session_id,
                rollout_id=self.context.rollout_id or self.context.session_id,
            )
            conversation_adapter = TypeAdapter(list[NeMoGymResponseInputItem])
            execution_started = monotonic()
        step_count = 0

        async def query(messages):
            params = self.params.model_dump(exclude_none=True)
            params["input"] = responses_input(messages)
            # mini-SWE executes bash calls; task MCP tools are discovered in setup
            # and made available through the task-local CLI described in the prompt.
            params["tools"] = [{"type": "function", **BASH_TOOL["function"], "strict": False}]
            if self.observability_enabled:
                invocation.conversation = conversation_adapter.validate_python(params["input"])
            response = await self.query(params)
            responses.append(response)
            output_items.extend(response.output)
            if self.observability_enabled:
                invocation.conversation.extend(response.output)
                turn_model_calls = []
                if response.id:
                    reference = ModelCallRef(
                        model_ref=ModelServerRef(type="responses_api_models", name=self.model_name),
                        response_id=response.id,
                    )
                    invocation.model_calls.append(reference)
                    turn_model_calls.append(reference)
                else:
                    trajectory.gaps.append(
                        ObservationGap(
                            code="model_call_reference_unavailable",
                            invocation_id=invocation.invocation_id,
                            detail=f"turn:{len(trajectory.turns) + 1}",
                        )
                    )
                # Record the decision before parsing: rejected model output is still
                # an observed decision, and must not disappear during recovery.
                trajectory.turns.append(
                    TrajectoryTurn(
                        invocation_id=invocation.invocation_id,
                        task_id=trajectory.task_id,
                        rollout_id=trajectory.rollout_id,
                        turn_no=len(trajectory.turns) + 1,
                        timestamp=time(),
                        question=params["input"],
                        answer=[item.model_dump(mode="json") for item in response.output],
                        reasoning_content=[
                            item.model_dump(mode="json") for item in response.output if item.type == "reasoning"
                        ]
                        or None,
                        step_count=step_count,
                        model_calls=turn_model_calls,
                    )
                )
            content = "\n".join(
                part.text
                for item in response.output
                if item.type == "message"
                for part in item.content
                if part.type == "output_text"
            )
            calls = [
                NeMoGymChatCompletionMessageToolCall(
                    id=item.call_id, type="function", function={"name": item.name, "arguments": item.arguments}
                )
                for item in response.output
                if item.type == "function_call"
            ]
            actions = parse_toolcall_actions(
                calls,
                format_error_template=MINI_CONFIG["model"]["format_error_template"],
            )
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": [call.model_dump() for call in calls],
                "extra": {
                    "actions": actions,
                    "response_output": [item.model_dump(exclude_none=True) for item in response.output],
                },
            }

        async def command(action: dict[str, str]) -> dict[str, Any]:
            nonlocal step_count
            observation = None
            if self.observability_enabled:
                observation = ToolCallObservation(
                    invocation_id=invocation.invocation_id,
                    tool_call_id=action["tool_call_id"],
                    tool_name="bash",
                    started_at=time(),
                    timing_source="executor",
                    status="incomplete",
                )
                observations.records.append(observation)
                started = monotonic()
            try:
                result = await self.sandbox.exec(
                    "setsid --wait bash -c "
                    + quote(f"echo $$ >> /tmp/{self.context.session_id}.pids; " + action["command"]),
                    user=self.context.user,
                    cwd=self.context.workdir,
                    env=MINI_CONFIG["environment"]["env"],
                    timeout_s=min(budget, self.config.step_timeout_sec),
                )
                if observation is not None:
                    observation.status = (
                        "timeout"
                        if result.error_type == "timeout"
                        else "failed"
                        if result.error_type or result.return_code != 0
                        else "completed"
                    )
                    observation.error_type = result.error_type
            except asyncio.CancelledError:
                if observation is not None:
                    observation.status = "cancelled"
                    observation.error_type = "CancelledError"
                raise
            except Exception as exc:
                if observation is not None:
                    observation.status = "failed"
                    observation.error_type = type(exc).__name__
                raise
            finally:
                if observation is not None:
                    completed_at = time()
                    if completed_at >= observation.started_at:
                        observation.completed_at = completed_at
                    else:
                        observations.gaps.append(
                            ObservationGap(code="tool_clock_moved_backwards", invocation_id=invocation.invocation_id)
                        )
                    observation.duration_ms = (monotonic() - started) * 1000
                    step_count += 1
                    if trajectory.turns:
                        trajectory.turns[-1].step_count = step_count
            if result.error_type and result.error_type != "timeout":
                raise RuntimeError(f"Sandbox execution failed: {result.error_type}")
            output = (result.stdout or "") + (result.stderr or "")
            images = []
            try:
                tool_result = json.loads(output)
                for part in tool_result.get("content", []):
                    if part.get("type") == "image":
                        images.append(f"data:{part['mimeType']};base64,{part.pop('data')}")
                if images:
                    output = json.dumps(tool_result)
            except (ValueError, AttributeError, KeyError, TypeError):
                pass
            outcome = {"output": output, "returncode": result.return_code, "images": images}
            # Observe before LocalEnvironment._check_finished raises Submitted.
            # mini-SWE deliberately omits that final ordinary tool message.
            messages = model.format_observation_messages({"extra": {"actions": [action]}}, [outcome])
            item = NeMoGymFunctionCallOutput.model_validate(responses_input(messages)[0])
            output_items.append(item)
            if self.observability_enabled:
                invocation.conversation.append(item)
            return outcome

        model = GymModel(bridge, query)
        agent = DefaultAgent(
            model,
            SandboxEnvironment(bridge, command, self.system_info),
            system_template=MINI_CONFIG["agent"]["system_template"],
            instance_template=MINI_CONFIG["agent"]["instance_template"],
            step_limit=self.config.step_limit,
            cost_limit=0,
            output_path=self.directory / "trajectory.json",
        )
        worker = asyncio.create_task(asyncio.to_thread(agent.run, self.context.instruction + self.extra_instruction))
        termination = HarnessOutcome(reason="completed")
        try:
            info = await asyncio.wait_for(asyncio.shield(worker), budget)
            if info.get("exit_status") != "Submitted":
                termination = HarnessOutcome(reason="nonzero_exit", detail=info.get("exit_status"))
        except asyncio.CancelledError:
            termination = HarnessOutcome(reason="cancelled")
        except TimeoutError:
            termination = HarnessOutcome(reason="timeout")
        except Exception as exc:
            termination = HarnessOutcome(reason="infrastructure_error", detail=f"{type(exc).__name__}: {exc}")
        finally:
            await bridge.aclose()
            # Cancel pending I/O and join the synchronous loop before verification.
            await asyncio.gather(worker, return_exceptions=True)
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.model_name,
            object="response",
            output=output_items,
            tool_choice=self.params.tool_choice,
            tools=self.params.tools,
            parallel_tool_calls=self.params.parallel_tool_calls,
            # An empty or partial sum is not a measured rollout total.
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in responses])
            if responses and all(r.usage is not None for r in responses)
            else None,
        )
        extra = {"mini_swe_trajectory": agent.serialize(), "harness_version": mini_swe_version}
        if self.observability_enabled:
            invocation.status = (
                "completed"
                if termination.reason == "completed"
                else "failed"
                if termination.reason == "infrastructure_error"
                else "incomplete"
            )
            invocation.duration_ms = (monotonic() - execution_started) * 1000
            extra["ng_agent_observations"] = observations.model_dump(mode="json")
            extra["ng_trajectory"] = trajectory.model_dump(mode="json")
        termination.artifacts = [str(self.directory / "trajectory.json")]
        self.result = (response, termination, extra)

        return self.result
