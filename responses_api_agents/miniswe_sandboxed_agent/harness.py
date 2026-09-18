# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE execution on a caller-owned sandbox with an injected model callback."""

import asyncio
import json
from pathlib import Path
from shlex import quote
from threading import Lock
from time import time
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
from pydantic import BaseModel, Field

from nemo_gym.openai_utils import NeMoGymChatCompletionMessageToolCall, NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.sandbox import AsyncSandbox


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


class HarnessOutcome(BaseModel):
    reason: str
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class HarnessContext(BaseModel):
    session_id: str
    instruction: str
    user: str | int | None = None
    workdir: str | None = None
    setup_timeout_sec: float = Field(default=360, gt=0)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None


class WorkerBridge:
    """Synchronous mini-SWE loop, asynchronous Gym I/O, explicit cancellation."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.pending = set()
        self.lock = Lock()

    def call(self, factory):
        with self.lock:
            if self.closed:
                raise RuntimeError("Episode is closed")
            future = asyncio.run_coroutine_threadsafe(factory(), self.loop)
            self.pending.add(future)
        try:
            return future.result()
        finally:
            with self.lock:
                self.pending.discard(future)

    def close(self):
        with self.lock:
            self.closed = True
            pending = list(self.pending)
        for future in pending:
            future.cancel()


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
        output = self.bridge.call(lambda: self._execute(action["command"]))
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
    ):
        self.sandbox = sandbox
        self.context = context
        self.config = config
        self.params = params
        self.query = query
        self.model_name = model_name
        self.directory = directory
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

        async def query(messages):
            params = self.params.model_dump(exclude_none=True)
            params["input"] = responses_input(messages)
            # mini-SWE executes bash calls; task MCP tools are discovered in setup
            # and made available through the task-local CLI described in the prompt.
            params["tools"] = [{"type": "function", **BASH_TOOL["function"], "strict": False}]
            response = await self.query(params)
            responses.append(response)
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

        async def command(text):
            result = await self.sandbox.exec(
                "setsid --wait bash -c " + quote(f"echo $$ >> /tmp/{self.context.session_id}.pids; " + text),
                user=self.context.user,
                cwd=self.context.workdir,
                env=MINI_CONFIG["environment"]["env"],
                timeout_s=min(budget, self.config.step_timeout_sec),
            )
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
            return {"output": output, "returncode": result.return_code, "images": images}

        agent = DefaultAgent(
            GymModel(bridge, query),
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
            bridge.close()
            # Cancel pending I/O and join the synchronous loop before verification.
            await asyncio.gather(worker, return_exceptions=True)
        response = NeMoGymResponse(
            id="resp_" + uuid4().hex,
            created_at=int(time()),
            model=self.model_name,
            object="response",
            output=[item for part in responses for item in part.output],
            tool_choice=self.params.tool_choice,
            tools=self.params.tools,
            parallel_tool_calls=self.params.parallel_tool_calls,
            usage=NeMoGymResponseUsage.sum_from_list([r.usage for r in responses if r.usage]),
        )
        termination.artifacts = [str(self.directory / "trajectory.json")]
        self.result = (
            response,
            termination,
            {"mini_swe_trajectory": agent.serialize(), "harness_version": mini_swe_version},
        )

        return self.result
