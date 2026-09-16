# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mini-SWE 2.1 DefaultAgent on a borrowed environment, with Gym model routing."""

import asyncio
import json
from pathlib import Path
from shlex import quote
from threading import Lock
from typing import Any

from fastapi import Request
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseUsage
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status
from resources_servers.terminal_bench_4.agent import artifact_directory, empty_response, run_borrowed
from resources_servers.terminal_bench_4.handoff import AgentTermination, SandboxedVerifyResponse


SYSTEM = """You are an assistant operating a task environment. Respond with exactly one bash action in a
```mswea_bash_command code block. Commands run in separate shells; filesystem changes persist.
Complete the task in the environment. To finish, run only: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
After submission you cannot change the environment. Task-specific instructions take precedence."""


class MiniSWESandboxedConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    sandbox_providers: dict[str, Any]
    step_limit: int = Field(default=0, ge=0)
    step_timeout_sec: int = Field(default=600, gt=0)


class MiniSWERunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


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
        messages = format_observation_messages(
            outputs,
            observation_template="<returncode>{{output.returncode}}</returncode>\n{{output.output}}",
        )
        for observation, output in zip(messages, outputs, strict=True):
            if output.get("images"):
                observation["content"] = [{"type": "input_text", "text": observation["content"]}] + [
                    {"type": "input_image", "image_url": uri} for uri in output["images"]
                ]
        return messages

    def get_template_vars(self):
        return {}

    def serialize(self):
        return {"info": {"model_transport": "nemo_gym_responses"}}


class BorrowedEnvironment:
    def __init__(self, bridge, execute):
        self.bridge, self._execute = bridge, execute

    def execute(self, action):
        output = self.bridge.call(lambda: self._execute(action["command"]))
        LocalEnvironment._check_finished(self, output)
        return output

    def get_template_vars(self):
        return {"system": "Linux"}

    def serialize(self):
        return {"info": {"environment_type": "gym_borrowed_sandbox"}}


class MiniSWESandboxedAgent(SimpleResponsesAPIAgent):
    config: MiniSWESandboxedConfig

    async def responses(self, body):
        raise NotImplementedError("This harness requires /run with a resources session")

    async def run(self, request: Request, body: MiniSWERunRequest) -> SandboxedVerifyResponse:
        extra_instruction = ""

        async def setup(sandbox, seed):
            nonlocal extra_instruction
            result = await sandbox.exec("command -v setsid", user=seed.user)
            if result.return_code:
                raise RuntimeError("mini-SWE requires setsid for process cleanup")
            if seed.skills_dir:
                extra_instruction += f"\nTask skills are in {seed.skills_dir}. Read the relevant SKILL.md files.\n"
            if seed.mcp_servers:
                directory = artifact_directory(self.config.name, seed.session_id)
                (directory / "mcp.json").write_text(json.dumps(seed.mcp_servers))
                remote = f"/tmp/{seed.session_id}-mcp"
                command = f"python3 -m venv {remote} && {remote}/bin/pip -q install mcp==1.29.0 httpx-aiohttp==0.2.0"
                result = await sandbox.exec(command, user=seed.user, timeout_s=seed.setup_timeout_sec)
                if result.return_code:
                    raise RuntimeError(f"Task MCP client setup failed: {result.stderr}")
                await sandbox.upload(Path(__file__).with_name("mcp_client.py"), remote + "/client.py")
                await sandbox.upload(directory / "mcp.json", remote + "/servers.json")
                cli = f"{remote}/bin/python {remote}/client.py"
                daemon = f"echo $$ >> /tmp/{seed.session_id}.pids; exec {cli} serve"
                started = await sandbox.exec(
                    "bash -c "
                    + quote(
                        f"setsid --fork bash -c {quote(daemon)} > {remote}/server.log 2>&1 < /dev/null; "
                        f"for i in $(seq 1 60); do [ -S {remote}/server.sock ] && exit 0; sleep 1; done; "
                        f"cat {remote}/server.log; exit 1"
                    ),
                    user=seed.user,
                    timeout_s=65,
                )
                if started.return_code:
                    raise RuntimeError(f"Task MCP session setup failed: {started.stdout}")
                listed = await sandbox.exec(cli + " list", user=seed.user, timeout_s=60)
                if listed.return_code:
                    raise RuntimeError(f"Task MCP discovery failed: {listed.stderr}")
                extra_instruction += (
                    f"\nTask MCP tools (JSON schemas): {listed.stdout}\n"
                    f"Call with: {cli} call SERVER TOOL 'JSON_ARGUMENTS'.\n"
                )

        async def execute(sandbox, seed, budget):
            bridge = WorkerBridge()
            responses = []
            directory = artifact_directory(self.config.name, seed.session_id)

            async def query(messages):
                params = body.responses_create_params.model_dump(exclude_none=True)
                params["input"] = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
                params.pop("tools", None)
                params.pop("tool_choice", None)
                model_response = await self.server_client.post(
                    server_name=self.config.model_server.name,
                    url_path=self.url_path_for_run(url_path="/v1/responses", body=body),
                    json=params,
                    cookies=request.cookies,
                )
                await raise_for_status(model_response)
                response = NeMoGymResponse.model_validate(await get_response_json(model_response))
                responses.append(response)
                content = "\n".join(
                    part.text
                    for item in response.output
                    if item.type == "message"
                    for part in item.content
                    if part.type == "output_text"
                )
                actions = parse_regex_actions(
                    content,
                    action_regex=r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```",
                    format_error_template="Return exactly one bash action in a mswea_bash_command code block.",
                )
                return {"role": "assistant", "content": content, "extra": {"actions": actions}}

            async def command(text):
                result = await sandbox.exec(
                    "setsid --wait bash -c " + quote(f"echo $$ >> /tmp/{seed.session_id}.pids; " + text),
                    user=seed.user,
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
                BorrowedEnvironment(bridge, command),
                system_template=SYSTEM,
                instance_template="{{task}}",
                step_limit=self.config.step_limit,
                cost_limit=0,
                output_path=directory / "trajectory.json",
            )
            worker = asyncio.create_task(asyncio.to_thread(agent.run, seed.instruction + extra_instruction))
            termination = AgentTermination(reason="completed")
            try:
                info = await asyncio.wait_for(asyncio.shield(worker), budget)
                if info.get("exit_status") != "Submitted":
                    termination = AgentTermination(reason="nonzero_exit", detail=info.get("exit_status"))
            except TimeoutError:
                termination = AgentTermination(reason="timeout")
            except Exception as exc:
                termination = AgentTermination(reason="infrastructure_error", detail=f"{type(exc).__name__}: {exc}")
            finally:
                bridge.close()
                # Cancel pending I/O and join the synchronous loop before verification.
                await asyncio.gather(worker, return_exceptions=True)
            response = empty_response(body.responses_create_params, self.config.model_server.name)
            response.output = [item for part in responses for item in part.output]
            response.usage = NeMoGymResponseUsage.sum_from_list([r.usage for r in responses if r.usage])
            termination.artifacts = [str(directory / "trajectory.json")]
            return response, termination, {"mini_swe_trajectory": agent.serialize(), "harness_version": "2.1.0"}

        return SandboxedVerifyResponse.model_validate(
            await run_borrowed(self, request, body, setup=setup, execute=execute)
        )


if __name__ == "__main__":
    MiniSWESandboxedAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = MiniSWESandboxedAgent.run_webserver()
