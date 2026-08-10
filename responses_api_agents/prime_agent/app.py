# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import json
import logging
import os
import shlex
import shutil
from asyncio import Semaphore
from pathlib import Path
from time import time
from typing import Any, Optional
from uuid import uuid4

from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.prime_agent.setup_prime_agent import ensure_prime_agent


LOG = logging.getLogger(__name__)


def parse_prime_agent_events(stdout: str) -> tuple[list[Any], dict[str, int]]:
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            continue

        if role == "assistant":
            usage = message.get("usage") or {}
            input_tokens += int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
            output_tokens += int(usage.get("output") or 0)
            cached_tokens += int(usage.get("cacheRead") or 0)
            if message.get("stopReason") in {"error", "aborted"}:
                LOG.warning("Prime Agent stopped with an error: %s", message.get("errorMessage") or "unknown error")
                return [], {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": cached_tokens,
                }
            texts = [
                block["text"] for block in content if isinstance(block, dict) and (block.get("text") or "").strip()
            ]
            if texts:
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text="\n".join(texts), annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                args = block.get("arguments")
                arguments = json.dumps(args) if isinstance(args, (dict, list)) else str(args or "")
                call_id = block.get("id") or f"call-{uuid4().hex[:8]}"
                output_items.append(
                    NeMoGymResponseFunctionToolCall(
                        arguments=arguments,
                        call_id=call_id,
                        name=block.get("name", ""),
                        type="function_call",
                        id=call_id,
                        status="completed",
                    )
                )

        elif role == "toolResult":
            call_id = message.get("toolCallId", "")
            result_text = "".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
            output_items.append(
                NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=result_text,
                    status="incomplete" if message.get("isError") else "completed",
                )
            )

    return output_items, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or "")
    return str(getattr(content, "text", ""))


def _extract_instruction(body_input) -> tuple[str, Optional[str]]:
    user_message = ""
    system_messages = []
    for item in body_input:
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        text = _content_text(content)
        if role in {"system", "developer"} and text:
            system_messages.append(text)
        elif role == "user" and text:
            user_message = text

    return user_message, "\n\n".join(system_messages) or None


class PrimeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "prime-agent"
    model: str = "policy/model"
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/prime_agent/workspaces"
    kernel_venv: Optional[str] = "outputs/prime_agent/kernel-venv"
    thinking: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = Field(default_factory=list)
    models_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    prime_agent_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class PrimeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PrimeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class PrimeAgent(SimpleResponsesAPIAgent):
    config: PrimeAgentConfig
    sem: Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if command == "prime-agent" and shutil.which(command) is None:
            ensure_prime_agent(self.config.prime_agent_version)
        if not command or shutil.which(command) is None:
            LOG.warning("Prime Agent command %r is not on PATH", self.config.command)

    def _workspace_root(self) -> Path:
        root = Path(self.config.workspace_root).expanduser() / f"prime_agent_{uuid4().hex[:8]}"
        if not root.is_absolute():
            root = Path.cwd() / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _kernel_venv(self) -> Optional[Path]:
        if self.config.kernel_venv is None:
            return None
        path = Path(self.config.kernel_venv).expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    def _env(self, home: Path) -> dict[str, str]:
        agent_dir = home / ".prime" / "agent"
        env = {
            **os.environ,
            "HOME": str(home),
            "PRIME_AGENT_CODING_AGENT_DIR": str(agent_dir),
            "PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND": "1",
            "PI_SKIP_VERSION_CHECK": "1",
        }
        kernel_venv = self._kernel_venv()
        if kernel_venv is not None:
            env["PRIME_AGENT_KERNEL_VENV"] = str(kernel_venv)
        env.update({key: value for key, value in self.config.env.items() if value})
        return env

    def _effective_model(self) -> str:
        return f"nemo/{self.config.model}" if self.config.model_server else self.config.model

    def _build_models_config(self, rollout_id: Optional[str] = None) -> dict[str, Any]:
        config = copy.deepcopy(self.config.models_config)
        if self.config.model_server is None:
            return config
        providers = config.setdefault("providers", {})
        providers["nemo"] = {
            "baseUrl": self.resolve_model_base_url(self.config.model_server.name, rollout_id),
            "api": "openai-completions",
            "apiKey": "EMPTY",  # pragma: allowlist secret
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [
                {
                    "id": self.config.model,
                    "reasoning": True,
                    "input": ["text"],
                    "contextWindow": self.config.context_window,
                    "maxTokens": self.config.max_output_tokens,
                }
            ],
        }
        return config

    def _build_command(
        self,
        instruction: str,
        system_prompt: Optional[str],
        daemon_socket: Optional[Path] = None,
    ) -> list[str]:
        effective_model = self._effective_model()
        provider, separator, model_id = effective_model.partition("/")
        cmd = [*self.config.command_parts, "--print", "--mode", "json", "--no-session"]
        if daemon_socket is not None:
            cmd += ["--daemon-socket", str(daemon_socket)]
        if separator:
            cmd += ["--provider", provider, "--model", model_id]
        else:
            cmd += ["--model", effective_model]
        if self.config.thinking:
            cmd += ["--thinking", self.config.thinking]
        if system_prompt:
            cmd += ["--append-system-prompt", system_prompt]
        cmd += self.config.extra_args
        cmd.append(instruction)
        return cmd

    async def _run_prime_agent(
        self, instruction: str, system_prompt: Optional[str], rollout_id: Optional[str]
    ) -> tuple[list[Any], dict[str, int], str]:
        work_dir = self._workspace_root()
        home = work_dir / ".prime-home"
        agent_dir = home / ".prime" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models_config = self._build_models_config(rollout_id)
        if models_config:
            (agent_dir / "models.json").write_text(json.dumps(models_config, indent=2))
        env = self._env(home)
        cmd = self._build_command(instruction, system_prompt, work_dir / ".pa.sock")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(work_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                LOG.warning("Prime Agent timed out after %ds", self.config.timeout)
                return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, self.config.model

            if proc.returncode not in (0, None):
                LOG.warning("Prime Agent exited %d: %s", proc.returncode, stderr.decode(errors="replace")[-1000:])
                return [], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, self.config.model
            output_items, usage = parse_prime_agent_events(stdout.decode(errors="replace"))
            return output_items, usage, self.config.model
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        body = body.model_copy(deep=True)
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        user_message, input_system = _extract_instruction(body.input)
        system_parts = [part for part in [self.config.system_prompt, body.instructions, input_system] if part]
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        rollout_id = request.path_params.get("rollout_id") if request is not None else None

        output_items, usage, model_name = await self._run_prime_agent(user_message, system_prompt, rollout_id)

        if not any(
            getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            for item in output_items
        ):
            LOG.warning("Prime Agent produced no assistant message. Padding empty output")
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    content=[NeMoGymResponseOutputText(text="", annotations=[])],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)

        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=model_name,
            object="response",
            output=output_items,
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
            usage=NeMoGymResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
                output_tokens=output_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def run(self, request: Request, body: PrimeAgentRunRequest) -> PrimeAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies

            seed_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_resp)
            cookies = seed_resp.cookies

            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)

            verify_resp = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_resp_json},
                cookies=cookies,
            )
            await raise_for_status(verify_resp)
            verify_json = await get_response_json(verify_resp)

            gym_resp = NeMoGymResponse.model_validate(agent_resp_json)
            turns = sum(
                1
                for item in gym_resp.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_resp.output[-1] if gym_resp.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return PrimeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    PrimeAgent.run_webserver()
