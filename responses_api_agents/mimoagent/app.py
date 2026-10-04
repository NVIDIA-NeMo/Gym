# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


PROFILES_DIR = Path(__file__).parent / "profiles"
NATIVE_LOOPS = {"default", "bashonly-agent", "cc-agent", "codex-agent", "mimocode-agent"}
MCP_CONFIG_PATH = Path("/work/_setup/mcp_servers.json")


class MimoAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    profile: str
    agent_overrides: dict[str, Any] = Field(default_factory=dict)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    protocol: str | None = None
    cwd: str | None = None
    command_timeout: int = 600
    concurrency: int = 32


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text") or "") if isinstance(c, dict) else str(c) for c in content)
    return "" if content is None else str(content)


def _without_tool_name(message: dict) -> dict:
    if message.get("role") != "tool" or "name" not in message:
        return message
    return {k: v for k, v in message.items() if k != "name"}


def _task_from_input(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    if isinstance(body.input, str):
        return body.input
    parts = [
        _text(getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")) for m in body.input
    ]
    return "\n\n".join(p for p in parts if p)


def _output_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            if content := _text(message.get("content")):
                items.append(
                    {
                        "id": f"msg_{uuid4().hex}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": content, "annotations": []}],
                    }
                )
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append(
                    {
                        "id": f"fc_{uuid4().hex}",
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": _text(message.get("content")),
                }
            )
    return items


class MimoAgent(SimpleResponsesAPIAgent):
    ray_enabled = False
    config: MimoAgentConfig
    sem: asyncio.Semaphore = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = asyncio.Semaphore(self.config.concurrency)

    def _base_url(self) -> str:
        cfg = get_first_server_config_dict(self.server_client.global_config_dict, self.config.model_server.name)
        return str(self.server_client._build_server_base_url(cfg)).rstrip("/").removesuffix("/v1")

    def _build(self, body: NeMoGymResponseCreateParamsNonStreaming, prefix: str = ""):
        from mimoagent.agents.factory import make_agent
        from mimoagent.config import expand_env_vars
        from mimoagent.environments.local import LocalEnvironment
        from mimoagent.models import get_model

        profile = expand_env_vars(yaml.safe_load((PROFILES_DIR / f"{self.config.profile}.yaml").read_text()))
        agent_cfg = {**(profile.get("agent") or {}), **self.config.agent_overrides}
        agent_type = agent_cfg.pop("type", "default")
        model_block = profile.get("model") or {}
        protocol = self.config.protocol or model_block.get("protocol") or "chat"
        base_url = self._base_url() + prefix
        model_kwargs = {
            "api_key": os.environ.get("MIMOAGENT_API_KEY", "dummy-key"),
            "base_url": f"{base_url}/v1" if agent_type in NATIVE_LOOPS and protocol != "anthropic" else base_url,
            **self.config.model_kwargs,
        }
        if body.temperature is not None:
            model_kwargs["temperature"] = body.temperature
        if body.top_p is not None:
            model_kwargs["top_p"] = body.top_p
        model = get_model(
            config={
                "model_name": self.config.model or model_block.get("model_name") or body.model,
                "protocol": protocol,
                "model_kwargs": model_kwargs,
            }
        )
        query = model.query
        model.query = lambda messages, **kwargs: query([_without_tool_name(m) for m in messages], **kwargs)
        if protocol == "responses" and hasattr(model, "client"):
            create = model.client.responses.create

            def create_with_strict(**kwargs: Any) -> Any:
                if kwargs.get("tools"):
                    kwargs["tools"] = [
                        {"strict": False, **t} if t.get("type") == "function" else t for t in kwargs["tools"]
                    ]
                return create(**kwargs)

            model.client.responses.create = create_with_strict
        env = LocalEnvironment(cwd=self.config.cwd or os.getcwd(), timeout=self.config.command_timeout)
        agent = make_agent(agent_type, model, env, **agent_cfg)
        if MCP_CONFIG_PATH.exists() and hasattr(agent, "tool_registry"):
            from responses_api_agents.mimoagent.mcp_proxy import discover_mcp_tools

            mcp = json.loads(MCP_CONFIG_PATH.read_text())
            for tool in discover_mcp_tools(env, mcp["servers"], mcp["bridge_python"], mcp["bridge_script"]):
                agent.tool_registry.register(tool)
            agent._tool_definitions = agent.tool_registry.get_function_definitions()
        return agent

    def _run_agent(
        self, body: NeMoGymResponseCreateParamsNonStreaming, prefix: str
    ) -> tuple[str, str, list[dict[str, Any]]]:
        agent = self._build(body, prefix)
        try:
            status, result = agent.run(_task_from_input(body))
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        return status, result, list(agent.messages)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return await self._respond(body, self.url_path_for_request("", request))

    async def _respond(self, body: NeMoGymResponseCreateParamsNonStreaming, prefix: str) -> NeMoGymResponse:
        async with self.sem:
            status, result, messages = await asyncio.to_thread(self._run_agent, body, prefix)
        output = _output_items(messages)
        if result and not any(i["type"] == "message" and i["content"][0]["text"] == result for i in output):
            output += _output_items([{"role": "assistant", "content": result}])
        return NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time.time()),
            model=self.config.model or body.model or "",
            object="response",
            output=output,
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            metadata={"mimoagent_status": status, "mimoagent_profile": self.config.profile},
        )

    async def run(self, request: Request, body: BaseRunRequest) -> BaseVerifyResponse:
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed)
        cookies = request.cookies | seed.cookies
        resp = await self._respond(body.responses_create_params, self.url_path_for_run("", body))
        verify = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": json.loads(resp.model_dump_json())},
            cookies=cookies,
        )
        await raise_for_status(verify)
        return BaseVerifyResponse.model_validate(await get_response_json(verify))


if __name__ == "__main__":
    MimoAgent.run_webserver()
