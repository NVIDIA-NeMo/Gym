# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import shlex
import shutil
from asyncio import Semaphore
from typing import Any, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.agents.prime import PrimeAgentHarness
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.prime_agent.setup_prime_agent import ensure_prime_agent


LOG = logging.getLogger(__name__)


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
    _harness: PrimeAgentHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = PrimeAgentHarness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=self.config.model,
                    provider="nemo" if self.config.model_server else "",
                    settings={
                        "context_window": self.config.context_window,
                        "max_output_tokens": self.config.max_output_tokens,
                    },
                ),
                timeout_seconds=self.config.timeout,
                system_prompt=self.config.system_prompt,
                settings={
                    "command": self.config.command,
                    "env": self.config.env,
                    "workspace_root": self.config.workspace_root,
                    "kernel_venv": self.config.kernel_venv,
                    "thinking": self.config.thinking,
                    "extra_args": self.config.extra_args,
                    "models_config": self.config.models_config,
                },
            )
        )
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if command == "prime-agent":
            ensure_prime_agent(self.config.prime_agent_version)
        if not command or shutil.which(command) is None:
            LOG.warning("Prime Agent command %r is not on PATH", self.config.command)

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        rollout_id = request.path_params.get("rollout_id") if request is not None else None
        return await self._harness.run(body, model_base_url=self._resolve_model_base_url(rollout_id))

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
