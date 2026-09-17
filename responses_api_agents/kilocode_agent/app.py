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
from nemo_gym.agents.kilocode import KiloCodeHarness
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.kilocode_agent.setup_kilo import ensure_kilo


LOG = logging.getLogger(__name__)


class KiloCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "kilo"
    model: str = "openai/gpt-4o-mini"
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/kilocode_agent/workspaces"
    repo_dir: Optional[str] = None
    thinking: bool = False
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    kilo_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 32768
    max_output_tokens: int = 8192
    reasoning_field: Optional[str] = "reasoning_content"
    kilo_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class KiloCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class KiloCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class KiloCodeAgent(SimpleResponsesAPIAgent):
    config: KiloCodeAgentConfig
    sem: Semaphore = None
    _harness: KiloCodeHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = KiloCodeHarness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=self.config.model,
                    provider="nemo" if self.config.model_server else "",
                    api_key="" if self.config.model_server else self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                    settings={
                        "context_window": self.config.context_window,
                        "max_output_tokens": self.config.max_output_tokens,
                        "reasoning_field": self.config.reasoning_field,
                    },
                ),
                timeout_seconds=self.config.timeout,
                system_prompt=self.config.system_prompt,
                workspace=self.config.repo_dir,
                settings={
                    "command": self.config.command,
                    "env": self.config.env,
                    "workspace_root": self.config.workspace_root,
                    "thinking": self.config.thinking,
                    "setup_timeout": self.config.setup_timeout,
                    "extra_args": self.config.extra_args,
                    "kilo_config": self.config.kilo_config,
                },
            )
        )
        ensure_kilo(self.config.kilo_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("kilo command %r is not on PATH yet", self.config.command)

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        rollout_id = request.path_params.get("rollout_id")
        return await self._harness.run(body, model_base_url=self._resolve_model_base_url(rollout_id))

    async def run(self, request: Request, body: KiloCodeAgentRunRequest) -> KiloCodeAgentVerifyResponse:
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
            return KiloCodeAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    KiloCodeAgent.run_webserver()
