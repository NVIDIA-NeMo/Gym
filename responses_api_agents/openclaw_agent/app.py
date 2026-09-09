# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import shlex
import shutil
from asyncio import Semaphore
from collections.abc import Mapping
from typing import Any, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.agents.openclaw import OpenClawHarness
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.openclaw_agent.setup_openclaw import ensure_openclaw


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


class OpenClawAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    command: str = "openclaw"
    model: str = "nvinf/nvidia/meta/llama-3.3-70b-instruct"
    node_bin_dir: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/openclaw_agent/workspaces"
    openclaw_agent_id: str = "main"
    thinking: str = "off"
    system_prompt: Optional[str] = None
    setup_timeout: int = 900
    timeout: int = 900
    extra_args: list[str] = []
    openclaw_config: dict[str, Any] = Field(default_factory=dict)
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    openclaw_version: str

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class OpenClawAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class OpenClawAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class OpenClawAgent(SimpleResponsesAPIAgent):
    config: OpenClawAgentConfig
    sem: Semaphore = None
    _harness: OpenClawHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = OpenClawHarness(
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
                    "node_bin_dir": self.config.node_bin_dir,
                    "env": self.config.env,
                    "workspace_root": self.config.workspace_root,
                    "openclaw_agent_id": self.config.openclaw_agent_id,
                    "thinking": self.config.thinking,
                    "setup_timeout": self.config.setup_timeout,
                    "extra_args": self.config.extra_args,
                    "openclaw_config": self.config.openclaw_config,
                },
            )
        )
        ensure_openclaw(self.config.openclaw_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("openclaw command %r is not on PATH yet", self.config.command)

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> str:
        if self.config.model_server is None:
            return ""
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if not isinstance(rollout_id, str):
            return await self._harness.run(body, model_base_url=self._resolve_model_base_url())
        episode = await self._harness.run_episode(
            body,
            model_base_url=self._resolve_model_base_url(rollout_id),
            model_ref=self.config.model_server,
        )
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: OpenClawAgentRunRequest) -> OpenClawAgentVerifyResponse:
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

            rollout_id = self.rollout_id_from_run(body)
            agent_resp = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_resp)
            cookies = agent_resp.cookies
            agent_resp_json = await get_response_json(agent_resp)
            raw_observations = (
                agent_resp_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

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
            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return OpenClawAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    OpenClawAgent.run_webserver()
