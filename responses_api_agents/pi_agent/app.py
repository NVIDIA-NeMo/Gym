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

import logging
import shlex
import shutil
from asyncio import Semaphore
from collections.abc import Mapping
from typing import Any, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.agents.pi import PiHarness
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.pi_agent.setup_pi import ensure_pi


LOG = logging.getLogger(__name__)
_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


class PiAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 8
    command: str = "pi"
    model: str = "nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct"
    env: dict[str, str] = Field(default_factory=dict)
    workspace_root: str = "outputs/pi_agent/workspaces"
    thinking: Optional[str] = None
    system_prompt: Optional[str] = None
    timeout: int = 900
    extra_args: list[str] = []
    models_config: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 262144
    max_output_tokens: int = 131072
    pi_version: Optional[str] = None

    @property
    def command_parts(self) -> list[str]:
        return shlex.split(self.command)


class PiAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class PiAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class PiAgent(SimpleResponsesAPIAgent):
    config: PiAgentConfig
    sem: Semaphore = None
    _harness: PiHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = PiHarness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=self.config.model,
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
                    "thinking": self.config.thinking,
                    "extra_args": self.config.extra_args,
                    "models_config": self.config.models_config,
                },
            )
        )
        ensure_pi(self.config.pi_version)
        command = self.config.command_parts[0] if self.config.command_parts else ""
        if not command or shutil.which(command) is None:
            LOG.warning("pi command %r is not on PATH yet", self.config.command)

    def _resolve_model_base_url(self, rollout_id: Optional[str] = None) -> Optional[str]:
        if self.config.model_server is None:
            return None
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        episode = await self._harness.run_episode(
            body,
            model_base_url=self._resolve_model_base_url(rollout_id),
            model_ref=self.config.model_server,
            rollout_id=rollout_id,
            collect_observations=isinstance(rollout_id, str),
        )
        if not isinstance(rollout_id, str):
            return episode.response
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: PiAgentRunRequest) -> PiAgentVerifyResponse:
        async with self.sem:
            cookies = request.cookies
            seed_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/seed_session",
                json=body.model_dump(),
                cookies=cookies,
            )
            await raise_for_status(seed_response)
            cookies = seed_response.cookies

            rollout_id = self.rollout_id_from_run(body)
            agent_response = await self.server_client.post(
                server_name=self.config.name,
                url_path=self.url_path_for_run("/v1/responses", body),
                json=body.responses_create_params,
                cookies=cookies,
            )
            await raise_for_status(agent_response)
            cookies = agent_response.cookies
            agent_response_json = await get_response_json(agent_response)
            raw_observations = (
                agent_response_json.pop(_INTERNAL_OBSERVATIONS_KEY, None) if rollout_id is not None else None
            )
            observations = (
                AgentObservationBundle.model_validate(raw_observations) if isinstance(raw_observations, dict) else None
            )

            verify_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": agent_response_json},
                cookies=cookies,
            )
            await raise_for_status(verify_response)
            verify_json = await get_response_json(verify_response)

            gym_response = NeMoGymResponse.model_validate(agent_response_json)
            turns = sum(
                1
                for item in gym_response.output
                if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
            )
            last = gym_response.output[-1] if gym_response.output else None
            naturally = getattr(last, "type", None) == "message" and getattr(last, "role", None) == "assistant"

            return PiAgentVerifyResponse.model_validate(
                verify_json
                | {"turns_used": turns, "finished_naturally": naturally}
                | ({"ng_agent_observations": observations} if observations is not None else {})
            )


if __name__ == "__main__":
    PiAgent.run_webserver()
