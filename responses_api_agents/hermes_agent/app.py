# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from asyncio import Semaphore
from collections.abc import Mapping
from typing import Any, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.agents.hermes import HermesHarness
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import get_response_json, raise_for_status


_INTERNAL_OBSERVATIONS_KEY = "_ng_agent_observations"


class HermesAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model: Optional[str] = None
    concurrency: int = Field(default=1, ge=1, le=1)
    max_turns: int = 90
    max_tokens: Optional[int] = None
    enabled_toolsets: Optional[list[str]] = None
    disabled_toolsets: Optional[list[str]] = None
    temperature: float | None = None
    terminal_backend: str = "local"
    terminal_timeout: int = 180
    system_prompt: Optional[str] = None
    timeout: int = 900
    compression_enabled: bool = True
    compression_threshold: float = 0.85
    chat_template_kwargs_enabled: bool = True
    api_key: Optional[str] = None
    delegation_max_iterations: int = 50
    checkpoints_enabled: bool = False


class HermesAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class HermesAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: AgentObservationBundle | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class HermesAgent(SimpleResponsesAPIAgent):
    config: HermesAgentConfig
    sem: Semaphore = None
    _harness: HermesHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = HermesHarness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=self.config.model or str(self.config.model_server.name),
                    provider="auto",
                    api_key=self.config.api_key or "",
                    settings={"max_tokens": self.config.max_tokens, "temperature": self.config.temperature},
                ),
                max_turns=self.config.max_turns,
                timeout_seconds=self.config.timeout,
                system_prompt=self.config.system_prompt,
                settings={
                    "enabled_toolsets": self.config.enabled_toolsets,
                    "disabled_toolsets": self.config.disabled_toolsets,
                    "terminal_backend": self.config.terminal_backend,
                    "terminal_timeout": self.config.terminal_timeout,
                    "compression_enabled": self.config.compression_enabled,
                    "compression_threshold": self.config.compression_threshold,
                    "chat_template_kwargs_enabled": self.config.chat_template_kwargs_enabled,
                    "delegation_max_iterations": self.config.delegation_max_iterations,
                    "checkpoints_enabled": self.config.checkpoints_enabled,
                },
            )
        )

    def _model_base_url(self, rollout_id: Optional[str] = None) -> str:
        return self.resolve_model_base_url(self.config.model_server.name, rollout_id)

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        if not isinstance(rollout_id, str):
            return await self._harness.run(body, model_base_url=self._model_base_url())
        episode = await self._harness.run_episode(
            body,
            model_base_url=self._model_base_url(rollout_id),
            model_ref=self.config.model_server,
        )
        return episode.response.model_copy(
            update={_INTERNAL_OBSERVATIONS_KEY: episode.observations.model_dump(mode="json")}
        )

    async def run(self, request: Request, body: HermesAgentRunRequest) -> HermesAgentVerifyResponse:
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
            return HermesAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    HermesAgent.run_webserver()
