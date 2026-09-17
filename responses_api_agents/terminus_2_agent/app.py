# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from asyncio import Semaphore
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.agents.terminus_2 import Terminus2Harness
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


class Terminus2AgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    concurrency: int = 1
    model: Optional[str] = None
    workspace_root: Optional[str] = None
    system_prompt: Optional[str] = None
    max_turns: Optional[int] = None
    parser_name: Literal["json", "xml"] = "json"
    temperature: float = 0.7
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "default"]] = None
    collect_rollout_details: bool = False
    enable_summarize: bool = True
    proactive_summarization_threshold: int = 8000
    max_thinking_tokens: Optional[int] = None
    model_info: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_input_tokens": 262144,
            "max_output_tokens": 81920,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }
    )
    trajectory_config: dict[str, Any] = Field(default_factory=lambda: {"raw_content": False})
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40
    store_all_messages: bool = False
    record_terminal_session: bool = False
    interleaved_thinking: bool = False
    model_timeout_sec: float = 2400
    command_timeout_sec: float = 1800
    timeout: float = 10800
    keep_logs: bool = False
    logs_root: str = "outputs/terminus_2_agent/runs"


class Terminus2AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class Terminus2AgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    agent_timeout_error: int = 0
    context_length_exceeded_error: int = 0


class Terminus2Agent(SimpleResponsesAPIAgent):
    config: Terminus2AgentConfig
    sem: Semaphore = None
    _harness: Terminus2Harness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = Terminus2Harness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=self.config.model,
                    settings={
                        "temperature": self.config.temperature,
                        "reasoning_effort": self.config.reasoning_effort,
                        "collect_rollout_details": self.config.collect_rollout_details,
                        "max_thinking_tokens": self.config.max_thinking_tokens,
                        "model_info": self.config.model_info,
                        "timeout_seconds": self.config.model_timeout_sec,
                    },
                ),
                timeout_seconds=self.config.timeout,
                max_turns=self.config.max_turns,
                system_prompt=self.config.system_prompt,
                workspace=Path(self.config.workspace_root) if self.config.workspace_root else None,
                settings={
                    "parser_name": self.config.parser_name,
                    "enable_summarize": self.config.enable_summarize,
                    "proactive_summarization_threshold": self.config.proactive_summarization_threshold,
                    "trajectory_config": self.config.trajectory_config,
                    "tmux_pane_width": self.config.tmux_pane_width,
                    "tmux_pane_height": self.config.tmux_pane_height,
                    "store_all_messages": self.config.store_all_messages,
                    "record_terminal_session": self.config.record_terminal_session,
                    "interleaved_thinking": self.config.interleaved_thinking,
                    "command_timeout_seconds": self.config.command_timeout_sec,
                    "keep_logs": self.config.keep_logs,
                    "logs_root": self.config.logs_root,
                },
            )
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        if self.config.model is None and body.model is None:
            global_config = getattr(self.server_client, "global_config_dict", {})
            if isinstance(global_config, Mapping) and global_config.get("policy_model_name"):
                body = body.model_copy(update={"model": global_config["policy_model_name"]})
        path_params = getattr(request, "path_params", None)
        rollout_id = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        model_base_url = self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        async with self.sem:
            return await self._harness.run(body, model_base_url=model_base_url)

    async def run(self, request: Request, body: Terminus2AgentRunRequest) -> Terminus2AgentVerifyResponse:
        cookies = request.cookies
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        agent_response = await self.server_client.post(
            server_name=self.config.name,
            url_path=self.url_path_for_run("/v1/responses", body),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(agent_response)
        cookies = agent_response.cookies
        response_json = await get_response_json(agent_response)

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": response_json},
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        verify_json = await get_response_json(verify_response)

        gym_response = NeMoGymResponse.model_validate(response_json)
        turns = sum(
            1
            for item in gym_response.output
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant"
        )
        try:
            metadata = json.loads((gym_response.metadata or {}).get("terminus_2", "{}"))
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        flags = metadata.get("agent_error_flags", {})
        timed_out = bool(metadata.get("timed_out"))

        return Terminus2AgentVerifyResponse.model_validate(
            verify_json
            | {
                "turns_used": turns,
                "finished_naturally": bool(metadata.get("finished_naturally")) and not timed_out,
                "agent_timeout_error": int(timed_out),
                "context_length_exceeded_error": int(flags.get("context_length_exceeded", False)),
            }
        )


if __name__ == "__main__":
    Terminus2Agent.run_webserver()
