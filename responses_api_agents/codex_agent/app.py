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
import subprocess
from asyncio import Semaphore
from typing import Any, Literal, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.codex import CodexHarness
from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.codex_agent.setup_codex import ensure_codex


LOG = logging.getLogger(__name__)


class CodexAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # When model_server is set, the Codex model provider's base_url is resolved from the Gym model
    # server's URL (every Gym model server speaks the streaming Responses dialect on /v1/responses).
    # When None, openai_base_url is used directly (default: the real OpenAI API).
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    model: Optional[str] = None
    openai_api_key: str = ""  # pragma: allowlist secret
    openai_base_url: Optional[str] = None
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] = "danger-full-access"
    timeout: int = 600
    system_prompt: Optional[str] = None
    reasoning_effort: Optional[str] = None
    codex_version: str
    cwd: Optional[str] = None
    stream_idle_timeout_ms: Optional[int] = None
    extra_config: dict[str, Any] = Field(default_factory=dict)


class CodexAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class CodexAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False


class CodexAgent(SimpleResponsesAPIAgent):
    config: CodexAgentConfig
    sem: Semaphore = None
    _harness: CodexHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        model = self.config.model or ("gym-policy-model" if self.config.model_server else None)
        self._harness = CodexHarness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=model,
                    provider="openai",
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                    settings={
                        "reasoning_effort": self.config.reasoning_effort,
                        "stream_idle_timeout_ms": self.config.stream_idle_timeout_ms,
                    },
                ),
                timeout_seconds=self.config.timeout,
                system_prompt=self.config.system_prompt,
                workspace=self.config.cwd,
                settings={
                    "sandbox_mode": self.config.sandbox_mode,
                    "extra_config": self.config.extra_config,
                },
            )
        )
        ensure_codex(self.config.codex_version)
        try:
            version = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
            LOG.warning("codex version: %s", version or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine codex version: %s", exc)

    def _resolve_call_base_url(self, rollout_id: Optional[str]) -> str:
        """Return the model-call base URL, including Gym's rollout prefix when configured."""
        if self.config.model_server:
            return self.resolve_model_base_url(self.config.model_server.name, rollout_id)
        return self.config.openai_base_url or "https://api.openai.com/v1"

    def _resources_server_base_url(self) -> str:
        config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(config)

    def _rollout_mcp_servers(self, seed_response_json: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Translate Gym seed metadata into Codex MCP configuration."""
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None

        server_name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        entry: dict[str, Any] = {
            "url": f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}",
        }
        headers = metadata.get("headers")
        if isinstance(headers, dict) and headers:
            entry["http_headers"] = {str(key): str(value) for key, value in headers.items()}
        else:
            LOG.warning(
                "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
                "session token and will reject the calls.",
                server_name,
            )
        return {server_name: entry}

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        return await self._harness.run(body, model_base_url=self._resolve_call_base_url(None))

    async def run(self, request: Request, body: CodexAgentRunRequest) -> CodexAgentVerifyResponse:
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
            seed_response_json = await get_response_json(seed_response)

            skills_path = ((body.model_extra or {}).get(SKILLS_REF_KEY_NAME) or {}).get("path")
            rollout_id = self.rollout_id_from_run(body)
            agent_response = await self._harness.run(
                body.responses_create_params,
                mcp_servers=self._rollout_mcp_servers(seed_response_json),
                skills_path=skills_path,
                model_base_url=self._resolve_call_base_url(rollout_id),
            )
            agent_response_json = agent_response.model_dump(mode="json")

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

            return CodexAgentVerifyResponse.model_validate(
                verify_json | {"turns_used": turns, "finished_naturally": naturally}
            )


if __name__ == "__main__":
    CodexAgent.run_webserver()
