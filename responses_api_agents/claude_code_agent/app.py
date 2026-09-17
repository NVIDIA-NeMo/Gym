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
import tempfile
from asyncio import Semaphore
from pathlib import Path
from typing import Any, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents.claude_code import ClaudeCodeHarness
from nemo_gym.agents.config import AgentHarnessConfig, AgentModelConfig
from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY, BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import SKILLS_REF_KEY_NAME, get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import apply_rollout_prefix, get_response_json, raise_for_status
from responses_api_agents.claude_code_agent.setup_claude_code import ensure_claude_code


LOG = logging.getLogger(__name__)


class ClaudeCodeAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    # When model_server is set, ANTHROPIC_BASE_URL is resolved from the Gym model
    # server's URL (requires the server to expose POST /v1/messages).
    # When None, anthropic_base_url is used directly.
    model_server: Optional[ModelServerRef] = None
    concurrency: int = 32
    model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # pragma: allowlist secret
    anthropic_base_url: Optional[str] = None
    max_turns: Optional[int] = 30  # None -> unlimited turns
    timeout: int = 300
    system_prompt: Optional[str] = None
    workspace: Optional[str] = None
    allowed_tools: Optional[str] = None
    disallowed_tools: Optional[str] = None
    claude_code_version: Optional[str] = None
    thinking: Optional[str] = None
    max_thinking_tokens: Optional[int] = None
    # Runtime capability knobs. The default (bare=True, no mcp_config/settings) reproduces the original
    # isolated behavior: Claude Code skips hooks, LSP, plugin sync, attribution, auto-memory, background
    # prefetches, keychain reads, and CLAUDE.md auto-discovery (skills still resolve via /skill-name).
    bare: bool = True
    mcp_config: Optional[str] = None
    settings: Optional[str] = None


class ClaudeCodeAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ClaudeCodeAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    turns_used: int = 0
    finished_naturally: bool = False
    ng_agent_observations: Optional[AgentObservationBundle] = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ClaudeCodeAgent(SimpleResponsesAPIAgent):
    config: ClaudeCodeAgentConfig
    sem: Semaphore = None
    _harness: ClaudeCodeHarness = PrivateAttr()
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.sem = Semaphore(self.config.concurrency)
        self._harness = ClaudeCodeHarness(
            AgentHarnessConfig(
                model=AgentModelConfig(
                    model=self.config.model,
                    provider="anthropic",
                    api_key=self.config.anthropic_api_key,
                    base_url=self.config.anthropic_base_url,
                ),
                timeout_seconds=self.config.timeout,
                max_turns=self.config.max_turns,
                system_prompt=self.config.system_prompt,
                workspace=self.config.workspace,
                settings={
                    "allowed_tools": self.config.allowed_tools,
                    "disallowed_tools": self.config.disallowed_tools,
                    "thinking": self.config.thinking,
                    "max_thinking_tokens": self.config.max_thinking_tokens,
                    "bare": self.config.bare,
                    "mcp_config": self.config.mcp_config,
                    "settings_file": self.config.settings,
                },
            )
        )
        ensure_claude_code(self.config.claude_code_version)
        try:
            version = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            LOG.warning("claude-code version: %s", version or "(unknown)")
        except Exception as exc:
            LOG.warning("could not determine claude-code version: %s", exc)

    def _resolve_base_url(self) -> str:
        if self.config.model_server:
            config = get_first_server_config_dict(
                self.server_client.global_config_dict,
                self.config.model_server.name,
            )
            return self.server_client._build_server_base_url(config)
        return self.config.anthropic_base_url or ""

    def _resolve_call_base_url(self, rollout_id: Optional[str]) -> str:
        """Return the CLI model-call URL with its rollout prefix."""
        base_url = self._resolve_base_url()
        if base_url and self.config.model_server:
            return apply_rollout_prefix(
                base_url,
                rollout_id,
                token_capture=self._token_id_capture_enabled(),
            )
        return base_url

    def _resources_server_base_url(self) -> str:
        config = get_first_server_config_dict(
            self.server_client.global_config_dict,
            self.config.resources_server.name,
        )
        return self.server_client._build_server_base_url(config)

    def _write_rollout_mcp_config(self, seed_response_json: dict[str, Any], output_dir: Path) -> Optional[str]:
        metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
        if not isinstance(metadata, dict):
            return None

        server_name = str(metadata.get("server_name") or self.config.resources_server.name)
        url_path = str(metadata.get("url_path") or "/mcp")
        url = f"{self._resources_server_base_url().rstrip('/')}/{url_path.lstrip('/')}"
        headers = metadata.get("headers")
        return self._harness.write_mcp_config(
            server_name=server_name,
            url=url,
            output_dir=output_dir,
            transport=str(metadata.get("transport") or "http"),
            headers=headers if isinstance(headers, dict) else None,
        )

    async def responses(
        self,
        request: Request,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        return await self._harness.run(
            body,
            model_base_url=self._resolve_call_base_url(request.path_params.get("rollout_id")),
        )

    async def run(self, request: Request, body: ClaudeCodeAgentRunRequest) -> ClaudeCodeAgentVerifyResponse:
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
            model_base_url = self._resolve_call_base_url(rollout_id)

            with tempfile.TemporaryDirectory(prefix="nemo_gym_claude_mcp_") as mcp_config_dir:
                mcp_config = self._write_rollout_mcp_config(seed_response_json, Path(mcp_config_dir))
                if rollout_id is not None:
                    episode = await self._harness.run_episode(
                        body.responses_create_params,
                        mcp_config=mcp_config,
                        skills_path=skills_path,
                        model_base_url=model_base_url,
                        model_ref=self.config.model_server,
                    )
                    agent_response, observations = episode.response, episode.observations
                else:
                    agent_response = await self._harness.run(
                        body.responses_create_params,
                        mcp_config=mcp_config,
                        skills_path=skills_path,
                        model_base_url=model_base_url,
                    )
                    observations = None
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

            result = verify_json | {"turns_used": turns, "finished_naturally": naturally}
            if observations is not None:
                result["ng_agent_observations"] = observations.model_dump(mode="json")
            return ClaudeCodeAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    ClaudeCodeAgent.run_webserver()
