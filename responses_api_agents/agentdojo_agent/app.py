# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Official AgentDojo backend for the shared AgentDojo-family adapter."""

from __future__ import annotations

from typing import Literal

from fastapi import Body
from pydantic import ConfigDict, Field

from responses_api_agents.agentdojo_family.app import (
    AgentDojoFamilyAgent,
    AgentDojoFamilyAgentConfig,
    AgentDojoFamilyRunRequest,
    AgentDojoFamilyVerifyResponse,
)


AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")

# `agentdojo.agent_pipeline.agent_pipeline.DEFENSES` at the pinned v0.1.35 commit. Upstream's
# `AgentPipeline.from_config` raises "Invalid defense name" for anything else, so an unknown
# name is rejected at the request boundary instead of masking every rollout of a run.
AgentDojoDefense = Literal[
    "tool_filter",
    "transformers_pi_detector",
    "spotlighting_with_delimiting",
    "repeat_user_prompt",
]
AgentDojoVerifyResponse = AgentDojoFamilyVerifyResponse


class AgentDojoAgentConfig(AgentDojoFamilyAgentConfig):
    # Rollouts share one process and upstream pipeline state is not designed for concurrent
    # use, so the agent runs one rollout at a time. Scale out with more agent processes.
    concurrency: Literal[1] = 1
    default_defense: AgentDojoDefense | None = None


class AgentDojoRunRequest(AgentDojoFamilyRunRequest):
    model_config = ConfigDict(extra="allow")

    suite: Literal["banking", "slack", "travel", "workspace"]
    defense: AgentDojoDefense | None = Field(default=None)


class AgentDojoAgent(AgentDojoFamilyAgent):
    config: AgentDojoAgentConfig

    async def run(self, body: AgentDojoRunRequest = Body()) -> AgentDojoVerifyResponse:
        return await super().run(body)


if __name__ == "__main__":
    AgentDojoAgent.run_webserver()
