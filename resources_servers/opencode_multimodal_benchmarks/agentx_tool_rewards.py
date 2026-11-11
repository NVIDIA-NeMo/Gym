# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-only tool rewards over the independent Agent-X evaluator."""

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.opencode_multimodal_benchmarks.agentx import (
    AgentXConfig,
    AgentXServer,
    AgentXVerifyResponse,
)
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    OpenCodeToolReward,
    OpenCodeToolRewardConfig,
    ToolRewardVerifier,
)
from resources_servers.opencode_multimodal_benchmarks.verifier import VisualVerifyRequest


class AgentXToolRewardConfig(AgentXConfig):
    tool_rewards: OpenCodeToolRewardConfig | None = Field(default_factory=OpenCodeToolRewardConfig)


class AgentXToolRewardVerifyResponse(AgentXVerifyResponse):
    opencode_tool_reward: OpenCodeToolReward | None = None
    reward_components: dict[str, float] | None = None


class AgentXToolRewardServer(ToolRewardVerifier, AgentXServer):
    config: AgentXToolRewardConfig

    async def verify(self, body: VisualVerifyRequest) -> AgentXToolRewardVerifyResponse:
        return await AgentXServer.verify(self, body)

    def _make_response(self, body, prediction, score, raw) -> AgentXToolRewardVerifyResponse:
        result = super()._make_response(body, prediction, score, raw)
        # Shape inside the evaluator's existing grading observation, after it has a valid grade.
        return AgentXToolRewardVerifyResponse.model_validate(self._apply_tool_rewards(body, result).model_dump())

    def compute_metrics(
        self,
        tasks: list[list[dict[str, Any]]],
        *,
        expected_repeats: Mapping[tuple[str, str], set[int]] | None = None,
    ) -> dict[str, Any]:
        # Only benchmark metrics use correctness. Shared reward statistics retain the original rows.
        scoring_tasks = [
            [
                row
                | {
                    "reward": (row.get("reward_components") or {}).get(
                        "verifier", (row.get("opencode_tool_reward") or {}).get("base_reward", row["reward"])
                    )
                }
                if row["verifier_metadata"]["benchmark"] == "AgentX"
                else row
                for row in repeats
            ]
            for repeats in tasks
        ]
        return super().compute_metrics(scoring_tasks, expected_repeats=expected_repeats)


if __name__ == "__main__":
    AgentXToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentXToolRewardServer.run_webserver()
