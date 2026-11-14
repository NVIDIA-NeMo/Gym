# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool-reward schedule over the existing mcqa verifier."""

from nemo_gym.server_runtime import observe_programmatic_grading
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.mcqa.app import MCQAResourcesServer, MCQAResourcesServerConfig, MCQAVerifyRequest
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardEvidence,
    ToolRewardScoringConfig,
    ToolRewardVerifier,
    ToolRewardVerifyResponse,
)


class ToolRewardConfig(ToolRewardScoringConfig, MCQAResourcesServerConfig):
    pass


class ToolRewardVerifyRequest(ToolRewardEvidence, MCQAVerifyRequest):
    pass


class ToolRewardServer(ToolRewardVerifier, MCQAResourcesServer):
    config: ToolRewardConfig

    async def verify(self, body: ToolRewardVerifyRequest) -> ToolRewardVerifyResponse:
        # Bind the complete calculation, including restoration and penalties, rather than an intermediate grade.
        return observe_programmatic_grading(
            lambda: self._apply_tool_rewards(body, self._programmatic_score(self._grading_body(body)))
        )


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
