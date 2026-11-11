# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool-reward schedule over the existing mcqa verifier."""

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
        return ToolRewardVerifyResponse.model_validate(await super().verify(body))


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
