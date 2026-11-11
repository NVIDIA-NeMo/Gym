# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool-reward schedule over the existing math_with_judge verifier."""

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.math_with_judge.app import (
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
)
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardScoringConfig,
    ToolRewardVerifier,
    ToolRewardVerifyResponse,
)


class ToolRewardConfig(ToolRewardScoringConfig, LibraryJudgeMathResourcesServerConfig):
    pass


class ToolRewardServer(ToolRewardVerifier, LibraryJudgeMathResourcesServer):
    config: ToolRewardConfig

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> ToolRewardVerifyResponse:
        return ToolRewardVerifyResponse.model_validate(await super().verify(body))


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
