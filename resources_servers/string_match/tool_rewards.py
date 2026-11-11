# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool-reward schedule over the existing string_match verifier."""

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardScoringConfig,
    ToolRewardVerifier,
    ToolRewardVerifyResponse,
)
from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchResourcesServerConfig,
    StringMatchVerifyRequest,
)


class ToolRewardConfig(ToolRewardScoringConfig, StringMatchResourcesServerConfig):
    pass


class ToolRewardServer(ToolRewardVerifier, StringMatchResourcesServer):
    config: ToolRewardConfig

    async def verify(self, body: StringMatchVerifyRequest) -> ToolRewardVerifyResponse:
        return ToolRewardVerifyResponse.model_validate(await super().verify(body))


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
