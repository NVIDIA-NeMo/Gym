# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional tool-reward schedule over the existing string_match verifier."""

from typing import ClassVar

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.opencode_multimodal_benchmarks.rewards import (
    ToolRewardEvidence,
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


class ToolRewardVerifyRequest(ToolRewardEvidence, StringMatchVerifyRequest):
    pass


class ToolRewardServer(ToolRewardVerifier, StringMatchResourcesServer):
    config: ToolRewardConfig
    _answer_block_separator: ClassVar[str] = "\n"

    async def verify(self, body: ToolRewardVerifyRequest) -> ToolRewardVerifyResponse:
        return ToolRewardVerifyResponse.model_validate(await super().verify(body))


if __name__ == "__main__":
    ToolRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolRewardServer.run_webserver()
