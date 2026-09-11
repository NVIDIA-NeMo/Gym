# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unscored response collector for human evaluation of Safe-Child-LLM."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class SafeChildLLMConfig(BaseResourcesServerConfig):
    pass


class SafeChildLLMVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SafeChildLLMVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    annotation_status: str = "pending_human_review"


class SafeChildLLMResourcesServer(SimpleResourcesServer):
    config: SafeChildLLMConfig

    async def verify(self, body: SafeChildLLMVerifyRequest) -> SafeChildLLMVerifyResponse:
        return SafeChildLLMVerifyResponse(
            **body.model_dump(exclude={"annotation_status", "reward"}),
            reward=0.0,
            annotation_status="pending_human_review",
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        rollouts = [rollout for task in tasks for rollout in task]
        return {"pending_human_review": len(rollouts)}


if __name__ == "__main__":
    SafeChildLLMResourcesServer.run_webserver()
