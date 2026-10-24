# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP configuration and routes for frozen Agentic IF verification."""

from pathlib import Path
from typing import ClassVar, Literal

from nemo_gym.base_resources_server import BaseResourcesServerConfig, ReverifyMode, SimpleResourcesServer
from resources_servers.agentic_if_realworld.benchmark.core import DEFAULT_JUDGE_MODEL
from resources_servers.agentic_if_realworld.verifier import (
    VERIFIER_FIXTURE as VERIFIER_FIXTURE,
)
from resources_servers.agentic_if_realworld.verifier import (
    FixtureVerifier as FixtureVerifier,
)
from resources_servers.agentic_if_realworld.verifier import (
    Verifier as Verifier,
)
from resources_servers.agentic_if_realworld.verifier import (
    VerifyRequest as VerifyRequest,
)
from resources_servers.agentic_if_realworld.verifier import (
    VerifyResponse as VerifyResponse,
)
from resources_servers.agentic_if_realworld.verifier import (
    cached_dataset as cached_dataset,
)
from resources_servers.agentic_if_realworld.verifier import (
    cached_preflight as cached_preflight,
)
from resources_servers.agentic_if_realworld.verifier import (
    project_gym_response as project_gym_response,
)


class ResourcesConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    dataset_dir: str = ""
    receipt_dir: str
    judge_endpoint: str
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_api_key_env: str = "AGENTIC_IF_JUDGE_API_KEY"
    judge_reasoning: str = "medium"
    synthetic: bool = False
    split_registry_path: str = ""
    ground_truth_path: str = ""
    response_source: Literal["raw_receipt", "gym_response"] = "raw_receipt"


class ResourcesServer(SimpleResourcesServer):
    config: ResourcesConfig

    async def verify(self, body: VerifyRequest) -> VerifyResponse:
        if not self.config.synthetic:
            cached_preflight(self.config.split_registry_path, self.config.ground_truth_path, self.config.dataset_dir)
        verifier = Verifier(
            dataset=cached_dataset(self.config.dataset_dir, self.config.synthetic),
            receipt_dir=Path(self.config.receipt_dir),
            endpoint=self.config.judge_endpoint,
            model=self.config.judge_model,
            key_env=self.config.judge_api_key_env,
            reasoning=self.config.judge_reasoning,
            response_source=self.config.response_source,
        )
        return await verifier.verify(body)


if __name__ == "__main__":
    ResourcesServer.run_webserver()
