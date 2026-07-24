# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resources server for synthetic tool-use domain generation."""

from fastapi import FastAPI

from nemo_gym.base_resources_server import BaseResourcesServer, BaseResourcesServerConfig
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import SimpleServer
from resources_servers.synthetic_tool_use.assets import generation_asset_hashes
from resources_servers.synthetic_tool_use.common.artifacts import RunArtifactStore
from resources_servers.synthetic_tool_use.common.clients import GymModelGenerator
from resources_servers.synthetic_tool_use.common.models import (
    SeedGenerationConfig,
    StageGenerationRequest,
    StageGenerationResponse,
)
from resources_servers.synthetic_tool_use_domain_generation.assets import load_domain_prompt
from resources_servers.synthetic_tool_use_domain_generation.stage import DomainGenerationStage


class DomainGenerationResourcesServerConfig(BaseResourcesServerConfig):
    generation: SeedGenerationConfig
    model_server: ModelServerRef


class DomainGenerationResourcesServer(BaseResourcesServer, SimpleServer):
    config: DomainGenerationResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/generate")(self.generate)
        return app

    async def generate(self, body: StageGenerationRequest) -> StageGenerationResponse:
        generation = self.config.generation
        asset_hashes = generation_asset_hashes(generation.generation_profile)
        store = RunArtifactStore.create(generation, asset_hashes)
        store.record_asset_hashes(asset_hashes)
        generator = GymModelGenerator(
            server_client=self.server_client,
            model_server=self.config.model_server,
            role=generation.domain_model,
        )
        await DomainGenerationStage(generation, load_domain_prompt(), store, generator).run(resume=body.resume)
        return StageGenerationResponse(report=store.write_generation_report())


if __name__ == "__main__":
    DomainGenerationResourcesServer.run_webserver()
