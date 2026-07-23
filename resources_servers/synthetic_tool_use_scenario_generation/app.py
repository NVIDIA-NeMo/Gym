# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resources server for synthetic tool-use scenario generation."""

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
from resources_servers.synthetic_tool_use_scenario_generation.assets import load_scenario_prompts
from resources_servers.synthetic_tool_use_scenario_generation.stage import ScenarioGenerationStage


class ScenarioGenerationResourcesServerConfig(BaseResourcesServerConfig):
    generation: SeedGenerationConfig
    model_server: ModelServerRef


class ScenarioGenerationResourcesServer(BaseResourcesServer, SimpleServer):
    config: ScenarioGenerationResourcesServerConfig

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
            role=generation.scenario_model,
        )
        await ScenarioGenerationStage(generation, load_scenario_prompts(), store, generator).run(
            resume=body.resume,
            source_indexes=store.select_source_indexes(body.domain_start, body.domain_end),
        )
        return StageGenerationResponse(report=store.write_generation_report())


if __name__ == "__main__":
    ScenarioGenerationResourcesServer.run_webserver()
