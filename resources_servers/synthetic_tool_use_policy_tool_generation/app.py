# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym resources server for synthetic tool-use policy and tool generation."""

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
from resources_servers.synthetic_tool_use_policy_tool_generation.profiles import load_profile
from resources_servers.synthetic_tool_use_policy_tool_generation.stage import PolicyToolsGenerationStage


class PolicyToolGenerationResourcesServerConfig(BaseResourcesServerConfig):
    generation: SeedGenerationConfig
    model_server: ModelServerRef
    judge_model_server: ModelServerRef | None = None


class PolicyToolGenerationResourcesServer(BaseResourcesServer, SimpleServer):
    config: PolicyToolGenerationResourcesServerConfig

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
            role=generation.policy_tools_model,
        )

        judge = None
        if generation.policy_tools.judge_enabled:
            if generation.judge_model is None or self.config.judge_model_server is None:
                raise ValueError("judge_model and judge_model_server are required when judging is enabled")
            judge = GymModelGenerator(
                server_client=self.server_client,
                model_server=self.config.judge_model_server,
                role=generation.judge_model,
            )

        await PolicyToolsGenerationStage(
            generation,
            load_profile(generation.generation_profile),
            store,
            generator,
            judge,
        ).run(
            resume=body.resume,
            source_indexes=store.select_source_indexes(body.domain_start, body.domain_end),
        )
        return StageGenerationResponse(report=store.write_generation_report())


if __name__ == "__main__":
    PolicyToolGenerationResourcesServer.run_webserver()
