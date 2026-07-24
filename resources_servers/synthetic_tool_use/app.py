# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym orchestration server for synthetic conversational tool-use generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import BaseResourcesServer, BaseResourcesServerConfig
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import SimpleServer, get_response_json, raise_for_status
from resources_servers.synthetic_tool_use.assets import generation_asset_hashes
from resources_servers.synthetic_tool_use.common.artifacts import RunArtifactStore, atomic_write_json
from resources_servers.synthetic_tool_use.common.models import (
    SeedGenerationConfig,
    StageGenerationRequest,
    StageGenerationResponse,
    StageState,
)
from resources_servers.synthetic_tool_use_simulation.scripts.build_synthetic_tool_use_dataset import (
    build_sample_dataset,
    validate_domain_static_artifacts,
)


GenerationStage = Literal["domains", "policy_tools", "scenarios"]
STAGE_ORDER: tuple[GenerationStage, ...] = ("domains", "policy_tools", "scenarios")


class SyntheticToolUsePipelineConfig(BaseResourcesServerConfig):
    generation: SeedGenerationConfig
    domain_generation_server: ResourcesServerRef
    policy_tool_generation_server: ResourcesServerRef
    scenario_generation_server: ResourcesServerRef


class PipelineGenerationRequest(StageGenerationRequest):
    stages: list[GenerationStage] = Field(default_factory=lambda: list(STAGE_ORDER), min_length=1)


class MaterializeRequest(BaseModel):
    dataset_name: str | None = None
    output_path: Path | None = None
    parallel_tool_calls: bool = False


class SyntheticToolUsePipelineServer(BaseResourcesServer, SimpleServer):
    config: SyntheticToolUsePipelineConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/generate")(self.generate)
        app.post("/validate")(self.validate)
        app.post("/materialize")(self.materialize)
        return app

    def _store(self) -> RunArtifactStore:
        generation = self.config.generation
        return RunArtifactStore.create(generation, generation_asset_hashes(generation.generation_profile))

    async def _run_stage(
        self,
        stage: GenerationStage,
        body: StageGenerationRequest,
    ) -> StageGenerationResponse:
        server_ref = {
            "domains": self.config.domain_generation_server,
            "policy_tools": self.config.policy_tool_generation_server,
            "scenarios": self.config.scenario_generation_server,
        }[stage]
        response = await self.server_client.post(
            server_name=server_ref.name,
            url_path="/generate",
            json=body,
        )
        await raise_for_status(response)
        return StageGenerationResponse.model_validate(await get_response_json(response))

    async def generate(self, body: PipelineGenerationRequest) -> StageGenerationResponse:
        requested = set(body.stages)
        stage_request = StageGenerationRequest(
            resume=body.resume,
            domain_start=body.domain_start,
            domain_end=body.domain_end,
        )
        for stage in STAGE_ORDER:
            if stage in requested:
                await self._run_stage(stage, stage_request)
        return StageGenerationResponse(report=self._store().write_generation_report())

    async def validate(self) -> dict[str, Any]:
        store = self._store()
        report: dict[str, Any] = {
            "domains_seen": 0,
            "domains_eligible": 0,
            "domains_incomplete": 0,
            "domains_valid": 0,
            "failures": [],
        }
        for entry in store.load_manifest().domains:
            report["domains_seen"] += 1
            if entry.stages["scenarios"].state != StageState.COMPLETE:
                report["domains_incomplete"] += 1
                continue
            report["domains_eligible"] += 1
            domain_dir = store.domains_dir / entry.artifact_dir
            try:
                validate_domain_static_artifacts(domain_dir)
                report["domains_valid"] += 1
            except Exception as exc:
                report["failures"].append(
                    {
                        "domain_id": entry.domain_id,
                        "source_index": entry.source_index,
                        "reason": getattr(exc, "reason", "validation_error"),
                        "detail": str(exc),
                    }
                )
        atomic_write_json(store.run_dir / "validation_report.json", report)
        return report

    async def materialize(self, body: MaterializeRequest) -> dict[str, Any]:
        validation = await self.validate()
        if validation["failures"]:
            raise ValueError(f"cannot materialize: {len(validation['failures'])} domains failed validation")
        if validation["domains_valid"] == 0:
            raise ValueError("cannot materialize: no completed domains passed validation")

        generation = self.config.generation
        store = self._store()
        dataset_name = body.dataset_name or generation.source_name
        output_path = body.output_path or store.run_dir / f"{dataset_name}.jsonl"
        return build_sample_dataset(
            source_dirs=[store.domains_dir],
            output_path=output_path,
            report_path=output_path.with_suffix(".report.json"),
            max_rows=None,
            dataset_name=dataset_name,
            source_names=[generation.source_name],
            max_rows_per_domain=None,
            scan_domains_per_source=None,
            domain_generator_model=generation.domain_model.model,
            policy_tools_model=generation.policy_tools_model.model,
            scenario_generator_model=generation.scenario_model.model,
            parallel_tool_calls=body.parallel_tool_calls,
        )


if __name__ == "__main__":
    SyntheticToolUsePipelineServer.run_webserver()
