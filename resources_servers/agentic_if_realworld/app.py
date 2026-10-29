# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP configuration and routes for frozen Agentic IF verification."""

from pathlib import Path
from typing import ClassVar, Literal

from nemo_gym.base_resources_server import BaseResourcesServerConfig, ReverifyMode, SimpleResourcesServer
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentic_if_realworld.benchmark.core import (
    DEFAULT_JUDGE_MODEL,
    judge,
    summarize,
    summary_metrics,
)
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

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        """Keep the frozen full-cohort denominator for native, reverify and reaggregate runs."""
        if not self.config.synthetic:
            cached_preflight(self.config.split_registry_path, self.config.ground_truth_path, self.config.dataset_dir)
        dataset = cached_dataset(self.config.dataset_dir, self.config.synthetic)
        for row in body.verify_responses:
            identifier = row.get("item_id")
            if not isinstance(identifier, str) or identifier not in dataset.inputs or row.get("task_id") != identifier:
                raise ValueError("Aggregate result has an unknown or inconsistent frozen task identifier")
            if row.get("verifier_metadata") != {
                "item_id": identifier,
                "dataset_sha256": dataset.manifest_sha256,
                "input_sha256": judge.sha(dataset.request(identifier)),
            }:
                raise ValueError("Aggregate result provenance differs from frozen dataset")
            scored = row.get("scoring_status") == "scored"
            if (
                row.get("mask_sample") is not (not scored)
                or row.get("evaluation_completed", scored) is not scored
                or row.get("reward") != float(scored and row.get("item_verdict") == "pass")
            ):
                raise ValueError("Aggregate result has inconsistent grade, reward or masking")
        metrics = summary_metrics(summarize(body.verify_responses, dataset))
        aggregate = await super().aggregate_metrics(body)
        aggregate.agent_metrics.update(metrics)
        aggregate.key_metrics = metrics
        return aggregate


if __name__ == "__main__":
    ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ResourcesServer.run_webserver()  # noqa: F401
