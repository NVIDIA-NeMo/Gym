# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Environment resources server."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

import resources_servers.swe_bench.harnesses  # noqa: F401
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import SandboxSpec
from resources_servers.swe_bench.harness import get_harness
from resources_servers.swe_bench.task_builder import build_swetask, problem_info_from_row
from resources_servers.swe_bench.verify_task import report_to_reward, verify_task


Topology = Literal["none", "env_sandboxed", "agent_in_env", "whole_interaction"]


class SweBenchResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_provider: dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    container_formatter: str = "swebench/sweb.eval.x86_64.{instance_id}"
    eval_timeout_s: float = 1800.0
    flat_eval: bool = True
    default_topology: Topology = "agent_in_env"


class PlacementDescriptor(BaseModel):
    topology: Topology


class SandboxDescriptor(BaseModel):
    spec: dict[str, Any]


class EgressDescriptor(BaseModel):
    env: dict[str, str] = Field(default_factory=dict)


class SweBenchSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[dict[str, Any]] = None


class SweBenchSeedSessionResponse(BaseSeedSessionResponse):
    placement: PlacementDescriptor
    sandbox: SandboxDescriptor
    egress: EgressDescriptor
    verifier_metadata: dict[str, Any]


class SweBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[dict[str, Any]] = None


class SweBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    resolved: bool = False
    patch_exists: bool = False
    mask_sample: bool = False
    error_kind: Optional[str] = None


def _spec_to_dict(spec: SandboxSpec) -> dict[str, Any]:
    payload = dataclasses.asdict(spec)
    resources = payload.get("resources")
    if resources is not None and hasattr(resources, "__dataclass_fields__"):
        payload["resources"] = dataclasses.asdict(resources)
    return payload


class SweBenchResourcesServer(SimpleResourcesServer):
    config: SweBenchResourcesServerConfig

    def _task_from_body(self, body: SweBenchSeedSessionRequest | SweBenchVerifyRequest) -> tuple[dict[str, Any], Any]:
        responses_metadata = (body.responses_create_params.metadata or {}) if body.responses_create_params else {}
        verifier_metadata = dict(body.verifier_metadata or {})
        problem_info = problem_info_from_row(verifier_metadata, responses_metadata)
        task = build_swetask(
            problem_info,
            container_formatter=self.config.container_formatter,
            flat_eval=self.config.flat_eval,
        )
        return problem_info, task

    async def seed_session(self, body: SweBenchSeedSessionRequest) -> SweBenchSeedSessionResponse:
        problem_info, task = self._task_from_body(body)
        harness = get_harness(task.benchmark)
        if self.config.flat_eval and hasattr(harness, "with_flat_eval"):
            harness = harness.with_flat_eval()
        spec = harness.build_spec(task)

        verifier_metadata = {
            **problem_info,
            "instance_id": task.instance_id,
            "dataset_name": task.metadata.get("dataset_name", problem_info.get("dataset_name", "")),
            "split": task.split,
            "benchmark": task.benchmark,
            "flat_eval": self.config.flat_eval,
        }

        return SweBenchSeedSessionResponse(
            placement=PlacementDescriptor(topology=self.config.default_topology),
            sandbox=SandboxDescriptor(spec=_spec_to_dict(spec)),
            egress=EgressDescriptor(env={}),
            verifier_metadata=verifier_metadata,
        )

    async def verify(self, body: SweBenchVerifyRequest) -> SweBenchVerifyResponse:
        _, task = self._task_from_body(body)
        verifier_metadata = dict(body.verifier_metadata or {})
        model_patch = verifier_metadata.get("model_patch") or verifier_metadata.get("git_patch") or ""
        task = dataclasses.replace(task, model_patch=model_patch)

        report = await verify_task(
            self.config.sandbox_provider,
            task,
            eval_timeout_s=self.config.eval_timeout_s,
        )
        reward = report_to_reward(report)
        masked = report.error_kind is not None

        return SweBenchVerifyResponse(
            **body.model_dump(),
            reward=reward,
            resolved=report.resolved,
            patch_exists=report.patch_exists,
            mask_sample=masked,
            error_kind=report.error_kind,
        )


if __name__ == "__main__":
    SweBenchResourcesServer.run_webserver()
