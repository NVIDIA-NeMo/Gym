# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Environment resources server."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from pydantic import Field

import resources_servers.swe_bench.harnesses  # noqa: F401
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import SandboxSpec
from resources_servers.swe_bench.harness import get_harness
from resources_servers.swe_bench.session import (
    EgressDescriptor,
    PlacementDescriptor,
    SandboxDescriptor,
    SessionDescriptor,
    SweBenchSeedSessionRequest,
    SweBenchVerifyRequest,
    SweBenchVerifyResponse,
)
from resources_servers.swe_bench.task import (
    ENVIRONMENT_NAME,
    SweTask,
    parse_submission,
    parse_task_from_request,
)
from resources_servers.swe_bench.verify_task import report_to_reward, verify_task


Topology = Literal["none", "env_sandboxed", "agent_in_env", "whole_interaction"]


class SweBenchResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_provider: dict[str, Any] = Field(default_factory=lambda: {"docker": {}})
    container_formatter: str = "swebench/sweb.eval.x86_64.{instance_id}"
    eval_timeout_s: float = 1800.0
    flat_eval: bool = True
    default_topology: Topology = "agent_in_env"


def _spec_to_dict(spec: SandboxSpec) -> dict[str, Any]:
    payload = dataclasses.asdict(spec)
    resources = payload.get("resources")
    if resources is not None and hasattr(resources, "__dataclass_fields__"):
        payload["resources"] = dataclasses.asdict(resources)
    return payload


class SweBenchResourcesServer(SimpleResourcesServer):
    config: SweBenchResourcesServerConfig

    def _parse_task(self, body: SweBenchSeedSessionRequest | SweBenchVerifyRequest) -> SweTask:
        return parse_task_from_request(
            body,
            container_formatter=self.config.container_formatter,
            flat_eval=self.config.flat_eval,
            environment=ENVIRONMENT_NAME,
        )

    async def seed_session(self, body: SweBenchSeedSessionRequest) -> SessionDescriptor:
        task = self._parse_task(body)
        harness = get_harness(task.harness_family)
        if self.config.flat_eval and hasattr(harness, "with_flat_eval"):
            harness = harness.with_flat_eval()
        spec = harness.build_spec(task)

        verifier_metadata = task.privileged_verifier_metadata(flat_eval=self.config.flat_eval)
        if body.verifier_metadata:
            verifier_metadata = {**body.verifier_metadata, **verifier_metadata}

        return SessionDescriptor(
            environment=ENVIRONMENT_NAME,
            task=task.public_view(environment=ENVIRONMENT_NAME),
            placement=PlacementDescriptor(topology=self.config.default_topology),
            sandbox=SandboxDescriptor(spec=_spec_to_dict(spec)),
            egress=EgressDescriptor(env={}),
            verifier_metadata=verifier_metadata,
        )

    async def verify(self, body: SweBenchVerifyRequest) -> SweBenchVerifyResponse:
        task = self._parse_task(body)
        task = task.with_submission(parse_submission(body.verifier_metadata))

        report = await verify_task(
            self.config.sandbox_provider,
            task,
            eval_timeout_s=self.config.eval_timeout_s,
        )
        reward = report_to_reward(report)
        masked = report.error_kind is not None

        return SweBenchVerifyResponse(
            **body.model_dump(),
            task_id=task.task_id,
            environment=ENVIRONMENT_NAME,
            reward=reward,
            resolved=report.resolved,
            patch_exists=report.patch_exists,
            mask_sample=masked,
            error_kind=report.error_kind,
        )


if __name__ == "__main__":
    SweBenchResourcesServer.run_webserver()
