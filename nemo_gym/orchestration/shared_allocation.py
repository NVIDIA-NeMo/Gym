# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The explicit contract for several pinned drivers sharing one Slurm allocation."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.orchestration.api import (
    GymInstallConfig,
    JobConfig,
    OtelConfig,
    SlurmComputeConfig,
    SubmitConfig,
    VllmNodeServiceConfig,
)
from nemo_gym.orchestration.executors.script_templates import bash_var


class PolicyAdmissionConfig(BaseModel):
    """One admission service, independent of the member drivers' Gym installations."""

    model_config = ConfigDict(extra="forbid")
    container: str
    gym_install: GymInstallConfig
    max_inflight: int = Field(gt=0)
    max_queued: int = Field(ge=0)
    port: int = Field(default=18001, ge=1024, le=65535)
    health_check_timeout_seconds: int = Field(default=1800, ge=5)


def _run_keys(settings: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key, value in settings.items():
        key = prefix + key.lstrip("+")
        if isinstance(value, dict):
            keys.update(_run_keys(value, key + "."))
        else:
            keys.add(key)
    return keys


class SharedAllocationConfig(BaseModel):
    """A group of otherwise ordinary submission configs with identical shared infrastructure.

    Each member contains exactly one benchmark, named by its key. Pins, images, preparation, scoring,
    exports and environment remain per member. Shared infrastructure must match in full, including pool order.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["shared_allocation"]
    members: dict[str, SubmitConfig] = Field(min_length=2)
    job: JobConfig
    admission: PolicyAdmissionConfig
    first_driver_port: int = Field(default=20000, ge=1024, le=65535)
    driver_port_count: int = Field(default=400, ge=3)

    @property
    def common(self) -> SubmitConfig:
        return next(iter(self.members.values()))

    @property
    def otel(self) -> OtelConfig:
        return self.common.otel

    def member_ports(self, name: str) -> dict[str, int]:
        first = self.first_driver_port + list(self.members).index(name) * self.driver_port_count
        return {
            "head_server.port": first,
            "port_range_low": first + 1,
            "port_range_high": first + self.driver_port_count - 1,
        }

    @model_validator(mode="after")
    def validate_shared_contract(self) -> "SharedAllocationConfig":
        common = self.common
        compute = next(iter(common.compute.values()))
        if not isinstance(compute, SlurmComputeConfig):
            raise ValueError("shared_allocation requires Slurm")
        if compute.lustre_md_stats:
            raise ValueError("shared_allocation does not yet support lustre_md_stats")
        if self.job.export:
            raise ValueError("Put exports in each member's job or benchmark, not the shared job")
        if common.driver.policy_host != "localhost" or common.driver.policy_model is None:
            raise ValueError("shared_allocation requires driver.policy_model and policy_host: localhost")
        if not all(isinstance(service, VllmNodeServiceConfig) for service in common.services.values()):
            raise ValueError(
                "shared_allocation currently supports vllm services; Ray and disaggregated services are unsupported"
            )
        policy = common.services[common.driver.policy_model]
        if policy.server_per_node:
            raise ValueError("shared_allocation requires one policy endpoint, not server_per_node")
        last = self.first_driver_port + len(self.members) * self.driver_port_count - 1
        if last > 65535:
            raise ValueError("The shared allocation's driver port ranges exceed 65535")
        if any(bash_var(name) == bash_var("gym_admission") for name in common.services):
            raise ValueError("gym_admission is reserved for the shared policy admission service")
        reserved = {4317, 4318, 13133}
        for service in common.services.values():
            reserved.add(service.port)
            reserved.add(service.data_parallel_rpc_port)
            if service.health_check and service.health_check.port:
                reserved.add(service.health_check.port)
        if self.admission.port in reserved:
            raise ValueError("The admission port conflicts with a shared service")
        reserved.add(self.admission.port)
        if any(self.first_driver_port <= port <= last for port in reserved):
            raise ValueError("Driver port ranges overlap shared service ports")
        for name, member in self.members.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in set(common.services) | {
                "logs",
                "otel",
                "artifacts",
                "job.sh",
                "members.json",
                "eval.exit",
                "driver",
                "gym_admission",
                "otel_collector",
            }:
                raise ValueError(f"Invalid shared-allocation member name: {name!r}")
            if list(member.driver.benchmarks) != [name]:
                raise ValueError(f"Member {name!r} must contain exactly driver.benchmarks.{name}")
            if member.compute != common.compute or list(next(iter(member.compute.values())).node_pools) != list(
                compute.node_pools
            ):
                raise ValueError(f"Member {name!r} has different compute settings or node-pool order")
            if member.services != common.services or member.otel != common.otel:
                raise ValueError(f"Member {name!r} has different shared services or telemetry settings")
            if (member.driver.policy_model, member.driver.policy_host) != (
                common.driver.policy_model,
                common.driver.policy_host,
            ):
                raise ValueError(f"Member {name!r} uses a different policy placement")
            benchmark = member.driver.benchmarks[name]
            if benchmark.resume_config is not None:
                raise ValueError(
                    f"Member {name!r} requests resumable execution; grouped recovery is not yet supported. Use ordinary submission to retain it."
                )
            if benchmark.command is not None:
                raise ValueError(f"Member {name!r} uses command; the shared admission route requires gym eval run")
            keys = _run_keys(benchmark.run)
            if keys & {"head_server.port", "port_range_low", "port_range_high"}:
                raise ValueError(f"Member {name!r} sets ports owned by the shared allocation")
            if any(key.startswith("policy_model.") and key.endswith("base_url") for key in keys):
                raise ValueError(
                    f"Member {name!r} overrides the policy endpoint directly; use the policy_base_url binding"
                )
        return self
