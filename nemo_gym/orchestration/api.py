# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, model_validator


# Reject unknown fields on all config models so typos in YAML surface immediately.
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthCheckConfig(_StrictModel):
    path: str = "/health"
    # port defaults to None so VllmServiceConfig can fill it from service.port when omitted.
    port: int | None = None
    timeout_seconds: int = 60


class BaseServiceConfig(_StrictModel):
    container: str
    # Resolved to the sole compute resource name at validation time when not set.
    placement: str | None = None
    health_check: HealthCheckConfig | None = None
    env: dict[str, str] = {}


class BaseModelServiceConfig(BaseServiceConfig):
    """Base for services that serve a model and can be wired as the policy model."""

    model: str
    port: int = 8000


class VllmServiceConfig(BaseModelServiceConfig):
    type: Literal["vllm"]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def _default_health_check(self) -> "VllmServiceConfig":
        # vLLM always exposes /health on its serving port; set it automatically
        # so the sbatch script gets a health check without the user having to repeat the port.
        if self.health_check is None:
            self.health_check = HealthCheckConfig(port=self.port)
        elif self.health_check.port is None:
            self.health_check.port = self.port
        return self


class RayServiceConfig(BaseServiceConfig):
    type: Literal["ray"]


# Discriminated union keyed on `type`; Pydantic rejects unknown type values at parse time.
ServiceConfig = Annotated[
    Annotated[VllmServiceConfig, Tag("vllm")] | Annotated[RayServiceConfig, Tag("ray")],
    Discriminator("type"),
]


class NodePool(_StrictModel):
    partition: str
    nodes: int = 1
    ntasks_per_node: int = 1
    # Structured field the executor uses for smart deployment decisions (e.g. multi-instance vLLM).
    gpus_per_node: int | None = None
    # Arbitrary #SBATCH directives forwarded verbatim for options we don't model explicitly.
    extra_args: dict[str, str] = {}


class BaseComputeConfig(_StrictModel):
    pass


class SlurmComputeConfig(BaseComputeConfig):
    type: Literal["slurm"]
    account: str
    hostname: str | None = None  # None means we're already on the login node; skip SSH.
    walltime: str | None = None
    node_pools: dict[str, NodePool] = {}
    extra_args: dict[str, str] = {}  # Job-level #SBATCH directives (e.g. --comment, --mail-user).


ComputeConfig = Annotated[
    Annotated[SlurmComputeConfig, Tag("slurm")],
    Discriminator("type"),
]


class BenchmarkRunConfig(_StrictModel):
    # Hydra overrides forwarded to `gym eval prepare`. Flattened to +key=value tokens.
    prepare: dict[str, Any] = {}
    # Hydra overrides forwarded to `gym eval run`. policy_model wiring is injected here at
    # validation time so all executors see it uniformly via flatten_run_args.
    run: dict[str, Any] = {}


class GymInstallConfig(_StrictModel):
    repo: str = "https://github.com/NVIDIA-NeMo/gym"
    ref: str  # Git tag or commit hash.


class DriverConfig(_StrictModel):
    container: str = "python:3.12"
    gym_install: GymInstallConfig | None = None
    # Name of a service in `services:` to use as the policy model. When set, injects
    # policy_base_url/policy_model_name/policy_api_key into each benchmark's run config.
    policy_model: str | None = None
    benchmarks: dict[str, BenchmarkRunConfig]
    env: dict[str, str] = {}


class JobConfig(_StrictModel):
    # Remote base directory. Each submit creates a timestamped subdirectory here.
    output_path: str


class SubmitConfig(_StrictModel):
    services: dict[str, ServiceConfig]
    compute: dict[str, ComputeConfig]
    driver: DriverConfig
    job: JobConfig

    @model_validator(mode="after")
    def _resolve_and_validate_placements(self) -> "SubmitConfig":
        compute_names = set(self.compute)

        if len(compute_names) > 1:
            raise ValueError(f"Multiple compute resources are not supported yet ({', '.join(sorted(compute_names))}).")

        sole_compute = next(iter(compute_names))

        for service_name, service in self.services.items():
            if service.placement is None:
                service.placement = sole_compute
            elif service.placement not in compute_names:
                raise ValueError(
                    f"Service '{service_name}' placement '{service.placement}' does not match any compute resource "
                    f"({', '.join(sorted(compute_names))})."
                )

        if self.driver.policy_model is not None:
            if self.driver.policy_model not in self.services:
                raise ValueError(
                    f"driver.policy_model '{self.driver.policy_model}' does not match any service "
                    f"({', '.join(sorted(self.services))})."
                )
            service = self.services[self.driver.policy_model]
            if isinstance(service, BaseModelServiceConfig):
                for bench_name, benchmark in self.driver.benchmarks.items():
                    conflicts = [
                        k for k in ("policy_base_url", "policy_model_name", "policy_api_key") if k in benchmark.run
                    ]
                    if conflicts:
                        raise ValueError(
                            f"Benchmark '{bench_name}' run config already sets {conflicts} "
                            f"but driver.policy_model is also set. Remove one."
                        )
                    benchmark.run["policy_base_url"] = f"http://localhost:{service.port}/v1"
                    benchmark.run["policy_model_name"] = service.model
                    # vLLM doesn't require auth; dummy key satisfies clients that require the header.
                    benchmark.run["policy_api_key"] = "dummy"  # pragma: allowlist secret

        return self
