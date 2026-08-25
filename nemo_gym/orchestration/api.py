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

import warnings
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, field_validator, model_validator


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
    # Pyxis-style bind mounts passed as --container-mounts.
    # Each entry is "src", "src:dst", or "src:dst:flags" (e.g. "/data:/data:ro").
    mounts: list[str] = []


class BaseModelServiceConfig(BaseServiceConfig):
    """Base for services that serve a model and can be wired as the policy model."""

    model: str
    port: int = 8000


class VllmServiceConfig(BaseModelServiceConfig):
    type: Literal["vllm"]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = False
    number_of_instances: int = 1

    @field_validator("number_of_instances")
    @classmethod
    def _validate_number_of_instances(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"number_of_instances must be >= 1, got {v}")
        return v

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
    # Pyxis-style bind mounts passed as --container-mounts.
    # Each entry is "src", "src:dst", or "src:dst:flags" (e.g. "/data:/data:ro").
    mounts: list[str] = []


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
        compute = self.compute[sole_compute]
        total_nodes = (
            sum(p.nodes for p in compute.node_pools.values()) if isinstance(compute, SlurmComputeConfig) else 1
        )
        is_multi_node = total_nodes > 1

        for service_name, service in self.services.items():
            if service.placement is None:
                service.placement = sole_compute
            elif service.placement not in compute_names:
                raise ValueError(
                    f"Service '{service_name}' placement '{service.placement}' does not match any compute resource "
                    f"({', '.join(sorted(compute_names))})."
                )

            if (
                is_multi_node
                and isinstance(service, VllmServiceConfig)
                and service.number_of_instances > 1
                and service.number_of_instances % total_nodes != 0
            ):
                raise ValueError(
                    f"Service '{service_name}' has number_of_instances={service.number_of_instances}, which must "
                    f"be evenly divisible by the number of nodes ({total_nodes}) for multi-node data-parallel "
                    "deployment - each node hosts an equal share of the data-parallel replicas."
                )

            if isinstance(service, VllmServiceConfig):
                self._validate_vllm_gpu_footprint(service_name, service, total_nodes)

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

    def _validate_vllm_gpu_footprint(self, service_name: str, service: "VllmServiceConfig", total_nodes: int) -> None:
        compute = self.compute[service.placement]
        if not isinstance(compute, SlurmComputeConfig):
            return

        gpus_per_node_values = [
            pool.gpus_per_node for pool in compute.node_pools.values() if pool.gpus_per_node is not None
        ]
        if not gpus_per_node_values:
            return

        max_gpus_per_node = max(gpus_per_node_values)
        tp_pp = service.tensor_parallel_size * service.pipeline_parallel_size

        if total_nodes > 1 and service.number_of_instances > 1:
            if tp_pp > max_gpus_per_node:
                # Each instance's own TP/PP footprint already exceeds a single node's GPU count, so
                # spreading multiple such instances across nodes would require every instance to
                # itself span multiple nodes. That's not supported: multi-node data-parallel only
                # distributes whole instances across nodes with tensor/pipeline parallelism kept
                # local to each node (see _build_vllm_multi_instance_multi_node_command).
                raise ValueError(
                    f"Service '{service_name}' sets number_of_instances={service.number_of_instances} with "
                    f"tensor_parallel_size={service.tensor_parallel_size} x "
                    f"pipeline_parallel_size={service.pipeline_parallel_size}={tp_pp}, which exceeds a single "
                    f"node's gpus_per_node ({max_gpus_per_node}). Multiple instances where each instance's own "
                    "tensor/pipeline-parallel footprint spans multiple nodes is not supported - reduce "
                    "tensor_parallel_size/pipeline_parallel_size to fit within one node, or set "
                    "number_of_instances=1 to let a single instance span nodes."
                )
            # Multi-node data-parallel: each node runs its own equal share of the replicas with
            # local tensor/pipeline parallelism (see _build_vllm_multi_instance_multi_node_command);
            # the per-node share, not the total footprint, has to fit in that node's GPU count.
            instances_per_node = service.number_of_instances // total_nodes
            gpus_needed = tp_pp * instances_per_node
            gpus_available = max_gpus_per_node
            footprint = (
                f"{instances_per_node} local replica(s) per node (number_of_instances="
                f"{service.number_of_instances} / {total_nodes} nodes) x tensor_parallel_size="
                f"{service.tensor_parallel_size} x pipeline_parallel_size={service.pipeline_parallel_size}"
            )
            scope = f"a single node's gpus_per_node ({max_gpus_per_node})"
        elif total_nodes > 1:
            # Single instance's TP/PP footprint spans the whole allocation via the ray backend.
            gpus_needed = tp_pp
            gpus_available = sum(
                pool.nodes * pool.gpus_per_node for pool in compute.node_pools.values() if pool.gpus_per_node
            )
            footprint = (
                f"tensor_parallel_size={service.tensor_parallel_size} x "
                f"pipeline_parallel_size={service.pipeline_parallel_size}"
            )
            scope = f"the total GPUs across all nodes ({gpus_available})"
        else:
            gpus_needed = tp_pp * service.number_of_instances
            gpus_available = max_gpus_per_node
            footprint = (
                f"tensor_parallel_size={service.tensor_parallel_size} x "
                f"pipeline_parallel_size={service.pipeline_parallel_size} x "
                f"number_of_instances={service.number_of_instances}"
            )
            scope = f"the node pool's gpus_per_node ({max_gpus_per_node})"

        if gpus_needed > gpus_available:
            raise ValueError(
                f"Service '{service_name}' requires {gpus_needed} GPUs ({footprint}), which exceeds {scope} "
                f"on compute '{service.placement}'. Reduce number_of_instances/tensor_parallel_size/"
                "pipeline_parallel_size, or add more nodes/GPUs."
            )
        elif gpus_needed < gpus_available:
            warnings.warn(
                f"Service '{service_name}' requires {gpus_needed} GPUs ({footprint}) but compute "
                f"'{service.placement}' provides {scope}, leaving {gpus_available - gpus_needed} GPU(s) idle. "
                "Increase number_of_instances/tensor_parallel_size or reduce gpus_per_node to use the full "
                "allocation.",
                stacklevel=2,
            )
