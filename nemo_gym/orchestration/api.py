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

import os
import re
import warnings
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, field_validator, model_validator


# Reject unknown fields on all config models so typos in YAML surface immediately.
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Canonical marker left on a resolved `env` value for `runtime:VAR` entries. Executors
# (e.g. slurm_script.py) detect this prefix and emit an unquoted shell reference instead
# of a literal, so the value is picked up from the job's actual environment at run time.
RUNTIME_ENV_PREFIX = "runtime:"


def resolve_env_dict(env: dict[str, str]) -> dict[str, str]:
    """Resolve `lit:`/`host:`/`runtime:` prefixes on `env` values. Every value must use one
    of these prefixes; a missing or misspelled prefix raises rather than being guessed at.

    - `lit:VALUE` -> literal VALUE.
    - `host:VAR` -> read from os.environ[VAR] on the machine running `gym eval submit`;
      raises if VAR isn't set there.
    - `runtime:VAR` -> left unresolved; canonicalized to `runtime:VAR` for executors to
      pick up and reference from the job's own environment at run time.
    """
    resolved = {}
    for key, raw in env.items():
        if raw.startswith("lit:"):
            resolved[key] = raw[len("lit:") :]
        elif raw.startswith("host:"):
            var = raw[len("host:") :]
            if not _ENV_VAR_NAME_RE.match(var):
                raise ValueError(f"env[{key!r}]: {var!r} is not a valid environment variable name for host:{var}")
            value = os.environ.get(var)
            if value is None:
                raise ValueError(
                    f"env[{key!r}] references host:{var}, but {var!r} is not set in the submitting shell's environment"
                )
            resolved[key] = value
        elif raw.startswith(RUNTIME_ENV_PREFIX):
            var = raw[len(RUNTIME_ENV_PREFIX) :]
            if not _ENV_VAR_NAME_RE.match(var):
                raise ValueError(f"env[{key!r}]: {var!r} is not a valid environment variable name for runtime:{var}")
            resolved[key] = f"{RUNTIME_ENV_PREFIX}{var}"
        else:
            raise ValueError(
                f"env[{key!r}]: {raw!r} must start with one of the prefixes 'lit:', 'host:', or 'runtime:'"
            )
    return resolved


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
    # Values may be prefixed `lit:` (literal), `host:VAR` (read from the submitting
    # machine's env), or `runtime:VAR` (resolved from the job's own env at run time).
    # Every value must use one of these prefixes. See resolve_env_dict.
    env: dict[str, str] = {}
    # Pyxis-style bind mounts passed as --container-mounts.
    # Each entry is "src", "src:dst", or "src:dst:flags" (e.g. "/data:/data:ro").
    mounts: list[str] = []
    # Raw shell statements run before the service command starts, in the same
    # shell (so export/unset and dynamic values like $(hostname -I) work
    # normally) -- e.g. working around an image or engine-version bug that
    # needs an env var set to a real address or a stale one unset before the
    # service binary runs. Unlike `env` (literal key=value pairs only) or a
    # service-specific extra_args (appended to that service's own command
    # line), this runs as its own statement(s) ahead of the command.
    pre_command: str = ""

    @field_validator("env")
    @classmethod
    def _resolve_env_prefixes(cls, v: dict[str, str]) -> dict[str, str]:
        return resolve_env_dict(v)


class BaseModelServiceConfig(BaseServiceConfig):
    """Base for services that serve a model and can be wired as the policy model."""

    model: str
    port: int = 8000
    served_model_name: str | None = None


class VllmServiceConfig(BaseModelServiceConfig):
    type: Literal["vllm"]
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = False
    number_of_instances: int = 1
    use_ray_serve: bool = False
    # Raw extra flags appended verbatim to `vllm serve` (e.g. "--max-model-len 8192").
    extra_args: str = ""

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


def effective_ray_serve(service: "VllmServiceConfig", total_nodes: int, gpus_per_node_values: list[int]) -> bool:
    """Whether the Ray Serve gateway manages this service's instances/routing instead of vLLM's own DP."""
    if service.use_ray_serve:
        return True
    if not gpus_per_node_values:
        return False
    max_gpus_per_node = max(gpus_per_node_values)
    tp_pp = service.tensor_parallel_size * service.pipeline_parallel_size
    return total_nodes > 1 and service.number_of_instances > 1 and tp_pp > max_gpus_per_node


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
    # Which responses_api_models asset serves as the policy, passed as
    # `--model-type`. Not every benchmark wants the same one: Gym permits exactly
    # one entry under `policy_model.responses_api_models`, so composing
    # openai_model against a benchmark that ships its own vllm_model policy (e.g.
    # lmarena_v3) fails validation with "Dictionary should have at most 1 item
    # after validation, not 2", and overrides keyed on `vllm_model.*` land on a
    # server that was never composed. Set to "" to compose no policy model config
    # at all, for a benchmark whose own config already declares a complete one.
    policy_model_type: str = "openai_model"
    benchmarks: dict[str, BenchmarkRunConfig]
    # Values may be prefixed `lit:` (literal), `host:VAR` (read from the submitting
    # machine's env), or `runtime:VAR` (resolved from the job's own env at run time).
    # Every value must use one of these prefixes. See resolve_env_dict.
    env: dict[str, str] = {}
    # Pyxis-style bind mounts passed as --container-mounts.
    # Each entry is "src", "src:dst", or "src:dst:flags" (e.g. "/data:/data:ro").
    mounts: list[str] = []

    @field_validator("env")
    @classmethod
    def _resolve_env_prefixes(cls, v: dict[str, str]) -> dict[str, str]:
        return resolve_env_dict(v)


class JobConfig(_StrictModel):
    # Remote base directory. Each submit creates a timestamped subdirectory here.
    output_path: str


class ObservabilityConfig(_StrictModel):
    """An OpenTelemetry collector beside every benchmark job: scrapes each model service's
    Prometheus `/metrics`, receives OTLP from the job's own processes on :4317/:4318, and ships
    both to an OTLP/HTTP backend while keeping a copy under `<job dir>/otel/`. On by default, so
    a run is observable unless it opts out; `endpoint` and `service_name` come from the
    deployment's own config (a cluster fragment, typically) and are required while enabled."""

    enabled: bool = True
    # Collector binary: a path on the compute nodes (the release tarball's static `otelcol-contrib`
    # on shared storage) when `container` is unset, else a path inside `container`.
    binary: str = "otelcol-contrib"
    # Optional image for the collector step. Unset runs the binary directly on the node, which is
    # what enroot-based clusters need: the upstream collector image is distroless, and enroot
    # cannot start a container without /bin/sh.
    container: str | None = None
    # OTLP/HTTP ingest base URL (`/v1/metrics` etc. are appended by the exporter).
    endpoint: str | None = None
    # Env var holding the ingest bearer token on the machine running `gym eval submit`. Read at
    # submit time and forwarded into the job's environment; never written into the job directory.
    token_env: str = "OBSERVABILITY_TOKEN"
    # Sent as the `service.name` resource attribute: the identity the backend routes the token by.
    service_name: str | None = None
    # Display identity of the scraped metrics in the backend (`service.name.override`).
    component: str = "gym-vllm"
    # Gym span groups to switch on (a preset or comma-separated names). `default` gives one trace
    # per rollout and the rollout duration metric; `verify` adds the verify duration and success
    # metrics, which are gated on that group.
    gym_span_groups: str = "default,verify"
    # Ship Gym's Python logging as OTel logs too (trace-correlated), through the same collector.
    gym_logs: bool = True
    # Node-level exporters that clusters commonly run as system services on every compute node;
    # scraped on localhost when set, skipped when null. DCGM gives per-GPU activity/memory/power,
    # node_exporter gives CPU/memory/network/disk. A closed port only logs scrape errors.
    gpu_metrics_port: int | None = 9400
    node_metrics_port: int | None = 9100
    scrape_interval_seconds: int = 15
    health_check_timeout_seconds: int = 300

    @field_validator("token_env")
    @classmethod
    def _validate_token_env(cls, v: str) -> str:
        if not _ENV_VAR_NAME_RE.match(v):
            raise ValueError(f"observability.token_env: {v!r} is not a valid environment variable name")
        return v

    @field_validator("scrape_interval_seconds", "health_check_timeout_seconds")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v


class SubmitConfig(_StrictModel):
    services: dict[str, ServiceConfig]
    compute: dict[str, ComputeConfig]
    driver: DriverConfig
    job: JobConfig
    observability: ObservabilityConfig = ObservabilityConfig()

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
        gpus_per_node_values = (
            [p.gpus_per_node for p in compute.node_pools.values() if p.gpus_per_node is not None]
            if isinstance(compute, SlurmComputeConfig)
            else []
        )

        for service_name, service in self.services.items():
            if service.placement is None:
                service.placement = sole_compute
            elif service.placement not in compute_names:
                raise ValueError(
                    f"Service '{service_name}' placement '{service.placement}' does not match any compute resource "
                    f"({', '.join(sorted(compute_names))})."
                )

            if not isinstance(service, VllmServiceConfig):
                continue

            is_ray_serve = effective_ray_serve(service, total_nodes, gpus_per_node_values)

            if (
                is_multi_node
                and service.number_of_instances > 1
                and service.number_of_instances % total_nodes != 0
                and not is_ray_serve
            ):
                raise ValueError(
                    f"Service '{service_name}' has number_of_instances={service.number_of_instances}, which must "
                    f"be evenly divisible by the number of nodes ({total_nodes}) for multi-node data-parallel "
                    "deployment - each node hosts an equal share of the data-parallel replicas."
                )

            self._validate_vllm_gpu_footprint(
                service_name, service, total_nodes, compute, gpus_per_node_values, is_ray_serve
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
                    benchmark.run["policy_model_name"] = service.served_model_name or service.model
                    # vLLM doesn't require auth; dummy key satisfies clients that require the header.
                    benchmark.run["policy_api_key"] = "dummy"  # pragma: allowlist secret

        return self

    def _validate_vllm_gpu_footprint(
        self,
        service_name: str,
        service: "VllmServiceConfig",
        total_nodes: int,
        compute: "SlurmComputeConfig",
        gpus_per_node_values: list[int],
        is_ray_serve: bool,
    ) -> None:
        if not gpus_per_node_values:
            return

        max_gpus_per_node = max(gpus_per_node_values)
        tp_pp = service.tensor_parallel_size * service.pipeline_parallel_size

        if total_nodes > 1 and is_ray_serve:
            # Ray Serve's placement-group scheduler packs the aggregate footprint across the cluster.
            gpus_needed = tp_pp * service.number_of_instances
            gpus_available = sum(
                pool.nodes * pool.gpus_per_node for pool in compute.node_pools.values() if pool.gpus_per_node
            )
            footprint = (
                f"tensor_parallel_size={service.tensor_parallel_size} x "
                f"pipeline_parallel_size={service.pipeline_parallel_size} x "
                f"number_of_instances={service.number_of_instances} (ray_serve gateway)"
            )
            scope = f"the total GPUs across all nodes ({gpus_available})"
        elif total_nodes > 1 and service.number_of_instances > 1:
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
