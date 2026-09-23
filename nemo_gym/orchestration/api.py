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
    # Name of a node pool in the placed compute's `node_pools`. Pins this service to
    # that pool's slice of the allocation instead of letting it land wherever srun
    # starts, which is how two services get nodes of their own -- a scorer or judge
    # that cannot share a GPU with the policy, or a prefill/decode split. Pools take
    # contiguous node ranges in declaration order. None means the whole allocation.
    node_pool: str | None = None
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
    # Marks this service as one tier of a prefill/decode disaggregated deployment.
    # "producer" computes prefill and hands its KV cache off; "consumer" receives that
    # cache and decodes. A router service (type: router) fronts the pair. Unset means
    # an ordinary self-contained vLLM deployment, which is what most configs are.
    kv_role: Literal["producer", "consumer"] | None = None
    # The vLLM KV connector moving cache between the tiers.
    kv_connector: str = "NixlConnector"
    # What vLLM does when a KV transfer fails. "fail" surfaces the error rather than
    # silently recomputing, which would read as a slow run instead of a broken one.
    kv_load_failure_policy: str = "fail"
    # NIXL's side-channel port. Each tier needs its own, since both run on nodes of
    # the same allocation and the port is bound per host.
    nixl_side_channel_port: int = 5600
    # Port the data-parallel ranks of this service coordinate on. Two tiers sharing an
    # allocation need different ones, the way they need different side-channel ports.
    data_parallel_rpc_port: int = 13345

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


class RouterServiceConfig(BaseModelServiceConfig):
    """vllm-router fronting a prefill/decode pair.

    This is the address clients use: `driver.policy_model` names the router, not
    either tier, and the router forwards each phase to the tier that owns it.
    """

    type: Literal["router"]
    # Names of the two vLLM services this router fronts. They must carry kv_role
    # "producer" and "consumer" respectively.
    prefill: str
    decode: str
    prefill_policy: str = "cache_aware"
    decode_policy: str = "cache_aware"
    intra_node_data_parallel_size: int = 1
    # An agentic benchmark holds a request open for a long time; the router must not
    # be the thing that gives up on it.
    request_timeout_secs: int = 86400
    log_level: str = "error"

    @model_validator(mode="after")
    def _default_health_check(self) -> "RouterServiceConfig":
        if self.health_check is None:
            self.health_check = HealthCheckConfig(port=self.port)
        elif self.health_check.port is None:
            self.health_check.port = self.port
        return self


class RayServiceConfig(BaseServiceConfig):
    type: Literal["ray"]
    # "head" starts a cluster; "worker" joins the one at `address`. A worker is how a
    # second node joins the driver's Ray cluster and offers its GPUs to actors the
    # benchmark schedules (e.g. a scorer that runs off the policy's node).
    mode: Literal["head", "worker"] = "head"
    # Required for mode="worker": the head's host:port.
    address: str | None = None
    # Head only; ignored by a worker, which takes the port from `address`.
    port: int = 6379
    # Custom Ray resources this node advertises, e.g. {"extra_gpu": 4}. A benchmark
    # asks for these by name rather than by num_gpus when it manages device placement
    # itself.
    resources: dict[str, float] = {}
    num_cpus: int | None = None
    num_gpus: int | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "RayServiceConfig":
        if self.mode == "worker" and not self.address:
            raise ValueError("A ray service with mode='worker' needs `address` set to the head's host:port.")
        if self.mode == "head" and self.address:
            raise ValueError("A ray service with mode='head' starts its own cluster; remove `address`.")
        return self


# Discriminated union keyed on `type`; Pydantic rejects unknown type values at parse time.
ServiceConfig = Annotated[
    Annotated[VllmServiceConfig, Tag("vllm")]
    | Annotated[RayServiceConfig, Tag("ray")]
    | Annotated[RouterServiceConfig, Tag("router")],
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

        pool_names = set(compute.node_pools) if isinstance(compute, SlurmComputeConfig) else set()

        for service_name, service in self.services.items():
            if service.placement is None:
                service.placement = sole_compute
            elif service.placement not in compute_names:
                raise ValueError(
                    f"Service '{service_name}' placement '{service.placement}' does not match any compute resource "
                    f"({', '.join(sorted(compute_names))})."
                )

            if service.node_pool is not None and service.node_pool not in pool_names:
                raise ValueError(
                    f"Service '{service_name}' node_pool '{service.node_pool}' does not match any node pool of "
                    f"compute '{service.placement}' ({', '.join(sorted(pool_names)) or 'none declared'})."
                )

            if isinstance(service, VllmServiceConfig) and service.kv_role is not None and service.node_pool is None:
                raise ValueError(
                    f"Service '{service_name}' sets kv_role='{service.kv_role}' but no node_pool. A prefill/decode "
                    "tier needs nodes of its own: the two tiers run side by side and the router addresses each "
                    "tier's head by its pool."
                )

            if not isinstance(service, VllmServiceConfig):
                continue

            # A pinned service is sized against its own pool, not the whole job: one node of a
            # ten-node allocation is a single-node deployment with that pool's GPUs, and judging
            # it by the allocation total both mis-builds the command and mis-reports idle GPUs.
            service_pools = (
                {service.node_pool: compute.node_pools[service.node_pool]}
                if service.node_pool is not None and isinstance(compute, SlurmComputeConfig)
                else (compute.node_pools if isinstance(compute, SlurmComputeConfig) else {})
            )
            service_nodes = sum(p.nodes for p in service_pools.values()) or total_nodes
            service_gpus = [p.gpus_per_node for p in service_pools.values() if p.gpus_per_node is not None]

            is_ray_serve = effective_ray_serve(service, service_nodes, service_gpus)

            if (
                service_nodes > 1
                and service.number_of_instances > 1
                and service.number_of_instances % service_nodes != 0
                and not is_ray_serve
            ):
                raise ValueError(
                    f"Service '{service_name}' has number_of_instances={service.number_of_instances}, which must "
                    f"be evenly divisible by the number of nodes ({service_nodes}) for multi-node data-parallel "
                    "deployment - each node hosts an equal share of the data-parallel replicas."
                )

            self._validate_vllm_gpu_footprint(
                service_name, service, service_nodes, service_pools, service_gpus, is_ray_serve
            )

        self._validate_routers(compute)

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

    def _validate_routers(self, compute: "ComputeConfig") -> None:
        """Check that every router fronts a real prefill/decode pair.

        A router that names a missing or mis-roled service produces a script that
        starts, serves nothing, and fails as a timeout much later.
        """
        pools = list(compute.node_pools) if isinstance(compute, SlurmComputeConfig) else []

        for name, router in self.services.items():
            if not isinstance(router, RouterServiceConfig):
                continue

            for field, expected in (("prefill", "producer"), ("decode", "consumer")):
                tier_name = getattr(router, field)
                tier = self.services.get(tier_name)
                if tier is None:
                    raise ValueError(
                        f"Router '{name}' names {field} service '{tier_name}', which is not in services "
                        f"({', '.join(sorted(self.services))})."
                    )
                if not isinstance(tier, VllmServiceConfig):
                    raise ValueError(
                        f"Router '{name}' names {field} service '{tier_name}', which is a "
                        f"'{tier.type}' service; a router fronts vllm services."
                    )
                if tier.kv_role != expected:
                    raise ValueError(
                        f"Router '{name}' names {field} service '{tier_name}', whose kv_role is "
                        f"{tier.kv_role!r}; it has to be '{expected}' to serve as the {field} tier."
                    )

            prefill = self.services[router.prefill]
            decode = self.services[router.decode]
            assert isinstance(prefill, VllmServiceConfig) and isinstance(decode, VllmServiceConfig)
            for field, value in (
                ("nixl_side_channel_port", prefill.nixl_side_channel_port == decode.nixl_side_channel_port),
                ("data_parallel_rpc_port", prefill.data_parallel_rpc_port == decode.data_parallel_rpc_port),
            ):
                if value:
                    raise ValueError(
                        f"Router '{name}': prefill '{router.prefill}' and decode '{router.decode}' share "
                        f"{field} {getattr(prefill, field)}. Each tier binds the port on its own hosts, so the "
                        "two tiers need different ones."
                    )

            # driver.policy_model points clients at http://localhost:<port>, and the
            # driver runs on the allocation's first node. A router anywhere else is
            # reachable by nothing.
            if router.node_pool is not None and pools and router.node_pool != pools[0]:
                raise ValueError(
                    f"Router '{name}' is pinned to node_pool '{router.node_pool}', but the driver reaches it over "
                    f"localhost and runs on the first node. Pin it to '{pools[0]}' or leave node_pool unset."
                )

    def _validate_vllm_gpu_footprint(
        self,
        service_name: str,
        service: "VllmServiceConfig",
        total_nodes: int,
        node_pools: dict[str, "NodePool"],
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
            gpus_available = sum(pool.nodes * pool.gpus_per_node for pool in node_pools.values() if pool.gpus_per_node)
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
            gpus_available = sum(pool.nodes * pool.gpus_per_node for pool in node_pools.values() if pool.gpus_per_node)
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
