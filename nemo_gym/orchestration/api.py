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

from pydantic import BaseModel, ConfigDict, Discriminator, PrivateAttr, Tag, field_validator, model_validator


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
    # Port the data-parallel ranks of a multi-node service coordinate on.
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


class VllmPDTierConfig(VllmServiceConfig):
    """One tier of a `vllm_pd` service as deployed. Built by the service, never written by a user."""

    nixl_side_channel_port: int
    server_per_node: bool
    kv_transfer_config: dict[str, str]


# Set on the vllm_pd service, never on a tier: the service derives each tier's value.
_PD_SERVICE_ONLY_FIELDS = ("server_per_node", "nixl_side_channel_port")
_DECODE_NIXL_PORT_OFFSET = 1


class VllmPDServiceConfig(BaseModelServiceConfig):
    """Prefill/decode disaggregated vLLM: two tiers behind a vllm-router.

    Deployed as three services named `<name>-prefill`, `<name>-decode` and `<name>`
    (the router); `driver.policy_model` names the router.
    """

    type: Literal["vllm_pd"]
    prefill: VllmServiceConfig
    decode: VllmServiceConfig
    # Run an independent server on every node of each tier's pool instead of one data-parallel
    # engine across them. The router then lists each node as its own endpoint.
    server_per_node: bool = False
    # Prefill's NIXL side-channel port; decode uses this plus 1.
    nixl_side_channel_port: int = 5600
    prefill_policy: str = "cache_aware"
    decode_policy: str = "cache_aware"
    intra_node_data_parallel_size: int = 1
    # An agentic benchmark holds a request open for a long time; the router must not give up on it.
    request_timeout_secs: int = 86400
    log_level: str = "error"
    kv_connector: str = "NixlConnector"
    # "fail" surfaces a broken KV transfer instead of silently recomputing the prefill.
    kv_load_failure_policy: str = "fail"
    _tiers: dict[str, VllmPDTierConfig] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _fill_tier_defaults(cls, data: Any) -> Any:
        # A tier inherits the model identity and container, and takes the next ports after
        # the router's and prefill's, unless it sets its own.
        if not isinstance(data, dict) or not all(isinstance(data.get(t), dict) for t in ("prefill", "decode")):
            return data
        for tier in ("prefill", "decode"):
            misplaced = [f for f in _PD_SERVICE_ONLY_FIELDS if f in data[tier]]
            if misplaced:
                raise ValueError(
                    f"{', '.join(misplaced)} is set on the {tier} tier. Set it on the vllm_pd service itself; "
                    "it applies to both tiers."
                )
        inherited = {"type": "vllm", **{k: data[k] for k in ("container", "model", "served_model_name") if k in data}}
        port = data.get("port", cls.model_fields["port"].default)
        prefill = {**inherited, "port": port + 1, **data["prefill"]}
        decode = {**inherited, "port": port + 2, **data["decode"]}
        rpc_port = prefill.get(
            "data_parallel_rpc_port", VllmServiceConfig.model_fields["data_parallel_rpc_port"].default
        )
        decode.setdefault("data_parallel_rpc_port", rpc_port + 1)
        return {**data, "prefill": prefill, "decode": decode}

    @model_validator(mode="after")
    def _build_tiers(self) -> "VllmPDServiceConfig":
        # Built once and kept private: the deploy-only values stay out of the saved config.
        for name, tier, role, nixl_port in (
            ("prefill", self.prefill, "kv_producer", self.nixl_side_channel_port),
            ("decode", self.decode, "kv_consumer", self.nixl_side_channel_port + _DECODE_NIXL_PORT_OFFSET),
        ):
            # model_construct: the tier is already validated, and env prefixes must not resolve twice.
            self._tiers[name] = VllmPDTierConfig.model_construct(
                _fields_set=tier.model_fields_set,
                **dict(tier),
                nixl_side_channel_port=nixl_port,
                server_per_node=self.server_per_node,
                kv_transfer_config={
                    "kv_connector": self.kv_connector,
                    "kv_role": role,
                    "kv_load_failure_policy": self.kv_load_failure_policy,
                },
            )
        if self.health_check is None:
            self.health_check = HealthCheckConfig(port=self.port)
        elif self.health_check.port is None:
            self.health_check.port = self.port
        return self

    def tiers(self, name: str) -> dict[str, VllmPDTierConfig]:
        return {f"{name}-prefill": self._tiers["prefill"], f"{name}-decode": self._tiers["decode"]}


class RayServiceConfig(BaseServiceConfig):
    type: Literal["ray"]
    # Node pools the cluster spans. The head starts on the driver's node and every
    # other node of these pools joins it as a worker. Empty: a single-node cluster.
    node_pools: list[str] = []
    port: int = 6379
    # Custom Ray resources advertised by each spanned pool's nodes, keyed by pool,
    # e.g. {"aux": {"extra_gpu": 4}}. A benchmark asks for these by name when it
    # manages device placement itself.
    resources: dict[str, dict[str, float]] = {}
    num_cpus: int | None = None
    num_gpus: int | None = None
    # Raw extra flags appended verbatim to `ray start` on every node (e.g. fixed ports).
    extra_args: str = ""

    @model_validator(mode="after")
    def _validate_shape(self) -> "RayServiceConfig":
        if self.node_pool is not None:
            raise ValueError(
                f"A ray service spans `node_pools`, not a single `node_pool`; use node_pools: [{self.node_pool}]."
            )
        if len(set(self.node_pools)) != len(self.node_pools):
            raise ValueError(f"A ray service's node_pools lists a pool more than once: {self.node_pools}.")
        unspanned = sorted(set(self.resources) - set(self.node_pools))
        if unspanned:
            raise ValueError(
                f"A ray service sets resources for {', '.join(unspanned)}, which it does not span "
                f"(node_pools: {self.node_pools}). Add the pool to node_pools or drop its resources."
            )
        return self


# Discriminated union keyed on `type`; Pydantic rejects unknown type values at parse time.
ServiceConfig = Annotated[
    Annotated[VllmServiceConfig, Tag("vllm")]
    | Annotated[RayServiceConfig, Tag("ray")]
    | Annotated[VllmPDServiceConfig, Tag("vllm_pd")],
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


class ResumeConfig(_StrictModel):
    # Non-timeout failures (job script bugs, OOM, etc.) are resubmitted at most this many
    # times, so a broken benchmark doesn't requeue forever.
    max_retries: int = 3
    # Optional cap on the chain's total accumulated walltime, as a Slurm-style
    # duration string (e.g. "48:00:00"). None means resume until max_retries is
    # hit on a non-timeout failure, or the job completes/is cancelled.
    max_walltime: str | None = None


class BenchmarkRunConfig(_StrictModel):
    # Hydra overrides forwarded to `gym eval prepare`. Flattened to +key=value tokens.
    prepare: dict[str, Any] = {}
    # Hydra overrides forwarded to `gym eval run`. policy_model wiring is injected here at
    # validation time so all executors see it uniformly via flatten_run_args.
    run: dict[str, Any] = {}
    # Shell script the driver runs INSTEAD of `gym eval run`, for a benchmark whose
    # harness is not Gym's own runner -- one that provisions external machines and
    # drives Gym from there, for instance. `prepare` still runs first, and the
    # driver exports NEMO_GYM_BENCH_DIR plus the policy's base URL, model name and
    # API key (when driver.policy_model is set) so the script can reach the served
    # model without repeating its address.
    command: str | None = None

    @model_validator(mode="after")
    def _validate_command(self) -> "BenchmarkRunConfig":
        if self.command is not None and self.run:
            raise ValueError(
                "A benchmark sets both `command` and `run`, but `run` only configures `gym eval run`, which "
                "`command` replaces. Fold those settings into the command, or drop it."
            )
        if self.command is not None and self.resume_config is not None:
            raise ValueError(
                "A benchmark sets both `command` and `resumable`, but a resumed job only passes "
                "`resume_from_cache` to `gym eval run`, so a `command` would restart from zero. "
                "Drop `resumable`, or run through `gym eval run`."
            )
        return self

    # True enables auto-resume with ResumeConfig defaults; pass a ResumeConfig to tune
    # max_retries/max_walltime. False (default) leaves fire-and-forget submission unchanged.
    # Only executors that declare `supports_resumable = True` may set this (see
    # SubmitConfig validation in submit.py). Not allowed with `command`.
    resumable: bool | ResumeConfig = False

    @property
    def resume_config(self) -> ResumeConfig | None:
        if self.resumable is False:
            return None
        return ResumeConfig() if self.resumable is True else self.resumable


class GymInstallConfig(_StrictModel):
    repo: str = "https://github.com/NVIDIA-NeMo/gym"
    ref: str  # Git tag or commit hash.


class DriverConfig(_StrictModel):
    container: str = "python:3.12"
    gym_install: GymInstallConfig | None = None
    # Name of a service in `services:` to use as the policy model. When set, injects
    # policy_base_url/policy_model_name/policy_api_key into each benchmark's run config.
    policy_model: str | None = None
    # Host in the injected policy_base_url. localhost suits clients on the driver's
    # node; an agent in a remote sandbox needs an address it can route to, e.g.
    # "${oc.env:HEAD_NODE_IP}", resolved when the driver runs.
    policy_host: str = "localhost"
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


class OtelConfig(_StrictModel):
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
    token_env: str = "OTEL_TOKEN"
    # Sent as the `service.name` resource attribute: the identity the backend routes the token by.
    service_name: str | None = None
    # Display identity of the scraped metrics in the backend (`service.name.override`).
    component: str = "gym-vllm"
    # Gym's own nemo-lens telemetry; needs `driver.gym_install`, which installs the `telemetry` extra.
    gym_telemetry: bool = True
    # Gym span groups to switch on: a preset or comma-separated names (`default`, `verify`, `sandbox`, ...).
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
            raise ValueError(f"otel.token_env: {v!r} is not a valid environment variable name")
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
    otel: OtelConfig = OtelConfig()

    @property
    def deployed_services(self) -> dict[str, ServiceConfig]:
        """`services` with each vllm_pd service expanded into its two tiers and its router."""
        deployed: dict[str, ServiceConfig] = {}
        for name, service in self.services.items():
            if isinstance(service, VllmPDServiceConfig):
                deployed.update(service.tiers(name))
            deployed[name] = service
        return deployed

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

        for name, service in self.services.items():
            if isinstance(service, VllmPDServiceConfig):
                taken = sorted(set(service.tiers(name)) & set(self.services))
                if taken:
                    raise ValueError(
                        f"vllm_pd service '{name}' deploys its tiers as {', '.join(service.tiers(name))}, "
                        f"but services already has {', '.join(taken)}. Rename one."
                    )

        for service_name, service in self.deployed_services.items():
            if service.placement is None:
                service.placement = sole_compute
            elif service.placement not in compute_names:
                raise ValueError(
                    f"Service '{service_name}' placement '{service.placement}' does not match any compute resource "
                    f"({', '.join(sorted(compute_names))})."
                )

            if isinstance(service, RayServiceConfig):
                unknown = [pool for pool in service.node_pools if pool not in pool_names]
                if unknown:
                    raise ValueError(
                        f"Service '{service_name}' node_pools {unknown} do not match any node pool of compute "
                        f"'{service.placement}' ({', '.join(sorted(pool_names)) or 'none declared'})."
                    )

            if service.node_pool is not None and service.node_pool not in pool_names:
                raise ValueError(
                    f"Service '{service_name}' node_pool '{service.node_pool}' does not match any node pool of "
                    f"compute '{service.placement}' ({', '.join(sorted(pool_names)) or 'none declared'})."
                )

            if isinstance(service, VllmPDTierConfig) and service.node_pool is None:
                raise ValueError(
                    f"Service '{service_name}' is a prefill/decode tier without a node_pool. Each tier needs nodes "
                    "of its own: the two tiers run side by side and the router addresses each tier by its pool."
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
            if isinstance(service, VllmPDTierConfig) and service.server_per_node:
                if service.number_of_instances != 1:
                    raise ValueError(
                        f"Service '{service_name}' runs one server per node (server_per_node), so each node is one "
                        f"instance; number_of_instances must be 1, got {service.number_of_instances}."
                    )
                # Each node serves on its own, so size it as a single-node deployment.
                service_nodes = 1
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

        self._validate_pd_services(compute)

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
                    benchmark.run["policy_base_url"] = f"http://{self.driver.policy_host}:{service.port}/v1"
                    benchmark.run["policy_model_name"] = service.served_model_name or service.model
                    # vLLM doesn't require auth; dummy key satisfies clients that require the header.
                    benchmark.run["policy_api_key"] = "dummy"  # pragma: allowlist secret

        return self

    def _validate_pd_services(self, compute: "ComputeConfig") -> None:
        pools = list(compute.node_pools) if isinstance(compute, SlurmComputeConfig) else []

        for name, pd in self.services.items():
            if not isinstance(pd, VllmPDServiceConfig):
                continue

            prefill_name, decode_name = pd.tiers(name)
            if pd.prefill.data_parallel_rpc_port == pd.decode.data_parallel_rpc_port:
                raise ValueError(
                    f"Services '{prefill_name}' and '{decode_name}' share data_parallel_rpc_port "
                    f"{pd.prefill.data_parallel_rpc_port}. Each tier binds the port on its own hosts, so the two "
                    "tiers need different ones."
                )

            # The driver reaches the router over localhost from the allocation's first node.
            if pd.node_pool is not None and pools and pd.node_pool != pools[0]:
                raise ValueError(
                    f"vllm_pd service '{name}' pins its router to node_pool '{pd.node_pool}', but the driver reaches "
                    f"it over localhost and runs on the first node. Pin it to '{pools[0]}' or leave node_pool unset."
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
