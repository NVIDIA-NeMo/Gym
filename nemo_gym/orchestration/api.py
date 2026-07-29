from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthCheckConfig(_StrictModel):
    path: str = "/health"
    port: int | None = None
    timeout_seconds: int = 60


class BaseServiceConfig(_StrictModel):
    container: str
    placement: str | None = None
    health_check: HealthCheckConfig | None = None


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
        if self.health_check is None:
            self.health_check = HealthCheckConfig(port=self.port)
        elif self.health_check.port is None:
            self.health_check.port = self.port
        return self


class RayServiceConfig(BaseServiceConfig):
    type: Literal["ray"]


ServiceConfig = Annotated[
    Annotated[VllmServiceConfig, Tag("vllm")] | Annotated[RayServiceConfig, Tag("ray")],
    Discriminator("type"),
]


class NodePool(_StrictModel):
    partition: str
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int | None = None
    extra_args: dict[str, str] = {}


class BaseComputeConfig(_StrictModel):
    pass


class SlurmComputeConfig(BaseComputeConfig):
    type: Literal["slurm"]
    account: str
    hostname: str | None = None
    walltime: str | None = None
    node_pools: dict[str, NodePool] = {}
    extra_args: dict[str, str] = {}


ComputeConfig = Annotated[
    Annotated[SlurmComputeConfig, Tag("slurm")],
    Discriminator("type"),
]


class BenchmarkRunConfig(_StrictModel):
    run: dict[str, Any] = {}


class GymInstallConfig(_StrictModel):
    repo: str = "https://github.com/NVIDIA-NeMo/gym"
    ref: str


class DriverConfig(_StrictModel):
    container: str = "python:3.12"
    gym_install: GymInstallConfig | None = None
    policy_model: str | None = None
    benchmarks: dict[str, BenchmarkRunConfig]


class JobConfig(_StrictModel):
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
            raise ValueError(
                f"Multiple compute resources are not supported yet ({', '.join(sorted(compute_names))})."
            )

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
                    conflicts = [k for k in ("policy_base_url", "policy_model_name") if k in benchmark.run]
                    if conflicts:
                        raise ValueError(
                            f"Benchmark '{bench_name}' run config already sets {conflicts} "
                            f"but driver.policy_model is also set. Remove one."
                        )
                    benchmark.run["policy_base_url"] = f"http://localhost:{service.port}/v1"
                    benchmark.run["policy_model_name"] = service.model

        return self
