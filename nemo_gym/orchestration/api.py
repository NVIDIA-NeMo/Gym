from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Tag, model_validator


class HealthCheckConfig(BaseModel):
    path: str = "/health"
    port: int
    timeout_seconds: int = 60


class BaseServiceConfig(BaseModel):
    container: str
    placement: str | None = None
    health_check: HealthCheckConfig | None = None


class VllmServiceConfig(BaseServiceConfig):
    type: Literal["vllm"]
    model: str
    port: int = 8000
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    trust_remote_code: bool = False


class RayServiceConfig(BaseServiceConfig):
    type: Literal["ray"]


ServiceConfig = Annotated[
    Annotated[VllmServiceConfig, Tag("vllm")] | Annotated[RayServiceConfig, Tag("ray")],
    Discriminator("type"),
]


class BaseComputeConfig(BaseModel):
    pass


class SlurmComputeConfig(BaseComputeConfig):
    type: Literal["slurm"]
    hostname: str | None = None
    walltime: str | None = None
    node_pools: dict[str, dict] | None = None


ComputeConfig = Annotated[
    Annotated[SlurmComputeConfig, Tag("slurm")],
    Discriminator("type"),
]


class BenchmarkRunConfig(BaseModel):
    name: str


class DriverConfig(BaseModel):
    container: str
    benchmarks: list[BenchmarkRunConfig]


class SubmitConfig(BaseModel):
    services: dict[str, ServiceConfig]
    compute: dict[str, ComputeConfig]
    driver: DriverConfig

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

        return self
