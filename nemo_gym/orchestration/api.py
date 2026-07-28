from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Tag, model_validator


class BaseServiceConfig(BaseModel):
    container: str
    placement: str | None = None


class VllmServiceConfig(BaseServiceConfig):
    type: Literal["vllm"]


class RayServiceConfig(BaseServiceConfig):
    type: Literal["ray"]


ServiceConfig = Annotated[
    Annotated[VllmServiceConfig, Tag("vllm")] | Annotated[RayServiceConfig, Tag("ray")],
    Discriminator("type"),
]


class ComputeConfig(BaseModel):
    type: str
    hostname: str | None = None
    walltime: str | None = None
    node_pools: dict[str, dict] | None = None


class SubmitConfig(BaseModel):
    services: dict[str, ServiceConfig]
    compute: dict[str, ComputeConfig]

    @model_validator(mode="after")
    def _resolve_and_validate_placements(self) -> "SubmitConfig":
        compute_names = set(self.compute)
        sole_compute = next(iter(compute_names)) if len(compute_names) == 1 else None

        for service_name, service in self.services.items():
            if service.placement is None:
                if sole_compute is None:
                    raise ValueError(
                        f"Service '{service_name}' has no placement and there are multiple compute resources "
                        f"({', '.join(sorted(compute_names))}). Set placement explicitly."
                    )
                service.placement = sole_compute
            elif service.placement not in compute_names:
                raise ValueError(
                    f"Service '{service_name}' placement '{service.placement}' does not match any compute resource "
                    f"({', '.join(sorted(compute_names))})."
                )

        return self
