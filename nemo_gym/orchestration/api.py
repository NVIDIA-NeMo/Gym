from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Tag


class BaseServiceConfig(BaseModel):
    container: str
    placement: str


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
