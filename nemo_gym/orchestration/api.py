from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Tag


class VllmServiceConfig(BaseModel):
    type: Literal["vllm"]
    container: str
    placement: str | None = None


class RayServiceConfig(BaseModel):
    type: Literal["ray"]
    container: str
    placement: str | None = None


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
