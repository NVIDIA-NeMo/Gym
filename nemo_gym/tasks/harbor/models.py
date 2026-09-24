# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic mirror of Harbor's ``task.toml`` (schema 1.4).

Field names are Harbor's. Deprecated spellings (``version``, ``memory``, ``storage``,
``allow_internet``, ``image``) are read and normalized but never written back.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HarborSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HarborHealthcheck(HarborSettings):
    command: str
    interval_sec: float = Field(default=5, ge=0)
    timeout_sec: float = Field(default=30, gt=0)
    start_period_sec: float = Field(default=0, ge=0)
    start_interval_sec: float = Field(default=5, ge=0)
    retries: int = Field(default=3, gt=0)


class HarborMCPServer(HarborSettings):
    name: str
    transport: Literal["stdio", "sse", "streamable-http"] = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_transport(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("transport") == "http":
            data = {**data, "transport": "streamable-http"}
        return data

    @model_validator(mode="after")
    def require_endpoint(self) -> "HarborMCPServer":
        if not (self.command if self.transport == "stdio" else self.url):
            raise ValueError(f"MCP server {self.name!r}: stdio needs `command`, other transports need `url`")
        return self


class HarborPhaseSettings(HarborSettings):
    network_mode: Literal["public", "no-network"] | None = None
    allowed_hosts: list[str] | None = None


def _size_to_mb(raw: str) -> int:
    text = raw.strip().upper()
    units = {"G": 1024, "M": 1, "K": 1 / 1024}
    if not text or text[-1] not in units:
        raise ValueError(f"Cannot parse size {raw!r}; expected a number followed by K, M or G")
    return int(float(text[:-1]) * units[text[-1]])


class HarborEnvironment(HarborPhaseSettings):
    """``[environment]``: the sandbox the agent works in.

    ``docker_image`` is optional. A task without one builds from
    ``environment/Dockerfile`` or, when that file only names a base image, runs the
    base image directly.
    """

    docker_image: str | None = None
    build_timeout_sec: float = Field(default=600, gt=0)
    os: Literal["linux", "windows"] = "linux"
    cpus: int | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int | None = Field(default=None, ge=0)
    gpu_types: list[str] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[HarborMCPServer] = Field(default_factory=list)
    skills_dir: str | None = None
    healthcheck: HarborHealthcheck | None = None
    workdir: str | None = None
    network_mode: Literal["public", "no-network"] = "public"

    @model_validator(mode="before")
    @classmethod
    def read_deprecated_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "image" in data:
            image = data.pop("image")
            if data.get("docker_image") not in (None, image):
                raise ValueError("Conflicting `image` and `docker_image`")
            data["docker_image"] = image
        allow = data.pop("allow_internet", None)
        if allow is not None:
            data.setdefault("network_mode", "public" if allow else "no-network")
        for old, new in (("memory", "memory_mb"), ("storage", "storage_mb")):
            if old in data:
                size = _size_to_mb(str(data.pop(old)))
                if new in data and data[new] != size:
                    raise ValueError(f"Conflicting `{old}` and `{new}`")
                data[new] = size
        return data


class HarborAgent(HarborPhaseSettings):
    timeout_sec: float = Field(default=28800, gt=0)
    user: str | int | None = None


class HarborVerifier(HarborPhaseSettings):
    timeout_sec: float = Field(default=600, gt=0)
    user: str | int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    environment_mode: Literal["separate", "shared"] | None = None
    environment: HarborEnvironment | None = None

    @model_validator(mode="after")
    def shared_has_no_environment(self) -> "HarborVerifier":
        if self.environment_mode == "shared" and self.environment is not None:
            raise ValueError("A shared verifier cannot declare its own [verifier.environment]")
        return self


class HarborSolution(HarborSettings):
    env: dict[str, str] = Field(default_factory=dict)


class HarborTaskConfig(HarborSettings):
    """One ``task.toml``."""

    schema_version: str = "1.4"
    task: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    solution: HarborSolution = Field(default_factory=HarborSolution)
    agent: HarborAgent = Field(default_factory=HarborAgent)
    environment: HarborEnvironment = Field(default_factory=HarborEnvironment)
    verifier: HarborVerifier = Field(default_factory=HarborVerifier)
    artifacts: list[Any] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def read_deprecated_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "version" in data:
            data.setdefault("schema_version", str(data.pop("version")))
        if data.get("steps") or data.get("multi_step_reward_strategy"):
            raise ValueError("Multi-step tasks ([[steps]]) are not supported yet")
        return data

    @property
    def is_shared_verifier(self) -> bool:
        """Harbor runs the verifier in the agent's container unless a separate one is declared."""
        return self.verifier.environment is None and self.verifier.environment_mode != "separate"
