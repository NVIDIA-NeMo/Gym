# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic mirror of Harbor's ``task.toml`` (schema 1.4).

Field names are Harbor's. Deprecated spellings (``version``, ``memory``, ``storage``,
``allow_internet``, ``image``) are read and normalized but never written back. Keys
this mirror does not know are ignored, as Harbor ignores them; :meth:`HarborSettings.unknown_keys`
lists them so the loader can say so.
"""

from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HarborSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    # Keys a `before` validator reads and renames; they are accepted, not unknown.
    deprecated_keys: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def unknown_keys(cls, data: Any, prefix: str = "") -> list[str]:
        """Dotted paths of the keys in ``data`` that no field of this model tree declares."""
        if not isinstance(data, dict):
            return []
        unknown: list[str] = []
        for key, value in data.items():
            field = cls.model_fields.get(key)
            if field is None:
                if key not in cls.deprecated_keys:
                    unknown.append(f"{prefix}{key}")
                continue
            nested = _nested_settings(field.annotation)
            if nested is None:
                continue
            if isinstance(value, list):
                for index, item in enumerate(value):
                    unknown += nested.unknown_keys(item, f"{prefix}{key}[{index}].")
            else:
                unknown += nested.unknown_keys(value, f"{prefix}{key}.")
        return unknown


def _nested_settings(annotation: Any) -> type[HarborSettings] | None:
    if isinstance(annotation, type) and issubclass(annotation, HarborSettings):
        return annotation
    for arg in get_args(annotation):
        nested = _nested_settings(arg)
        if nested is not None:
            return nested
    return None


class HarborHealthcheck(HarborSettings):
    """Mirrors Harbor's ``HealthcheckConfig``: plain numbers with Harbor's defaults and no bounds.

    Published tasks pin values at the lower bound (``retries = 0``: one attempt once the start
    period ends), and Harbor loads them; bounding the fields here rejected those tasks.
    """

    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3


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

    deprecated_keys: ClassVar[frozenset[str]] = frozenset({"image", "allow_internet", "memory", "storage"})

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


class HarborCollectHook(HarborSettings):
    """``[[verifier.collect]]``: a command run in a service before its artifacts are collected."""

    command: str
    service: str = "main"
    timeout_sec: float = Field(default=60, gt=0)
    user: str | int | None = None


class HarborVerifier(HarborPhaseSettings):
    timeout_sec: float = Field(default=600, gt=0)
    user: str | int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    environment_mode: Literal["separate", "shared"] | None = None
    environment: HarborEnvironment | None = None
    collect: list[HarborCollectHook] = Field(default_factory=list)

    @model_validator(mode="after")
    def shared_has_no_environment(self) -> "HarborVerifier":
        if self.environment_mode == "shared" and self.environment is not None:
            raise ValueError("A shared verifier cannot declare its own [verifier.environment]")
        return self


class HarborSolution(HarborSettings):
    env: dict[str, str] = Field(default_factory=dict)


class HarborArtifact(HarborSettings):
    """One ``artifacts`` entry: a path copied out of the agent's sandbox after it finishes.

    A bare string is the ``source``. ``destination`` is where it lands relative to the run's
    artifacts folder (default: the source path without its leading slash). ``service`` names a
    Compose sidecar; ``None`` or ``"main"`` is the agent's own container.
    """

    source: str
    destination: str | None = None
    service: str | None = None
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def from_string(cls, data: Any) -> Any:
        return {"source": data} if isinstance(data, str) else data

    @model_validator(mode="after")
    def contained_paths(self) -> "HarborArtifact":
        for path in (self.source, self.destination):
            if path and (".." in PurePosixPath(path).parts or "\\" in path):
                raise ValueError(f"Artifact paths must stay inside the sandbox and the artifacts folder: {path!r}")
        if self.destination and PurePosixPath(self.destination).is_absolute():
            raise ValueError(f"Artifact destination must be relative: {self.destination!r}")
        return self

    @property
    def host_path(self) -> PurePosixPath:
        return PurePosixPath(self.destination or self.source.lstrip("/"))


class HarborTaskConfig(HarborSettings):
    """One ``task.toml``."""

    deprecated_keys: ClassVar[frozenset[str]] = frozenset({"version", "steps", "multi_step_reward_strategy"})

    schema_version: str = "1.4"
    task: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    solution: HarborSolution = Field(default_factory=HarborSolution)
    agent: HarborAgent = Field(default_factory=HarborAgent)
    environment: HarborEnvironment = Field(default_factory=HarborEnvironment)
    verifier: HarborVerifier = Field(default_factory=HarborVerifier)
    artifacts: list[HarborArtifact] = Field(default_factory=list)

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
        """Harbor runs the verifier in the agent's container unless the task says otherwise.

        As in Harbor, ``[verifier].environment_mode = "separate"`` or a ``[verifier.environment]``
        block selects the separate mode, where a fresh container receives ``/logs/artifacts`` and
        the declared ``artifacts``. Only ``environment_mode = "shared"`` keeps a sized verifier shared.
        """
        return self.verifier.environment_mode != "separate" and self.verifier.environment is None
