# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process placement configuration shared by Gym agent harnesses."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentDependenciesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    python_version: str = Field(default="3.13.14", pattern=r"^\d+\.\d+\.\d+$")
    uv_version: str = Field(default="0.12.3", pattern=r"^\d+\.\d+\.\d+$")
    # Defaults to the checkout containing Gym. Installed distributions need an explicit checkout.
    source_root: str | None = None
    # Relative to source_root; inferred for built-in responses_api_agents modules.
    harness_path: str | None = None


class AgentRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["local", "sandbox"] = "local"
    sandbox_source: Literal["environment", "runtime"] = "environment"
    provider: str = "sandbox"
    spec: dict[str, Any] = Field(default_factory=dict)
    python: str = "python3"
    dependencies: AgentDependenciesConfig = Field(default_factory=AgentDependenciesConfig)
    env: dict[str, str] = Field(default_factory=dict)
    setup_command: str | None = None
    setup_timeout_s: float = Field(default=900, gt=0)
    timeout_s: float = Field(default=10800, gt=0)
    concurrency: int = Field(default=8, gt=0)
    # Local file -> absolute destination in the sandbox (e.g. a wheel or installer).
    uploads: dict[str, str] = Field(default_factory=dict)
    # Overrides applied only inside the sandbox, e.g. repository and command paths.
    agent_config: dict[str, Any] = Field(default_factory=dict)
    # Reachable root URLs of referenced Gym servers.
    server_urls: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_placement(self):
        if "runtime" in self.agent_config:
            raise ValueError("runtime.agent_config cannot override runtime placement")
        if self.sandbox_source == "environment" and self.spec:
            raise ValueError("runtime.spec requires sandbox_source=runtime; the environment owns its task spec")
        return self
