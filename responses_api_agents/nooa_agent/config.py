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

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any, Literal

from nooa import Agent
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef


class NOOAArgumentBinding(BaseModel):
    """Map one entrypoint argument from a Gym run row."""

    model_config = ConfigDict(extra="forbid")

    source: str
    transform: str = "identity"

    @field_validator("source")
    @classmethod
    def validate_source(cls, source: str) -> str:
        from responses_api_agents.nooa_agent.mapping import validate_source_path

        validate_source_path(source)
        return source

    @field_validator("transform")
    @classmethod
    def validate_transform(cls, transform: str) -> str:
        from responses_api_agents.nooa_agent.mapping import get_transform

        get_transform(transform)
        return transform


class NOOAInvocationConfig(BaseModel):
    """Configuration for constructing and invoking a NOOA agent."""

    model_config = ConfigDict(extra="forbid")

    agent_class: str
    entrypoint: str
    execution_mode: Literal["embedded"] = "embedded"
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    arguments: dict[str, NOOAArgumentBinding]

    @field_validator("agent_class")
    @classmethod
    def validate_agent_class_path(cls, value: str) -> str:
        module_name, separator, class_name = value.partition(":")
        if not separator or not module_name or not class_name or "." in class_name:
            raise ValueError("agent_class must use the format 'module.path:ClassName'")
        return value

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint_name(cls, value: str) -> str:
        if not value.isidentifier() or value.startswith("_"):
            raise ValueError("entrypoint must be a public Python method name")
        return value

    @model_validator(mode="after")
    def validate_argument_names(self) -> "NOOAInvocationConfig":
        invalid = sorted(name for name in self.arguments if not name.isidentifier() or name.startswith("_"))
        if invalid:
            raise ValueError(f"argument mapping names must be public Python identifiers: {invalid}")
        if "llm" in self.init_kwargs:
            raise ValueError("init_kwargs.llm is reserved; Gym always injects the rollout LLM")
        return self


class NOOAAgentConfig(BaseResponsesAPIAgentConfig):
    """Gym server configuration for the NOOA adapter."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    nooa: NOOAInvocationConfig
    max_steps: int = Field(default=10, gt=0)
    concurrency: int = Field(default=8, gt=0)
    run_timeout_secs: float = Field(default=2100, gt=0)


def load_agent_class(path: str) -> type[Agent]:
    """Import and validate a ``module:Class`` NOOA agent reference."""

    module_name, _, class_name = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"could not import NOOA agent module {module_name!r}") from error

    try:
        candidate = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"module {module_name!r} has no attribute {class_name!r}") from error

    if not inspect.isclass(candidate) or not issubclass(candidate, Agent):
        raise ValueError(f"{path!r} must resolve to a subclass of nooa.Agent")
    return candidate


def validate_invocation(config: NOOAInvocationConfig) -> tuple[type[Agent], Callable[..., Any]]:
    """Validate imports and constructor/entrypoint signatures at server startup."""

    agent_class = load_agent_class(config.agent_class)
    try:
        inspect.signature(agent_class).bind(llm=object(), **config.init_kwargs)
    except TypeError as error:
        raise ValueError(f"init_kwargs do not match {config.agent_class}: {error}") from error

    entrypoint = getattr(agent_class, config.entrypoint, None)
    if entrypoint is None or not callable(entrypoint):
        raise ValueError(f"{config.agent_class} has no callable entrypoint {config.entrypoint!r}")
    if not inspect.iscoroutinefunction(entrypoint):
        raise ValueError(f"entrypoint {config.entrypoint!r} must be async")

    signature = inspect.signature(entrypoint)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )

    unknown = set(config.arguments) - set(parameters)
    if unknown and not accepts_kwargs:
        raise ValueError(f"arguments not accepted by {config.entrypoint}: {sorted(unknown)}")

    positional_only = {
        name for name, parameter in parameters.items() if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    }
    mapped_positional_only = positional_only & set(config.arguments)
    if mapped_positional_only:
        raise ValueError(f"entrypoint parameters must accept keyword arguments: {sorted(mapped_positional_only)}")

    required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
    }
    missing = required - set(config.arguments)
    if missing:
        raise ValueError(f"required entrypoint arguments are not mapped: {sorted(missing)}")

    return agent_class, entrypoint
