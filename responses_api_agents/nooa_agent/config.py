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
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from nooa import Agent
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


NOOAInvocationAdapter = Callable[
    [Agent, NeMoGymResponseCreateParamsNonStreaming],
    Awaitable[object],
]


class NOOAInvocationConfig(BaseModel):
    """Configuration for constructing and invoking a NOOA agent."""

    model_config = ConfigDict(extra="forbid")

    agent_class: str
    invocation_adapter: str
    execution_mode: Literal["embedded", "sandboxed"] = "embedded"
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    model_aliases: dict[str, str] = Field(default_factory=dict)

    @field_validator("agent_class")
    @classmethod
    def validate_agent_class_path(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("agent_class must use the format 'module.path:ClassName'")

        module_name, class_name = parts
        if not module_name or not class_name or "." in class_name:
            raise ValueError("agent_class must use the format 'module.path:ClassName'")
        return value

    @field_validator("invocation_adapter")
    @classmethod
    def validate_invocation_adapter_path(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("invocation_adapter must use the format 'module.path:function_name'")

        module_name, function_name = parts
        if not module_name or not function_name or "." in function_name:
            raise ValueError("invocation_adapter must use the format 'module.path:function_name'")
        return value

    @field_validator("model_aliases")
    @classmethod
    def validate_model_aliases(cls, aliases: dict[str, str]) -> dict[str, str]:
        if any(not alias.strip() or not server.strip() for alias, server in aliases.items()):
            raise ValueError("model_aliases must map non-empty NOOA names to non-empty Gym model-server names")
        return aliases

    @field_validator("init_kwargs")
    @classmethod
    def validate_init_kwargs(cls, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "llm" in init_kwargs:
            raise ValueError("init_kwargs.llm is reserved; Gym always injects the rollout LLM")
        return init_kwargs


class NOOAAgentConfig(BaseResponsesAPIAgentConfig):
    """Gym server configuration for the NOOA adapter."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    nooa: NOOAInvocationConfig
    max_policy_calls: int = Field(default=10, gt=0)
    run_timeout_secs: float = Field(default=2100, gt=0)


def load_agent_class(path: str) -> type[Agent]:
    """Import and validate a ``module:Class`` NOOA agent reference."""

    module_name, _, class_name = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, TypeError) as error:
        raise ValueError(f"could not import NOOA agent module {module_name!r}") from error

    try:
        candidate = getattr(module, class_name)
    except AttributeError as error:
        raise ValueError(f"module {module_name!r} has no attribute {class_name!r}") from error

    if not inspect.isclass(candidate) or not issubclass(candidate, Agent):
        raise ValueError(f"{path!r} must resolve to a subclass of nooa.Agent")
    return candidate


def load_invocation_adapter(path: str) -> NOOAInvocationAdapter:
    """Import and validate an async ``module:function`` invocation adapter."""

    module_name, _, function_name = path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, TypeError) as error:
        raise ValueError(f"could not import NOOA invocation adapter module {module_name!r}") from error

    try:
        candidate = getattr(module, function_name)
    except AttributeError as error:
        raise ValueError(f"module {module_name!r} has no attribute {function_name!r}") from error

    if not callable(candidate) or not inspect.iscoroutinefunction(candidate):
        raise ValueError(f"{path!r} must resolve to an async function")
    try:
        inspect.signature(candidate).bind(object(), object())
    except TypeError as error:
        raise ValueError(f"invocation adapter must accept positional agent and request arguments: {error}") from error
    return candidate


def validate_invocation(config: NOOAInvocationConfig) -> tuple[type[Agent], NOOAInvocationAdapter]:
    """Validate agent construction and the Responses invocation adapter at startup."""

    agent_class = load_agent_class(config.agent_class)
    try:
        inspect.signature(agent_class).bind(llm=object(), **config.init_kwargs)
    except TypeError as error:
        raise ValueError(f"init_kwargs do not match {config.agent_class}: {error}") from error

    return agent_class, load_invocation_adapter(config.invocation_adapter)
