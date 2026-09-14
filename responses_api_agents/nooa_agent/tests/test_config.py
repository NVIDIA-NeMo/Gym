# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nooa import Agent
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.config import (
    NOOAAgentConfig,
    NOOAInvocationConfig,
    load_agent_class,
    load_invocation_adapter,
    validate_invocation,
)


class ExampleAgent(Agent):
    pass


class NotAnAgent:
    pass


class ConstructorAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label


async def invoke(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    return agent, request


def synchronous_adapter(agent: Agent, request: NeMoGymResponseCreateParamsNonStreaming) -> object:
    return agent, request


async def one_argument_adapter(agent: Agent) -> object:
    return agent


NOT_AN_ADAPTER = object()


def invocation_config(**overrides: Any) -> NOOAInvocationConfig:
    values = {
        "agent_class": f"{__name__}:ExampleAgent",
        "invocation_adapter": f"{__name__}:invoke",
    }
    values.update(overrides)
    return NOOAInvocationConfig.model_validate(values)


def agent_config(**overrides: Any) -> NOOAAgentConfig:
    values = {
        "name": "nooa",
        "host": "127.0.0.1",
        "port": 9000,
        "entrypoint": "app.py",
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "model_server": {"type": "responses_api_models", "name": "policy"},
        "nooa": invocation_config().model_dump(),
    }
    values.update(overrides)
    return NOOAAgentConfig.model_validate(values)


@pytest.mark.parametrize("field", ["agent_class", "invocation_adapter"])
def test_rejects_malformed_import_path(field: str) -> None:
    with pytest.raises(ValidationError, match="must use the format"):
        invocation_config(**{field: "missing-colon"})


def test_agent_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        agent_config(max_step=20)

    error = exc_info.value.errors()[0]
    assert error["type"] == "extra_forbidden"
    assert error["loc"] == ("max_step",)


def test_agent_config_names_policy_call_budget_explicitly() -> None:
    assert agent_config(max_policy_calls=6).max_policy_calls == 6

    with pytest.raises(ValidationError, match="max_steps"):
        agent_config(max_steps=6)


def test_sandboxed_execution_mode_is_reserved_for_a_future_runner() -> None:
    assert invocation_config(execution_mode="sandboxed").execution_mode == "sandboxed"


def test_validate_invocation_returns_agent_and_adapter() -> None:
    agent_class, adapter = validate_invocation(invocation_config())

    assert agent_class is ExampleAgent
    assert adapter is invoke


def test_validate_invocation_checks_constructor_with_injected_llm() -> None:
    config = invocation_config(
        agent_class=f"{__name__}:ConstructorAgent",
        init_kwargs={"label": "production"},
    )

    agent_class, _ = validate_invocation(config)

    assert agent_class is ConstructorAgent


def test_validate_invocation_rejects_invalid_constructor_kwargs() -> None:
    config = invocation_config(
        agent_class=f"{__name__}:ConstructorAgent",
        init_kwargs={"unknown": True},
    )

    with pytest.raises(ValueError, match="init_kwargs"):
        validate_invocation(config)


def test_rejects_static_llm_override() -> None:
    with pytest.raises(ValidationError, match="Gym always injects"):
        invocation_config(init_kwargs={"llm": "provider-model"})


@pytest.mark.parametrize("aliases", [{"": "server"}, {"helper": " "}, {" ": "server"}])
def test_rejects_empty_model_aliases(aliases: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="model_aliases"):
        invocation_config(model_aliases=aliases)


def test_model_aliases_come_from_configuration() -> None:
    config = invocation_config(model_aliases={"helper": "helper_model"})
    assert config.model_aliases == {"helper": "helper_model"}
    assert invocation_config().model_aliases == {}


def test_load_invocation_adapter_rejects_missing_function() -> None:
    with pytest.raises(ValueError, match="has no attribute"):
        load_invocation_adapter(f"{__name__}:missing")


@pytest.mark.parametrize("name", ["synchronous_adapter", "NOT_AN_ADAPTER"])
def test_load_invocation_adapter_requires_async_function(name: str) -> None:
    with pytest.raises(ValueError, match="async function"):
        load_invocation_adapter(f"{__name__}:{name}")


def test_load_invocation_adapter_requires_agent_and_request_arguments() -> None:
    with pytest.raises(ValueError, match="agent and request"):
        load_invocation_adapter(f"{__name__}:one_argument_adapter")


def test_load_agent_class_rejects_non_agent_class() -> None:
    with pytest.raises(ValueError, match="subclass of nooa.Agent"):
        load_agent_class(f"{__name__}:NotAnAgent")


def test_load_agent_class_reports_relative_import_as_config_error() -> None:
    with pytest.raises(ValueError, match="could not import NOOA agent module") as exc_info:
        load_agent_class(".relative.agents:Agent")

    assert isinstance(exc_info.value.__cause__, TypeError)
