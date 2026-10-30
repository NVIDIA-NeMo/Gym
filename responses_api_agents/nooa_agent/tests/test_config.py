# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from nooa import Agent
from pydantic import ValidationError

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
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
    assert agent_config().max_policy_calls is None
    assert agent_config(max_policy_calls=None).max_policy_calls is None
    assert agent_config(max_policy_calls=6).max_policy_calls == 6

    with pytest.raises(ValidationError, match="max_steps"):
        agent_config(max_steps=6)


@pytest.mark.parametrize("limit", [0, -1])
def test_agent_config_rejects_nonpositive_policy_call_limits(limit: int) -> None:
    with pytest.raises(ValidationError, match="max_policy_calls"):
        agent_config(max_policy_calls=limit)


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


@pytest.mark.parametrize("field", ["resources_server", "run_timeout_secs"])
def test_episode_policy_is_not_agent_configuration(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        agent_config(**{field: 1})


def test_several_workers_are_accepted() -> None:
    assert agent_config(num_workers=2).num_workers == 2


@pytest.mark.parametrize("field", ["sandbox_runner_timeout_seconds", "session_close_timeout_seconds"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_session_timeouts_must_be_positive_and_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        agent_config(**{field: value})


def test_session_timeout_defaults_and_fractional_budgets() -> None:
    default = agent_config()
    assert default.sandbox_runner_timeout_seconds == 21420
    assert default.session_close_timeout_seconds == 30
    configured = agent_config(sandbox_runner_timeout_seconds=0.5, session_close_timeout_seconds=0.25)
    assert configured.sandbox_runner_timeout_seconds == 0.5
    assert configured.session_close_timeout_seconds == 0.25


def test_shipped_yaml_declares_worker_and_close_defaults_below_episode_timeout() -> None:
    from pathlib import Path

    import yaml

    config = yaml.safe_load((Path(__file__).parents[1] / "configs/nooa_agent.yaml").read_text())
    agent = config["nooa_agent"]["responses_api_agents"]["nooa_agent"]
    environment = config["nooa_environment_server"]["environment_servers"]["single_agent_turn"]
    defaults = agent_config()
    assert agent["sandbox_runner_timeout_seconds"] == defaults.sandbox_runner_timeout_seconds
    assert agent["session_close_timeout_seconds"] == defaults.session_close_timeout_seconds
    # These clocks start at different times; this does not reserve cleanup time within the episode.
    assert agent["sandbox_runner_timeout_seconds"] < environment["default_episode_timeout_seconds"]
