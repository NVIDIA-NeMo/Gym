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

from typing import Any

import pytest
from nooa import Agent
from pydantic import BaseModel, ValidationError

from responses_api_agents.nooa_agent.config import (
    NOOAArgumentBinding,
    NOOAInvocationConfig,
    load_agent_class,
    validate_invocation,
)
from responses_api_agents.nooa_agent.mapping import (
    latest_user_text,
    materialize_arguments,
    normalize_responses_input,
    resolve_source,
)


class ExampleAgent(Agent):
    async def analyze(self, text: str, customer_id: str, optional: bool = False) -> str:
        """Analyze customer feedback."""

        ...


class NotAnAgent:
    pass


class ConstructorAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label

    async def analyze(self, text: str) -> str:
        return text


class SyncAgent(Agent):
    def analyze(self, text: str) -> str:
        return text


class InputItem(BaseModel):
    role: str
    content: str


def invocation_config(**overrides: Any) -> NOOAInvocationConfig:
    values = {
        "agent_class": f"{__name__}:ExampleAgent",
        "entrypoint": "analyze",
        "arguments": {
            "text": {
                "source": "responses_create_params.input",
                "transform": "latest_user_text",
            },
            "customer_id": {"source": "agent_inputs.customer_id"},
        },
    }
    values.update(overrides)
    return NOOAInvocationConfig.model_validate(values)


def test_materialize_arguments_from_complete_run_row() -> None:
    config = invocation_config()
    row = {
        "responses_create_params": {
            "input": [
                {"role": "developer", "content": "Classify customer feedback."},
                {"role": "user", "content": [{"type": "input_text", "text": "Shipping was slow."}]},
            ]
        },
        "agent_inputs": {"customer_id": "customer-42"},
    }

    assert materialize_arguments(row, config.arguments) == {
        "text": "Shipping was slow.",
        "customer_id": "customer-42",
    }


@pytest.mark.parametrize(
    "source",
    [
        "answer",
        "label",
        "expected_output",
        "ground_truth",
        "target",
        "verifier_metadata.answer",
        "agent_ref.name",
        "_ng_rollout_id",
        "response.output",
        "reward",
        "responses_create_params.model",
        "responses_create_params.input_extra",
        "agent_inputs.items.__class__",
    ],
)
def test_rejects_sources_outside_agent_visible_allowlist(source: str) -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        NOOAArgumentBinding(source=source)


@pytest.mark.parametrize(
    "source",
    [
        "responses_create_params.input",
        "responses_create_params.input.0.content",
        "agent_inputs",
        "agent_inputs.customer_id",
    ],
)
def test_allows_explicitly_agent_visible_sources(source: str) -> None:
    assert NOOAArgumentBinding(source=source).source == source


def test_rejects_unknown_transform() -> None:
    with pytest.raises(ValidationError, match="unknown transform"):
        NOOAArgumentBinding(source="agent_inputs.customer_id", transform="lambda value: value")


def test_resolves_pydantic_models_and_sequence_indexes() -> None:
    row = {"agent_inputs": {"items": [InputItem(role="user", content="hello")]}}

    assert resolve_source(row, "agent_inputs.items.0.content") == "hello"


@pytest.mark.parametrize("source", ["", "items..content", "items.-1", "items.not-valid"])
def test_rejects_malformed_source_paths(source: str) -> None:
    with pytest.raises(ValueError, match="dotted path"):
        resolve_source({}, source)


def test_missing_source_reports_full_path_and_segment() -> None:
    with pytest.raises(ValueError, match=r"'agent_inputs\.customer\.profile\.id'.*'profile'"):
        resolve_source({"agent_inputs": {"customer": {}}}, "agent_inputs.customer.profile.id")


def test_latest_user_text_uses_last_user_message() -> None:
    assert (
        latest_user_text(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        )
        == "second"
    )


def test_normalize_responses_input_dumps_models() -> None:
    assert normalize_responses_input([InputItem(role="user", content="hello")]) == [
        {"role": "user", "content": "hello"}
    ]


def test_validate_invocation_returns_agent_and_entrypoint() -> None:
    agent_class, entrypoint = validate_invocation(invocation_config())

    assert agent_class is ExampleAgent
    assert entrypoint is ExampleAgent.analyze


def test_validate_invocation_rejects_missing_required_mapping() -> None:
    config = invocation_config(
        arguments={
            "text": {
                "source": "responses_create_params.input",
                "transform": "latest_user_text",
            }
        }
    )

    with pytest.raises(ValueError, match="customer_id"):
        validate_invocation(config)


def test_validate_invocation_rejects_unknown_argument() -> None:
    config = invocation_config(
        arguments={
            "text": {"source": "responses_create_params.input", "transform": "latest_user_text"},
            "customer_id": {"source": "agent_inputs.customer_id"},
            "answer": {"source": "agent_inputs.answer"},
        }
    )

    with pytest.raises(ValueError, match="answer"):
        validate_invocation(config)


def test_validate_invocation_checks_constructor_with_injected_llm() -> None:
    config = invocation_config(
        agent_class=f"{__name__}:ConstructorAgent",
        init_kwargs={"label": "production"},
        arguments={"text": {"source": "responses_create_params.input", "transform": "latest_user_text"}},
    )

    agent_class, _ = validate_invocation(config)

    assert agent_class is ConstructorAgent


def test_validate_invocation_rejects_invalid_constructor_kwargs() -> None:
    config = invocation_config(
        agent_class=f"{__name__}:ConstructorAgent",
        init_kwargs={"unknown": True},
        arguments={"text": {"source": "responses_create_params.input", "transform": "latest_user_text"}},
    )

    with pytest.raises(ValueError, match="init_kwargs"):
        validate_invocation(config)


def test_rejects_static_llm_override() -> None:
    with pytest.raises(ValidationError, match="Gym always injects"):
        invocation_config(init_kwargs={"llm": "provider-model"})


def test_validate_invocation_rejects_synchronous_entrypoint() -> None:
    config = invocation_config(
        agent_class=f"{__name__}:SyncAgent",
        arguments={"text": {"source": "responses_create_params.input", "transform": "latest_user_text"}},
    )

    with pytest.raises(ValueError, match="must be async"):
        validate_invocation(config)


def test_rejects_private_or_invalid_argument_names() -> None:
    with pytest.raises(ValidationError, match="public Python identifiers"):
        invocation_config(
            arguments={
                "not-valid": {"source": "agent_inputs.customer_id"},
                "_private": {"source": "agent_inputs.customer_id"},
            }
        )


def test_load_agent_class_rejects_non_agent_class() -> None:
    with pytest.raises(ValueError, match="subclass of nooa.Agent"):
        load_agent_class(f"{__name__}:NotAnAgent")
