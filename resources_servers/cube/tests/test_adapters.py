# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from resources_servers.cube.adapters import action_schemas_to_openai_tools, observation_to_input_messages
from resources_servers.cube.schemas import CubeEnvStateEasyInputMessage


@dataclass
class _FakeActionSchema:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class _FakeObservation:
    messages: list[dict[str, Any]]

    def to_llm_messages(self) -> list[dict[str, Any]]:
        return self.messages


def test_action_schemas_to_openai_tools_strict() -> None:
    schemas = [
        _FakeActionSchema(
            name="click",
            description="Click",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer", "description": "x"}},
                "required": ["x"],
            },
        )
    ]
    tools = action_schemas_to_openai_tools(schemas)  # type: ignore[arg-type]
    assert len(tools) == 1
    assert tools[0]["name"] == "click"
    assert tools[0]["strict"] is True
    assert tools[0]["parameters"]["additionalProperties"] is False
    assert tools[0]["parameters"]["required"] == ["x"]


def test_action_schemas_openai_strict_required_covers_all_properties() -> None:
    """OpenAI rejects strict tools if ``required`` omits any ``properties`` key (e.g. optional ``button``)."""
    schemas = [
        _FakeActionSchema(
            name="click",
            description="Click",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "button": {"type": "string"},
                },
                "required": ["x"],
            },
        )
    ]
    tools = action_schemas_to_openai_tools(schemas)  # type: ignore[arg-type]
    assert tools[0]["parameters"]["required"] == ["button", "x"]


def test_observation_text_to_messages() -> None:
    obs = _FakeObservation(messages=[{"role": "user", "content": "hello"}])
    msgs = observation_to_input_messages(obs)  # type: ignore[arg-type]
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hello"


def test_observation_screenshot_is_env_state() -> None:
    obs = _FakeObservation(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}],
            }
        ]
    )
    msgs = observation_to_input_messages(obs)  # type: ignore[arg-type]
    assert len(msgs) == 1
    assert isinstance(msgs[0], CubeEnvStateEasyInputMessage)
    assert msgs[0].is_env_state is True


def test_tool_role_becomes_user_text() -> None:
    obs = _FakeObservation(
        messages=[
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        ]
    )
    msgs = observation_to_input_messages(obs)  # type: ignore[arg-type]
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert "[tool result]" in str(msgs[0].content)
