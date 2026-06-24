# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OSWorld runner registry wiring."""

from __future__ import annotations

import pytest

from responses_api_agents.osworld_agent.runner_registry import (
    DEFAULT_RUNNER_NAME,
    resolve_runner_spec,
)


def test_default_runner_preserves_existing_pyautogui_path() -> None:
    spec = resolve_runner_spec(DEFAULT_RUNNER_NAME)

    assert spec.kind == "gym_policy"
    assert spec.env_class_path == "desktop_env.desktop_env.DesktopEnv"
    assert spec.action_space == "pyautogui"
    assert spec.observation_type == "screenshot"
    assert spec.agent_class_path is None


def test_prompt_agent_computer_13_runner_uses_native_prompt_agent() -> None:
    spec = resolve_runner_spec("prompt_agent_computer_13")

    assert spec.kind == "prompt_agent"
    assert spec.agent_class_path == "mm_agents.agent.PromptAgent"
    assert spec.action_space == "computer_13"
    assert spec.observation_type == "screenshot"


def test_runner_overrides_are_applied() -> None:
    spec = resolve_runner_spec(
        "prompt_agent",
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        env_class_path="custom.Env",
        agent_class_path="custom.Agent",
        agent_kwargs={"foo": "bar"},
    )

    assert spec.kind == "prompt_agent"
    assert spec.action_space == "computer_13"
    assert spec.observation_type == "screenshot_a11y_tree"
    assert spec.env_class_path == "custom.Env"
    assert spec.agent_class_path == "custom.Agent"
    assert spec.agent_kwargs == {"foo": "bar"}


def test_unknown_runner_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown OSWorld runner_name"):
        resolve_runner_spec("does_not_exist")
