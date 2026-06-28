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


def test_prompt_agent_default_matches_upstream_osworld_defaults() -> None:
    spec = resolve_runner_spec("prompt_agent")

    assert spec.kind == "prompt_agent"
    assert spec.agent_class_path == "mm_agents.agent.PromptAgent"
    assert spec.action_space == "computer_13"
    assert spec.observation_type == "screenshot_a11y_tree"


def test_pointer_agent_runner_uses_upstream_pointer_agent() -> None:
    spec = resolve_runner_spec("pointer_agent")

    assert spec.kind == "pointer_agent"
    assert spec.env_class_path == "desktop_env.desktop_env_pointer.DesktopEnv"
    assert spec.agent_class_path == "mm_agents.pointer.PointerAgent"
    assert spec.action_space == "pyautogui"
    assert spec.observation_type == "screenshot"
    assert spec.agent_kwargs == {"provider_name": "anthropic"}


def test_m3_agent_runner_uses_official_osworld_scaffold() -> None:
    spec = resolve_runner_spec("m3_agent")

    assert spec.kind == "m3_agent"
    assert spec.agent_class_path == "mm_agents.m3.M3Agent"
    assert spec.action_space == "pyautogui"
    assert spec.observation_type == "screenshot"


@pytest.mark.parametrize(
    ("runner_name", "action_space", "observation_type"),
    [
        ("prompt_agent_screenshot_pyautogui", "pyautogui", "screenshot"),
        ("prompt_agent_computer_13", "computer_13", "screenshot"),
        ("prompt_agent_a11y_tree_pyautogui", "pyautogui", "a11y_tree"),
        ("prompt_agent_a11y_tree_computer_13", "computer_13", "a11y_tree"),
        ("prompt_agent_screenshot_a11y_tree_pyautogui", "pyautogui", "screenshot_a11y_tree"),
        ("prompt_agent_screenshot_a11y_tree_computer_13", "computer_13", "screenshot_a11y_tree"),
        ("prompt_agent_som_pyautogui", "pyautogui", "som"),
    ],
)
def test_native_prompt_agent_runner_matrix(
    runner_name: str,
    action_space: str,
    observation_type: str,
) -> None:
    spec = resolve_runner_spec(runner_name)

    assert spec.kind == "prompt_agent"
    assert spec.agent_class_path == "mm_agents.agent.PromptAgent"
    assert spec.action_space == action_space
    assert spec.observation_type == observation_type


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
