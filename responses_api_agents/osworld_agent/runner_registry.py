# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSWorld runner registry.

The Gym wrapper should preserve OSWorld's native agent/runner contracts where
possible. This registry keeps those contracts explicit: each runner declares
which DesktopEnv class, action space, observation mode, and agent class it
expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, Literal, Optional


RunnerKind = Literal["gym_policy", "prompt_agent"]


@dataclass(frozen=True)
class RunnerSpec:
    """Static wiring for one OSWorld-style runner."""

    name: str
    kind: RunnerKind
    env_class_path: str = "desktop_env.desktop_env.DesktopEnv"
    action_space: str = "pyautogui"
    observation_type: str = "screenshot"
    agent_class_path: Optional[str] = None
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)


DEFAULT_RUNNER_NAME = "gym_pyautogui"


_PROMPT_AGENT = "mm_agents.agent.PromptAgent"


RUNNER_REGISTRY: Dict[str, RunnerSpec] = {
    # Backward-compatible path used by the current Gym integration. It keeps
    # Gym in charge of prompt construction and model calls.
    DEFAULT_RUNNER_NAME: RunnerSpec(
        name=DEFAULT_RUNNER_NAME,
        kind="gym_policy",
        action_space="pyautogui",
        observation_type="screenshot",
    ),
    # Native OSWorld PromptAgent path. We patch PromptAgent.call_llm to route
    # its already-constructed messages to the Gym policy model, but keep
    # PromptAgent's prompt/action parsing behavior. The unqualified
    # "prompt_agent" entry mirrors OSWorld's upstream PromptAgent defaults.
    "prompt_agent": RunnerSpec(
        name="prompt_agent",
        kind="prompt_agent",
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_screenshot_pyautogui": RunnerSpec(
        name="prompt_agent_screenshot_pyautogui",
        kind="prompt_agent",
        action_space="pyautogui",
        observation_type="screenshot",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_computer_13": RunnerSpec(
        name="prompt_agent_computer_13",
        kind="prompt_agent",
        action_space="computer_13",
        observation_type="screenshot",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_a11y_tree_pyautogui": RunnerSpec(
        name="prompt_agent_a11y_tree_pyautogui",
        kind="prompt_agent",
        action_space="pyautogui",
        observation_type="a11y_tree",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_a11y_tree_computer_13": RunnerSpec(
        name="prompt_agent_a11y_tree_computer_13",
        kind="prompt_agent",
        action_space="computer_13",
        observation_type="a11y_tree",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_screenshot_a11y_tree_pyautogui": RunnerSpec(
        name="prompt_agent_screenshot_a11y_tree_pyautogui",
        kind="prompt_agent",
        action_space="pyautogui",
        observation_type="screenshot_a11y_tree",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_screenshot_a11y_tree_computer_13": RunnerSpec(
        name="prompt_agent_screenshot_a11y_tree_computer_13",
        kind="prompt_agent",
        action_space="computer_13",
        observation_type="screenshot_a11y_tree",
        agent_class_path=_PROMPT_AGENT,
    ),
    "prompt_agent_som_pyautogui": RunnerSpec(
        name="prompt_agent_som_pyautogui",
        kind="prompt_agent",
        action_space="pyautogui",
        observation_type="som",
        agent_class_path=_PROMPT_AGENT,
    ),
}


def load_attr(import_path: str) -> Any:
    """Load ``package.module.attr`` lazily."""

    module_name, attr_name = import_path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def resolve_runner_spec(
    runner_name: str,
    *,
    action_space: Optional[str] = None,
    observation_type: Optional[str] = None,
    env_class_path: Optional[str] = None,
    agent_class_path: Optional[str] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
) -> RunnerSpec:
    """Resolve configured runner values into an immutable spec."""

    if runner_name not in RUNNER_REGISTRY:
        allowed = ", ".join(sorted(RUNNER_REGISTRY))
        raise ValueError(f"Unknown OSWorld runner_name={runner_name!r}. Allowed: {allowed}")

    base = RUNNER_REGISTRY[runner_name]
    merged_agent_kwargs = dict(base.agent_kwargs)
    if agent_kwargs:
        merged_agent_kwargs.update(agent_kwargs)

    return RunnerSpec(
        name=base.name,
        kind=base.kind,
        env_class_path=env_class_path or base.env_class_path,
        action_space=action_space or base.action_space,
        observation_type=observation_type or base.observation_type,
        agent_class_path=agent_class_path or base.agent_class_path,
        agent_kwargs=merged_agent_kwargs,
    )
