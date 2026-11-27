# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit v4-only entrypoint into the unchanged Gym task/evaluator loop."""

from __future__ import annotations

import threading
from typing import Any, Callable

from .agent import Nano4Agent, stable_id
from .coordinate_compat import STRICT_COORDINATES
from .protocol import PROTOCOL_ID
from .session import CURRENT_SESSION, Session


RUNNER_NAME = "nemotron_nano4_reference_agent"
_REGISTER_LOCK = threading.Lock()


def register_runner(coordinate_policy: str = STRICT_COORDINATES) -> str:
    from responses_api_agents.osworld_agent.runner_registry import RUNNER_REGISTRY, RunnerSpec

    runner_name = RUNNER_NAME if coordinate_policy == STRICT_COORDINATES else "nemotron_nano4_agent"
    from .coordinate_compat import COMPAT_PROTOCOL_ID

    if "model_protocol_id" not in RunnerSpec.__dataclass_fields__:
        raise RuntimeError("Nano4 requires the model_protocol_agent runner and Python-suite support from Gym PR #5101")
    spec = RunnerSpec(
        name=runner_name,
        kind="model_protocol_agent",
        env_class_path="responses_api_agents.osworld_agent.nano4.environment.Nano4DesktopEnv",
        agent_class_path="responses_api_agents.osworld_agent.nano4.agent.Nano4Agent",
        model_protocol_id=PROTOCOL_ID if coordinate_policy == STRICT_COORDINATES else COMPAT_PROTOCOL_ID,
    )
    with _REGISTER_LOCK:
        existing = RUNNER_REGISTRY.get(runner_name)
        if existing is not None and existing != spec:
            raise ValueError("A conflicting v4 runner is already registered")
        RUNNER_REGISTRY[runner_name] = spec
    return runner_name


def run_task(
    task_config: dict[str, Any],
    messages_model_fn: Callable[..., Any],
    *,
    reference_root: str,
    model: str,
    max_steps: int = 100,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    top_p: float = 0.95,
    screen_size: tuple[int, int] = (1920, 1080),
    actions_per_chunk: int = 10,
    parse_retries: int = 5,
    coordinate_policy: str = STRICT_COORDINATES,
    parse_error_feedback: bool | None = None,
    client_password: str = "password",
    **environment_options: Any,
) -> Any:
    """Run v4 without editing or selecting any v3/v3.5 adapter implementation.

    This is a separate direct-library entrypoint, matching the previous OSWorld
    evaluation driver's use of Gym. It does not route through legacy protocol
    or history resolvers and cannot silently fall back to them.
    """
    if environment_options.get("resources_server_url"):
        raise ValueError("V4 requires a controller-capable Gym Sandbox/local DesktopEnv for canonical feedback")
    forbidden = {
        "runner_name",
        "agent_class_path",
        "env_class_path",
        "agent_kwargs",
        "history_policy",
        "model_protocol_id",
        "agent_contract",
        "max_trajectory_length",
        "policy_model_name",
        "policy_max_tokens",
        "policy_temperature",
        "policy_top_p",
    }.intersection(environment_options)
    if forbidden:
        raise ValueError(f"V4 routing and history are owned by its independent entrypoint: {sorted(forbidden)}")
    agent_options = {
        "reference_root": reference_root,
        "actions_per_chunk": actions_per_chunk,
        "parse_retries": parse_retries,
        "coordinate_policy": coordinate_policy,
        "parse_error_feedback": parse_error_feedback,
    }
    prototype = Nano4Agent(model, max_steps, screen_size=screen_size, client_password=client_password, **agent_options)
    contract = prototype.agent_contract
    contract["agent_contract_id"] = stable_id("nemotron-v4-agent-contract", contract)
    runner_name = register_runner(coordinate_policy)
    from responses_api_agents.osworld_agent.client import run_osworld_task

    def no_generic_model_path(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("The independent v4 messages path was not selected")

    session = Session(reference=prototype.reference)
    token = CURRENT_SESSION.set(session)
    try:
        return run_osworld_task(
            task_config,
            no_generic_model_path,
            runner_name=runner_name,
            env_class_path="responses_api_agents.osworld_agent.nano4.environment.Nano4DesktopEnv",
            agent_class_path="responses_api_agents.osworld_agent.nano4.agent.Nano4Agent",
            agent_kwargs=agent_options,
            history_policy=prototype.history_contract,
            model_protocol_id=prototype.model_protocol_id,
            agent_contract=contract,
            messages_model_fn=messages_model_fn,
            policy_model_name=model,
            policy_max_tokens=max_tokens,
            policy_temperature=temperature,
            policy_top_p=top_p,
            max_steps=max_steps,
            max_trajectory_length=actions_per_chunk + 1,
            screen_size=screen_size,
            client_password=client_password,
            **environment_options,
        )
    finally:
        CURRENT_SESSION.reset(token)
