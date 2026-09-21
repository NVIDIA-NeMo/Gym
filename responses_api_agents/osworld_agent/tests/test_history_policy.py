# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for OSWorld protocol/history contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from responses_api_agents.osworld_agent.adapter_agents import NemotronV3NanoOmniAgent
from responses_api_agents.osworld_agent.agent_contract import (
    assert_train_eval_parity,
    resolve_agent_contract,
)
from responses_api_agents.osworld_agent.history_policy import (
    HistoryPolicySpec,
    HistoryPolicyState,
    plan_history,
)
from responses_api_agents.osworld_agent.runner_registry import resolve_runner_spec


def _nemotron_runner(**agent_kwargs: Any):
    return resolve_runner_spec(
        "nemotron_v3_nano_omni_agent",
        agent_kwargs=agent_kwargs,
    )


def test_fixed_and_hysteresis_have_distinct_semantic_identities() -> None:
    fixed = HistoryPolicySpec.fixed(3)
    hysteresis = HistoryPolicySpec.hysteresis(low_water=3, high_water=10)

    assert fixed.policy_id != hysteresis.policy_id
    assert fixed.to_config() == {
        "schema_version": 1,
        "name": "fixed",
        "params": {"keep_images": 3},
    }
    assert hysteresis.to_config() == {
        "schema_version": 1,
        "name": "hysteresis",
        "params": {"low_water": 3, "high_water": 10},
    }


def test_hysteresis_accumulates_to_ten_then_returns_to_three() -> None:
    spec = HistoryPolicySpec.hysteresis(low_water=3, high_water=10)
    state = HistoryPolicyState()
    image_counts = []
    compacted = []

    for completed_turns in range(12):
        plan = plan_history(spec, state, completed_turns=completed_turns)
        state = plan.next_state
        image_counts.append(len(plan.image_turns))
        compacted.append(plan.compaction_triggered)

    assert image_counts == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4]
    assert compacted == [False] * 10 + [True, False]


def test_legacy_3_10_3_and_explicit_policy_resolve_to_same_contract() -> None:
    legacy = resolve_agent_contract(
        runner_spec=_nemotron_runner(max_live_images=10),
        max_steps=200,
        max_trajectory_length=3,
        explicit_history_policy=None,
        model_protocol_id=None,
        rollout_purpose="training",
        parity_mode="strict",
    )
    explicit = resolve_agent_contract(
        runner_spec=_nemotron_runner(),
        max_steps=200,
        max_trajectory_length=3,
        explicit_history_policy={
            "name": "hysteresis",
            "params": {"low_water": 3, "high_water": 10},
        },
        model_protocol_id=None,
        rollout_purpose="evaluation",
        parity_mode="strict",
    )

    assert legacy.contract_id == explicit.contract_id
    assert legacy.history_policy == explicit.history_policy


def test_strict_parity_rejects_and_declared_parity_allows_a_policy_difference() -> None:
    fixed = resolve_agent_contract(
        runner_spec=_nemotron_runner(),
        max_steps=200,
        max_trajectory_length=3,
        explicit_history_policy={"name": "fixed", "params": {"keep_images": 3}},
        model_protocol_id=None,
        rollout_purpose="evaluation",
        parity_mode="strict",
    )
    hysteresis = resolve_agent_contract(
        runner_spec=_nemotron_runner(),
        max_steps=200,
        max_trajectory_length=3,
        explicit_history_policy={
            "name": "hysteresis",
            "params": {"low_water": 3, "high_water": 10},
        },
        model_protocol_id=None,
        rollout_purpose="training",
        parity_mode="strict",
    )

    with pytest.raises(ValueError, match="contracts differ"):
        assert_train_eval_parity(hysteresis, fixed, parity_mode="strict")
    assert_train_eval_parity(hysteresis, fixed, parity_mode="declared")


def test_fixed_three_prompt_rendering_remains_byte_for_byte_compatible() -> None:
    """Golden hash captured before HistoryPolicy extraction at 381b5bead."""

    agent = NemotronV3NanoOmniAgent(
        model="policy",
        max_steps=20,
        max_image_history_length=3,
        parse_retries=1,
    )
    payloads: list[dict[str, Any]] = []

    def call_llm(payload: dict[str, Any], _model: str) -> dict[str, Any]:
        payloads.append(payload)
        return {
            "content": "## Action:\nClick.\n## Code:\n```python\npyautogui.click(0.5, 0.5)\n```",
            "reasoning_content": f"Thought {len(payloads)}",
        }

    agent.call_llm = call_llm  # type: ignore[method-assign]
    for index in range(5):
        agent.predict(
            "Complete the task.",
            {"screenshot": f"png-{index + 1}".encode()},
        )

    messages = [payload["messages"] for payload in payloads]
    serialized = json.dumps(messages, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(serialized).hexdigest() == (
        "849f1759a4759d79bc93929de672c44ece06f3ece0654b01c25f221e7c9a6a07"
    )
    assert [payload["_osworld_log_context"]["history_policy_name"] for payload in payloads] == ["fixed"] * 5


def test_unknown_protocol_and_ambiguous_history_configuration_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown model_protocol_id"):
        resolve_agent_contract(
            runner_spec=_nemotron_runner(),
            max_steps=200,
            max_trajectory_length=3,
            explicit_history_policy=None,
            model_protocol_id="unregistered-v3x",
            rollout_purpose="training",
            parity_mode="strict",
        )

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_agent_contract(
            runner_spec=_nemotron_runner(max_live_images=10),
            max_steps=200,
            max_trajectory_length=3,
            explicit_history_policy={"name": "fixed", "params": {"keep_images": 3}},
            model_protocol_id=None,
            rollout_purpose="training",
            parity_mode="strict",
        )
