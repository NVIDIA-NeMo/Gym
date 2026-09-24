# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the real-model NOOA stateful smoke rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _arguments(item: dict[str, Any]) -> dict[str, Any]:
    arguments = item.get("arguments", {})
    return json.loads(arguments) if isinstance(arguments, str) else arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()

    with args.rollouts.open(encoding="utf-8") as rollouts_file:
        rollouts = [json.loads(line) for line in rollouts_file if line.strip()]

    assert len(rollouts) == 1, f"expected one rollout, found {len(rollouts)}"
    rollout = rollouts[0]
    response = rollout["response"]
    assert response["status"] == "completed", response.get("error")
    assert response["error"] is None
    assert response["usage"]["input_tokens"] > 0
    assert response["usage"]["output_tokens"] > 0
    assert rollout["reward"] == 1.0
    assert rollout["initial_count"] == 3
    assert rollout["expected_count"] == 6

    resource_names = {"increment_counter", "get_counter_value"}
    calls = [item for item in response["output"] if item.get("type") == "function_call"]
    resource_calls = [(item["name"], _arguments(item)) for item in calls if item.get("name") in resource_names]
    assert resource_calls == [
        ("increment_counter", {"count": 1}),
        ("increment_counter", {"count": 2}),
        ("get_counter_value", {}),
    ]

    trajectory = rollout["ng_trajectory"]
    assert trajectory["turns"]
    assert any(turn["model_calls"] for turn in trajectory["turns"])
    tool_calls = [call for call in trajectory["tool_calls"] if call["tool_name"] in resource_names]
    assert [call["tool_name"] for call in tool_calls] == [name for name, _ in resource_calls]
    assert all(call["status"] == "completed" for call in tool_calls)
    assert json.loads(tool_calls[-1]["output"]) == {"count": 6}

    observations = rollout["ng_agent_observations"]
    assert observations["source"] == "nooa"
    assert rollout["agent_ref"] == {"name": "example_session_state_mgmt_nooa_agent"}


if __name__ == "__main__":
    main()
