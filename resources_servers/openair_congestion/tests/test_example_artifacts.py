# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import pytest
from openair_congestion.tools import TOOLS


DATA_DIR = Path(__file__).parents[1] / "data"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_committed_example_metrics_and_rollouts_are_complete():
    examples = _jsonl(DATA_DIR / "example.jsonl")
    metrics = json.loads((DATA_DIR / "example_metrics.json").read_text())
    rollouts = _jsonl(DATA_DIR / "example_rollouts.jsonl")

    assert len(examples) == metrics["Number of examples"] == len(rollouts) == 5
    for index, (example, rollout) in enumerate(zip(examples, rollouts, strict=True)):
        assert rollout["seed"] == example["seed"]
        assert rollout["scenario_id"] == example["scenario_id"]
        assert rollout["_ng_task_index"] == index
        assert rollout["_ng_rollout_index"] == 0
        assert rollout["example_policy"] == "deterministic_scripted_relief_v1"
        assert rollout["max_steps"] == 2
        assert len(rollout["trajectory"]) == 2
        assert rollout["trajectory"][0]["action"]["name"] != "noop"
        assert rollout["trajectory"][1]["action"]["name"] == "noop"
        assert math.isfinite(rollout["reward"])
        assert rollout["terminated"] or rollout["truncated"]
        assert rollout["trajectory"]
        assert rollout["reward"] == pytest.approx(sum(step["reward"] for step in rollout["trajectory"]))
        assert all(step["next_observation"] for step in rollout["trajectory"])
        assert rollout["trajectory"][-1]["next_observation"]
        assert all(step["causal_action_effects"] is True for step in rollout["trajectory"])
        assert all(step["training_usable"] is True for step in rollout["trajectory"])
        assert all(step["guardrail_accepted"] is True for step in rollout["trajectory"])


def test_committed_model_messages_do_not_name_hidden_regimes():
    examples = _jsonl(DATA_DIR / "example.jsonl")
    hidden_names = {
        "prb_exhaustion",
        "bursty",
        "interference",
        "prach_storm",
        "qos_competition",
    }

    for example in examples:
        create_params = example["responses_create_params"]
        input_content = " ".join(str(message.get("content", "")) for message in create_params["input"])
        normalized_input = input_content.lower().replace("-", "_").replace(" ", "_")
        assert not any(name in normalized_input for name in hidden_names)

        model_facing_content = " ".join(
            [input_content, *(str(tool.get("description", "")) for tool in create_params["tools"])]
        ).lower()
        assert "deliverable #1" not in model_facing_content
        assert "synthetic-live" not in model_facing_content
        assert "connected t1" not in model_facing_content


def test_committed_tool_schemas_match_the_canonical_action_space():
    expected = [{"type": "function", **tool["function"], "strict": False} for tool in TOOLS]

    for example in _jsonl(DATA_DIR / "example.jsonl"):
        assert example["responses_create_params"]["tools"] == expected
