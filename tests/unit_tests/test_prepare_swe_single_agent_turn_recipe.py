# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from scripts.prepare_swe_single_agent_turn_recipe import HARNESSES, SOURCES, _compose, _materialize_rows

from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


def test_materialization_preserves_task_content_and_order_in_canonical_nested_input(tmp_path: Path) -> None:
    rows = [
        {
            "agent_ref": {"name": f"{source}_opencode_sandboxed_agent"},
            "instance_id": f"task-{index}",
            "image_name": f"image:{index}",
            "responses_create_params": {"input": [{"role": "user", "content": f"task {index}"}]},
            "verifier_metadata": {"patch": "unchanged", "tests": [index]},
        }
        for index, source in enumerate(SOURCES)
    ]
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = source.read_bytes()
    assert _materialize_rows(source, output) == dict.fromkeys(SOURCES, 1)
    for before, after, taskset in zip(rows, map(json.loads, output.read_text().splitlines()), SOURCES, strict=True):
        assert after["task_id"] == {"taskset": taskset, "task_id": before["instance_id"]}
        assert "agent_ref" not in after
        params = before.pop("responses_create_params")
        before.pop("agent_ref")
        assert after["task_input"] == {"responses_create_params": params, "task_data": before}
        SingleAgentTurnTaskInput.model_validate(after["task_input"])
    assert source.read_bytes() == original


def test_unsupported_rows_are_rejected_instead_of_filtered(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"agent_ref": {"name": "denovo_swe_opencode_sandboxed_agent"}}) + "\n")
    with pytest.raises(ValueError, match="No rows are filtered"):
        _materialize_rows(source, tmp_path / "output.jsonl")


@pytest.mark.parametrize("harness", HARNESSES)
def test_composition_switches_direct_taskset_routes_without_changing_trainer_settings(harness: str) -> None:
    original = OmegaConf.create(
        {
            "policy": {"generation": {"temperature": 1.0, "top_p": 1.0}, "optimizer": {"lr": 1e-6}},
            "grpo": {"masked_reward_policy": "include"},
            "env": {"nemo_gym": {"sandbox": {"opensandbox": {"connection": {}}}}},
        }
    )
    result = _compose(original, harness=harness)
    assert result.policy == original.policy
    assert result.grpo == original.grpo
    gym = result.env.nemo_gym
    for source in SOURCES:
        assert gym.environment_server_routes[source] == f"{source}_environment"
        env = gym[f"{source}_environment"].environment_servers
        assert list(env) == ["single_agent_turn"]
        assert env.single_agent_turn.mask_incomplete_for_training is True
    assert HARNESSES[harness] in gym
