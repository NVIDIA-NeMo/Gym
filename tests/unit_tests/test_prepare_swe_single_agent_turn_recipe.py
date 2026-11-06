# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from scripts.prepare_swe_single_agent_turn_recipe import HARNESSES, SOURCES, _compose, _retag_rows


@pytest.mark.parametrize("harness", HARNESSES)
def test_retagging_preserves_every_task_field_and_row_order(tmp_path: Path, harness: str) -> None:
    rows = [
        {
            "agent_ref": {"name": f"{source}_opencode_sandboxed_agent", "type": "responses_api_agents"},
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

    assert _retag_rows(source, output, harness=harness) == dict.fromkeys(SOURCES, 1)
    for before, after, taskset in zip(rows, map(json.loads, output.read_text().splitlines()), SOURCES, strict=True):
        assert after.pop("agent_ref") == {
            "name": f"{taskset}_{HARNESSES[harness]}",
            "type": "responses_api_agents",
        }
        before.pop("agent_ref")
        assert after == before
    assert source.read_bytes() == original


def test_unsupported_rows_are_rejected_instead_of_filtered(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"agent_ref": {"name": "denovo_swe_opencode_sandboxed_agent"}}) + "\n")
    with pytest.raises(ValueError, match="No rows are filtered"):
        _retag_rows(source, tmp_path / "output.jsonl", harness="pi")


@pytest.mark.parametrize("harness", HARNESSES)
def test_composition_preserves_trainer_settings_and_gives_each_alias_one_environment(harness: str) -> None:
    original = OmegaConf.create(
        {
            "policy": {"generation": {"temperature": 1.0, "top_p": 1.0}, "optimizer": {"lr": 1e-6}},
            "grpo": {"masked_sample_rewards_in_baseline": True},
            "env": {
                "mask_sample_rules": [{"field": "opencode_finished", "equals": False, "name": "harness_unfinished"}],
                "nemo_gym": {
                    "config_paths": [],
                    "sandbox": {"opensandbox": {"connection": {"api_key": "${oc.env:OPENSANDBOX_API_KEY}"}}},
                    **{f"{source}_opencode_sandboxed_agent": {} for source in (*SOURCES, "denovo_swe")},
                    **{
                        f"{source}_resources_server": {"resources_servers": {source: {"evaluation_timeout": 1200}}}
                        for source in SOURCES
                    },
                },
            },
        }
    )
    result = _compose(original, harness=harness)
    assert result.policy == original.policy
    assert result.grpo == original.grpo
    assert original.env.mask_sample_rules[0].field == "opencode_finished"
    assert result.env.mask_sample_rules[0].field == "agent_execution_finished"
    assert result.env.mask_sample_rules[0].equals is False
    gym = result.env.nemo_gym
    aliases = []
    for source in SOURCES:
        env = gym[f"{source}_environment"].environment_servers.single_agent_turn_legacy
        aliases.append(env.agent_server.name)
        assert env.resources_server.name == f"{source}_resources_server"
        assert env.resources_tool_transports == []
        assert gym[env.agent_server.name]._inherit_from == HARNESSES[harness]
        assert gym[f"{source}_resources_server"].resources_servers[source].evaluation_timeout == 1200
    assert len(set(aliases)) == len(SOURCES)
