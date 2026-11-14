# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper
from responses_api_agents.deepcad_opencode.prepare_manifest import prepare


def test_lazy_manifest_routes_to_configured_agent(tmp_path):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "prepared.jsonl"
    row = {
        "verifier_metadata": {"source_step_path": "/missing/reference.step"},
        "responses_create_params": {"input": []},
    }
    source.write_text(json.dumps(row) + "\n")
    assert prepare(source, output) == 1
    prepared = json.loads(output.read_text())
    config = OmegaConf.load(Path(__file__).parents[1] / "configs/deepcad_opencode.yaml")
    RolloutCollectionHelper.resolve_task_sources([prepared], config)
    assert prepared["agent_ref"]["name"] == "deepcad_opencode_agent"
    assert prepared["verifier_metadata"] == row["verifier_metadata"]
    assert prepared["responses_create_params"] == row["responses_create_params"]
