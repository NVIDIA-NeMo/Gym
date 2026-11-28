# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.models.token_capture.config import token_capture_enabled_for_agent


def test_shipped_agent_participates_in_collection_capture_without_all_agents(tmp_path):
    root = Path(__file__).resolve().parents[4]
    config = OmegaConf.load(
        root / "responses_api_agents/simple_agent_with_compaction/configs/simple_agent_with_compaction.yaml"
    )
    config.token_id_capture = {"enabled": True, "all_agents": False, "dir": str(tmp_path)}
    assert token_capture_enabled_for_agent(config, "simple_agent_with_compaction")
