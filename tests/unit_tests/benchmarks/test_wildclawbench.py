# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check component discovery without importing its runtime into the core suite."""

from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.cli import env


def test_agent_is_in_actual_all_components_discovery(monkeypatch):
    captured = []
    monkeypatch.setattr(env, "get_global_config_dict", lambda: OmegaConf.create({"setup_only": True}))
    monkeypatch.setattr(env, "_setup_all", lambda paths, *args: captured.extend(paths))
    env.test_all()
    assert Path("responses_api_agents/wildclawbench_agent") in captured
