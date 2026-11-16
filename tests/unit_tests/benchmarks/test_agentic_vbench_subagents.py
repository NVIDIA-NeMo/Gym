# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Agentic-VBench training entry's subagent overlay is a session configuration the OpenCode harness accepts."""

from pathlib import Path

from omegaconf import OmegaConf

from responses_api_agents.opencode_agent.app import _configured_subagents, _session_opencode_config_error


ROOT = Path(__file__).resolve().parents[3]
OVERLAY = ROOT / "environments/harbor/agentic_vbench_train/subagents.yaml"


def _overlay_opencode_config() -> dict:
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAY), resolve=True)
    return overlay["opencode_agent"]["responses_api_agents"]["opencode_agent"]["opencode_config"]


def test_the_subagent_overlay_defines_one_report_only_analyst_the_session_accepts() -> None:
    config = _overlay_opencode_config()
    assert _session_opencode_config_error(config) is None
    assert _configured_subagents(config) == ["analyst"]
    analyst = config["agent"]["analyst"]
    assert analyst["mode"] == "subagent"
    assert analyst["steps"] == 10
    # Depth one: the analyst may not delegate; everything else it inherits from the session's permission profile.
    assert analyst["permission"] == {"task": "deny"}
    assert "/workspace/output" in analyst["description"] and "/workspace/output" in analyst["prompt"]
    assert "model" not in analyst
    # The built-in subagents are hidden so the task tool offers the analyst alone.
    assert config["agent"]["general"] == {"disable": True}
    assert config["agent"]["explore"] == {"disable": True}


def test_the_subagent_overlay_leaves_the_session_configuration_otherwise_alone() -> None:
    overlay = OmegaConf.to_container(OmegaConf.load(OVERLAY), resolve=True)
    assert set(overlay) == {"opencode_agent"}
    harness = overlay["opencode_agent"]["responses_api_agents"]["opencode_agent"]
    assert set(harness) == {"opencode_config"}
    assert set(harness["opencode_config"]) == {"agent"}
