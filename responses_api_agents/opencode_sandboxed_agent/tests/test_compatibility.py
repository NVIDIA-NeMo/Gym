# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from responses_api_agents.opencode_agent.legacy import LegacyOpenCodeAgent, LegacyOpenCodeAgentConfig
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


def test_existing_entrypoint_reuses_the_legacy_implementation() -> None:
    assert OpenCodeSandboxedAgent is LegacyOpenCodeAgent
    assert OpenCodeSandboxedAgentConfig is LegacyOpenCodeAgentConfig
    config_path = Path(__file__).parents[1] / "configs" / "benchmark.yaml"
    config = OmegaConf.load(config_path)
    settings = config.opencode_benchmark_agent.responses_api_agents.opencode_sandboxed_agent
    assert settings.entrypoint == "app.py"
    assert settings.execution_failure_reward_zero is True
    assert settings.output_token_policy == "remaining_context"
    assert settings.preinstalled_opencode is True
