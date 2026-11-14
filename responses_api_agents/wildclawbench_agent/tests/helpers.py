# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct an agent with deterministic policy and image metadata for component tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from responses_api_agents.wildclawbench_agent.app import WildClawAgent, WildClawConfig


def make_agent(tmp_path: Path, worker_python: str) -> WildClawAgent:
    """Build the real agent while replacing startup policy/image inspection."""
    config = WildClawConfig(
        name="wildclaw_agent",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        source_root=str(tmp_path),
        worker_python=worker_python,
        output_root=str(tmp_path / "episodes"),
        model_name="policy",
        model_server=dict(type="responses_api_models", name="policy_model"),
        sandbox_runtime="docker",
    )
    with (
        patch(
            "responses_api_agents.wildclawbench_agent.app.get_global_config_dict",
            return_value=OmegaConf.create(
                {
                    "policy_model": {
                        "responses_api_models": {
                            "vllm_model": {
                                "entrypoint": "wildclawbench.py",
                                "uses_reasoning_parser": True,
                                "propagate_context_overflow_errors": True,
                            }
                        }
                    }
                }
            ),
        ),
        patch(
            "responses_api_agents.wildclawbench_agent.app.subprocess.run",
            return_value=SimpleNamespace(stdout="sha256:" + "a" * 64),
        ),
    ):
        return WildClawAgent(config=config, server_client=MagicMock(spec=ServerClient))
