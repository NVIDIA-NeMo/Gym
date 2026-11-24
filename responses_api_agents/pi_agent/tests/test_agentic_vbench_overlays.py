# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Agentic-VBench entries' Pi overlays are configurations the Pi harness accepts.

A harness-suite test: importing the harness module from a core test would count the whole harness in the core
coverage total.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from responses_api_agents.pi_agent.app import PiAgentConfig


ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ROOT / "responses_api_agents/pi_agent/configs/pi_agent.yaml"
OVERLAYS = [
    ROOT / "environments/harbor/agentic_vbench/harbor_sandbox_pi.yaml",
    ROOT / "environments/harbor/agentic_vbench_train/harbor_train_pi.yaml",
]


def _harness_config(overlay: Path, **settings: int) -> PiAgentConfig:
    merged = OmegaConf.merge(
        OmegaConf.load(SHIPPED), OmegaConf.load(overlay), {"policy_model_name": "policy", **settings}
    )
    block = OmegaConf.to_container(merged["pi_agent"]["responses_api_agents"]["pi_agent"], resolve=True)
    # The server's own keys, which the launcher fills in.
    return PiAgentConfig(host="127.0.0.1", port=1, name="pi_agent", **block)


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda path: path.name)
def test_the_overlay_over_the_shipped_config_is_a_pi_configuration_with_image_input(overlay: Path) -> None:
    config = _harness_config(overlay)
    # The prompt has no image; without this Pi omits every frame its tools read.
    assert config.image_input is True
    assert config.pi_version == "0.80.2"
    # The benchmark overlay lets each task's own budget apply; the training overlay caps an episode at 1800 s.
    assert config.timeout == (10800 if overlay == OVERLAYS[0] else 1800)


def test_the_training_overlay_turns_capture_on_and_compaction_off_within_the_policy_budgets() -> None:
    config = _harness_config(OVERLAYS[1])
    assert config.token_id_capture is True
    assert config.auto_compaction is False
    assert (config.timeout, config.context_window, config.max_output_tokens) == (1800, 49152, 8192)
    tuned = _harness_config(OVERLAYS[1], agentic_vbench_context_tokens=65536, agentic_vbench_output_tokens=4096)
    assert (tuned.context_window, tuned.max_output_tokens) == (65536, 4096)
