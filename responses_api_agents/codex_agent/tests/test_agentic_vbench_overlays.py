# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Agentic-VBench entries' Codex overlays are configurations the Codex harness and the vLLM model server accept.

A harness-suite test: importing the harness module from a core test would count the whole harness in the core
coverage total.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from responses_api_agents.codex_agent.app import CodexAgentConfig
from responses_api_models.vllm_model.app import VLLMModelConfig


ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ROOT / "responses_api_agents/codex_agent/configs/codex_agent.yaml"
OVERLAYS = [
    ROOT / "environments/harbor/agentic_vbench/harbor_sandbox_codex.yaml",
    ROOT / "environments/harbor/agentic_vbench_train/harbor_train_codex.yaml",
]


def _harness_config(overlay: Path, **settings: int) -> CodexAgentConfig:
    merged = OmegaConf.merge(
        OmegaConf.load(SHIPPED), OmegaConf.load(overlay), {"policy_model_name": "policy", **settings}
    )
    block = OmegaConf.to_container(merged["codex_agent"]["responses_api_agents"]["codex_agent"], resolve=True)
    # The server's own keys, which the launcher fills in.
    return CodexAgentConfig(host="127.0.0.1", port=1, name="codex_agent", **block)


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda path: path.name)
def test_the_overlay_over_the_shipped_config_is_a_codex_configuration(overlay: Path) -> None:
    config = _harness_config(overlay)
    assert config.codex_version == "0.144.4"
    # The benchmark overlay lets each task's own budget apply; the training overlay caps an episode at 1800 s.
    assert config.timeout == (10800 if overlay == OVERLAYS[0] else 1800)


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda path: path.name)
def test_the_overlay_turns_on_a_switch_the_vllm_model_server_has(overlay: Path) -> None:
    switch = OmegaConf.to_container(OmegaConf.load(overlay))["policy_model"]["responses_api_models"]["vllm_model"]
    assert switch == {"inline_tool_output_images": True}
    # `view_image` returns a frame inside the tool result; without the switch the conversion refuses the request.
    assert VLLMModelConfig.model_fields["inline_tool_output_images"].default is False


def test_the_training_overlay_turns_capture_on_within_the_policy_context() -> None:
    config = _harness_config(OVERLAYS[1])
    assert config.token_id_capture is True
    assert config.model_context_window == 49152
    assert _harness_config(OVERLAYS[1], agentic_vbench_context_tokens=65536).model_context_window == 65536
