# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.mark.parametrize("harness", ["opencode", "mini_swe_agent"])
@pytest.mark.parametrize("override", [None, 128000])
def test_tb4_composition_shares_agent_and_model_cap(monkeypatch, harness, override):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(root)
    cli = {
        "config_paths": [str(root / f"benchmarks/terminal_bench_4/{harness}.yaml")],
        "policy_base_url": "http://model-server/v1",
        "policy_api_key": "synthetic",
        "policy_model_name": "synthetic",
    }
    if override is not None:
        cli["tb4_max_output_tokens"] = override
    with patch.object(GlobalConfigDictParser, "parse_global_config_dict_from_cli", return_value=OmegaConf.create(cli)):
        config = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(offline=True, hide_secrets=False))
    config = OmegaConf.to_container(config, resolve=True)
    expected = override or 131072
    assert config["policy_model"]["responses_api_models"]["vllm_model"]["sampling_overrides"]["max_tokens"] == expected
    if harness == "opencode":
        agent = config["terminal_bench_4_opencode_sandboxed_agent"]["responses_api_agents"]["opencode_sandboxed_agent"]
        assert agent["opencode_max_output_tokens"] == expected
        assert agent["opencode_max_context_window"] == 262144
    else:
        agent = config["terminal_bench_4_mini_swe_agent"]["responses_api_agents"]["mini_swe_agent_sandboxed_agent"]
        assert agent["mini_config_overrides"]["model"]["model_kwargs"]["max_output_tokens"] == expected
