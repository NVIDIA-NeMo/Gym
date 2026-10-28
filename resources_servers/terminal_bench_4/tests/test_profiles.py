# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_both_tb4_recipes_use_standard_sessions(monkeypatch):
    from pathlib import Path

    from omegaconf import OmegaConf

    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
        get_first_server_config_dict,
    )

    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    for harness in ("opencode", "miniswe"):
        config = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
                initial_global_config_dict=OmegaConf.create(
                    {
                        "config_paths": [
                            f"benchmarks/terminal_bench_4/{harness}.yaml",
                            "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        ],
                        "policy_base_url": "http://unused.invalid/v1",
                        "policy_api_key": "unused",
                        "policy_model_name": "test-model",
                    }
                ),
            )
        )
        resource = get_first_server_config_dict(config, "terminal_bench_4")
        agent = get_first_server_config_dict(config, f"terminal_bench_4_{harness}")
        environment = get_first_server_config_dict(config, f"terminal_bench_4_{harness}_environment_server")
        assert "single_agent_turn" in config[f"terminal_bench_4_{harness}_environment_server"].environment_servers
        assert environment.agent_server.name == f"terminal_bench_4_{harness}"
        assert agent.datasets[0].prepare_script == "benchmarks/terminal_bench_4/prepare.py"
        if harness == "opencode":
            assert resource.environment.sandbox_provider == "tb4_sandbox"
            assert agent.opencode_version == "1.17.11"
            assert agent.opencode_config.tools is None
            assert agent.opencode_config.permission.task == "deny"
            assert agent.opencode_config.permission.webfetch == "deny"
            assert agent.opencode_config.permission.websearch == "deny"
        else:
            assert resource.environment.sandbox_provider == "tb4_sandbox"
            assert agent.harness.step_limit == 500
