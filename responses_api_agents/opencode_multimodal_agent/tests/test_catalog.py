# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise complete recipes, including inherited policies and episode routing."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.opencode_multimodal_agent.eval import OpenCodeMultimodalEvalConfig


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "benchmark,sessions",
    [
        (benchmark, sessions)
        for benchmark in ["vstar", "hrbench8k", "vtcbench", "zerobench"]
        for sessions in [False, True, "opensandbox"]
    ],
)
def test_catalog_and_compatibility_route_once(monkeypatch, benchmark, sessions):
    monkeypatch.chdir(ROOT)
    prefix = "benchmarks/opencode_multimodal_benchmarks"
    suffix = f"{benchmark}/opencode.yaml" if sessions else f"{benchmark}/opencode_relay.yaml"
    profile = "opencode_sessions.yaml" if sessions else "opencode_eval_agent.yaml"
    if sessions == "opensandbox":
        profile = "opencode_sessions_opensandbox.yaml"
    if sessions in ("opensandbox", "opensandbox-provider"):
        monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-placeholder")
    initial = OmegaConf.create(
        {
            "config_paths": [
                f"{prefix}/{suffix}",
                f"{prefix}/{profile}",
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "policy_model_name": "visual-model",
            "policy_base_url": "http://model.invalid/v1",
            "policy_api_key": "none",
            "visual_agent": {
                "responses_api_agents": {
                    "opencode_multimodal_agent": {"sandbox_config": {"image": "visual-runtime:test"}}
                }
            },
        }
        if sessions
        else {
            "config_paths": [
                f"{prefix}/{suffix}",
                f"{prefix}/{profile}",
                "responses_api_models/vllm_model/configs/vllm_model.yaml",
            ],
            "policy_model_name": "visual-model",
            "policy_base_url": "http://model.invalid/v1",
            "policy_api_key": "none",
        }
    )
    initial["opencode_multimodal_benchmarks"] = {
        "resources_servers": {"opencode_multimodal_benchmarks": {"judge_model": "test-judge"}}
    }
    if sessions == "opensandbox-provider":
        initial.config_paths.append("nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml")
    if sessions == "opensandbox":
        monkeypatch.setenv("GYM_JUDGE_BASE_URL", "https://judge.invalid/v1")
        monkeypatch.setenv("GYM_JUDGE_API_KEY", "test-placeholder")
        initial.config_paths.append(f"{prefix}/judge.yaml")
        settings = initial.visual_agent.responses_api_agents.opencode_multimodal_agent
        for asset in ("executable", "plugin_dependencies", "ripgrep"):
            settings[f"sandbox_{asset}_path"] = f"/opt/runtime/{asset}"
            settings[f"sandbox_{asset}_sha256"] = "a" * 64
    cfg = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    environments = [
        server
        for instance in cfg.values()
        if OmegaConf.is_dict(instance)
        for server in instance.get("environment_servers", {}).values()
    ]
    assert len(environments) == 1
    name = "visual_agent" if sessions else "visual_opencode"
    assert environments[0].agent_server.name == name
    agent = OpenCodeMultimodalEvalConfig.model_validate(
        OmegaConf.to_container(cfg[name].responses_api_agents.opencode_multimodal_agent, resolve=True) | {"name": name}
    )
    original = yaml.safe_load((ROOT / prefix / "opencode_eval_agent.yaml").read_text())["visual_opencode"][
        "responses_api_agents"
    ]["opencode_multimodal_agent"]
    assert agent.build_system_prompt == original["build_system_prompt"]
    assert agent.context_window == agent.max_output_tokens == 131072
    assert agent.opencode_config.get("tools", {}).get("webfetch", False) is False
    assert agent.opencode_config.get("tools", {}).get("websearch", False) is False
    assert agent.network_access == "model_only"
    assert agent.max_turns == agent.turn_budget.hard_limit == 65
    assert agent.turn_budget.model_dump() == OmegaConf.to_container(
        cfg.policy_model.responses_api_models.vllm_model.turn_budget
    )
    if sessions:
        if sessions == "opensandbox-provider":
            assert "opensandbox" in cfg.sandbox
            assert agent.sandbox_executable_path is None
            assert agent.sandbox_plugin_dependencies_path is None
        if sessions == "opensandbox":
            assert agent.host == cfg.head_server.host == "127.0.0.1"
            assert "visual_opencode" not in cfg
            assert cfg.visual_judge.responses_api_models.openai_model.host == "127.0.0.1"
            assert (
                cfg.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks.host == "127.0.0.1"
            )
            assert cfg.visual_environment.environment_servers.single_agent_turn.host == "127.0.0.1"
            assert cfg.use_absolute_ip
            assert agent.sandbox_plugin_dependencies_path == "/opt/runtime/plugin_dependencies"
        assert cfg.environment_routing_mode == "taskset"
        assert cfg.environment_server_routes.visual_tasks == "visual_environment"
        assert "single_agent_turn" in cfg.visual_environment.environment_servers
        dataset = cfg.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks.datasets[0]
        assert dataset.taskset == "visual_tasks" and "agent" not in dataset
        catalog = yaml.safe_load((ROOT / prefix / benchmark / "config.yaml").read_text())
        expected = catalog["opencode_multimodal_benchmarks"]["resources_servers"]["opencode_multimodal_benchmarks"]
        assert OmegaConf.to_container(dataset) == expected["datasets"][0]
        assert agent.command == "opencode" and not any(agent.env.values())
        assert agent.sandbox is None and agent.sandbox_provider == "sandbox"
        assert agent.tool_rewards is None and not agent.token_id_capture
        assert "datasets" not in cfg[name].responses_api_agents.opencode_multimodal_agent
    else:
        assert agent.command == original["command"] and agent.sandbox is not None
