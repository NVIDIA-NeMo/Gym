# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.swe_together.app import SWETConfig, SWETEnvironmentServer
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.config import AggregateMetricsRequest, GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.config.global_config import taskset_environment_server_name
from nemo_gym.rollouts import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper


@pytest.mark.asyncio
@pytest.mark.parametrize("smoke", [False, True])
async def test_recipe_routes_to_environment_and_aggregates_measurements(monkeypatch, smoke):
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    monkeypatch.delenv("SWE_TOGETHER_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("NEMO_GYM_ALLOW_UNSUPPORTED_PAIRING", raising=False)
    for role in ("POLICY", "SIMULATOR", "JUDGE", "INTERACTION"):
        monkeypatch.setenv(role + "_API_KEY", "unused")
    for key, value in {
        "POLICY_MODEL": "openai/gpt-5.5",
        "OPENSANDBOX_API_KEY": "unused",
        "OPENSANDBOX_SHARED_STORAGE_PATH": "/mnt/shared",
        "SWE_TOGETHER_POLICY_EGRESS_TARGET": "10.0.0.1",
        "SWE_TOGETHER_JUDGE_EGRESS_TARGET": "10.0.0.2",
        "SWE_TOGETHER_DEPENDENCY_PROXY_URL": "http://10.0.0.3:3128",
    }.items():
        monkeypatch.setenv(key, value)
    paths = ["benchmarks/swe_together/opencode.yaml"]
    if smoke:
        paths.append("benchmarks/swe_together/smoke.yaml")
    paths.append("benchmarks/swe_together/opensandbox.yaml")
    initial = OmegaConf.create({"config_paths": paths, "allow_unsupported_pairing": False})
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    environment = taskset_environment_server_name(config, "swe_together:full109")
    assert environment == "swe_together"
    RolloutCollectionHelper._validate_environment_servers([{NG_ENVIRONMENT_SERVER_KEY: environment}], config)
    assert not any("resources_servers" in value for value in config.values() if isinstance(value, dict))
    env_config = SWETConfig(name="swe_together", **config["swe_together"]["environment_servers"]["swe_together"])
    renewal = config["sandbox"]["opensandbox"]["create"]["renew_interval_s"]
    assert renewal == 3600
    assert renewal < env_config.sandbox_config.get("ttl_s", 7200)
    assert renewal < env_config.judge_sandbox_config.get("ttl_s", 7200)
    assert env_config.sandbox_config["metadata"] == {"nemo.nvidia.com/oom-kill": "process"}
    assert env_config.judge_sandbox_config["metadata"] == {}
    assert env_config.interaction_timeout_seconds == (600 if smoke else 4800)
    assert env_config.agent_server.name == "swe_together_candidate"
    assert env_config.judge_agent.name == "swe_together_judge"
    native = config["swe_together_candidate"]["responses_api_agents"]["opencode_agent"]["opencode_config"]
    assert native["permission"]["bash"] == "allow"
    assert native["tools"]["webfetch"] is False and native["tools"]["websearch"] is False
    server = SWETEnvironmentServer.model_construct(config=env_config)
    records = [
        {"task_id": f"task-{index}", "_ng_task_index": index, "_ng_rollout_index": 0, "reward": 1, "judge_score": 1}
        for index in range(109)
    ]
    metrics = (await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=records))).agent_metrics
    assert metrics["planned"] == (109 if smoke else 218)
    assert metrics["missing"] == (0 if smoke else 109)
    assert metrics["complete_tasks"] == (109 if smoke else 0)

    benchmark = BenchmarkConfig.from_initial_config_dict(Path("benchmarks/swe_together/opencode.yaml"), initial)
    assert benchmark is not None
    assert benchmark.environment_server == "swe_together"
    assert benchmark.num_repeats == (1 if smoke else 2)
