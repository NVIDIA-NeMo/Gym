# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config import DatasetConfig
from resources_servers.video_mcqa.app import VideoMCQAResourcesServerConfig
from resources_servers.video_mcqa.sandbox import PLUGIN_PATH, SANDBOX_IMAGE_DIR
from resources_servers.video_mcqa.tests.test_app import make_server
from resources_servers.videomme_v2.app import VideoMMEV2ResourcesServerConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.simple_agent.app import SimpleAgentConfig


@pytest.mark.parametrize("server_name", ["videomme_v1", "lvbench", "videomme_v2", "video_mcqa"])
def test_actual_composed_single_turn_configs_resolve_and_validate(tmp_path, server_name) -> None:
    server, _, _ = make_server(tmp_path, benchmark="lvbench" if server_name == "lvbench" else "videomme_v1")
    path = Path(__file__).parents[2] / server_name / "configs" / f"{server_name}_single_turn.yaml"
    values = {f"{server_name}_cohort_manifest": str(server.config.cohort_manifest)}
    config = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(path), OmegaConf.create(values)), resolve=True)
    config_type = VideoMMEV2ResourcesServerConfig if server_name == "videomme_v2" else VideoMCQAResourcesServerConfig
    resource = config_type(
        name=server_name, host="127.0.0.1", port=8001, **config[server_name]["resources_servers"][server_name]
    )
    assert resource.cohort_manifest == server.config.cohort_manifest
    agent_name = f"{server_name}_single_turn"
    (component,) = config[agent_name]["responses_api_agents"]
    assert component == "simple_agent"
    agent_config = config[agent_name]["responses_api_agents"]["simple_agent"]
    agent = SimpleAgentConfig(name=agent_name, host="127.0.0.1", port=8002, **agent_config)
    assert agent.resources_server.name == server_name and agent.max_steps == 1
    (example,) = [DatasetConfig.model_validate(item) for item in agent_config["datasets"]]
    assert (example.type, example.license) == ("example", "Apache 2.0")
    assert example.jsonl_fpath == f"resources_servers/{server_name}/data/example.jsonl"
    (environment_type,) = config[agent_name + "_environment"]["environment_servers"]
    assert environment_type == "single_agent_turn_legacy"
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=agent_name + "_environment",
        host="127.0.0.1",
        port=8003,
        **config[agent_name + "_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert environment.agent_server.name == agent_name
    assert environment.resources_server.name == server_name


@pytest.mark.parametrize("server_name", ["videomme_v1", "lvbench", "videomme_v2"])
def test_actual_composed_agentic_configs_resolve_and_validate(tmp_path, monkeypatch, server_name) -> None:
    monkeypatch.setenv("VIDEO_MCQA_SANDBOX_IMAGE", "registry.example/video-mcqa-sandbox:test")
    cohort = Path(__file__).parents[2] / server_name / "data" / "example_agentic_cohort.json"
    path = Path(__file__).parents[2] / server_name / "configs" / f"{server_name}_agentic.yaml"
    values = {
        f"{server_name}_agentic_cohort_manifest": str(cohort),
        f"{server_name}_agentic_video_roots": [str(tmp_path)],
        "policy_model_name": "policy",
    }
    loaded = OmegaConf.load(path)
    assert list(loaded.config_paths) == ["nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"]
    config = OmegaConf.to_container(OmegaConf.merge(loaded, OmegaConf.create(values)), resolve=True)
    name = f"{server_name}_agentic"
    config_type = VideoMMEV2ResourcesServerConfig if server_name == "videomme_v2" else VideoMCQAResourcesServerConfig
    resource = config_type(name=name, port=8001, **config[name]["resources_servers"][server_name])
    assert resource.sandbox_provider == "sandbox" and resource.sandbox_network_access == "model_only"
    assert resource.sandbox_config.image == "registry.example/video-mcqa-sandbox:test"
    assert (resource.sandbox_media_root, resource.sandbox_workdir) == ("/testbed", "/testbed/work")
    assert resource.sandbox_model_server.name == "policy_model" and resource.video_roots == [tmp_path]
    agent_config = config[f"{name}_opencode"]["responses_api_agents"]["opencode_agent"]
    agent = OpenCodeAgentConfig(name=f"{name}_opencode", port=8002, **agent_config)
    # Every server but the policy model stays on loopback: the sandbox's egress rule opens the model's host.
    assert resource.host == agent.host == "127.0.0.1"
    assert agent.resources_server.name == name and agent.plugins == [str(PLUGIN_PATH)]
    assert agent.image_input and agent.opencode_config["compaction"] == {"auto": False}
    assert agent.network_access == "inherit" and agent.sandbox_provider is None
    # The image's Dockerfile installs the binary and the plugin dependencies these pins name.
    dockerfile = (SANDBOX_IMAGE_DIR / "Dockerfile").read_text()
    assert f"{agent.sandbox_executable_sha256}  {agent.sandbox_executable_path}" in dockerfile
    assert agent.sandbox_plugin_dependencies_sha256 in dockerfile
    assert f"COPY video_tools.mjs {PLUGIN_PATH}" in dockerfile
    environment = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=f"{name}_environment",
        port=8003,
        **config[f"{name}_environment"]["environment_servers"]["single_agent_turn_legacy"],
    )
    assert environment.agent_server.name == f"{name}_opencode" and environment.resources_server.name == name
    assert environment.max_concurrent_episodes == agent.concurrency


@pytest.mark.parametrize("server_name", ["videomme_v1", "lvbench", "videomme_v2"])
def test_agentic_llm_extraction_overlays_add_a_loopback_judge(tmp_path, monkeypatch, server_name) -> None:
    monkeypatch.setenv("VIDEO_MCQA_SANDBOX_IMAGE", "registry.example/video-mcqa-sandbox:test")
    configs = Path(__file__).parents[2] / server_name / "configs"
    plain = OmegaConf.load(configs / f"{server_name}_agentic.yaml")
    overlay = OmegaConf.load(configs / f"{server_name}_agentic_llm_extraction.yaml")
    values = {
        f"{server_name}_agentic_cohort_manifest": str(tmp_path / "cohort.json"),
        f"{server_name}_agentic_video_roots": [str(tmp_path)],
        "policy_model_name": "policy",
        "judge_base_url": "https://judge.example.invalid/v1",
        "judge_api_key": "placeholder",
        "judge_model_name": "judge-model",
    }
    merged = OmegaConf.to_container(OmegaConf.merge(plain, overlay, OmegaConf.create(values)), resolve=True)
    name = f"{server_name}_agentic"
    config_type = VideoMMEV2ResourcesServerConfig if server_name == "videomme_v2" else VideoMCQAResourcesServerConfig
    resource = config_type(name=name, port=8001, **merged[name]["resources_servers"][server_name])
    assert resource.judge_model_server.name == "video_mcqa_answer_judge"
    resource.check_answer_judge("llm_extraction_v1")
    judge = merged["video_mcqa_answer_judge"]["responses_api_models"]["openai_model"]
    # The lent sandbox may reach every port of the policy model's host; the judge stays off that address.
    assert judge["host"] == "127.0.0.1" and judge["openai_model"] == "judge-model"
    # The overlay changes only the resources server's judge fields: the agent, environment and sandbox are the plain
    # agentic config's.
    plain_values = OmegaConf.to_container(OmegaConf.merge(plain, OmegaConf.create(values)), resolve=True)
    for key in (f"{name}_opencode", f"{name}_environment"):
        assert merged[key] == plain_values[key]
    added = {"judge_model_server", "judge_chat_completion_create_params"}
    resources = merged[name]["resources_servers"][server_name]
    assert {k: v for k, v in resources.items() if k not in added} == plain_values[name]["resources_servers"][
        server_name
    ]
