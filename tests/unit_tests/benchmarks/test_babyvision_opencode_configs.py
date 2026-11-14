# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BabyVision's configs start opencode_agent in a sandbox that the resources server lends, behind an Environment Server,
with the manifest's rows."""

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[3]


def resolve(config_path: str, **overrides: object) -> tuple[GlobalConfigDictParser, dict]:
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(ROOT / config_path)], **overrides},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return parser, config


@pytest.mark.parametrize("benchmark", ["babyvision_opencode", "babyvision_opencode_question_only"])
def test_benchmark_config_routes_the_rows_through_an_agent_session(benchmark):
    parser, config = resolve(f"benchmarks/{benchmark}/config.yaml")
    started = {instance.name for instance in parser.filter_for_server_instance_configs(config)}
    assert started == {
        "policy_model",
        "visual_coding_opencode",
        "visual_coding_opencode_judge_model",
        "visual_coding_opencode_agent",
        "visual_coding_opencode_environment_server",
    }

    environment = config["visual_coding_opencode_environment_server"]["environment_servers"]
    pairing = environment["single_agent_turn_legacy"]
    assert pairing["resources_server"]["name"] == "visual_coding_opencode"
    assert pairing["agent_server"]["name"] == "visual_coding_opencode_agent"

    # The resources server lends each episode a sandbox that reaches only the policy model.
    resources = config["visual_coding_opencode"]["resources_servers"]["visual_coding_opencode"]
    assert resources["sandbox_provider"] in config
    assert resources["sandbox_network_access"] == "model_only"
    assert resources["sandbox_model_server"]["name"] == "policy_model"
    assert resources["sandbox_config"]["resources"] == {"cpu": 2, "memory_mib": 8192}

    # The agent borrows that sandbox, so it keeps the default network and starts no sandbox of its own.
    agent = config["visual_coding_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    assert "sandbox_provider" not in agent and "network_access" not in agent
    assert agent["opencode_config"]["compaction"]["auto"] is False
    assert "agent" not in agent["opencode_config"]  # no step limit, so token capture can join every call

    # The benchmark's rows replace the example rows, and match its manifest.
    manifest = yaml.safe_load((ROOT / "benchmarks" / benchmark / "manifest.yaml").read_text())
    assert manifest["agent_server"] == "opencode_agent"
    assert [OmegaConf.to_container(dataset) for dataset in agent["datasets"]] == manifest["datasets"]


def test_only_the_policy_model_takes_a_launcher_s_host_address():
    """The lent sandbox may reach every port of the policy model's host, so no other server listens there.

    NeMo RL sets default_host to the node's address. A server's own host wins over it, so the judge (it spends the
    judge's key for any caller), the agent (a request without a session runs OpenCode on the host) and the resources
    and Environment Servers stay on loopback.
    """
    parser, config = resolve("benchmarks/babyvision_opencode/config.yaml", default_host="10.1.2.3")
    hosts = {
        instance.name: instance.get_inner_run_server_config_dict()["host"]
        for instance in parser.filter_for_server_instance_configs(config)
    }
    assert hosts == {
        "policy_model": "10.1.2.3",
        "visual_coding_opencode": "127.0.0.1",
        "visual_coding_opencode_judge_model": "127.0.0.1",
        "visual_coding_opencode_agent": "127.0.0.1",
        "visual_coding_opencode_environment_server": "127.0.0.1",
    }


def test_episodes_are_admitted_at_the_agent_s_rate_and_end_before_the_sandbox_expires():
    _, config = resolve("benchmarks/babyvision_opencode/config.yaml")
    pairing = config["visual_coding_opencode_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    agent = config["visual_coding_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    sandbox = config["visual_coding_opencode"]["resources_servers"]["visual_coding_opencode"]["sandbox_config"]
    # Each seed starts a sandbox, so admitting more episodes than OpenCode runs would leave sandboxes idle.
    assert pairing["max_concurrent_episodes"] == agent["concurrency"]
    # The episode covers the sandbox start and OpenCode's run, and ends before the sandbox's TTL.
    deadline = pairing["default_episode_timeout_seconds"]
    assert sandbox["ready_timeout_s"] + agent["timeout"] < deadline < sandbox["ttl_s"]


def test_the_sandbox_image_carries_the_pinned_plugin_dependencies():
    """The image carries OpenCode's plugin dependencies (from agentic_if's lockfile) and binary, each to its pin."""
    _, config = resolve("resources_servers/visual_coding_opencode/configs/visual_coding_opencode.yaml")
    agent = config["visual_coding_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    image = ROOT / "resources_servers" / "visual_coding_opencode" / "sandbox_image"
    for name in ("package.json", "package-lock.json"):
        assert (image / "opencode" / name).read_bytes() == (
            ROOT / "benchmarks/agentic_if/opencode" / name
        ).read_bytes()
    dockerfile = (image / "Dockerfile").read_text()
    archive = agent["sandbox_plugin_dependencies_path"]
    assert f"COPY --from=plugin-archive /{Path(archive).name} {Path(archive).parent}/" in dockerfile
    # The build checks the archive against the pin, so the image and the config cannot drift apart.
    assert f"{agent['sandbox_plugin_dependencies_sha256']}  /{Path(archive).name}" in dockerfile
    # The same holds for the OpenCode binary: the build downloads the pinned version and checks it against the pin.
    assert f"/download/v{agent['opencode_version']}/" in dockerfile
    assert f"{agent['sandbox_executable_sha256']}  {agent['sandbox_executable_path']}" in dockerfile
