# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chartography's OpenCode config starts opencode_agent in a sandbox that the resources server lends, behind an
Environment Server, with the manifest's rows."""

from pathlib import Path, PurePosixPath

import yaml
from omegaconf import OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = "benchmarks/chartography_opencode/config.yaml"
# The setting with the judge its header names, as `gym env start` and a training launch load it.
JUDGE_PAIRING = (
    "resources_servers/chartography/configs/chartography_opencode.yaml",
    "resources_servers/chartography/configs/judge_model_openai.yaml",
)
# The no-tools setting, whose grading the OpenCode setting repeats.
PLAIN = (
    "resources_servers/chartography/configs/chartography.yaml",
    "resources_servers/chartography/configs/judge_model_openai.yaml",
)


def resolve(*config_paths: str, **overrides: object) -> tuple[GlobalConfigDictParser, dict]:
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(ROOT / path) for path in config_paths], **overrides},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return parser, config


def test_benchmark_config_routes_the_rows_through_an_agent_session():
    parser, config = resolve(BENCHMARK)
    started = {instance.name for instance in parser.filter_for_server_instance_configs(config)}
    assert started == {
        "policy_model",
        "judge_model",
        "chartography_opencode",
        "chartography_opencode_agent",
        "chartography_opencode_environment_server",
    }

    environment = config["chartography_opencode_environment_server"]["environment_servers"]
    pairing = environment["single_agent_turn_legacy"]
    assert pairing["resources_server"]["name"] == "chartography_opencode"
    assert pairing["agent_server"]["name"] == "chartography_opencode_agent"
    # OpenCode works with its own tools in the sandbox; the resources server grants none.
    assert list(pairing["resources_tool_transports"]) == []

    # The resources server grades as in the simple_agent setting, and lends each episode a sandbox that reaches
    # only the policy model.
    resources = config["chartography_opencode"]["resources_servers"]["chartography"]
    _, plain_config = resolve(*PLAIN)
    plain = plain_config["chartography"]["resources_servers"]["chartography"]
    for field in plain.keys() - {"description", "value", "host"}:
        assert resources[field] == plain[field], field
    assert "sandbox_provider" not in plain
    assert resources["sandbox_provider"] in config
    assert resources["sandbox_network_access"] == "model_only"
    assert resources["sandbox_model_server"]["name"] == "policy_model"

    # The agent borrows that sandbox, so it keeps the default network and starts no sandbox of its own.
    agent = config["chartography_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    assert agent["resources_server"]["name"] == "chartography_opencode"
    assert "sandbox_provider" not in agent and "network_access" not in agent
    assert agent["thinking"] is True and agent["image_input"] is True
    assert agent["opencode_config"]["compaction"]["auto"] is False
    assert "agent" not in agent["opencode_config"]  # no step limit, so token capture can join every call
    assert {name for name, on in agent["opencode_config"]["tools"].items() if not on} == {
        "webfetch",
        "websearch",
        "task",
    }

    # The benchmark's rows replace the example rows, and match its manifest.
    manifest = yaml.safe_load((ROOT / "benchmarks/chartography_opencode/manifest.yaml").read_text())
    assert manifest["agent_server"] == "opencode_agent"
    datasets = [OmegaConf.to_container(dataset) for dataset in agent["datasets"]]
    assert [{k: v for k, v in d.items() if k in manifest["datasets"][0]} for d in datasets] == manifest["datasets"]


def test_only_the_policy_model_takes_a_launcher_s_host_address():
    """The lent sandbox may reach every port of the policy model's host, so no other server listens there.

    A server's own host wins over default_host, so the judge (it spends the judge's key for any caller), the agent
    (a request without a session runs OpenCode on the host) and the resources and Environment Servers of the
    OpenCode setting stay on loopback.
    """
    parser, config = resolve(BENCHMARK, default_host="10.1.2.3")
    hosts = {
        instance.name: instance.get_inner_run_server_config_dict()["host"]
        for instance in parser.filter_for_server_instance_configs(config)
    }
    assert hosts["policy_model"] == "10.1.2.3"
    assert {name: host for name, host in hosts.items() if name != "policy_model"} == dict.fromkeys(
        hosts.keys() - {"policy_model"}, "127.0.0.1"
    )


def test_the_judge_stays_on_loopback_with_the_setting_s_judge_config():
    """The judge spends its key for any caller, so it stays off the address the lent sandbox may reach."""
    parser, config = resolve(*JUDGE_PAIRING, default_host="10.1.2.3")
    judge = next(i for i in parser.filter_for_server_instance_configs(config) if i.name == "judge_model")
    assert judge.get_inner_run_server_config_dict()["host"] == "127.0.0.1"


def test_episodes_are_admitted_at_the_agent_s_rate_and_end_before_the_sandbox_expires():
    _, config = resolve(BENCHMARK)
    pairing = config["chartography_opencode_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    agent = config["chartography_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    sandbox = config["chartography_opencode"]["resources_servers"]["chartography"]["sandbox_config"]
    # Each seed starts a sandbox, so admitting more episodes than OpenCode runs would leave sandboxes idle.
    assert pairing["max_concurrent_episodes"] == agent["concurrency"]
    # The episode covers the sandbox start and OpenCode's run, and ends before the sandbox's TTL.
    deadline = pairing["default_episode_timeout_seconds"]
    assert sandbox["ready_timeout_s"] + agent["timeout"] < deadline < sandbox["ttl_s"]


def test_the_agent_pins_the_opencode_build_of_the_shared_sandbox_image():
    """The sandbox image is visual_coding_opencode's, so the agent pins the same OpenCode binary and plugin
    dependencies at the same paths; the image's build checks the plugin archive against that digest."""
    _, ours = resolve(BENCHMARK)
    _, theirs = resolve("resources_servers/visual_coding_opencode/configs/visual_coding_opencode.yaml")
    agent = ours["chartography_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    image_agent = theirs["visual_coding_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    for field in (
        "opencode_version",
        "sandbox_executable_path",
        "sandbox_executable_sha256",
        "sandbox_plugin_dependencies_path",
        "sandbox_plugin_dependencies_sha256",
    ):
        assert agent[field] == image_agent[field], field
    resources = ours["chartography_opencode"]["resources_servers"]["chartography"]
    image_resources = theirs["visual_coding_opencode"]["resources_servers"]["visual_coding_opencode"]
    assert resources["sandbox_workdir"] == image_resources["sandbox_workdir"]  # the image's WORKDIR
    dockerfile = (ROOT / "resources_servers/visual_coding_opencode/sandbox_image/Dockerfile").read_text()
    # The build installs this OpenCode release, checks the binary against the pinned digest and copies the
    # plugin archive to the pinned path.
    assert f"/download/v{agent['opencode_version']}/" in dockerfile
    assert f"{agent['sandbox_executable_sha256']}  {agent['sandbox_executable_path']}" in dockerfile
    assert PurePosixPath(agent["sandbox_plugin_dependencies_path"]).name in dockerfile
    assert f"WORKDIR {resources['sandbox_workdir']}" in dockerfile


def test_only_the_opencode_setting_scores_unfinished_runs_zero():
    _, ours = resolve(BENCHMARK)
    _, plain = resolve(*PLAIN)
    assert ours["chartography_opencode"]["resources_servers"]["chartography"]["score_unfinished_as_zero"] is True
    assert "score_unfinished_as_zero" not in plain["chartography"]["resources_servers"]["chartography"]
