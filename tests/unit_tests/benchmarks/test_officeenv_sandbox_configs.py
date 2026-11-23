# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OfficeEnv sandbox server's harness configs and its sandboxless arm, resolved with a model config: each names its
instances after itself, so several compose in one run without overriding each other. Resolving them needs no
OfficeEnv, which the server's own tests do."""

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config.global_config import (
    ROLLOUT_STOP_AGENT_TYPES,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
)
from nemo_gym.rollouts.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = "resources_servers/officeenv_sandbox/configs"
MODEL = "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
DOCKER = "nemo_gym/sandbox/providers/docker/configs/docker.yaml"
SHARED = "officeenv_sandbox_resources_server"
# Each harness config by the harness its instances are named after: the agent type it inherits, how the harness names
# MCP tools on its server (None: as the episodes' server does), and how its episodes reach their tools (Pi's over MCP;
# the others' rows are commands)
HARNESSES = {
    "pi": ("officeenv_sandbox.yaml", "pi_agent", "company_", ["mcp"]),
    "codex": ("officeenv_sandbox_codex.yaml", "codex_agent", "mcp__company__", []),
    "opencode": ("officeenv_sandbox_opencode.yaml", "opencode_agent", "company_", []),
    "openclaw": ("officeenv_sandbox_openclaw.yaml", "openclaw_agent", None, []),
    "stirrup": ("officeenv_sandbox_stirrup.yaml", "stirrup_agent", None, []),
    "miniswe": ("officeenv_sandbox_miniswe.yaml", "miniswe_sandboxed_agent", None, []),
    "terminus_2": ("officeenv_sandbox_terminus_2.yaml", "terminus_2_sandboxed_agent", None, []),
}
# The generic instances the harness configs inherit from, which no resolved config keeps under their own names
GENERIC = {"single_agent_turn_legacy", *(agent_type for _, agent_type, _, _ in HARNESSES.values())}
# The OfficeEnv server's instances (resources_servers/officeenv/configs/officeenv.yaml), and the sandboxless copies
OFFICEENV = ("officeenv_resources_server", "officeenv_simple_agent", "officeenv_environment_server")
SANDBOXLESS = (
    "officeenv_sandboxless_resources_server",
    "officeenv_sandbox_simple_agent",
    "officeenv_sandbox_environment_server",
)
ENVIRONMENT_SERVER_CONFIGS = {
    "single_agent_turn": SingleAgentTurnEnvironmentServerConfig,
    "single_agent_turn_legacy": SingleAgentTurnLegacyEnvironmentServerConfig,
}


def resolve(*config_paths: str, **overrides: object) -> tuple[list[str], DictConfig, dict]:
    """The servers ``gym env start`` would start from the configs and a model config, the merged config, and the same
    resolved to plain values. ``overrides`` win over the config files, as the command line's do."""
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(ROOT / path) for path in (MODEL, *config_paths)],
                    "policy_base_url": "http://127.0.0.1:1/v1",
                    "policy_api_key": "unused",  # pragma: allowlist secret
                    "policy_model_name": "policy",
                    **overrides,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    started = [instance.name for instance in parser.filter_for_server_instance_configs(config)]
    return started, config, OmegaConf.to_container(config, resolve=True)


def harness_config(harness: str) -> str:
    return f"{CONFIGS}/{HARNESSES[harness][0]}"


def resources_server_of(harness: str) -> str:
    return f"officeenv_sandbox_{harness}_resources_server"


def check_no_shared_server(started: list[str], resolved: dict) -> None:
    """Each harness's server inherits the episodes' server, which then neither starts idle nor is named anywhere."""
    assert SHARED not in started and SHARED not in resolved
    assert f'"{SHARED}"' not in json.dumps(resolved)


def check_harness(resolved: dict, harness: str) -> None:
    """The harness's agent, its environment server and the resources server it runs on, bound to one another."""
    _, agent_type, prefix, transports = HARNESSES[harness]
    agent = f"officeenv_sandbox_{harness}_agent"
    environment = resolved[f"officeenv_sandbox_{harness}_environment_server"]["environment_servers"]
    assert list(environment) == ["single_agent_turn_legacy"]
    binding = environment["single_agent_turn_legacy"]
    assert binding["agent_server"] == {"type": "responses_api_agents", "name": agent}
    assert binding["resources_server"] == {"type": "resources_servers", "name": resources_server_of(harness)}
    assert binding["resources_tool_transports"] == transports
    # A harness whose run the environment server stops at its task's end is one whose run can be stopped in flight.
    if binding.get("end_on_task_state"):
        assert agent_type in ROLLOUT_STOP_AGENT_TYPES and binding.get("max_episode_steps") is None
    SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(binding | {"name": f"{harness} environment"})
    assert list(resolved[agent]["responses_api_agents"]) == [agent_type]
    settings = resolved[agent]["responses_api_agents"][agent_type]
    if binding["max_concurrent_episodes"] is not None:
        assert binding["max_concurrent_episodes"] == settings["concurrency"]  # its own harness's cap
    server = resolved[resources_server_of(harness)]["resources_servers"]["officeenv_sandbox"]
    # The MCP endpoint lists itself under a company's name, never an instance's
    assert server["mcp_server_name"] == "company" and server["expose_tools_over_mcp"]
    assert server["sandbox_config"].get("mcp_tool_prefix", "") == (prefix or "")


@pytest.mark.parametrize("harness", HARNESSES)
def test_each_harness_config_names_its_instances_after_its_harness(harness: str) -> None:
    started, _, resolved = resolve(harness_config(harness), DOCKER)

    check_harness(resolved, harness)
    instances = [f"officeenv_sandbox_{harness}_{kind}" for kind in ("resources_server", "agent", "environment_server")]
    assert sorted(started) == sorted(["policy_model", *instances])
    check_no_shared_server(started, resolved)
    assert not GENERIC & set(resolved) and "oe" not in resolved


def test_pis_run_ends_once_its_task_has_ended_on_every_exposure() -> None:
    """Pi's environment server stops the run at the task's end (end_on_task_state, no step loop), so Pi takes no
    finish tool, which a cli or api_search row's tools would not offer."""
    _, _, resolved = resolve(harness_config("pi"), DOCKER)

    binding = resolved["officeenv_sandbox_pi_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert binding["end_on_task_state"] is True and binding.get("max_episode_steps") is None
    assert resolved["officeenv_sandbox_pi_agent"]["responses_api_agents"]["pi_agent"]["finish_tool_names"] == []


def test_stirrups_run_ends_once_its_task_has_ended() -> None:
    """Stirrup's environment server stops the run at the task's end (end_on_task_state, no step loop), which it can do
    only for an agent type whose run a rollout stop ends."""
    _, _, resolved = resolve(harness_config("stirrup"), DOCKER)

    environment = resolved["officeenv_sandbox_stirrup_environment_server"]["environment_servers"]
    binding = environment["single_agent_turn_legacy"]
    assert binding["end_on_task_state"] is True and binding.get("max_episode_steps") is None
    assert "stirrup_agent" in ROLLOUT_STOP_AGENT_TYPES


def test_the_harness_configs_compose_in_one_run_without_overriding_each_other() -> None:
    started, config, resolved = resolve(*(harness_config(harness) for harness in HARNESSES), DOCKER)

    for harness in HARNESSES:
        check_harness(resolved, harness)
    # Each harness keeps its own cap: Pi's and OpenCode's 8 beside Codex's, OpenClaw's and Stirrup's 32
    caps = {
        harness: resolved[f"officeenv_sandbox_{harness}_environment_server"]["environment_servers"][
            "single_agent_turn_legacy"
        ]["max_concurrent_episodes"]
        for harness in HARNESSES
    }
    assert caps == {
        "pi": 8,
        "codex": 32,
        "opencode": 8,
        "openclaw": 32,
        "stirrup": 32,
        "miniswe": None,
        "terminus_2": None,
    }
    servers = [resources_server_of(harness) for harness in HARNESSES]
    agents = [f"officeenv_sandbox_{harness}_agent" for harness in HARNESSES]
    environments = [f"officeenv_sandbox_{harness}_environment_server" for harness in HARNESSES]
    assert sorted(started) == sorted(["policy_model", *servers, *agents, *environments])
    check_no_shared_server(started, resolved)  # no idle episodes' server beside the seven harnesses'
    assert not GENERIC & set(resolved)
    # A row that names a harness's agent reaches that harness's environment server, the one that fronts it
    servers_by_agent = _environment_servers_by_agent(config)
    routes = {agent: _environment_server_for_agent(agent, servers_by_agent) for agent in agents}
    assert routes == dict(zip(agents, environments))


def test_a_setting_of_the_episodes_server_reaches_each_harness_server() -> None:
    # Each harness's server inherits the episodes' server as the merged config has it, the command line's settings
    # included, so one setting reaches the server of every harness in the run, and each keeps its own.
    shared = {"host": "10.1.2.3", "sandbox_config": {"image": "officeenv-sandbox:documents"}}
    paths = [harness_config("pi"), harness_config("codex"), harness_config("miniswe"), DOCKER]
    started, _, resolved = resolve(*paths, **{SHARED: {"resources_servers": {"officeenv_sandbox": shared}}})

    check_no_shared_server(started, resolved)
    for harness in ("pi", "codex", "miniswe"):
        check_harness(resolved, harness)
        server = resolved[resources_server_of(harness)]["resources_servers"]["officeenv_sandbox"]
        assert server["host"] == "10.1.2.3" and server["sandbox_config"]["image"] == "officeenv-sandbox:documents"


@pytest.mark.parametrize("harnesses", [(), ("codex",), ("miniswe", "terminus_2")], ids=["alone", "codex", "two"])
def test_the_opensandbox_config_reaches_each_harness_server(monkeypatch, harnesses: tuple[str, ...]) -> None:
    # Listed after the other harness configs, as its README section says: the first of them pulls in the episodes'
    # config, which Gym merges once, there, so it does not reset the OpenSandbox settings that come after it.
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")  # pragma: allowlist secret
    paths = [*(harness_config(harness) for harness in harnesses), f"{CONFIGS}/officeenv_opensandbox.yaml"]
    started, _, resolved = resolve(*paths)

    check_no_shared_server(started, resolved)
    for harness in ("pi", *harnesses):
        check_harness(resolved, harness)
        sandbox = resolved[resources_server_of(harness)]["resources_servers"]["officeenv_sandbox"]["sandbox_config"]
        assert sandbox["image"].startswith("python:3.12-bookworm@sha256:") and sandbox["resources"]["cpu"] == 2


def test_the_episodes_config_runs_its_server_on_its_own() -> None:
    # Composed with no harness config, the episodes' server starts under its own name, for a config that binds a
    # harness to it.
    started, _, resolved = resolve(f"{CONFIGS}/officeenv_sandbox_episodes.yaml", DOCKER)

    assert sorted(started) == sorted(["policy_model", SHARED])
    server = resolved[SHARED]["resources_servers"]["officeenv_sandbox"]
    assert server["mcp_server_name"] == "company" and server["sandbox_config"]["image"] == "officeenv-sandbox:pilot"


def without(settings: dict, *keys: str) -> dict:
    """``settings`` without ``keys`` and the address each server is given."""
    return {key: value for key, value in settings.items() if key not in {*keys, "host", "port"}}


@pytest.mark.parametrize("harnesses", [(), ("pi", "codex")], ids=["alone", "beside Pi and Codex"])
def test_the_sandboxless_arm_copies_the_officeenv_servers_instances(harnesses: tuple[str, ...]) -> None:
    paths = [f"{CONFIGS}/officeenv_sandboxless.yaml", *(harness_config(harness) for harness in harnesses)]
    started, config, resolved = resolve(*paths, *([DOCKER] if harnesses else []))
    resources, agent, environment = OFFICEENV
    copied_resources, copied_agent, copied_environment = SANDBOXLESS

    # The OfficeEnv server's instances keep their names and their bindings.
    assert {*OFFICEENV, *SANDBOXLESS} <= set(started)
    for harness in harnesses:
        check_harness(resolved, harness)
    main_agent = resolved[agent]["responses_api_agents"]["simple_agent"]
    assert main_agent["resources_server"]["name"] == resources
    assert [dataset["taskset"] for dataset in main_agent["datasets"]] == ["officeenv:example"]
    assert resolved["environment_server_routes"] == {"officeenv:example": environment}

    # Each copy is its source with its references pointing at the other copies, and the agent declares no dataset.
    assert without(resolved[copied_resources]["resources_servers"]["officeenv"]) == without(
        resolved[resources]["resources_servers"]["officeenv"]
    )
    copy = resolved[copied_agent]["responses_api_agents"]["simple_agent"]
    assert without(copy, "resources_server", "datasets") == without(main_agent, "resources_server", "datasets")
    assert copy["resources_server"] == {"type": "resources_servers", "name": copied_resources}
    assert copy["datasets"] == []
    main_environment = resolved[environment]["environment_servers"]
    copied = resolved[copied_environment]["environment_servers"]
    assert list(copied) == list(main_environment)  # the server type the OfficeEnv server's environment server runs
    (server_type,) = main_environment
    references = ("resources_server", "agent_server")
    assert without(copied[server_type], *references) == without(main_environment[server_type], *references)
    assert copied[server_type]["resources_server"] == {"type": "resources_servers", "name": copied_resources}
    assert copied[server_type]["agent_server"] == {"type": "responses_api_agents", "name": copied_agent}
    ENVIRONMENT_SERVER_CONFIGS[server_type].model_validate(copied[server_type] | {"name": copied_environment})

    # A flat row that names the copied agent (agent_ref) reaches the copied environment server, and one that names the
    # OfficeEnv server's agent its own.
    servers_by_agent = _environment_servers_by_agent(config)
    assert _environment_server_for_agent(copied_agent, servers_by_agent) == copied_environment
    assert _environment_server_for_agent(agent, servers_by_agent) == environment
