# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OfficeEnv sandbox server's harness configs, resolved with a model config: each names its instances after its
harness, so several compose in one run without overriding each other. Resolving them needs no OfficeEnv, which the
server's own tests do."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollouts.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = "resources_servers/officeenv_sandbox/configs"
MODEL = "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
DOCKER = "nemo_gym/sandbox/providers/docker/configs/docker.yaml"
SHARED = "officeenv_sandbox_resources_server"
# Each harness config by the harness its instances are named after: the agent type it inherits, how the harness names
# MCP tools where it runs on a copy of the shared resources server of its own (None: on the shared one), and how its
# episodes reach their tools (Pi's over MCP; the others' rows are commands)
HARNESSES = {
    "pi": ("officeenv_sandbox.yaml", "pi_agent", "company_", ["mcp"]),
    "codex": ("officeenv_sandbox_codex.yaml", "codex_agent", "mcp__company__", []),
    "opencode": ("officeenv_sandbox_opencode.yaml", "opencode_agent", "company_", []),
    "openclaw": ("officeenv_sandbox_openclaw.yaml", "openclaw_agent", None, []),
    "miniswe": ("officeenv_sandbox_miniswe.yaml", "miniswe_sandboxed_agent", None, []),
    "terminus_2": ("officeenv_sandbox_terminus_2.yaml", "terminus_2_sandboxed_agent", None, []),
}
# The generic instances the harness configs inherit from, which no resolved config keeps under their own names
GENERIC = {"single_agent_turn_legacy", *(agent_type for _, agent_type, _, _ in HARNESSES.values())}


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
    return SHARED if HARNESSES[harness][2] is None else f"officeenv_sandbox_{harness}_resources_server"


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
    own = [f"officeenv_sandbox_{harness}_resources_server"] if HARNESSES[harness][2] else []
    instances = [f"officeenv_sandbox_{harness}_agent", f"officeenv_sandbox_{harness}_environment_server"]
    assert sorted(started) == sorted(["policy_model", SHARED, *own, *instances])
    assert not GENERIC & set(resolved) and "oe" not in resolved


def test_the_harness_configs_compose_in_one_run_without_overriding_each_other() -> None:
    started, config, resolved = resolve(*(harness_config(harness) for harness in HARNESSES), DOCKER)

    for harness in HARNESSES:
        check_harness(resolved, harness)
    # Each harness keeps its own cap: Pi's and OpenCode's 8 beside Codex's and OpenClaw's 32
    caps = {
        harness: resolved[f"officeenv_sandbox_{harness}_environment_server"]["environment_servers"][
            "single_agent_turn_legacy"
        ]["max_concurrent_episodes"]
        for harness in HARNESSES
    }
    assert caps == {"pi": 8, "codex": 32, "opencode": 8, "openclaw": 32, "miniswe": None, "terminus_2": None}
    copies = [f"officeenv_sandbox_{harness}_resources_server" for harness in ("pi", "codex", "opencode")]
    agents = [f"officeenv_sandbox_{harness}_agent" for harness in HARNESSES]
    environments = [f"officeenv_sandbox_{harness}_environment_server" for harness in HARNESSES]
    assert sorted(started) == sorted(["policy_model", SHARED, *copies, *agents, *environments])
    assert not GENERIC & set(resolved)
    # A row that names a harness's agent reaches that harness's environment server, the one that fronts it
    servers_by_agent = _environment_servers_by_agent(config)
    routes = {agent: _environment_server_for_agent(agent, servers_by_agent) for agent in agents}
    assert routes == dict(zip(agents, environments))


def test_a_setting_of_the_shared_server_reaches_each_harness_copy() -> None:
    # A copy takes the shared server as the merged config has it, the command line's settings included, so one setting
    # reaches the server of every harness in the run, and each copy keeps its own.
    shared = {"host": "10.1.2.3", "sandbox_config": {"image": "officeenv-sandbox:documents"}}
    paths = [harness_config("pi"), harness_config("codex"), harness_config("miniswe"), DOCKER]
    _, _, resolved = resolve(*paths, **{SHARED: {"resources_servers": {"officeenv_sandbox": shared}}})

    for harness in ("pi", "codex", "miniswe"):
        check_harness(resolved, harness)
        server = resolved[resources_server_of(harness)]["resources_servers"]["officeenv_sandbox"]
        assert server["host"] == "10.1.2.3" and server["sandbox_config"]["image"] == "officeenv-sandbox:documents"


def test_the_opensandbox_config_reaches_the_copy_of_a_harness_composed_before_it(monkeypatch) -> None:
    # Listed after the other harness configs, as its README section says: the first of them pulls in the episodes'
    # config, which Gym merges once, with that config's includes, so it does not reset the OpenSandbox settings.
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.invalid")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")  # pragma: allowlist secret
    _, _, resolved = resolve(harness_config("codex"), f"{CONFIGS}/officeenv_opensandbox.yaml")

    for harness in ("pi", "codex"):
        check_harness(resolved, harness)
        sandbox = resolved[resources_server_of(harness)]["resources_servers"]["officeenv_sandbox"]["sandbox_config"]
        assert sandbox["image"].startswith("python:3.12-bookworm@sha256:") and sandbox["resources"]["cpu"] == 2

