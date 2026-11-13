# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BrowseComp's OpenCode presets. The session preset runs opencode_agent in the sandbox that opencode_search lends,
behind single_agent_turn_legacy, with the settings of the relay preset's deprecated agent where opencode_agent has
them; the relay preset keeps its explicit legacy_agent relay."""

import os
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError
from pydantic import BaseModel

from benchmarks.browsecomp.prepare_opencode import OUTPUT_FPATH
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseRunServerTypeConfig
from nemo_gym.config.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
)
from nemo_gym.server_runtime.sandbox_lending import STOP_TIMEOUT_S
from resources_servers.opencode_search.app import SearchResourcesServer, SearchResourcesServerConfig
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig, _session_opencode_config_error
from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION


ROOT = Path(__file__).resolve().parents[3]
SESSIONS = "benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml"
RELAY = "benchmarks/browsecomp/configs/browsecomp_opencode.yaml"
AGENT = "browsecomp_opencode_agent"
ENVIRONMENT = "browsecomp_opencode_environment_server"
RESOURCES = "search_browsecomp_resources_server"
JUDGE = "search_judge_model"
RELAY_AGENT = "opencode_sandboxed_search_agent"
RELAY_ENVIRONMENT = "opencode_sandboxed_search_environment_server"
LOOPBACK = "127.0.0.1"
RECEIPTS = "resources_servers/opencode_search/opencode_bundle/nemo_gym_receipts.js"
# Catalog entries of a server block: listings read them, and pairing checks read allowed_agents.
CATALOG_KEYS = {"allowed_agents", "description", "value", "verified"}
# Stand-ins for what a run supplies: the files it pins, and the judge it names.
PINS = {
    "BROWSECOMP_OPENCODE_EXECUTABLE_PATH": "/opt/pinned/opencode",
    "BROWSECOMP_OPENCODE_EXECUTABLE_SHA256": "0" * 64,
    "BROWSECOMP_OPENCODE_RIPGREP_PATH": "/opt/pinned/rg",
    "BROWSECOMP_OPENCODE_RIPGREP_SHA256": "1" * 64,
}
JUDGE_NAMED = {"search_judge_base_url": "http://judge.invalid/v1", "search_judge_model_name": "judge-model"}
# The other settings the presets read from the run's environment.
RUN_VARIABLES = (
    "NEMO_GYM_DISABLE_BASH",
    "NEMO_GYM_FIX_GREP_DESCRIPTION",
    "OPENCODE_WEBSEARCH_TYPE",
    "NEMO_GYM_EXA_CACHE_ROOT",
    "NEMO_GYM_EXA_CACHE_LEGACY_ROOT",
    "BROWSECOMP_OPENCODE_EXPORT_DIR",
)
# opencode_search's bounds on its commands in a lent sandbox: creating the workdir at the seed, and reading the plugin
# log at verify.
WORKDIR_CREATE_S = 60
PLUGIN_LOG_READ_S = 60
NODE = shutil.which("node")


@pytest.fixture(autouse=True)
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply the pins, and keep this process's own settings out of the composed configs."""
    for name, value in PINS.items():
        monkeypatch.setenv(name, value)
    for name in RUN_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def resolve(path: str, **overrides: Any) -> tuple[GlobalConfigDictParser, DictConfig]:
    """Compose a preset with the vllm_model policy config, refusing to generate a relay for an agent without one."""
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                        str(ROOT / path),
                    ],
                    "policy_base_url": "http://policy.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "policy-model",
                    "error_on_agent_without_environment_server": True,
                    **overrides,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return parser, config


def sessions(**overrides: Any) -> tuple[GlobalConfigDictParser, DictConfig]:
    return resolve(SESSIONS, **(JUDGE_NAMED | overrides))


def settings(config: DictConfig, name: str) -> dict[str, Any]:
    return OmegaConf.to_container(get_first_server_config_dict(config, name), resolve=True)


def environment_servers(config: DictConfig) -> dict[str, list[str]]:
    return {
        name: list(instance.environment_servers)
        for name, instance in config.items()
        if OmegaConf.is_dict(instance) and "environment_servers" in instance
    }


def decision(opencode_config: Mapping[str, Any], tool: str) -> str:
    """OpenCode 1.17.11's permission for a tool: a tools map becomes rules ahead of the permission map, which replaces
    their actions in place, and the last rule whose key matches the tool wins."""
    rules = {name: "allow" if enabled else "deny" for name, enabled in (opencode_config.get("tools") or {}).items()}
    rules.update(opencode_config.get("permission") or {})
    return [action for name, action in rules.items() if name in ("*", tool)][-1]


def test_each_episode_runs_opencode_agent_in_the_sandbox_the_resources_server_lends() -> None:
    parser, config = sessions()

    started = {instance.name for instance in parser.filter_for_server_instance_configs(config)}
    assert started == {"policy_model", JUDGE, RESOURCES, AGENT, ENVIRONMENT}
    assert OmegaConf.missing_keys(config) == set()
    # One Environment Server, and no relay generated for any agent.
    assert environment_servers(config) == {ENVIRONMENT: ["single_agent_turn_legacy"]}
    pairing = settings(config, ENVIRONMENT)
    SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(pairing | {"name": ENVIRONMENT})
    assert (pairing["resources_server"]["name"], pairing["agent_server"]["name"]) == (RESOURCES, AGENT)
    # OpenCode works with its own tools in the sandbox; the resources server grants none.
    assert pairing["resources_tool_transports"] == []
    # Every episode is verified, with no tool servers on either side.
    assert not {"tool_servers", "skip_verification", "skip_verification_reward"} & pairing.keys()
    assert not {"tool_servers", "skip_verification"} & settings(config, AGENT).keys()
    # A run reported as a failed response of kind model_context_limit is graded; opencode_agent reports one that hit
    # the context window as an incomplete response, which the default policy grades anyway.
    assert pairing["agent_outcome_policy"] == {"by_failure_kind": {"model_context_limit": "verify"}}
    assert list(config[AGENT].responses_api_agents) == ["opencode_agent"]
    assert settings(config, AGENT)["resources_server"]["name"] == RESOURCES
    # Single-harness on purpose: OpenCode plugins shape the tool surface, and the server serves no tools of its own,
    # over MCP either.
    assert list(config[RESOURCES].resources_servers) == ["opencode_search"]
    assert settings(config, RESOURCES)["allowed_agents"] == ["opencode_agent"]
    assert not settings(config, RESOURCES).get("expose_tools_over_mcp", False)


@pytest.mark.parametrize(
    ("name", "config_class"), [(AGENT, OpenCodeAgentConfig), (RESOURCES, SearchResourcesServerConfig)]
)
def test_every_setting_is_one_gym_reads(name: str, config_class: type[BaseModel]) -> None:
    """Both config classes ignore keys they do not declare, so a misspelled setting would compose and do nothing.
    Gym's config layer reads the other keys: those of every server block, such as datasets, and catalog entries."""
    _, config = sessions()
    block = settings(config, name)

    read_by_gym = BaseRunServerTypeConfig.model_fields.keys() | CATALOG_KEYS
    assert block.keys() - config_class.model_fields.keys() - read_by_gym == set()
    config_class.model_validate(block | {"name": name})


def test_only_the_agent_environment_server_and_judge_are_pinned_to_loopback() -> None:
    """A lent sandbox reaches every port of an address it can reach. The agent (a request without a session runs
    OpenCode on this host), the Environment Server and the judge (it spends its key for any caller) keep loopback
    whatever address the run gives, while the resources server takes it, as its search cache proxy needs."""
    parser, config = sessions(default_host="192.0.2.20")

    hosts = {
        instance.name: instance.get_inner_run_server_config_dict()["host"]
        for instance in parser.filter_for_server_instance_configs(config)
    }
    assert hosts == {
        "policy_model": "192.0.2.20",
        RESOURCES: "192.0.2.20",
        AGENT: LOOPBACK,
        ENVIRONMENT: LOOPBACK,
        JUDGE: LOOPBACK,
    }
    # Given this host's own address instead, use_absolute_ip passes it on to the model servers only.
    assert config.use_absolute_ip is True
    assert list(config.absolute_ip_server_types) == ["responses_api_models"]


@pytest.mark.parametrize(("address", "starts"), [({"default_host": "192.0.2.20"}, True), ({}, False)])
def test_with_the_search_cache_the_resources_server_starts_only_where_sandboxes_reach_it(
    address: dict[str, str], starts: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A lent sandbox searches through the resources server's /mcp proxy. On loopback the server refuses to start,
    rather than lend sandboxes whose every search fails."""
    monkeypatch.setenv("NEMO_GYM_EXA_CACHE_ROOT", str(tmp_path / "search-cache"))
    _, config = sessions(**address)
    resources = SearchResourcesServerConfig.model_validate(settings(config, RESOURCES) | {"name": RESOURCES})

    with nullcontext() if starts else pytest.raises(ValueError, match="cannot reach the search cache proxy"):
        SearchResourcesServer(config=resources, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    "arms",
    [{}, {"NEMO_GYM_DISABLE_BASH": "1", "NEMO_GYM_FIX_GREP_DESCRIPTION": "1", "OPENCODE_WEBSEARCH_TYPE": "deep"}],
    ids=["baseline", "opt_in_arms"],
)
def test_the_lent_sandbox_gets_the_relay_agents_opencode_env(
    arms: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencode_agent takes no environment of its own, so the plugins' settings reach OpenCode as the lent sandbox's
    environment. Equal values for every key keep the two presets from drifting apart."""
    for name, value in arms.items():
        monkeypatch.setenv(name, value)
    _, config = sessions()
    _, relay = resolve(RELAY)
    resources = settings(config, RESOURCES)

    env = resources["episode_sandbox_env"]
    # The receipts plugin writes the log that verify reads.
    assert env.pop("NEMO_GYM_PLUGIN_LOG") == resources["plugin_log_path"]
    assert env == settings(relay, RELAY_AGENT)["opencode_env"]


def test_opencode_resolves_tool_permissions_as_on_the_relay() -> None:
    """The relay's permission rules, with webfetch and websearch allowed by permission rules rather than a tools map:
    webfetch, websearch and task allowed, skill denied, and every bash and edit pattern in the relay's order. No task
    rule, so OpenCode's truncation notice still names the task tool; the turn budget plugin removes the tool from every
    request."""
    _, config = sessions()
    _, relay = resolve(RELAY)
    ours = settings(config, AGENT)["opencode_config"]
    theirs = settings(relay, RELAY_AGENT)["opencode_config"]

    assert set(ours) == {"permission", "compaction"}
    web = ("webfetch", "websearch")
    assert {key: rule for key, rule in ours["permission"].items() if key not in web} == {
        key: rule for key, rule in theirs["permission"].items() if key not in web
    }
    assert next(iter(ours["permission"])) == next(iter(theirs["permission"])) == "*"
    # The last matching pattern wins, so each pattern map keeps the relay's order.
    pattern_maps = [key for key, rule in theirs["permission"].items() if isinstance(rule, dict)]
    for key in pattern_maps:
        assert list(ours["permission"][key].items()) == list(theirs["permission"][key].items())
    # The other rules resolve as on the relay, and so do three tools that no permission rule names.
    keys = [key for key in theirs["permission"] if key not in pattern_maps]
    keys += ["task", "external_directory", "doom_loop"]
    assert {key: decision(ours, key) for key in keys} == {key: decision(theirs, key) for key in keys}
    # The session's workdir is /workspace, so reading /tmp or a truncated output is an external_directory request,
    # which only the "*" rule allows.
    tools = ("webfetch", "websearch", "task", "skill", "external_directory", "doom_loop")
    assert [decision(ours, tool) for tool in tools] == ["allow", "allow", "allow", "deny", "allow", "allow"]


def test_the_agent_runs_opencode_as_the_relay_agent_does() -> None:
    _, config = sessions()
    _, relay = resolve(RELAY)
    agent = OpenCodeAgentConfig.model_validate(settings(config, AGENT) | {"name": AGENT})
    theirs = settings(relay, RELAY_AGENT)

    assert agent.opencode_version == theirs.get("opencode_version", OPENCODE_VERSION)
    # The model ID that selects OpenCode's model-family prompts and tools: the relay agent's dummy_model.
    assert agent.sandbox_model_id == "dummy_model"
    assert agent.thinking is True
    assert agent.context_window == agent.max_output_tokens == theirs["opencode_max_context_window"]
    assert agent.timeout == theirs["sandbox_timeout"]
    # opencode_agent puts no deadline on a single model call, and neither does the relay agent here.
    assert theirs["opencode_model_call_timeout"] is None
    assert agent.opencode_config["compaction"] == theirs["opencode_config"]["compaction"] == {"reserved": 70144}
    assert agent.exclude_compaction_summaries is theirs["exclude_compaction_summaries"] is True
    # OpenCode's stock system prompt and no step limit; a session accepts the rest of the configuration.
    assert agent.system_prompt is None and agent.env == {} and "agent" not in agent.opencode_config
    assert _session_opencode_config_error(agent.opencode_config) is None
    # A borrowed sandbox keeps the resources server's network, open to the web.
    assert agent.network_access == "inherit"
    pins = (
        agent.sandbox_executable_path,
        agent.sandbox_executable_sha256,
        agent.sandbox_ripgrep_path,
        agent.sandbox_ripgrep_sha256,
    )
    assert pins == tuple(PINS.values())


@pytest.mark.parametrize("unset", [*PINS, *JUDGE_NAMED])
def test_a_run_that_leaves_a_pin_or_the_judge_unset_does_not_compose(
    unset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No pin falls back to a download, and no judge to a default."""
    monkeypatch.delenv(unset, raising=False)

    with pytest.raises(InterpolationResolutionError, match=unset):
        resolve(SESSIONS, **{name: value for name, value in JUDGE_NAMED.items() if name != unset})


def test_the_agent_loads_the_staged_plugins_receipts_first_then_the_relay_agents_in_its_order() -> None:
    _, config = sessions()
    _, relay = resolve(RELAY)
    files = settings(config, RESOURCES)["episode_sandbox_files"]
    plugins = settings(config, AGENT)["plugins"]
    relay_plugins = settings(relay, RELAY_AGENT)["opencode_plugins"]

    assert sorted(plugins) == sorted(files)
    # The relay agent's plugin files, unchanged, at the paths it uploads them to.
    assert plugins == [f"/tmp/nemo-gym-plugin-{Path(name).name}" for name in (RECEIPTS, *relay_plugins)]
    assert [files[path] for path in plugins] == [
        RECEIPTS,
        *(f"responses_api_agents/opencode_sandboxed_agent/plugins/{name}" for name in relay_plugins),
    ]


# What OpenCode does with its plugins: it imports each, calls its factory, and runs the hooks the factories returned on
# each model call; here, one call.
_PLUGIN_HOST = """
import { pathToFileURL } from "node:url";
const hooks = [];
for (const path of process.argv.slice(1)) {
  const plugin = (await import(pathToFileURL(path).href)).default;
  hooks.push(await plugin({ client: { session: { abort: async () => {} } } }));
}
for (const hook of hooks) await hook["chat.params"]?.({ sessionID: "session", message: {} });
"""


@pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the OpenCode plugins")
def test_the_staged_plugins_print_every_line_verify_requires(tmp_path: Path) -> None:
    """verify refuses a run whose plugin log lacks a required line, so a line no plugin prints would refuse every run."""
    _, config = sessions()
    resources = settings(config, RESOURCES)
    log = tmp_path / "plugins.log"
    # The lent sandbox's environment, with the log moved off this host's /tmp.
    env = {name: value for name, value in resources["episode_sandbox_env"].items() if value}
    sources = [str(ROOT / resources["episode_sandbox_files"][path]) for path in settings(config, AGENT)["plugins"]]

    ran = subprocess.run(
        [NODE, "--input-type=module", "-e", _PLUGIN_HOST, *sources],
        env=os.environ | env | {"NEMO_GYM_PLUGIN_LOG": str(log)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert ran.returncode == 0, ran.stderr
    plugin_log = log.read_text()
    assert [marker for marker in resources["required_plugin_markers"] if marker not in plugin_log] == []


def test_an_episode_ends_before_its_lent_sandbox_expires() -> None:
    _, config = sessions()
    pairing = settings(config, ENVIRONMENT)
    agent = settings(config, AGENT)
    sandbox = settings(config, RESOURCES)["sandbox_config"]
    deadline = pairing["default_episode_timeout_seconds"]

    # Each admitted episode holds a sandbox; any one worker can run every admitted episode, so none waits for OpenCode.
    assert agent["concurrency"] >= pairing["max_concurrent_episodes"]
    # Everything before the judge, each step at its bound: the seed (the sandbox start, its workdir and OpenCode's
    # install), the run, the agent's close, and verify's plugin-log read and sandbox stop.
    before_judge = (
        sandbox["ready_timeout_s"]
        + WORKDIR_CREATE_S
        + agent["sandbox_install_timeout_seconds"]
        + agent["timeout"]
        + agent["session_close_timeout_seconds"]
        + PLUGIN_LOG_READ_S
        + STOP_TIMEOUT_S
    )
    assert before_judge < deadline
    # The judge is left out: verify stops the sandbox before it calls the judge, and a judge slower than the deadline
    # allows times the episode out. After the deadline the Environment Server closes the agent's session and then the
    # resources server's, which stops the sandbox, each within cleanup_timeout_seconds.
    assert deadline + 2 * pairing["cleanup_timeout_seconds"] < sandbox["ttl_s"]


def test_the_resources_server_is_the_relay_presets_but_for_its_episode_settings() -> None:
    _, config = sessions()
    _, relay = resolve(RELAY)
    ours, theirs = settings(config, RESOURCES), settings(relay, RESOURCES)

    differing = {key for key in (ours.keys() | theirs.keys()) - {"host", "port"} if ours.get(key) != theirs.get(key)}
    assert differing == {
        "allowed_agents",
        "sandbox_workdir",
        "sandbox_config",
        "episode_sandbox_files",
        "episode_sandbox_env",
        "plugin_log_path",
        "required_plugin_markers",
    }
    # The same sandbox, alive long enough for the episode.
    assert ours["sandbox_config"] == theirs["sandbox_config"] | {"ttl_s": ours["sandbox_config"]["ttl_s"]}


def test_the_judge_and_the_output_cap_are_the_relay_presets() -> None:
    _, relay = resolve(RELAY)
    relay_judge = settings(relay, JUDGE)
    # The run names the relay preset's judge.
    _, config = resolve(
        SESSIONS, search_judge_base_url=relay_judge["base_url"], search_judge_model_name=relay_judge["model"]
    )
    judge = settings(config, JUDGE)

    assert judge.pop("host") == LOOPBACK
    assert {key: value for key, value in judge.items() if key != "port"} == {
        key: value for key, value in relay_judge.items() if key not in ("host", "port")
    }
    sampling = settings(config, "policy_model")["sampling_overrides"]
    assert sampling == settings(relay, "policy_model")["sampling_overrides"]
    assert sampling == dict.fromkeys(("max_output_tokens", "max_tokens", "max_completion_tokens"))


def test_the_rows_are_the_relay_presets_questions_without_its_system_prompt_and_tools() -> None:
    _, config = sessions()
    _, relay = resolve(RELAY)
    (dataset,) = settings(config, AGENT)["datasets"]
    (relay_dataset,) = settings(relay, RELAY_AGENT)["datasets"]

    assert dataset == relay_dataset | {
        "jsonl_fpath": "benchmarks/browsecomp/data/browsecomp_opencode_benchmark.jsonl",
        "prepare_script": "benchmarks/browsecomp/prepare_opencode.py",
    }
    # The prepare script writes the file the dataset reads.
    assert ROOT / dataset["jsonl_fpath"] == OUTPUT_FPATH


def test_the_relay_preset_keeps_its_explicit_legacy_agent_relay() -> None:
    _, relay = resolve(RELAY)

    assert OmegaConf.missing_keys(relay) == set()
    assert environment_servers(relay) == {RELAY_ENVIRONMENT: ["legacy_agent"]}
    assert settings(relay, RELAY_ENVIRONMENT)["agent_server"]["name"] == RELAY_AGENT
    assert list(relay[RELAY_AGENT].responses_api_agents) == ["opencode_sandboxed_agent"]
    assert settings(relay, RELAY_AGENT)["resources_server"]["name"] == RESOURCES
