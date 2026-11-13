# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BrowseComp's OpenCode session preset against the config classes of opencode_agent and this server.

tests/unit_tests/benchmarks/test_browsecomp_opencode_configs.py checks the rest of the preset in the core suite, which
imports neither server, so the core suite's coverage does not count them."""

import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from resources_servers.opencode_search.app import (
    _EXPORT_COPY_TIMEOUT_S,
    SearchResourcesServer,
    SearchResourcesServerConfig,
)
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig, _session_opencode_config_error
from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION
from tests.unit_tests.benchmarks.test_browsecomp_opencode_configs import EXPORT_COPY_S


ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "benchmarks/browsecomp/README.md"
SESSIONS = "benchmarks/browsecomp/configs/browsecomp_opencode_sessions.yaml"
RELAY = "benchmarks/browsecomp/configs/browsecomp_opencode.yaml"
AGENT = "browsecomp_opencode_agent"
RESOURCES = "search_browsecomp_resources_server"
RELAY_AGENT = "opencode_sandboxed_search_agent"
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


@pytest.mark.parametrize(
    ("name", "config_class"), [(AGENT, OpenCodeAgentConfig), (RESOURCES, SearchResourcesServerConfig)]
)
def test_every_setting_is_one_gym_reads(name: str, config_class: type[BaseModel]) -> None:
    """Both config classes ignore keys they do not declare, so a misspelled setting would compose and do nothing.
    Gym's config layer reads the other keys: those of every server block, such as datasets, and catalog entries."""
    parser, config = sessions()
    (instance,) = (instance for instance in parser.filter_for_server_instance_configs(config) if instance.name == name)
    block = settings(config, name)

    # The keys of every server block are the fields of the config Gym's config layer reads the block as.
    read_by_gym = type(instance.get_inner_run_server_config()).model_fields.keys() | CATALOG_KEYS
    assert block.keys() - config_class.model_fields.keys() - read_by_gym == set()
    config_class.model_validate(block | {"name": name})


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


@pytest.mark.parametrize("export_dir", ["/tmp/nemo-gym-exports", None], ids=["set", "unset"])
def test_the_resources_server_copies_the_exports_from_where_the_agent_writes_them(
    export_dir: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BROWSECOMP_OPENCODE_EXPORT_DIR names both ends: the agent writes each session's export there, and verify copies
    the exports from there. The host directory is a run's own, so the preset leaves it to an overlay."""
    if export_dir is not None:
        monkeypatch.setenv("BROWSECOMP_OPENCODE_EXPORT_DIR", export_dir)
    _, config = sessions()
    resources = SearchResourcesServerConfig.model_validate(settings(config, RESOURCES) | {"name": RESOURCES})
    agent = OpenCodeAgentConfig.model_validate(settings(config, AGENT) | {"name": AGENT})

    assert resources.episode_export_dir == agent.export_dir == export_dir
    assert resources.episode_export_copy_dir is None


def test_the_readme_gives_the_judge_retry_limits_this_server_applies() -> None:
    """The README's time-limits section names each limit as `setting: value`. Neither preset sets the judge's retry
    limits, so they are this server's defaults, which the core suite's check of that section cannot read; a change to
    one that leaves the section stale fails here."""
    _, config = sessions()
    resources = SearchResourcesServerConfig.model_validate(settings(config, RESOURCES) | {"name": RESOURCES})
    section = README.read_text().split("\n### Time limits\n", 1)[1].split("\n### ", 1)[0]
    documented = {name: float(value) for name, value in re.findall(r"`([\w.]+): (\d+)`", section)}

    retry_limits = ("judge_retry_timeout_seconds", "judge_max_attempts", "judge_retry_delay_seconds")
    applied = {name: getattr(resources, name) for name in retry_limits}
    assert {name: documented.get(name) for name in retry_limits} == applied
    assert applied == {name: SearchResourcesServerConfig.model_fields[name].default for name in retry_limits}


def test_the_core_timing_check_counts_the_export_copy_at_this_servers_bound() -> None:
    """The core suite checks that an episode ends before its lent sandbox expires with the export copy at its bound, a
    constant of its own since the module that checks it imports neither this server nor opencode_agent. It must be
    the bound this server applies."""
    assert EXPORT_COPY_S == _EXPORT_COPY_TIMEOUT_S
