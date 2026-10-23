# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped example stack keeps the settings the benchmark's results depend on, and resolves as Gym loads it."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from aiohttp import ClientConnectionError
from omegaconf import OmegaConf

from nemo_gym.environment.validation import _resolve_manifest_composition
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_if import opencode_bundle
from resources_servers.agentic_if.app import AgenticIFResourcesServerConfig
from resources_servers.agentic_if.if_grading.configuration import load_config
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_models.openai_model import app as openai_model


GYM = Path(__file__).parents[3]
CONFIG_PATH = Path(__file__).parents[1] / "configs" / "agentic_if.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SERVER = CONFIG["agentic_if_resources"]["resources_servers"]["agentic_if"]
AGENT = CONFIG["agentic_if_opencode_agent"]["responses_api_agents"]["opencode_agent"]


def resolved(**overrides):
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(CONFIG_PATH)], **overrides},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return config, OmegaConf.to_container(config.agentic_if_resources.resources_servers.agentic_if, resolve=True)


def test_the_judge_model_server_serves_the_model_judge_yaml_names():
    reference = SERVER["judge_model_server"]
    assert reference["type"] == "responses_api_models"
    server = CONFIG[reference["name"]]["responses_api_models"]["openai_model"]
    # The model server replaces the request's model with its own, so the two must name the same judge.
    assert server["openai_model"] == load_config()["model"]["name"]
    assert server == {
        "entrypoint": "app.py",
        "openai_base_url": "${oc.select:judge_base_url,https://inference-api.nvidia.com/v1}",
        "openai_api_key": "${oc.select:judge_api_key,null}",
        "openai_model": "nvidia/zai-org/glm-5.3",
        "upstream_request_timeout_seconds": 600,
        "upstream_max_num_tries": 1,
        "upstream_retry_policy": {"max_attempts": 3, "backoff_initial_seconds": 5},
    }


@pytest.mark.asyncio
async def test_a_judge_outage_fails_after_the_bounded_retries(monkeypatch):
    """A refused connection counts against the retry policy, so an outage ends as a failed call, not a hung episode."""
    block = CONFIG["agentic_if_judge_model"]["responses_api_models"]["openai_model"]
    endpoint = {"openai_base_url": "https://judge.invalid/v1", "openai_api_key": "test"}  # pragma: allowlist secret
    config = openai_model.SimpleModelServerConfig(
        host="127.0.0.1", port=0, name="agentic_if_judge_model", **(block | endpoint)
    )
    server = openai_model.SimpleModelServer(
        config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )
    waits, calls = [], 0

    async def no_wait(seconds):
        waits.append(seconds)

    async def refused():
        nonlocal calls
        calls += 1
        raise ClientConnectionError("connection refused")

    monkeypatch.setattr(openai_model.asyncio, "sleep", no_wait)
    with pytest.raises(openai_model.UpstreamRetriesExhaustedError):
        await server._call_upstream(refused)
    assert calls == 3 and waits == [5, 10]


def test_the_example_stack_composes_upstream_components_around_this_server():
    # The run's config supplies the policy model; the composition is upstream single_agent_turn and opencode_agent.
    assert "policy_model" not in CONFIG
    assert CONFIG["environment_server_routes"] == {"agentic_if:example": "agentic_if_environment_server"}
    environment = CONFIG["agentic_if_environment_server"]
    assert environment["_inherit_from"] == "single_agent_turn"
    bound = environment["environment_servers"]["single_agent_turn"]
    assert bound["resources_server"]["name"] == "agentic_if_resources"
    assert bound["agent_server"]["name"] == "agentic_if_opencode_agent"
    # Admission at the agent's rate; the episode deadline covers the harness run and the rest of the episode; each
    # cleanup callback covers about five of the agent's close steps.
    assert (bound["max_concurrent_episodes"], bound["queue_timeout_seconds"]) == (8, 604800)
    assert bound["default_episode_timeout_seconds"] > AGENT["timeout"]
    assert bound["cleanup_timeout_seconds"] >= 5 * AGENT["session_close_timeout_seconds"]
    assert CONFIG["agentic_if_opencode_agent"]["_inherit_from"] == "opencode_agent"
    assert SERVER["allowed_agents"] == ["opencode_agent"] and SERVER["num_workers"] == 1
    (dataset,) = SERVER["datasets"]
    assert (dataset["name"], dataset["type"], dataset["taskset"]) == ("example", "example", "agentic_if:example")
    assert (GYM / dataset["jsonl_fpath"]).is_file()


def test_the_agent_loads_the_bundle_the_resources_server_stages():
    assert tuple(AGENT["plugins"]) == tuple(opencode_bundle.PLUGIN_PATHS)
    assert AGENT["export_dir"] == opencode_bundle.EXPORT_DIR
    assert AGENT["session_import_path"] == opencode_bundle.SESSION_IMPORT_PATH
    # OpenCode's limits and the harness budget the benchmark's results depend on.
    assert (AGENT["context_window"], AGENT["max_output_tokens"], AGENT["timeout"]) == (131072, 131072, 10800)
    # No tools map (it would re-enable skill and webfetch), and the three denies.
    assert AGENT["opencode_config"]["tools"] is None
    assert AGENT["opencode_config"]["permission"] == {"webfetch": "deny", "websearch": "deny", "skill": "deny"}


def test_the_agent_adds_no_system_prompt_of_its_own():
    # The agent joins its own system_prompt before the row's system message, and the boundary would then report
    # boundary_system_mismatch for every episode: the model must see the row's system text alone.
    assert "system_prompt" not in AGENT
    config, _ = resolved()
    agent = config.agentic_if_opencode_agent.responses_api_agents.opencode_agent
    assert agent.get("system_prompt") is None
    container = OmegaConf.to_container(agent, resolve=True)
    assert (
        OpenCodeAgentConfig.model_validate(container | {"host": "127.0.0.1", "port": 1, "name": "x"}).system_prompt
        is None
    )


def test_stock_gym_resolves_the_composition():
    composition = _resolve_manifest_composition(CONFIG_PATH)
    assert (composition.agent_server, composition.resources_server, composition.model_server) == (
        "opencode_agent",
        "agentic_if",
        "policy_model",
    )
    assert [dataset.name for dataset in composition.datasets] == ["example"]
    config, _ = resolved()
    agent = config.agentic_if_opencode_agent.responses_api_agents.opencode_agent
    assert agent.opencode_config.tools is None
    # The agent's default permission map is kept, with the denies after it: the last matching rule wins.
    permission = OmegaConf.to_container(agent.opencode_config.permission)
    assert list(permission)[0] == "*" and list(permission)[-3:] == ["webfetch", "websearch", "skill"]
    assert "opencode_agent" not in config and "single_agent_turn" not in config  # moved, not duplicated
    assert config.get("scale_swe_golden_patch_resources_server") is None


def test_the_resolved_server_config_validates_with_absolute_artifacts_and_known_task_backends(tmp_path):
    config, server = resolved()
    validated = AgenticIFResourcesServerConfig.model_validate(server | {"host": "127.0.0.1", "port": 1, "name": "x"})
    assert validated.artifacts_dir == Path(config["results_dir"]) / "agentic_if"
    assert validated.artifacts_dir.is_absolute()
    for backend in validated.task_resources.values():
        assert "resources_servers" in config[backend.name]
    _, chosen = resolved(agentic_if_artifacts_dir=str(tmp_path / "job" / "agentic_if"))
    assert chosen["artifacts_dir"] == str(tmp_path / "job" / "agentic_if")


@pytest.mark.parametrize("value", ["relative/agentic_if", ""])
def test_a_relative_artifacts_dir_is_refused(value):
    _, server = resolved(agentic_if_artifacts_dir=value)
    with pytest.raises(ValueError, match="absolute"):
        AgenticIFResourcesServerConfig.model_validate(server | {"host": "127.0.0.1", "port": 1, "name": "x"})
