# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig

import nemo_gym.environment.runtime_composition as runtime_composition
from nemo_gym import WORKING_DIR
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.authoring import find_environment_definition, load_environment
from nemo_gym.environment.episode_protocols import create_episode_protocol_runtime
from nemo_gym.environment.runtime_composition import (
    AgentRoleBinding,
    EpisodeProtocolRuntime,
    SandboxRuntime,
    compose_environment_run,
)
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import E2ERolloutCollectionConfig


HELLO_WORLD = WORKING_DIR / "environments/hello_world/environment.yaml"
HELLO_TASKSET = WORKING_DIR / "environments/hello_taskset/environment.yaml"
HELLO_MCP_TOOL = WORKING_DIR / "environments/hello_mcp_tool/environment.yaml"


def test_find_environment_definition_by_declared_name_and_path() -> None:
    assert find_environment_definition("hello-world") == HELLO_WORLD
    assert find_environment_definition(HELLO_WORLD.parent) == HELLO_WORLD


def test_borrowed_sandbox_profile_does_not_impose_model_compatibility() -> None:
    profile_path = WORKING_DIR / "responses_api_agents/hermes_agent/configs/borrowed_sandbox.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    hermes = profile["hermes_agent"]["responses_api_agents"]["hermes_agent"]

    assert hermes["enabled_toolsets"] == ["terminal"]
    assert "chat_template_kwargs_enabled" not in hermes


def test_compose_environment_run_materializes_internal_input_without_dataset_config(
    monkeypatch, tmp_path: Path
) -> None:
    loaded = load_environment(HELLO_TASKSET)
    materialize_tasks = runtime_composition.materialize_tasks
    materialization_calls = 0

    def count_materializations(*args, **kwargs):
        nonlocal materialization_calls
        materialization_calls += 1
        return materialize_tasks(*args, **kwargs)

    monkeypatch.setattr(runtime_composition, "materialize_tasks", count_materializations)
    sandbox = SandboxRuntime(
        config_paths=(Path("/supporting.yaml"),),
        runtime_image="nemo-gym-test:123",
        sandbox_provider_ref="sandbox",
        sandbox_config={},
    )
    episode_protocol = EpisodeProtocolRuntime(
        config_paths=(Path("/episode.yaml"),),
        environment_server_name="test_environment_server",
        environment_server_config={"test_environment_server": {"environment_servers": {}}},
        agent_roles={
            "agent": AgentRoleBinding(resources_server_name="environment_adapter_resources_server"),
        },
    )

    artifacts = compose_environment_run(
        loaded,
        tmp_path,
        sandbox=sandbox,
        episode_protocol=episode_protocol,
        adapter_config_path=Path("/adapter.yaml"),
    )

    assert len(artifacts.input_jsonl_path.read_text().splitlines()) == 2
    assert materialization_calls == 1
    config = yaml.safe_load(artifacts.config_path.read_text())
    assert config["rollout_input"] == {
        "type": "materialized_tasks",
        "path": str(artifacts.input_jsonl_path),
    }
    assert config["tasksets"] == {
        "example": {
            "task_input_contract": "nemo_gym.single_agent.v1",
        }
    }
    adapter = config["environment_adapter_resources_server"]["resources_servers"]["environment_adapter"]
    assert adapter["environment_root"] == str(HELLO_TASKSET.parent)
    assert adapter["runtime_image"] == "nemo-gym-test:123"
    assert "datasets" not in adapter
    assert "license" not in artifacts.config_path.read_text()
    assert config["agent_bindings"] == {
        "agent": {
            "resources_server": {
                "type": "resources_servers",
                "name": "environment_adapter_resources_server",
            }
        }
    }
    assert "environment_agent" not in config
    assert artifacts.config_paths == (
        Path("/supporting.yaml"),
        Path("/adapter.yaml"),
        Path("/episode.yaml"),
        artifacts.config_path,
    )


def test_compose_environment_run_rejects_unimplemented_mcp_servers(tmp_path: Path) -> None:
    loaded = load_environment(HELLO_MCP_TOOL)
    sandbox = SandboxRuntime(
        config_paths=(),
        runtime_image="nemo-gym-test:123",
        sandbox_provider_ref="sandbox",
        sandbox_config={},
    )
    episode_protocol = EpisodeProtocolRuntime(
        config_paths=(),
        environment_server_name="test_environment_server",
        environment_server_config={},
        agent_roles={},
    )

    with pytest.raises(ConfigError, match="does not yet support runtime.mcp_servers"):
        compose_environment_run(
            loaded,
            tmp_path,
            sandbox=sandbox,
            episode_protocol=episode_protocol,
            adapter_config_path=Path("/adapter.yaml"),
        )


def test_generated_composition_accepts_an_explicit_agent_type(tmp_path: Path) -> None:
    loaded = load_environment(HELLO_WORLD)
    sandbox = SandboxRuntime(
        config_paths=(WORKING_DIR / "nemo_gym/sandbox/providers/docker/configs/docker.yaml",),
        runtime_image="nemo-gym-test:123",
        sandbox_provider_ref="sandbox",
        sandbox_config={},
    )
    episode_protocol = create_episode_protocol_runtime(loaded)
    artifacts = compose_environment_run(
        loaded,
        tmp_path,
        sandbox=sandbox,
        episode_protocol=episode_protocol,
        adapter_config_path=WORKING_DIR / "resources_servers/environment_adapter/configs/environment_adapter.yaml",
    )
    config_paths = [
        WORKING_DIR / "responses_api_models/openai_model/configs/openai_model.yaml",
        WORKING_DIR / "responses_api_agents/hermes_agent/configs/borrowed_sandbox_openai_compatible.yaml",
        *artifacts.config_paths,
    ]

    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=DictConfig(
                {
                    "config_paths": [str(path) for path in config_paths],
                    "policy_base_url": "http://127.0.0.1:8000/v1",
                    "policy_api_key": "test",
                    "policy_model_name": "test-model",
                    "output_jsonl_fpath": str(tmp_path / "rollouts.jsonl"),
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    assert "hermes_agent" in config
    hermes = config["hermes_agent"]["responses_api_agents"]["hermes_agent"]
    assert hermes["enabled_toolsets"] == ["terminal"]
    assert hermes["chat_template_kwargs_enabled"] is False
    agent_ref = config["single_agent_environment_server"]["environment_servers"]["single_agent"]["agent_server"]
    assert agent_ref["name"] == "hermes_agent"
    rollout_config = E2ERolloutCollectionConfig.model_validate(config)
    assert rollout_config.rollout_input is not None
    assert rollout_config.rollout_input.type == "materialized_tasks"
    assert rollout_config.split is None
