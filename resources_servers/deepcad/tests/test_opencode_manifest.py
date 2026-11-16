# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollouts import RolloutCollectionHelper
from resources_servers.deepcad.dataset.augment_inputs import view_prompt
from resources_servers.deepcad.operations import add_operation_hint
from resources_servers.deepcad.prepare_opencode import prepare
from resources_servers.deepcad.tests.test_opencode_sessions import seed
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig, _session_opencode_config_error


CONFIG = Path(__file__).parents[1] / "configs/opencode.yaml"


def prepared(tmp_path, row=None):
    row = deepcopy(row or seed().task_data)
    row["responses_create_params"]["input"].insert(0, {"role": "system", "content": "Try to use box/fillet."})
    row["responses_create_params"]["input"][-1]["content"].insert(
        0, {"type": "input_text", "text": "Dimensions 1, 2, 3. Output ONLY Python."}
    )
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps(row) + "\n")
    assert prepare(source, output) == 1
    return row, json.loads(output.read_text())


def test_preparation_preserves_images_metadata_hints_and_sampling(tmp_path):
    original = deepcopy(seed().task_data)
    original["responses_create_params"]["temperature"] = 0.8
    original["responses_create_params"]["max_output_tokens"] = 32768
    source, result = prepared(tmp_path, original)
    assert result["verifier_metadata"] == source["verifier_metadata"]
    assert (
        result["responses_create_params"]["input"][-1]["content"][-1]
        == source["responses_create_params"]["input"][-1]["content"][-1]
    )
    assert result["responses_create_params"]["temperature"] == 0.8
    assert result["responses_create_params"]["max_output_tokens"] == 32768
    system = result["responses_create_params"]["input"][0]["content"]
    assert "Try to use box/fillet." in system and "read tool" in system
    assert "model.py" in system
    assert "Output ONLY" not in json.dumps(result["responses_create_params"])
    assert "Dimensions 1, 2, 3." in json.dumps(result["responses_create_params"])


def test_prepared_row_is_accepted_by_unmodified_stock_opencode(tmp_path):
    _, row = prepared(tmp_path)
    config = yaml.safe_load(CONFIG.read_text())["deepcad_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    config.update(name="deepcad_opencode_agent", port=8080)
    agent = OpenCodeAgent(config=OpenCodeAgentConfig(**config), server_client=MagicMock(spec=ServerClient))
    text, system, media = agent._session_input(
        NeMoGymResponseCreateParamsNonStreaming(**row["responses_create_params"])
    )
    assert "Dimensions" in text and "model.py" in system
    assert len(media) == 1
    assert _session_opencode_config_error(agent.config.opencode_config) is None


@pytest.mark.parametrize("broken", ["metadata", "images"])
def test_incomplete_inputs_fail_preparation(tmp_path, broken):
    row = deepcopy(seed().task_data)
    if broken == "metadata":
        row["verifier_metadata"] = {"source_step_path": "/missing/reference.step"}
    else:
        row["responses_create_params"]["input"][0]["content"] = "No image"
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises((ValueError, ValidationError)):
        prepare(source, output)


def test_routes_to_resources_owned_session_and_stock_agent(tmp_path):
    _, row = prepared(tmp_path)
    config = OmegaConf.load(CONFIG)
    RolloutCollectionHelper.resolve_task_sources([row], config)
    assert row["agent_ref"]["name"] == "deepcad_opencode_agent"
    servers = config.deepcad_opencode_environment_server.environment_servers
    assert list(servers) == ["single_agent_turn_legacy"]
    assert servers.single_agent_turn_legacy.resources_tool_transports == []
    assert list(config.deepcad_opencode.resources_servers) == ["deepcad"]
    assert config.deepcad_opencode.resources_servers.deepcad.entrypoint == "opencode_app.py"
    assert list(config.deepcad_opencode_agent.responses_api_agents) == ["opencode_agent"]
    assert config.deepcad_opencode_agent.responses_api_agents.opencode_agent.context_window == 131072
    assert config.head_server.host == "127.0.0.1"
    for block in (
        config.deepcad_opencode.resources_servers.deepcad,
        config.deepcad_opencode_agent.responses_api_agents.opencode_agent,
        servers.single_agent_turn_legacy,
    ):
        assert block.host == "127.0.0.1"


def test_cannot_overwrite_source_manifest(tmp_path):
    path = tmp_path / "source.jsonl"
    with pytest.raises(ValueError, match="separate"):
        prepare(path, path)


def test_composed_config_resolves_offline_without_a_legacy_agent_relay():
    root = CONFIG.parents[3]
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                OmegaConf.load(CONFIG),
                OmegaConf.load(root / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                {
                    "default_host": "192.0.2.10",
                    "policy_base_url": "http://192.0.2.10:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "validation-only",
                    "error_on_agent_without_environment_server": True,
                },
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert config.deepcad_opencode.resources_servers.deepcad.host == "127.0.0.1"
    assert config.policy_model.responses_api_models.vllm_model.host == "192.0.2.10"
    assert list(config.deepcad_opencode_environment_server.environment_servers) == ["single_agent_turn_legacy"]
    SingleAgentTurnLegacyEnvironmentServerConfig(
        name="deepcad_opencode_environment_server",
        **OmegaConf.to_container(
            config.deepcad_opencode_environment_server.environment_servers.single_agent_turn_legacy, resolve=True
        ),
    )


def test_operation_hint_after_single_turn_rules_survives_conversion(tmp_path):
    row = deepcopy(seed().task_data)
    row["responses_create_params"]["input"].insert(0, {"role": "system", "content": view_prompt("diagonal")})
    add_operation_hint(row, ["box", "fillet"], probability=1.0)
    source, output = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    source.write_text(json.dumps(row) + "\n")
    prepare(source, output)
    result = json.loads(output.read_text())
    system = result["responses_create_params"]["input"][0]["content"]
    assert "Try to use box/fillet operations." in system
    assert "do not read files" not in system
    assert "Output ONLY" not in system
    assert result["verifier_metadata"]["operation_hint_applied"] is True
