# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The five-harness recipe and its row preparer. This does not install or run native clients."""

import json
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
from environments.aga_visual_tasks.prepare import HARNESSES, native_row, prepare
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollouts import RolloutCollectionHelper
from resources_servers.aga_visual_tasks.tests.test_prepare import original


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "resources_servers/aga_visual_tasks/configs/multiharness.yaml"
BOUNDARY = {
    "opencode": ("OpenCodeAgent", "_session_input"),
    "pi": ("PiAgent", "_validate_request"),
    "codex": ("CodexAgent", "_sandbox_input"),
    "claude_code": ("ClaudeCodeAgent", "_validate_sandbox_request"),
    "openclaw": ("OpenClawAgent", "_sandbox_input"),
}


def _resolve(monkeypatch, recipes=None):
    monkeypatch.setenv("AGA_SANDBOX_IMAGE", "unused-image")
    monkeypatch.setenv("AGA_IMAGE_ROOT", "/images")
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                        *(recipes or [str(CONFIG)]),
                    ],
                    "default_host": "192.0.2.10",
                    "policy_base_url": "http://192.0.2.10:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "validation-only",
                    "anthropic_api_key": "",
                    "error_on_agent_without_environment_server": True,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


def _prepared(tmp_path: Path, harness: str) -> dict:
    root = tmp_path / "images"
    root.mkdir(exist_ok=True)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(original(root)) + "\n")
    output = tmp_path / f"out-{harness}"
    prepare(source, output, "code_assisted_vqa", root, harness, file_input=harness == "claude_code")
    return json.loads((output / "train.jsonl").read_text())


@pytest.mark.parametrize("harness", HARNESSES)
def test_each_harness_routes_a_prepared_row_through_its_stock_input_boundary(tmp_path, monkeypatch, harness) -> None:
    config = _resolve(monkeypatch)
    row = _prepared(tmp_path, harness)
    assert row["agent_ref"]["name"] == f"aga_{harness}_agent"
    assert row["_ng_environment_server"] == f"aga_{harness}_environment_server"

    rows = RolloutCollectionHelper().preprocess_examples([row], global_config_dict=config)
    RolloutCollectionHelper._validate_environment_servers(rows, config)
    env_name = f"aga_{harness}_environment_server"
    env = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=env_name,
        **OmegaConf.to_container(config[env_name].environment_servers.single_agent_turn_legacy, resolve=True),
    )
    assert env.agent_server.name == f"aga_{harness}_agent" and env.resources_server.name == "aga_visual_tasks"

    class_name, boundary = BOUNDARY[harness]
    module = import_module(f"responses_api_agents.{harness}_agent.app")
    values = OmegaConf.to_container(
        config[env.agent_server.name].responses_api_agents[harness + "_agent"], resolve=True
    )
    agent_config = getattr(module, class_name + "Config")(name=env.agent_server.name, **values)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = config
    agent = getattr(module, class_name)(config=agent_config, server_client=client)
    getattr(agent, boundary)(
        NeMoGymResponseCreateParamsNonStreaming.model_validate(rows[0]["responses_create_params"])
    )


def test_claude_code_takes_the_image_as_a_file_and_the_others_keep_it_inline(tmp_path) -> None:
    claude = _prepared(tmp_path, "claude_code")
    parts = claude["responses_create_params"]["input"][-1]["content"]
    assert all(part["type"] == "input_text" for part in parts)
    assert any("/images/target.png" in part["text"] for part in parts)
    assert claude["source_images"]  # the resources server still seeds the files

    inline = _prepared(tmp_path, "pi")
    assert any(part["type"] == "input_image" for part in inline["responses_create_params"]["input"][-1]["content"])


def test_claude_code_needs_file_input_and_other_harnesses_are_refused(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(original(root)) + "\n")
    with pytest.raises(ValueError, match="use --file-input"):
        prepare(source, tmp_path / "a", "code_assisted_vqa", root, "claude_code")
    with pytest.raises(ValueError, match="Unknown native harness"):
        prepare(source, tmp_path / "b", "code_assisted_vqa", root, "hermes")
    with pytest.raises(ValueError, match="names the image files"):
        native_row(original(root), "code_assisted_vqa", root, file_input=True)


def test_each_environment_server_admits_only_as_many_episodes_as_its_agent_runs(monkeypatch) -> None:
    config = _resolve(monkeypatch)
    for harness in HARNESSES:
        env = config[f"aga_{harness}_environment_server"].environment_servers.single_agent_turn_legacy
        agent = config[f"aga_{harness}_agent"].responses_api_agents[f"{harness}_agent"]
        assert agent.concurrency == env.max_concurrent_episodes, harness


def test_the_added_agents_use_an_alias_that_keeps_tool_output_images_inline(monkeypatch) -> None:
    config = _resolve(monkeypatch)

    assert config.aga_policy_model.responses_api_models.vllm_model.inline_tool_output_images
    assert not config.policy_model.responses_api_models.vllm_model.get("inline_tool_output_images", False)
    for harness in HARNESSES:
        agent = config[f"aga_{harness}_agent"].responses_api_agents[f"{harness}_agent"]
        assert agent.model_server.name == "aga_policy_model"
    assert config.aga_pi_agent.responses_api_agents.pi_agent.image_input is True
    assert config.aga_claude_code_agent.responses_api_agents.claude_code_agent.max_turns == 8


@pytest.mark.parametrize("cad_first", [False, True])
def test_the_recipe_does_not_change_another_benchmarks_agent(monkeypatch, cad_first) -> None:
    other = str(ROOT / "benchmarks/swebench/verified/opencode.yaml")
    agent = "swebench_verified_opencode_agent"
    baseline = _resolve(monkeypatch, [other])
    combined = _resolve(monkeypatch, [str(CONFIG), other] if cad_first else [other, str(CONFIG)])

    assert OmegaConf.to_container(combined[agent], resolve=False) == OmegaConf.to_container(
        baseline[agent], resolve=False
    )
    assert OmegaConf.to_container(combined.policy_model, resolve=True) == OmegaConf.to_container(
        baseline.policy_model, resolve=True
    )
