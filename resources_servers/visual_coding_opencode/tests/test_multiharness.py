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
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollouts import RolloutCollectionHelper
from resources_servers.visual_coding_opencode.app import MAX_INPUT_IMAGES
from resources_servers.visual_coding_opencode.prepare_multiharness import HARNESSES, prepare, prepare_row


ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "resources_servers/visual_coding_opencode/configs/multiharness.yaml"
PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
ROW = {
    "responses_create_params": {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_image", "image_url": PNG, "detail": "auto"},
                    {"type": "input_text", "text": "Which cell differs?"},
                ],
            }
        ]
    },
    "expected_answer": "(3,4)",
    "question": "Which cell differs?",
}


def test_an_inline_row_keeps_its_image_and_gains_the_files_and_routing() -> None:
    row = prepare_row(ROW, harness="pi", file_input=False)

    assert row["responses_create_params"] == ROW["responses_create_params"]
    assert row["input_images"] == [PNG]
    assert row["agent_ref"] == {"name": "visual_coding_pi_agent"}
    assert row["_ng_environment_server"] == "visual_coding_pi_environment_server"
    assert row["expected_answer"] == "(3,4)"


def test_a_file_input_row_names_the_workspace_files_and_has_no_inline_image() -> None:
    row = prepare_row(ROW, harness="claude_code", file_input=True)

    (message,) = row["responses_create_params"]["input"]
    assert message["role"] == "user"
    assert isinstance(message["content"], str)
    assert "Which cell differs?" in message["content"]
    assert message["content"].endswith("Image files:\n/workspace/input-0.png")
    assert "data:image" not in json.dumps(row["responses_create_params"])
    assert row["input_images"] == [PNG]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r["responses_create_params"]["input"][0]["content"].pop(0), "need an image"),
        (lambda r: r["responses_create_params"].update(input="plain text"), "one user message"),
        (
            lambda r: r["responses_create_params"]["input"][0]["content"].append({"type": "input_file"}),
            "only text and image",
        ),
    ],
)
def test_rows_that_are_not_one_user_message_of_text_and_images_are_refused(mutate, message) -> None:
    row = json.loads(json.dumps(ROW))
    mutate(row)
    with pytest.raises(ValueError, match=message):
        prepare_row(row, harness="opencode", file_input=False)


def test_an_unsupported_or_malformed_image_is_refused_for_every_harness() -> None:
    for image_url in ("data:image/bmp;base64,AAAA", "image/png;base64,AAAA", "not a data url", None):
        row = json.loads(json.dumps(ROW))
        row["responses_create_params"]["input"][0]["content"][0]["image_url"] = image_url
        with pytest.raises(ValueError, match="PNG, JPEG, WebP or GIF"):
            prepare_row(row, harness="pi", file_input=False)


def test_a_row_with_too_many_images_or_a_part_without_a_type_is_refused() -> None:
    many = json.loads(json.dumps(ROW))
    many["responses_create_params"]["input"][0]["content"] = [
        {"type": "input_image", "image_url": PNG, "detail": "auto"}
    ] * (MAX_INPUT_IMAGES + 1)
    with pytest.raises(ValueError, match="at most"):
        prepare_row(many, harness="opencode", file_input=False)
    untyped = json.loads(json.dumps(ROW))
    untyped["responses_create_params"]["input"][0]["content"].append({"text": "no type"})
    with pytest.raises(ValueError, match="only text and image"):
        prepare_row(untyped, harness="opencode", file_input=False)


def test_claude_code_needs_file_input_and_unknown_harnesses_are_refused(tmp_path) -> None:
    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps(ROW) + "\n")
    with pytest.raises(ValueError, match="use --file-input"):
        prepare(source, tmp_path / "out.jsonl", harness="claude_code")
    with pytest.raises(ValueError, match="Unsupported harness"):
        prepare(source, tmp_path / "out.jsonl", harness="hermes")
    with pytest.raises(ValueError, match="separate output"):
        prepare(source, source, harness="opencode")
    assert prepare(source, tmp_path / "ok.jsonl", harness="pi") == 1


def _resolve():
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                        str(CONFIG),
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


@pytest.mark.parametrize(
    "harness,class_name,boundary",
    [
        ("opencode", "OpenCodeAgent", "_session_input"),
        ("pi", "PiAgent", "_validate_request"),
        ("codex", "CodexAgent", "_sandbox_input"),
        ("claude_code", "ClaudeCodeAgent", "_validate_sandbox_request"),
        ("openclaw", "OpenClawAgent", "_sandbox_input"),
    ],
)
def test_each_harness_routes_a_prepared_row_through_its_stock_input_boundary(
    tmp_path, harness, class_name, boundary
) -> None:
    config = _resolve()
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(ROW) + "\n")
    prepared_path = tmp_path / f"{harness}.jsonl"
    prepare(source, prepared_path, harness=harness, file_input=harness == "claude_code")
    row = json.loads(prepared_path.read_text())

    rows = RolloutCollectionHelper().preprocess_examples([row], global_config_dict=config)
    RolloutCollectionHelper._validate_environment_servers(rows, config)
    env_name = f"visual_coding_{harness}_environment_server"
    env = SingleAgentTurnLegacyEnvironmentServerConfig(
        name=env_name,
        **OmegaConf.to_container(config[env_name].environment_servers.single_agent_turn_legacy, resolve=True),
    )
    assert rows[0]["agent_ref"]["name"] == env.agent_server.name == f"visual_coding_{harness}_agent"
    assert env.resources_server.name == "visual_coding_opencode"

    module = import_module(f"responses_api_agents.{harness}_agent.app")
    values = OmegaConf.to_container(
        config[env.agent_server.name].responses_api_agents[harness + "_agent"], resolve=True
    )
    agent_config = getattr(module, class_name + "Config")(name=env.agent_server.name, **values)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = config
    agent = getattr(module, class_name)(config=agent_config, server_client=client)
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(rows[0]["responses_create_params"])
    getattr(agent, boundary)(params)


def test_the_recipe_has_one_environment_server_per_harness() -> None:
    config = _resolve()
    names = {
        key for key, value in config.items() if isinstance(value, type(config)) and "environment_servers" in value
    }
    assert names == {f"visual_coding_{h}_environment_server" for h in HARNESSES}


def test_the_added_agents_use_an_alias_that_keeps_tool_output_images_inline() -> None:
    config = _resolve()

    alias = config.visual_coding_policy_model.responses_api_models.vllm_model
    assert alias.inline_tool_output_images
    # The shared policy model and the original OpenCode route are unchanged.
    assert not config.policy_model.responses_api_models.vllm_model.get("inline_tool_output_images", False)
    assert config.visual_coding_opencode_agent.responses_api_agents.opencode_agent.model_server.name == "policy_model"
    for harness in ("pi", "codex", "claude_code", "openclaw"):
        agent = config[f"visual_coding_{harness}_agent"].responses_api_agents[f"{harness}_agent"]
        assert agent.model_server.name == "visual_coding_policy_model"


def test_the_added_agents_keep_the_settings_the_recipe_sets() -> None:
    config = _resolve()
    for harness in ("pi", "openclaw"):
        agent = config[f"visual_coding_{harness}_agent"].responses_api_agents[f"{harness}_agent"]
        assert agent.image_input is True
        assert agent.context_window == 131072 and agent.max_output_tokens == 32768
    assert config.visual_coding_pi_agent.responses_api_agents.pi_agent.auto_compaction is False
    assert config.visual_coding_codex_agent.responses_api_agents.codex_agent.model_context_window == 131072


@pytest.mark.parametrize("cad_first", [False, True])
def test_the_recipe_does_not_change_another_benchmarks_agent(cad_first) -> None:
    other = str(ROOT / "benchmarks/swebench/verified/opencode.yaml")
    agent = "swebench_verified_opencode_agent"

    def resolve(recipes):
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.create(
                    {
                        "config_paths": [
                            str(ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                            *recipes,
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

    baseline = resolve([other])
    combined = resolve([str(CONFIG), other] if cad_first else [other, str(CONFIG)])
    assert OmegaConf.to_container(combined[agent], resolve=False) == OmegaConf.to_container(
        baseline[agent], resolve=False
    )
    assert OmegaConf.to_container(combined.policy_model, resolve=True) == OmegaConf.to_container(
        baseline.policy_model, resolve=True
    )
