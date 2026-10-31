# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the shipped recipe's routing, model budgets and request schemas without starting services."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.config import BenchmarkDatasetConfig, GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.task_data import normalize_task_fields
from nemo_gym.train_data_utils import TrainDataProcessor
from resources_servers.critpt_custom_grader import app as adapter
from resources_servers.critpt_custom_grader.task_data import TaskData


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "critpt_custom_grader_agentic.yaml"
AGENT = "critpt_custom_grader_stirrup_agent"
MODEL_SETTINGS = {
    "context_window_tokens": 32768,
    "max_completion_tokens_cap": 16384,
    "enable_thinking": False,
    "temperature": 0.0,
    "top_p": 1.0,
}


def resolved_recipe(monkeypatch, *, with_model_settings=True):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "recipe-test-key")
    monkeypatch.setenv("CRITPT_CUSTOM_GRADER_IMAGE", "example/grader@sha256:" + "1" * 64)
    monkeypatch.setenv("CRITPT_AGENTIC_EXECUTION_IMAGE", "example/execution@sha256:" + "2" * 64)
    monkeypatch.setenv("CRITPT_AGENTIC_ORCHESTRATION_IMAGE", "example/orchestration@sha256:" + "3" * 64)
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [str(CONFIG)], "error_on_agent_without_environment_server": True},
        {AGENT: {"responses_api_agents": {"stirrup_agent": MODEL_SETTINGS}}} if with_model_settings else {},
    )
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial, skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True
        )
    )


def test_agentic_recipe_routes_remote_tools_and_stateless_verification(monkeypatch):
    resolved = resolved_recipe(monkeypatch)
    block = OmegaConf.to_container(
        resolved["critpt_custom_grader"]["resources_servers"]["critpt_custom_grader"], resolve=True
    )
    config = adapter.CritPtCustomGraderConfig.model_validate({**block, "name": "critpt_custom_grader"})
    assert config.answer_source == "finish_tool"
    assert config.REVERIFY_MODE.value == "stateless"
    assert config.agentic_execution.command_timeout_s == 300
    agent = resolved[AGENT]["responses_api_agents"]["stirrup_agent"]
    assert agent["code_exec"] is False
    assert agent["finish_tool_names"] == ["finish"]
    assert agent["agent_max_turns"] == 50
    assert agent["concurrency"] <= config.agentic_execution.max_sessions
    dataset = BenchmarkDatasetConfig.model_validate(block["datasets"][0])
    assert dataset.num_repeats == 5
    assert dataset.prompt_config is not None
    assert (ROOT.parents[1] / dataset.prompt_config).resolve() == ROOT / "prompts" / "agentic.yaml"
    assert (ROOT.parents[1] / dataset.jsonl_fpath).resolve() == ROOT / "data" / "agentic_example.jsonl"
    environment = resolved["critpt_custom_grader_agentic_environment"]["environment_servers"][
        "single_agent_turn_legacy"
    ]
    assert environment["agent_server"]["name"] == AGENT
    assert environment["resources_server"]["name"] == "critpt_custom_grader"
    assert environment["resources_tool_transports"] == ["direct_http"]
    deny = {"defaultAction": "deny", "egress": []}
    assert config.agentic_execution.sandbox_spec.provider_options["network_policy"] == deny
    assert config.sandbox_spec.provider_options["network_policy"] == deny
    assert "network_policy" not in agent["sandbox_config"].get("provider_options", {})


def test_agentic_dataset_uses_resources_schema_and_environment_route(monkeypatch, tmp_path):
    resolved = resolved_recipe(monkeypatch)
    instances = GlobalConfigDictParser().filter_for_server_instance_configs(resolved)
    owner = next(instance for instance in instances if instance.name == "critpt_custom_grader")
    assert TrainDataProcessor._schema_home_for(owner, instances) == ("critpt_custom_grader", "resources_servers")
    dataset = owner.datasets[0]
    assert isinstance(dataset, BenchmarkDatasetConfig)
    assert dataset.taskset == "critpt_custom_grader_synthetic"
    benchmark = BenchmarkConfig.from_initial_config_dict(CONFIG, resolved)
    assert benchmark.agent_name == AGENT
    assert benchmark.environment_server == "critpt_custom_grader_agentic_environment"
    assert benchmark.num_repeats == 5

    local_input = tmp_path / "agentic_example.jsonl"
    local_input.write_text((ROOT / "data" / "agentic_example.jsonl").read_text())
    dataset.jsonl_fpath = str(local_input)
    dataset.prompt_config = ROOT / "prompts" / "agentic.yaml"
    prepared = TrainDataProcessor()._collate_samples_single_type("benchmark", [owner], task_data_validation="error")
    rows = [json.loads(line) for line in prepared[0].read_text().splitlines()]
    assert len(rows) == 5
    for row in rows:
        assert row["task_id"]["taskset"] == dataset.taskset
        assert "agent_ref" not in row
        fields, conflicts = normalize_task_fields(row["task_input"])
        assert conflicts == []
        TaskData.model_validate(fields)


def test_agentic_recipe_validates_complete_environment_config(monkeypatch):
    resolved = resolved_recipe(monkeypatch)
    name = "critpt_custom_grader_agentic_environment"
    block = OmegaConf.to_container(resolved[name]["environment_servers"]["single_agent_turn_legacy"], resolve=True)
    environment = SingleAgentTurnEnvironmentServerConfig.model_validate({**block, "name": name})
    assert environment.agent_server.name == AGENT
    assert environment.resources_server.name == "critpt_custom_grader"
    assert environment.max_concurrent_episodes == 8
    assert environment.queue_timeout_seconds is not None
    assert environment.queue_timeout_seconds > 0
    assert environment.default_episode_timeout_seconds is not None
    assert environment.cleanup_timeout_seconds > 0


def test_agentic_recipe_requires_explicit_model_settings(monkeypatch):
    with pytest.raises(ValueError, match="context_window_tokens"):
        resolved_recipe(monkeypatch, with_model_settings=False)


def test_agentic_example_renders_and_matches_the_endpoint_schemas():
    from resources_servers.critpt_custom_grader.agentic_tools import get_code_execution_params
    from resources_servers.critpt_custom_grader.submission import FinishRequest

    prompt = load_prompt_config(str(ROOT / "prompts" / "agentic.yaml"))
    rows = [json.loads(line) for line in (ROOT / "data" / "agentic_example.jsonl").read_text().splitlines()]
    validate_prompt_compatibility(rows, prompt)
    for row in rows:
        rendered = apply_prompt_to_row(row, prompt)
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate(rendered["responses_create_params"])
        assert len(body.input) == 2
        assert row["problem"] in rendered["responses_create_params"]["input"][1]["content"]
        assert row["code_template"] in rendered["responses_create_params"]["input"][1]["content"]
        fields, conflicts = normalize_task_fields(rendered)
        assert conflicts == []
        task = TaskData.model_validate(fields)
        assert task.problem_id.startswith("synthetic-")
        tools = {tool["name"]: tool for tool in row["responses_create_params"]["tools"]}
        assert set(tools) == {"code_exec", "finish"}
        for name, model in (("code_exec", get_code_execution_params()), ("finish", FinishRequest)):
            actual = model.model_json_schema()
            supplied = tools[name]["parameters"]
            assert set(supplied["required"]) == set(actual["required"])
            assert set(supplied["properties"]) == set(actual["properties"])
            for field, schema in supplied["properties"].items():
                assert schema["type"] == actual["properties"][field]["type"]
        assert "reference_source" not in json.dumps(rendered["responses_create_params"])
