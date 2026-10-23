# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from benchmarks.gdpval import prepare as preparation
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_name, _benchmark_config_paths
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.server_utils import ServerClient


ROOT = Path(__file__).resolve().parents[3]


def test_nooa_recipe_preserves_gdpval_tools_grading_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("nooa")
    pytest.importorskip("datasets")
    from nemo_gym.rollout_collection import (
        RolloutCollectionHelper,
        _environment_server_for_agent,
        _environment_servers_by_agent,
    )
    from nemo_gym.train_data_utils import TrainDataProcessor
    from resources_servers.gdpval.app import GDPValResourcesServerConfig
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig
    from responses_api_agents.nooa_agent.task_agent import _latest_user_text

    monkeypatch.setenv("TAVILY_API_KEY", "unused")
    monkeypatch.setenv("GDPVAL_SANDBOX_IMAGE", "fixture:1")
    monkeypatch.setenv("PERSIST_DELIVERABLES_DIR", str(tmp_path / "deliverables"))
    monkeypatch.setattr("nemo_gym.global_config._find_open_port_using_range", lambda **_: 12345)
    recipe = ROOT / "benchmarks/gdpval/nooa.yaml"
    assert recipe in _benchmark_config_paths(recipe.parent)
    assert _benchmark_config_name(recipe.relative_to(ROOT / "benchmarks")) == "gdpval/nooa"
    benchmark = BenchmarkConfig.from_config_path(recipe, strict=False)
    assert benchmark is not None
    assert benchmark.agent_name == "gdpval_nooa_agent"
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/gdpval/data/gdpval_benchmark.jsonl")
    assert benchmark.dataset.prepare_script == Path("benchmarks/gdpval/prepare.py")
    assert benchmark.dataset.prompt_config == Path("benchmarks/gdpval/prompts/harness-neutral.yaml")

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    _, baseline_configs = parser.load_extra_config_paths([str(recipe.parent / "config.yaml")])
    baseline = OmegaConf.merge(*baseline_configs)
    parser._recursively_swap_keys(baseline)
    resource_block = config.gdpval_resources_server.resources_servers.gdpval
    # Preserve every canonical resources/judge/provider setting except the harness pairing and dataset declaration.
    for name, block in baseline.items():
        if name not in {
            "config_paths",
            "gdpval_resources_server",
            "gdpval_stirrup_agent",
            "gdpval_environment_server",
        }:
            assert config[name] == block
    expected_resources = OmegaConf.to_container(baseline.gdpval_resources_server.resources_servers.gdpval)
    actual_resources = OmegaConf.to_container(resource_block)
    assert len(actual_resources.pop("datasets")) == 1
    actual_resources.pop("allowed_agents")
    expected_resources.pop("allowed_agents")
    assert actual_resources == expected_resources
    assert resource_block.allowed_agents == ["nooa_agent"]
    assert (
        config.rollout_collection_driver == "resources_servers.gdpval.multistage_orchestrator:run_rollout_collection"
    )
    environment_block = config.gdpval_environment_server.environment_servers.single_agent_turn_legacy
    expected_environment = OmegaConf.to_container(
        baseline.gdpval_environment_server.environment_servers.single_agent_turn_legacy
    )
    expected_environment["agent_server"]["name"] = benchmark.agent_name
    assert OmegaConf.to_container(environment_block) == expected_environment
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="gdpval_environment_server",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(environment_block, resolve=True),
    )
    agent = NOOAAgentConfig(
        name=benchmark.agent_name,
        host="localhost",
        port=8001,
        **OmegaConf.to_container(config.gdpval_nooa_agent.responses_api_agents.nooa_agent, resolve=True),
    )
    resources = GDPValResourcesServerConfig(
        name="gdpval_resources_server",
        host="localhost",
        port=8002,
        **OmegaConf.to_container(resource_block, resolve=True),
    )
    assert agent.nooa.execution_mode == "sandboxed"
    assert agent.resources_server == environment.resources_server
    assert agent.model_server.name == "policy_model"
    assert agent.max_policy_calls is None
    assert resources.entrypoint == "app.py"
    assert environment.resources_tool_transports == ["direct_http"]

    task = {
        "task_id": "fixture",
        "sector": "Finance",
        "occupation": "Auditor",
        "prompt": "Audit {this}.",
        "reference_files": ["reference_files/ledger.xlsx"],
        "reference_file_urls": ["https://example.invalid/private-download-url"],
        "rubric_json": [{"criterion": "private grading criterion", "score": 1}],
        "rubric_pretty": "private grading rubric",
    }
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: [task])
    monkeypatch.setattr(preparation, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(preparation, "OUTPUT_FPATH", tmp_path / "data/gdpval.jsonl")
    source = preparation.prepare()
    original = source.read_bytes()
    resource_block.datasets[0].jsonl_fpath = str(source)
    instances = parser.filter_for_server_instance_configs(config)
    assert [instance.name for instance in instances if instance.SERVER_TYPE == "environment_servers"] == [
        environment.name
    ]
    config.update(
        {"mode": "train_preparation", "output_dirpath": str(tmp_path / "collated"), "should_download": False}
    )
    TrainDataProcessor().run(config)
    collated = tmp_path / "collated/benchmark.jsonl"
    (row,) = RolloutCollectionHelper().preprocess_examples(
        [json.loads(line) for line in collated.read_text().splitlines()], global_config_dict=config
    )
    assert row["task_source"] == resources.name
    assert row["agent_ref"]["name"] == agent.name
    assert _environment_server_for_agent(agent.name, _environment_servers_by_agent(config)) == environment.name
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=environment, server_client=MagicMock(spec=ServerClient))
    request = adapter._episode_request_from_row(row)
    task_input = request.task.task_input
    prompt = _latest_user_text(task_input.responses_create_params)
    assert "Sector: Finance\nOccupation: Auditor\n\nAudit {this}." in prompt
    assert "- reference_files/ledger.xlsx" in prompt
    assert "code_exec" not in prompt
    for key in ("rubric_json", "rubric_pretty", "reference_file_urls"):
        assert task_input.task_data[key] == task[key]
        assert str(task[key]) not in prompt
    policy_input = task_input.responses_create_params.model_dump_json()
    assert "private grading" not in policy_input
    assert "private-download-url" not in policy_input
    tools = task_input.responses_create_params.model_dump(exclude_unset=True)["tools"]
    assert tools == json.loads((recipe.parent / "tools.json").read_text())
    assert {tool["name"] for tool in tools} == {"finish", "abandon_task_finish", "web_search", "fetch_web_page"}
    finish = next(tool for tool in tools if tool["name"] == "finish")
    assert set(finish["parameters"]["properties"]) == {"reason", "paths"}
    assert source.read_bytes() == original
