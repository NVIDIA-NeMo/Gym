# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_2_1 import prepare as preparation
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_name, _benchmark_config_paths
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


ROOT = Path(__file__).resolve().parents[3]


def test_nooa_recipe_prepares_and_routes_canonical_terminal_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("nooa")
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.train_data_utils import TrainDataProcessor
    from resources_servers.terminal_bench_2_1.app import TerminalBench21ResourcesServerConfig
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig
    from responses_api_agents.nooa_agent.task_agent import _latest_user_text

    monkeypatch.setattr("nemo_gym.global_config._find_open_port_using_range", lambda **_: 12345)
    recipe = ROOT / "benchmarks/terminal_bench_2_1/nooa.yaml"
    assert recipe in _benchmark_config_paths(recipe.parent)
    assert _benchmark_config_name(recipe.relative_to(ROOT / "benchmarks")) == "terminal_bench_2_1/nooa"
    benchmark = BenchmarkConfig.from_config_path(recipe, strict=False)
    assert benchmark is not None
    assert benchmark.agent_name == "terminal_bench_2_1_nooa_agent"
    assert benchmark.environment_server == "terminal_bench_2_1_nooa"
    assert benchmark.dataset.taskset == "terminal-bench-2.1-nooa"
    assert benchmark.dataset.prepare_script == Path("benchmarks/terminal_bench_2_1/prepare.py")
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl")

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    environment = SingleAgentTurnEnvironmentServerConfig(
        name=benchmark.environment_server,
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config.terminal_bench_2_1_nooa.environment_servers.single_agent_turn, resolve=True),
    )
    agent = NOOAAgentConfig(
        name=benchmark.agent_name,
        host="localhost",
        port=8001,
        **OmegaConf.to_container(config[benchmark.agent_name].responses_api_agents.nooa_agent, resolve=True),
    )
    resource_block = config[environment.resources_server.name].resources_servers.terminal_bench_2_1
    resources = TerminalBench21ResourcesServerConfig(
        name=environment.resources_server.name,
        host="localhost",
        port=8002,
        **OmegaConf.to_container(resource_block, resolve=True),
    )
    assert agent.nooa.execution_mode == "sandboxed"
    assert agent.sandbox_runner_timeout_seconds == 21600
    assert agent.resources_server == environment.resources_server
    assert agent.model_server.name == "policy_model"
    assert agent.num_workers == resources.num_workers == 1
    assert agent.max_policy_calls is None
    assert resources.entrypoint == "app.py"
    assert resources.evaluation_timeout == 1800
    assert environment.default_episode_timeout_seconds == 30000
    assert not environment.resources_tool_transports
    canonical = OmegaConf.load(ROOT / "resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml")
    sandbox = OmegaConf.to_container(
        canonical.terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.sandbox_config
    )
    assert resources.sandbox_config == sandbox | {"ttl_s": 30600}
    assert resources.sandbox_config["ttl_s"] >= (
        environment.default_episode_timeout_seconds + 3 * environment.cleanup_timeout_seconds
    )
    assert resource_block.allowed_agents == ["nooa_agent"]

    # The canonical preparer reads the task instruction and image, not verifier assets or per-task budgets.
    benchmark_dir = tmp_path / "benchmarks/terminal_bench_2_1"
    task_dir = benchmark_dir / "terminal-bench-2-1/tasks/fixture"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Solve the café task.\n")
    (task_dir / "task.toml").write_text(
        '[task]\nname = "terminal-bench/fixture"\n[environment]\ndocker_image = "fixture:1"\n'
        "[agent]\ntimeout_sec = 12\n[verifier]\ntimeout_sec = 34\n"
    )
    (task_dir / "tests").mkdir()
    (task_dir / "tests/test.sh").write_text("private verifier answer")
    monkeypatch.setattr(preparation, "BENCHMARK_DIR", benchmark_dir)
    monkeypatch.setattr(preparation, "OUTPUT_PATH", benchmark_dir / "data/benchmark.jsonl")
    monkeypatch.setattr(preparation, "NUM_TASKS", 1)
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
    (row,) = RolloutCollectionHelper()._preprocess_rows_from_config(
        RolloutCollectionConfig(
            input_jsonl_fpath=str(collated),
            output_jsonl_fpath=str(tmp_path / "unused.jsonl"),
            environment_server_routes=OmegaConf.to_container(config.environment_server_routes),
        )
    )
    assert row["_ng_environment_server"] == environment.name
    assert row["task_id"] == {"taskset": "terminal-bench-2.1-nooa", "task_id": "fixture"}
    task = SingleAgentTurnTaskInput.model_validate(row["task_input"])
    assert task.task_data == {
        "task_id": "fixture",
        "task_name": "terminal-bench/fixture",
        "docker_image": "fixture:1",
        "task_folder": "benchmarks/terminal_bench_2_1/terminal-bench-2-1/tasks/fixture",
    }
    assert _latest_user_text(task.responses_create_params) == "Solve the café task.\n"
    assert "private verifier answer" not in task.responses_create_params.model_dump_json()
    assert json.loads(original)["responses_create_params"] == task.responses_create_params.model_dump(
        exclude_unset=True
    )
    assert source.read_bytes() == original
