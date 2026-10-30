# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from omegaconf import DictConfig, OmegaConf

from benchmarks.terminal_bench_2_1 import prepare as preparation
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_paths
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "benchmarks/terminal_bench_2_1/nooa.yaml"


def _load_config(recipe: Path = RECIPE) -> DictConfig:
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs, {"policy_model_name": "test-model"})
    parser._recursively_swap_keys(config)
    return config


@pytest.fixture
def canonical_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The canonical preparer reads the instruction and image, not verifier assets or per-task budgets.
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
    return preparation.prepare()


@pytest.mark.parametrize("harness", ["nooa_agent", "pi_agent"])
def test_nooa_recipe_prepares_and_routes_canonical_terminal_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, canonical_source: Path, harness: str
) -> None:
    from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.train_data_utils import TrainDataProcessor
    from resources_servers.terminal_bench_2_1.app import TerminalBench21ResourcesServerConfig

    monkeypatch.setattr("nemo_gym.global_config._find_open_port_using_range", lambda **_: 12345)
    recipe = RECIPE
    # Catalog discovery reads literal dataset declarations; this run wrapper imports its dataset.
    assert recipe not in _benchmark_config_paths(recipe.parent)
    if harness == "pi_agent":
        profile = tmp_path / "pi.yaml"
        OmegaConf.save(
            {
                "config_paths": ["responses_api_agents/pi_agent/configs/pi_agent.yaml"],
                "agent": {"_inherit_from": "pi_agent"},
            },
            profile,
        )
        run_config = OmegaConf.load(RECIPE)
        run_config.config_paths[1] = str(profile)
        recipe = tmp_path / "run.yaml"
        OmegaConf.save(run_config, recipe)
    benchmark = BenchmarkConfig.from_config_path(recipe, strict=False)
    assert benchmark is not None
    assert benchmark.agent_name == "agent"
    assert benchmark.environment_server == "experiment"
    assert benchmark.dataset.taskset == "terminal-bench-2.1"
    assert benchmark.dataset.prepare_script == Path("benchmarks/terminal_bench_2_1/prepare.py")
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl")

    parser = GlobalConfigDictParser()
    config = _load_config(recipe)
    original_config = _load_config()
    assert config.benchmark == original_config.benchmark
    assert config.experiment == original_config.experiment
    assert config.environment_server_routes == original_config.environment_server_routes
    assert config.environment_routing_mode == "taskset"
    environment = SingleAgentTurnEnvironmentServerConfig(
        name=benchmark.environment_server,
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config.experiment.environment_servers.single_agent_turn, resolve=True),
    )
    assert list(config.agent.responses_api_agents) == [harness]
    agent = config.agent.responses_api_agents[harness]
    resource_block = config[environment.resources_server.name].resources_servers.terminal_bench_2_1
    resources = TerminalBench21ResourcesServerConfig(
        name=environment.resources_server.name,
        host="localhost",
        port=8002,
        **OmegaConf.to_container(resource_block, resolve=True),
    )
    assert agent.resources_server is None
    assert environment.agent_server.name == "agent"
    assert environment.resources_server.name == "benchmark"
    assert agent.model_server.name == "policy_model"
    assert agent.num_workers == resources.num_workers == 1
    assert resources.entrypoint == "app.py"
    assert resources.evaluation_timeout == 1800
    assert environment.default_episode_timeout_seconds == 30000
    assert environment.cleanup_timeout_seconds == 180
    assert not environment.resources_tool_transports
    canonical = OmegaConf.load(ROOT / "resources_servers/terminal_bench_2_1/configs/terminal_bench_2_1.yaml")
    sandbox = OmegaConf.to_container(
        canonical.terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.sandbox_config
    )
    assert resources.sandbox_config == sandbox | {"ttl_s": 30600}
    assert resources.sandbox_config["ttl_s"] >= (
        environment.default_episode_timeout_seconds + 3 * environment.cleanup_timeout_seconds
    )
    assert resource_block.allowed_agents == [
        *canonical.terminal_bench_2_1_resources_server.resources_servers.terminal_bench_2_1.allowed_agents,
        "nooa_agent",
    ]

    source = canonical_source
    original = source.read_bytes()
    resource_block.datasets[0].jsonl_fpath = str(source)
    instances = parser.filter_for_server_instance_configs(config)
    assert {(instance.name, instance.SERVER_TYPE) for instance in instances} == {
        ("benchmark", "resources_servers"),
        ("agent", "responses_api_agents"),
        ("experiment", "environment_servers"),
    }
    assert len(resource_block.datasets) == 1
    assert not agent.get("datasets")
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
            environment_routing_mode=config.environment_routing_mode,
        )
    )
    assert row["_ng_environment_server"] == environment.name
    assert row["task_id"] == {"taskset": "terminal-bench-2.1", "task_id": "fixture"}
    RolloutCollectionHelper._validate_environment_servers([row], config)
    task = SingleAgentTurnTaskInput.model_validate(row["task_input"])
    assert task.task_data == {
        "task_id": "fixture",
        "task_name": "terminal-bench/fixture",
        "docker_image": "fixture:1",
        "task_folder": "benchmarks/terminal_bench_2_1/terminal-bench-2-1/tasks/fixture",
    }
    assert json.loads(original)["responses_create_params"]["input"] == [
        {"role": "user", "content": "Solve the café task.\n"}
    ]
    assert "private verifier answer" not in task.responses_create_params.model_dump_json()
    assert json.loads(original)["responses_create_params"] == task.responses_create_params.model_dump(
        exclude_unset=True
    )
    assert source.read_bytes() == original


def test_nooa_sandboxed_profile_preserves_runtime_limits() -> None:
    pytest.importorskip("nooa")
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig

    config = _load_config()
    agent = NOOAAgentConfig(
        name="agent",
        host="localhost",
        port=8001,
        **OmegaConf.to_container(config.agent.responses_api_agents.nooa_agent, resolve=True),
    )
    assert agent.resources_server is None
    assert agent.nooa.execution_mode == "sandboxed"
    assert agent.sandbox_runner_timeout_seconds == 21600
    assert agent.session_close_timeout_seconds == 30
    assert agent.max_policy_calls is None


@pytest.mark.parametrize("command", ["prepare", "resolve", "run"])
def test_documented_cli_uses_imported_dataset_and_private_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_source: Path,
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import nemo_gym.cli.eval as cli_eval
    import nemo_gym.global_config as global_config
    from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
    from nemo_gym.cli.main import main
    from nemo_gym.rollout_collection import RolloutCollectionHelper

    monkeypatch.chdir(tmp_path)
    (tmp_path / "env.yaml").write_text("")
    monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
    monkeypatch.delenv(global_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, raising=False)
    monkeypatch.delenv(global_config.NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setattr(global_config, "_find_open_port_using_range", lambda **_: 12345)

    private = tmp_path / "private-policy-provider.yaml"
    dataset = OmegaConf.to_container(_load_config().benchmark.resources_servers.terminal_bench_2_1.datasets[0])
    dataset["jsonl_fpath"] = str(canonical_source)
    OmegaConf.save(
        {
            "policy_model_name": "test-model",
            "policy_base_url": "http://127.0.0.1:9999/v1",
            "policy_api_key": "test-key",
            "policy_model": {"responses_api_models": {"vllm_model": {"sampling_overrides": {"temperature": 0.37}}}},
            "benchmark": {
                "resources_servers": {
                    "terminal_bench_2_1": {
                        "datasets": [dataset],
                        "sandbox_config": {"resources": {"cpu": 8}},
                    }
                }
            },
        },
        private,
    )
    assert "split" not in OmegaConf.load(private)
    argv = ["gym", "env" if command == "resolve" else "eval", command, "--config", str(RECIPE)]
    if command != "prepare":
        argv.extend(
            [
                "--config",
                str(ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
                "--config",
                str(private),
            ]
        )
    output = tmp_path / "results.jsonl"
    collected = AsyncMock()
    prepared = []
    if command == "prepare":
        monkeypatch.setattr(cli_eval, "_prepare_datasets", lambda datasets, _: prepared.extend(datasets))
    elif command == "run":
        argv.extend(
            [
                "--split",
                "benchmark",
                "--limit",
                "1",
                "--concurrency",
                "1",
                "--num-repeats",
                "1",
                "--output",
                str(output),
            ]
        )
        monkeypatch.setattr(cli_eval.RunHelper, "start", lambda *_: None)
        monkeypatch.setattr(cli_eval.RunHelper, "shutdown", lambda *_: None)
        monkeypatch.setattr(cli_eval, "_interrupt_on_sigterm", lambda: None)
        monkeypatch.setattr(cli_eval, "_health_check_collected_rollouts", lambda *_: None)
        monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", collected)
    monkeypatch.setattr(sys, "argv", argv)
    main()

    if command == "prepare":
        (benchmark,) = prepared
        assert benchmark.agent_name == "agent"
        assert benchmark.environment_server == "experiment"
        assert benchmark.dataset.taskset == "terminal-bench-2.1"
        assert benchmark.dataset.jsonl_fpath == Path("benchmarks/terminal_bench_2_1/data/benchmark.jsonl")
    else:
        config = global_config.get_global_config_dict()
        assert config.policy_model.responses_api_models.vllm_model.sampling_overrides.temperature == 0.37
        assert config.benchmark.resources_servers.terminal_bench_2_1.sandbox_config.resources.cpu == 8
        if command == "resolve":
            assert "terminal-bench-2.1: experiment" in capsys.readouterr().out
        else:
            collected.assert_awaited_once()
            collection_config = collected.call_args.args[0]
            assert (
                Path(collection_config.input_jsonl_fpath)
                == output.with_suffix("") / "preprocessed_datasets/benchmark.jsonl"
            )
            (row,) = RolloutCollectionHelper()._preprocess_rows_from_config(collection_config)
            assert row["task_id"] == {"taskset": "terminal-bench-2.1", "task_id": "fixture"}
            assert row["_ng_environment_server"] == "experiment"
