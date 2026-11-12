# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check preparation entrypoints and interchangeable harness configuration."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from benchmarks.legal_agent_bench import prepare as benchmark_prepare
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment import BenchmarkConfig
from resources_servers.legal_agent_bench.app import LegalAgentBenchEnvironmentConfig, _build_verifier_env
from resources_servers.legal_agent_bench.prepare import INDEX_FILENAME


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PRESETS = ["native", "hermes", "claude_code", "codex"]


def resolve(harness="native", **overrides):
    initial = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / f"config_{harness}.yaml"),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"default_host": "127.0.0.1"},
        overrides,
    )
    return initial, GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)


def environment_config(resolved):
    values = OmegaConf.to_container(
        resolved.legal_agent_bench_environment_server.environment_servers.harbor, resolve=True
    )
    return LegalAgentBenchEnvironmentConfig(name="lab", **values)


def test_preparation_forwards_selection_cache_paths_and_preserves_input(monkeypatch, tmp_path):
    selection = tmp_path / "selection.jsonl"
    row = {
        "instance_id": "legal_agent_bench::practice__task",
        "responses_create_params": {"input": [], "temperature": 0.2},
    }
    selection.write_text(json.dumps(row) + "\n")
    original = selection.read_bytes()
    tasks, skills, output = tmp_path / "tasks", tmp_path / "skills", tmp_path / "output.jsonl"
    monkeypatch.setenv("LEGAL_AGENT_BENCH_TASK_CACHE_DIR", str(tasks))
    monkeypatch.setenv("LEGAL_AGENT_BENCH_SKILLS_DIR", str(skills))
    assets = Mock(return_value={"tasks": tasks, "skills": skills})
    materialize = Mock(return_value=output)
    monkeypatch.setattr(benchmark_prepare, "prepare_assets", assets)
    monkeypatch.setattr(benchmark_prepare, "prepare_environment", materialize)
    monkeypatch.setattr(
        benchmark_prepare,
        "get_global_config_dict",
        lambda: {"lab": {"agent_timeout_seconds": 300, "verifier_timeout_seconds": 600}},
    )
    assert benchmark_prepare.prepare(input=selection, output=output, image="prepared:image") == output
    assets.assert_called_once_with(
        "all", force=False, tasks_dir=str(tasks), skills_dir=str(skills), task_ids=("practice__task",)
    )
    materialize.assert_called_once_with(
        tasks_dir=tasks,
        skills_dir=skills,
        rows=[row],
        output=output,
        image="prepared:image",
        agent_timeout_seconds=300,
        verifier_timeout_seconds=600,
    )
    assert selection.read_bytes() == original


@pytest.mark.parametrize(
    "content,message",
    [
        ("not JSON\n", "Invalid LAB task index JSON on line 1"),
        ("[]\n", "expected an object"),
        ("{}\n", "Expected 2 LAB benchmark rows"),
    ],
)
def test_invalid_index_does_not_publish(monkeypatch, tmp_path, content, message):
    (tmp_path / INDEX_FILENAME).write_text(content)
    output = tmp_path / "output.jsonl"
    output.write_text("previous")
    monkeypatch.setattr(
        benchmark_prepare, "prepare_assets", Mock(return_value={"tasks": tmp_path, "skills": tmp_path})
    )
    monkeypatch.setattr(benchmark_prepare, "EXPECTED_TASK_COUNT", 2)
    with pytest.raises(ValueError, match=message):
        benchmark_prepare.prepare(output=output, image="prepared:image")
    assert output.read_text() == "previous"


def test_preparation_failure_preserves_previous_output(monkeypatch, tmp_path):
    output = tmp_path / "output.jsonl"
    output.write_text("previous")
    monkeypatch.setattr(benchmark_prepare, "prepare_assets", Mock(side_effect=RuntimeError("asset failure")))
    with pytest.raises(RuntimeError, match="asset failure"):
        benchmark_prepare.prepare(output=output)
    assert output.read_text() == "previous"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("reward_mode", ["full_task", "criteria_pass_rate"])
def test_harness_switch_preserves_grading_and_dataset(count, reward_mode, tmp_path):
    judges = [
        {"model_name": "route/claude-sonnet-4-6", "base_url": "https://claude.example/v1", "api_key": "test-c"},
        {"model_name": "route/gpt-5.5", "reasoning_effort": "high"},
    ][:count]
    expected = None
    for harness in PRESETS:
        initial, resolved = resolve(
            harness,
            lab_judges=judges,
            judge_base_url="https://shared.example/v1",
            judge_api_key="test-shared",
            prepare_script_args={"output": str(tmp_path / "selected.jsonl")},
            lab={
                "reward_mode": reward_mode,
                "judge_parallelism": 2,
                "agent_timeout_seconds": 300,
                "verifier_timeout_seconds": 600,
            },
            num_samples_in_parallel=2,
        )
        config = environment_config(resolved)
        verifier_env = _build_verifier_env(config)
        assert json.loads(verifier_env["LAB_JUDGES"]) == judges
        assert verifier_env["LAB_JUDGE_BASE_URL"] == "https://shared.example/v1"
        assert verifier_env["LAB_JUDGE_API_KEY"] == "test-shared"
        assert verifier_env["LAB_JUDGE_PARALLELISM"] == "2"
        assert config.reward_mode == reward_mode
        assert config.max_concurrent_episodes == 2
        assert config.prepared_tasksets_file == str(tmp_path / "selected.jsonl.tasksets.json")
        assert config.separate_verifier_for_artifacts
        assert not set(resolved.get("responses_create_params", {})) & {"temperature", "top_p"}
        benchmark = BenchmarkConfig.from_initial_config_dict(
            path=BENCHMARK_DIR / f"config_{harness}.yaml", initial_config_dict=initial, strict=False
        )
        assert benchmark.agent_name == "legal_agent_bench_benchmark_agent"
        assert benchmark.dataset.jsonl_fpath == tmp_path / "selected.jsonl"
        assert benchmark.dataset.prepare_script == Path("benchmarks/legal_agent_bench/prepare.py")
        current = verifier_env, benchmark.dataset
        if expected is None:
            expected = current
        assert current == expected
        agents = resolved.legal_agent_bench_benchmark_agent.responses_api_agents
        expected_type = "legal_agent_bench_native" if harness == "native" else f"{harness}_agent"
        assert list(agents) == [expected_type]
        agent = agents[expected_type]
        if harness == "native":
            assert agent.agent_timeout_seconds == 300
        else:
            assert agent.concurrency == 2
            assert (agent.sandbox_runner_timeout_seconds if harness == "hermes" else agent.timeout) == 300
        assert config.agent_server.name == benchmark.agent_name


@pytest.mark.parametrize("harness", PRESETS)
def test_every_harness_requires_explicit_judges(harness):
    _, resolved = resolve(harness)
    with pytest.raises(ValueError, match="Configure lab_judges"):
        environment_config(resolved)


def test_default_alias_selects_lab_native():
    def wiring(config):
        result = []
        for name in ("legal_agent_bench_benchmark_agent", "legal_agent_bench_environment_server"):
            entry = OmegaConf.to_container(config[name], resolve=True)
            settings = next(iter(next(iter(entry.values())).values()))
            settings.pop("port", None)  # Config parsing allocates a fresh available port.
            result.append(entry)
        return result

    _, native = resolve("native")
    initial = OmegaConf.merge(
        OmegaConf.load(BENCHMARK_DIR / "config.yaml"),
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"default_host": "127.0.0.1"},
    )
    default = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    assert wiring(default) == wiring(native)
