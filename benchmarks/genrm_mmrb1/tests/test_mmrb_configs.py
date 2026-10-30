# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve both judge interfaces without credentials, network, or processes."""

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import BenchmarkDatasetConfig, GlobalConfigDictParser, GlobalConfigDictParserConfig


@pytest.mark.parametrize("benchmark", ["genrm_mmrb1", "genrm_mmrb2"])
@pytest.mark.parametrize(
    ("mode", "config", "model_type"),
    [
        ("genrm", "genrm.yaml", "genrm_model"),
        ("llm_as_a_judge", "llm_as_a_judge.yaml", "vllm_model"),
    ],
)
def test_interfaces(benchmark, mode, config, model_type):
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [f"benchmarks/{benchmark}/{config}"],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "judge",
                }
            ),
        )
    )
    assert list(resolved.policy_model.responses_api_models) == [model_type]
    assert resolved.judge_mode == mode
    assert resolved.prepare_script_args.judge_mode == mode
    assert resolved.responses_create_params.max_output_tokens == 16384
    assert resolved.responses_create_params.top_p == 0.95
    dataset = resolved[f"{benchmark}_benchmark_agent"].responses_api_agents.simple_agent.datasets[0]
    assert str(dataset.jsonl_fpath).endswith(f"_{mode}_validation.jsonl")
    # Dataset paths must survive conversion to Path without unresolved interpolation.
    parsed = BenchmarkDatasetConfig.model_validate(OmegaConf.to_container(dataset, resolve=False))
    assert "${" not in str(parsed.jsonl_fpath)
    assert str(parsed.jsonl_fpath).endswith(f"_{mode}_validation.jsonl")
    agent = resolved[f"{benchmark}_benchmark_agent"].responses_api_agents.simple_agent
    resources = resolved[agent.resources_server.name].resources_servers.genrm_eval
    parser = resources.get("verdict_format", "genrm")
    assert parser == ("genrm" if mode == "genrm" else "brackets" if benchmark == "genrm_mmrb1" else "json_ab")
    if benchmark == "genrm_mmrb2":
        assert "Pillow" in parsed.prepare_dependencies


@pytest.mark.parametrize("benchmarks", [("genrm_mmrb1", "genrm_mmrb2"), ("genrm_mmrb2", "genrm_mmrb1")])
def test_composed_llm_configs_keep_preparers_independent(benchmarks):
    import inspect
    from importlib import import_module

    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [f"benchmarks/{benchmark}/llm_as_a_judge.yaml" for benchmark in benchmarks],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "judge",
                }
            ),
        )
    )
    args = OmegaConf.to_container(resolved.prepare_script_args, resolve=True)
    assert args == {"judge_mode": "llm_as_a_judge"}
    for benchmark, expected in [("genrm_mmrb1", "brackets"), ("genrm_mmrb2", "json_ab")]:
        agent = resolved[f"{benchmark}_benchmark_agent"].responses_api_agents.simple_agent
        assert agent.resources_server.name == f"{benchmark}_resources_server"
        resources = resolved[agent.resources_server.name].resources_servers.genrm_eval
        assert resources.verdict_format == expected
        inspect.signature(import_module(f"benchmarks.{benchmark}.prepare").prepare).bind(**args)


def test_runnable_modes_are_discoverable(monkeypatch):
    from itertools import count
    from pathlib import Path

    from nemo_gym.environment.benchmarks import BenchmarkConfig, _benchmark_config_name, _benchmark_config_paths

    ports = count(12000)
    monkeypatch.setattr("nemo_gym.config.global_config._find_open_port_using_range", lambda **kwargs: next(ports))
    root = Path(__file__).resolve().parents[3] / "benchmarks"
    for benchmark in ("genrm_mmrb1", "genrm_mmrb2"):
        configs = list(_benchmark_config_paths(root / benchmark))
        assert {path.name for path in configs} == {"genrm.yaml", "llm_as_a_judge.yaml"}
        for path in configs:
            token = _benchmark_config_name(path.relative_to(root))
            assert token == f"{benchmark}/{path.stem}"
            assert BenchmarkConfig.from_config_path(root / f"{token}.yaml", strict=False) is not None
