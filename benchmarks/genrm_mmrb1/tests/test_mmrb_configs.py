# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve both judge interfaces without credentials, network, or processes."""

import pytest
from omegaconf import OmegaConf

from nemo_gym.config_types import BenchmarkDatasetConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


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
    parser = resolved.genrm_eval.resources_servers.genrm_eval.get("verdict_format", "genrm")
    assert parser == ("genrm" if mode == "genrm" else "brackets" if benchmark == "genrm_mmrb1" else "json_ab")
