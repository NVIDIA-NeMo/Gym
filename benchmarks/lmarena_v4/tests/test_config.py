# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and scoring regressions for the v4 preset."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from benchmarks.lmarena_v4 import prepare as prepare_module
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from resources_servers.arena.app import ArenaResourcesServerConfig
from resources_servers.arena.metrics import ArenaMetrics


def resolve_config(version: str) -> DictConfig:
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [f"benchmarks/{version}/config.yaml"],
                    "policy_base_url": "https://example.com/v1",
                    "policy_api_key": "dummy",
                    "policy_model_name": "policy",
                }
            ),
        )
    )


@pytest.fixture
def metrics() -> ArenaMetrics:
    resolved = resolve_config("lmarena_v4")
    config = OmegaConf.to_container(
        resolved.lmarena_v4_benchmark_resources_server.resources_servers.arena, resolve=True
    )
    config.update(host="127.0.0.1", port=8080, name="arena")
    return ArenaMetrics(ArenaResourcesServerConfig.model_validate(config))


def test_config_and_manifest_use_v4_dataset() -> None:
    resolved = resolve_config("lmarena_v4")
    judge = resolved.judge_model.responses_api_models.vllm_model
    assert judge.model == "gcp/google/gemini-3.8-flash"
    assert judge.uses_reasoning_parser is True
    datasets = resolved.lmarena_v4_benchmark_agent.responses_api_agents.simple_agent.datasets
    manifest = OmegaConf.load(Path(__file__).resolve().parents[1] / "manifest.yaml")
    assert len(datasets) == len(manifest.datasets) == 1
    for field in ("name", "type", "jsonl_fpath", "prepare_script", "num_repeats"):
        assert str(datasets[0][field]) == str(manifest.datasets[0][field])
    assert datasets[0].name == "lmarena_v4"
    assert datasets[0].jsonl_fpath == "benchmarks/lmarena_v4/data/lmarena_v4_validation.jsonl"
    assert datasets[0].prepare_script == "benchmarks/lmarena_v4/prepare.py"
    assert not any(name.startswith("lmarena_v3") for name in resolved)
    assert "benchmarks/lmarena_v3/" not in OmegaConf.to_yaml(resolved)
    assert "configs/lmarena_v3.yaml" not in OmegaConf.to_yaml(resolved)
    assert "lmarena_v3_super35" not in OmegaConf.to_yaml(resolved)
    config_path = Path(__file__).resolve().parents[3] / "resources_servers/arena/configs/lmarena_v4.yaml"
    standalone = OmegaConf.load(config_path)
    registry = standalone.lmarena_v4_simple_agent.responses_api_agents.simple_agent.datasets[0].gitlab_identifier
    assert registry.dataset_name == "lmarena_v4"
    assert registry.artifact_fpath == "lmarena_v4_validation.jsonl"
    environment = resolved.lmarena_v4_benchmark_environment_server.environment_servers.single_agent_turn_legacy
    assert environment.agent_server.name == "lmarena_v4_benchmark_agent"
    assert environment.resources_server.name == "lmarena_v4_benchmark_resources_server"
    arena = resolved.lmarena_v4_benchmark_resources_server.resources_servers.arena
    assert list(arena.style_length_ratio_range) == [0.0, 1.75]
    assert arena.style_short_reference_max_tokens == 100
    assert arena.style_short_response_max_tokens == 175
    assert arena.judge_concurrency == 32
    assert "judge_verdict_format" not in arena
    assert "[[A>>B]]" in arena.judge_system_message
    assert "[[BB]]" in arena.judge_system_message
    assert arena.judge_responses_create_params.temperature == 0.0
    assert arena.judge_responses_create_params.max_output_tokens == 32768
    assert resolved.responses_create_params.max_output_tokens == 32768


def test_prepare_downloads_published_subset_to_v4_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "data" / "lmarena_v4_validation.jsonl"
    calls = []
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", output)
    monkeypatch.setattr(prepare_module, "download_jsonl_dataset", calls.append)
    assert prepare_module.prepare() == output
    assert output.parent.is_dir()
    assert len(calls) == 1
    assert calls[0].dataset_name == "lmarena_v4"
    assert calls[0].version == "0.0.1"
    assert calls[0].artifact_fpath == "lmarena_v4_validation.jsonl"
    assert calls[0].output_fpath == str(output)


@pytest.mark.parametrize(
    ("response_tokens", "reference_tokens", "accepted"),
    [
        (1, 200, True),
        (99, 200, True),
        (349, 200, True),
        (350, 200, False),
        (1, 100, True),
        (175, 100, True),
        (176, 100, False),
    ],
)
def test_only_rejects_long_answers(
    metrics: ArenaMetrics, response_tokens: int, reference_tokens: int, accepted: bool
) -> None:
    assert metrics._is_comparable_to_reference_length(response_tokens, reference_tokens) is accepted


def test_short_winning_answer_keeps_score(metrics: ArenaMetrics) -> None:
    rollout = {
        "category": "lmarena_v4",
        "policy_answer": "Yes.",
        "baseline_answer": "Baseline.",
        "style_reference_token_count": 200,
        "games": [{"verdict": "[[A>>B]]"}, {"verdict": "[[B>>A]]"}],
    }
    result = metrics.compute([[rollout]])
    assert result["win_rate"] == result["win_rate_no_SC"] == 1.0
    assert result["verbosity_acceptance_rate"] == 1.0
    rollout["policy_answer"] = "word " * 500
    result = metrics.compute([[rollout]])
    assert result["win_rate"] == 0.0
    assert result["win_rate_no_SC"] == 1.0
    assert result["verbosity_acceptance_rate"] == 0.0


def test_v3_defaults_unchanged() -> None:
    resolved = resolve_config("lmarena_v3")
    assert resolved.judge_model.responses_api_models.vllm_model.model == "gcp/google/gemini-3.1-pro-preview"
    assert list(resolved.lmarena_v3_benchmark_resources_server.resources_servers.arena.style_length_ratio_range) == [
        0.5,
        1.75,
    ]
    original = resolved.lmarena_v3_benchmark_resources_server.resources_servers.arena
    standalone = resolve_config("lmarena_v4").lmarena_v4_benchmark_resources_server.resources_servers.arena
    assert standalone.judge_prompt_template == original.judge_prompt_template
    assert standalone.judge_system_message == original.judge_system_message
