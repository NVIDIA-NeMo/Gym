# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config.config_types import ConfigMissingValuesError, ServerRefNotFoundError
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment.validation import _resolve_config_wiring, _StaticWiringParser, validate_environment


ROOT = Path(__file__).resolve().parents[3]
CATALOG = "benchmarks/opencode_multimodal_benchmarks"


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
@pytest.mark.parametrize("mode", ["default", "openai", "vllm", "default_eval", "training"])
def test_catalog_pairings_validate_with_selected_model(benchmark, mode, monkeypatch):
    monkeypatch.setenv("GYM_JUDGE_API_KEY", "test-only")
    monkeypatch.setenv("GYM_JUDGE_BASE_URL", "https://judge.invalid/v1")
    monkeypatch.setenv("POLICY_API_KEY", "test-only")
    monkeypatch.setenv("POLICY_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.delenv("NEMO_GYM_REASONING_FIELD", raising=False)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-only")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://sandbox.invalid")
    adapter = "openai_model" if mode == "openai" else "vllm_model"
    model_config = "vllm_model_for_training" if mode == "training" else adapter
    paths = [f"responses_api_models/{adapter}/configs/{model_config}.yaml", f"{CATALOG}/{benchmark}/config.yaml"]
    overrides = {
        "visual_agent": {"responses_api_agents": {"opencode_agent": {"sandbox_config": {"image": "test-image"}}}}
    }
    if mode in {"openai", "vllm"}:
        paths[-1] = f"{CATALOG}/{benchmark}/opencode.yaml"
    if mode == "vllm":
        paths.append(f"{CATALOG}/opencode_eval_vllm.yaml")
    if mode == "default_eval":
        paths[0] = f"{CATALOG}/default_eval.yaml"
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(ROOT / p) for p in paths],
                    "policy_model_name": "test-model",
                    "policy_base_url": "http://127.0.0.1:9999/v1",
                    "policy_api_key": "test-only",
                    **overrides,
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            skip_exporters=True,
            offline=True,
        )
    )
    assert "policy_model" in {s.name for s in parser.filter_for_server_instance_configs(config)}
    assert set(config.policy_model.responses_api_models) == {adapter}
    env = SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(config.visual_environment.environment_servers.single_agent_turn, resolve=True)
        | {"name": "visual_environment"}
    )
    assert env.cleanup_timeout_seconds == 180
    assert env.default_episode_timeout_seconds is None
    assert env.resources_tool_transports == []
    assert env.agent_server.name == "visual_agent"
    assert env.resources_server.name == "opencode_multimodal_benchmarks"
    agent = config.visual_agent.responses_api_agents
    assert set(agent) == {"opencode_agent"}
    assert agent.opencode_agent.model_server.name == "policy_model"
    assert agent.opencode_agent.resources_server.name == "opencode_multimodal_benchmarks"
    assert agent.opencode_agent.network_access == "model_only"
    assert agent.opencode_agent.concurrency == config.num_samples_in_parallel == 64
    assert agent.opencode_agent.max_output_tokens == 32768
    assert agent.opencode_agent.context_window == 131072
    assert config.sandbox.opensandbox.operations.background_poll_interval_s == 2
    assert agent.opencode_agent.image_input
    assert not agent.opencode_agent.token_id_capture
    assert agent.opencode_agent.opencode_config.permission.webfetch == "deny"
    assert agent.opencode_agent.opencode_config.permission.websearch == "deny"
    manifest = OmegaConf.load(ROOT / CATALOG / benchmark / "manifest.yaml")
    assert manifest.agent_server == "opencode_agent"
    resources = config.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks
    assert resources.datasets[0].num_repeats == (5 if benchmark == "zerobench" else 1)
    assert resources.judge_concurrency == 16
    assert resources.judge_required is (benchmark != "zerobench")
    if benchmark != "zerobench":
        assert resources.judge_model == "azure/openai/gpt-4o-mini"
        assert resources.judge_server == "visual_judge"
        assert config.visual_judge.responses_api_models.openai_model.openai_model == resources.judge_model
    else:
        assert resources.judge_model is None
        assert "visual_judge" not in config
    if mode in {"vllm", "default_eval"}:
        model = config.policy_model.responses_api_models.vllm_model
        assert not agent.opencode_agent.token_id_capture
        assert not model.return_token_id_information
        assert model.uses_reasoning_parser and model.uses_interleaved_reasoning
        assert model.chat_template_kwargs.truncate_history_thinking is False
        assert model.sampling_overrides.max_tokens is None
        assert model.sampling_overrides.max_completion_tokens is None
        assert model.extra_body is None and model.propagate_context_overflow_errors
    if mode == "default_eval":
        model = config.policy_model.responses_api_models.vllm_model
        assert "reasoning_field" not in model  # Preserve the model server's default.
        assert model.model == "test-model"
        assert model.sampling_overrides.temperature == 0.6
        assert model.sampling_overrides.top_p == 0.95
    if mode == "training":
        assert config.policy_model.responses_api_models.vllm_model.return_token_id_information


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
def test_manifest_validation_needs_no_deployment(benchmark, monkeypatch):
    for key in ("OPENSANDBOX_API_KEY", "OPENSANDBOX_DOMAIN", "GYM_JUDGE_API_KEY", "GYM_JUDGE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    report = validate_environment(ROOT / CATALOG / benchmark / "manifest.yaml", validate_datasets=False)
    implementations = {component.implementation for component in report.components}
    assert {"opencode_agent", "opencode_multimodal_benchmarks"} <= implementations
    assert report.datasets == ()


@pytest.mark.parametrize(
    "override, error, message",
    [
        ({"settings": {"unrelated_required_field": "???"}}, ConfigMissingValuesError, "unrelated_required_field"),
        (
            {
                "visual_environment": {
                    "environment_servers": {
                        "single_agent_turn": {
                            "resources_server": {"type": "resources_servers", "name": "missing_resources"}
                        }
                    }
                }
            },
            ServerRefNotFoundError,
            "missing_resources",
        ),
    ],
)
def test_static_validation_still_rejects_invalid_configuration(tmp_path, override, error, message):
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"config_paths": [str(ROOT / CATALOG / "vstar/config.yaml")], **override}), path)
    original = path.read_bytes()
    with pytest.raises(error, match=message):
        _resolve_config_wiring(path)
    assert path.read_bytes() == original


def _catalog_config(benchmark, **overrides):
    return {
        "config_paths": [
            str(ROOT / "responses_api_models/vllm_model/configs/vllm_model.yaml"),
            str(ROOT / CATALOG / benchmark / "config.yaml"),
        ],
        "policy_model_name": "test-model",
        "policy_base_url": "https://policy.invalid/v1",
        "policy_api_key": "test-only",
        **overrides,
    }


def _parse_catalog(config):
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(config),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            skip_exporters=True,
            offline=True,
        )
    )


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
def test_runnable_catalog_still_requires_sandbox_image(benchmark):
    with pytest.raises(ConfigMissingValuesError, match="sandbox_config.image"):
        _parse_catalog(_catalog_config(benchmark))


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
def test_catalog_agent_selector_preserves_verifier_and_routes(benchmark, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-only")
    monkeypatch.setenv("GYM_JUDGE_API_KEY", "test-only")
    monkeypatch.setenv("GYM_JUDGE_BASE_URL", "https://judge.invalid/v1")
    initial = _catalog_config(benchmark)
    initial["config_paths"].append(str(ROOT / "responses_api_agents/simple_agent/configs/simple_agent.yaml"))
    config = _parse_catalog(initial)
    agents = [
        server
        for server in GlobalConfigDictParser().filter_for_server_instance_configs(config)
        if server.SERVER_TYPE == "responses_api_agents"
    ]
    assert len(agents) == 1
    agent = agents[0]
    assert set(config[agent.name].responses_api_agents) == {"simple_agent"}
    assert (
        config[agent.name].responses_api_agents.simple_agent.resources_server.name == "opencode_multimodal_benchmarks"
    )
    environment_name = config.environment_server_routes.visual_tasks
    environment = next(iter(config[environment_name].environment_servers.values()))
    assert environment.agent_server.name == agent.name
    resources = config.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks
    assert resources.datasets[0].num_repeats == (5 if benchmark == "zerobench" else 1)
    assert ("visual_judge" in config) is (benchmark != "zerobench")


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench"])
def test_direct_http_judge_does_not_need_unused_server_credentials(benchmark, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-only")
    monkeypatch.delenv("GYM_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("GYM_JUDGE_BASE_URL", raising=False)
    config = _parse_catalog(
        _catalog_config(
            benchmark,
            visual_agent={"responses_api_agents": {"opencode_agent": {"sandbox_config": {"image": "test-image"}}}},
            visual_judge=None,
            opencode_multimodal_benchmarks={
                "resources_servers": {
                    "opencode_multimodal_benchmarks": {
                        "judge_server": None,
                        "judge_endpoint": "https://judge.invalid/v1/chat/completions",
                        "judge_api_key_env": "DIRECT_JUDGE_API_KEY",
                    }
                }
            },
        )
    )
    assert "visual_judge" not in {
        server.name for server in GlobalConfigDictParser().filter_for_server_instance_configs(config)
    }
    resources = config.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks
    assert resources.judge_server is None
    assert resources.judge_endpoint == "https://judge.invalid/v1/chat/completions"
    assert resources.judge_api_key_env == "DIRECT_JUDGE_API_KEY"


def test_static_placeholders_do_not_replace_broken_config_references():
    config = OmegaConf.create(
        {
            "visual_agent": {
                "responses_api_agents": {
                    "opencode_agent": {
                        "sandbox_config": {"image": "${missing_image}"},
                    }
                }
            }
        }
    )
    _StaticWiringParser().raise_on_missing_values(config)
    with pytest.raises(InterpolationKeyError, match="missing_image"):
        OmegaConf.to_container(config, resolve=True)
