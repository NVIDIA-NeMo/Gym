# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError, InterpolationResolutionError

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config.config_types import ConfigMissingValuesError, InheritPathNotFoundError, ServerRefNotFoundError
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment.validation import _resolve_config_wiring, validate_environment


ROOT = Path(__file__).resolve().parents[3]
CATALOG = "benchmarks/opencode_multimodal_benchmarks"


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench", "agentx"])
@pytest.mark.parametrize("mode", ["default", "openai", "vllm", "training"])
def test_catalog_pairings_validate_with_selected_model(benchmark, mode, monkeypatch):
    monkeypatch.setenv("GYM_JUDGE_API_KEY", "test-only")
    monkeypatch.setenv("GYM_JUDGE_BASE_URL", "https://judge.invalid/v1")
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
    assert next(iter(agent.values())).model_server.name == "policy_model"
    resources = config.opencode_multimodal_benchmarks.resources_servers.opencode_multimodal_benchmarks
    assert resources.datasets[0].num_repeats == (5 if benchmark == "zerobench" else 1)
    if mode == "vllm":
        model = config.policy_model.responses_api_models.vllm_model
        assert not agent.opencode_agent.token_id_capture
        assert not model.return_token_id_information
        assert model.uses_reasoning_parser and model.uses_interleaved_reasoning
        assert model.chat_template_kwargs.truncate_history_thinking is False
        assert model.sampling_overrides.max_tokens is None
        assert model.sampling_overrides.max_completion_tokens is None
        assert model.extra_body is None and model.propagate_context_overflow_errors
    if mode == "training":
        assert config.policy_model.responses_api_models.vllm_model.return_token_id_information


@pytest.fixture
def no_deployment(monkeypatch):
    for key in ("OPENSANDBOX_API_KEY", "OPENSANDBOX_DOMAIN", "GYM_JUDGE_API_KEY", "GYM_JUDGE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
def test_manifest_validation_needs_no_deployment(benchmark, no_deployment):
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
def test_static_validation_still_rejects_invalid_wiring(tmp_path, no_deployment, override, error, message):
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"config_paths": [str(ROOT / CATALOG / "vstar/config.yaml")], **override}), path)
    original = path.read_bytes()
    with pytest.raises(error, match=message):
        _resolve_config_wiring(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "image, error, message",
    [
        ("${missing_image}", InterpolationKeyError, "missing_image"),
        ("${copy:missing_image}", InheritPathNotFoundError, "missing_image"),
        ("${inherit_from:missing_image}", InheritPathNotFoundError, "missing_image"),
    ],
)
def test_static_validation_preserves_broken_deployment_references(tmp_path, no_deployment, image, error, message):
    path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "config_paths": [str(ROOT / CATALOG / "vstar/config.yaml")],
                "visual_agent": {"responses_api_agents": {"opencode_agent": {"sandbox_config": {"image": image}}}},
            }
        ),
        path,
    )
    with pytest.raises(error, match=message):
        wiring = _resolve_config_wiring(path)
        OmegaConf.to_container(wiring.resolved, resolve=True)


@pytest.mark.parametrize("directive", ["copy", "inherit_from"])
def test_static_validation_keeps_unset_inherited_image(tmp_path, no_deployment, directive):
    path = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "config_paths": [str(ROOT / CATALOG / "vstar/config.yaml")],
                "settings": {"unset_image": "???"},
                "visual_agent": {
                    "responses_api_agents": {
                        "opencode_agent": {"sandbox_config": {"image": "${" + directive + ":settings.unset_image}"}}
                    }
                },
            }
        ),
        path,
    )
    with pytest.raises(ConfigMissingValuesError, match="sandbox_config.image"):
        _resolve_config_wiring(path)


@pytest.mark.parametrize("benchmark", ["vstar", "hrbench8k", "vtcbench", "zerobench"])
@pytest.mark.parametrize("image", ["???", "test-image"])
def test_runtime_parser_still_requires_deployment(benchmark, image, no_deployment):
    config = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [str(ROOT / CATALOG / benchmark / "config.yaml")],
            "visual_agent": {"responses_api_agents": {"opencode_agent": {"sandbox_config": {"image": image}}}},
        },
    )
    error = ConfigMissingValuesError if image == "???" else InterpolationResolutionError
    message = "sandbox_config.image" if image == "???" else "OPENSANDBOX|GYM_JUDGE"
    with pytest.raises(error, match=message):
        resolved = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=config,
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        OmegaConf.to_container(resolved, resolve=True)
