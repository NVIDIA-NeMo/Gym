# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf
from starlette.requests import Request

import nemo_gym.client.server_client
from benchmarks.indic.arena_proxy_v3.generation_model import ArenaProxyGenerationConfig, ArenaProxyGenerationModel
from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.config.config_types import ConfigMissingValuesError
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.config.secret_utils import MASKED_VALUE, hide_secrets_in_container
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks/indic/arena_proxy_v3"


def _parse(paths: list[Path], *, overrides: dict | None = None, dotenv_path: Path | None = None) -> DictConfig:
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                {
                    "config_paths": [str(path) for path in paths],
                    "policy_base_url": "http://127.0.0.1:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "candidate",
                },
                overrides or {},
            ),
            dotenv_path=dotenv_path,
            skip_load_from_cli=True,
            skip_load_from_dotenv=dotenv_path is None,
            offline=True,
        )
    )


@pytest.fixture
def model_profile(tmp_path: Path) -> Path:
    path = tmp_path / "model-profile.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "policy_model": {
                    "responses_api_models": {"vllm_model": {"context_length": 131072, "extra_body": {"top_k": 64}}}
                }
            }
        ),
        path,
    )
    return path


@pytest.mark.parametrize("profile_last,expected_top_k", [(False, -1), (True, 64)])
def test_model_profile_composition_preserves_context_and_respects_order(
    model_profile: Path, profile_last: bool, expected_top_k: int
) -> None:
    paths = [BENCHMARK / "generation.yaml", model_profile]
    if not profile_last:
        paths.reverse()
    config = _parse(paths)
    model = config.policy_model.responses_api_models.vllm_model
    assert model.context_length == 131072
    assert model.extra_body.top_k == expected_top_k
    assert model.extra_body.seed == 42
    assert model.entrypoint == "../../benchmarks/indic/arena_proxy_v3/generation_model.py"
    assert config.responses_create_params == {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 120000}


def test_generation_requires_context_even_when_legacy_context_environment_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXT", "163840")
    with pytest.raises(ConfigMissingValuesError, match="policy_model.responses_api_models.vllm_model.context_length"):
        _parse([BENCHMARK / "generation.yaml"])


def test_legacy_environment_does_not_override_composed_configuration(
    monkeypatch: pytest.MonkeyPatch, model_profile: Path, tmp_path: Path
) -> None:
    for name, value in {
        "TEMPERATURE": "0.123",
        "TOP_P": "0.234",
        "TOP_K": "3",
        "CONTEXT": "999999",
        "ARENA_JUDGE_API_KEY": "unused-environment-canary",  # pragma: allowlist secret
        "ARENA_JUDGE_BASE_URL": "https://unused.example.invalid/v1",
        "ARENA_JUDGE_MODEL": "unused-model",
        "INPUT_JSONL": "unused-input.jsonl",
    }.items():
        monkeypatch.setenv(name, value)
    input_path = str(tmp_path / "prepared/arena_proxy_v3_benchmark.jsonl")
    config = _parse(
        [BENCHMARK / "config.yaml", BENCHMARK / "generation.yaml", model_profile],
        overrides={"input_jsonl": input_path},
    )
    model = config.policy_model.responses_api_models.vllm_model
    judge = config.indic_arena_proxy_v3_judge_model.responses_api_models.openai_model
    assert model.context_length == 131072
    assert model.extra_body == {"top_k": 64, "seed": 42}
    assert config.responses_create_params == {"temperature": 1.0, "top_p": 0.95, "max_output_tokens": 120000}
    assert judge.openai_api_key == "MISSING"
    assert judge.openai_base_url == "https://inference-api.nvidia.com/v1"
    assert judge.openai_model == "gcp/google/gemini-3.8-flash"
    agent = config.indic_arena_proxy_v3_agent.responses_api_agents.simple_agent
    resources = config.indic_arena_proxy_v3_resources_server.resources_servers.arena_judge
    assert agent.datasets[0].jsonl_fpath == resources.indic_metrics_input_fpath == input_path
    assert resources.judge_chat_completions_create_params == {
        "messages": [],
        "max_tokens": 32768,
        "temperature": 0.0,
        "reasoning_effort": "low",
    }


@pytest.mark.parametrize("custom_input", [False, True])
def test_collection_and_judge_use_the_same_prepared_population(tmp_path: Path, custom_input: bool) -> None:
    overrides = {"input_jsonl": str(tmp_path / "hi/arena_proxy_v3_benchmark.jsonl")} if custom_input else {}
    config = _parse(
        [BENCHMARK / "config.yaml"],
        overrides=OmegaConf.to_container(
            OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, overrides), resolve=False
        ),
    )
    judge_only = _parse([BENCHMARK / "judge.yaml"], overrides=overrides)
    expected = overrides.get("input_jsonl", "benchmarks/indic/arena_proxy_v3/data/arena_proxy_v3_benchmark.jsonl")
    assert config.indic_arena_proxy_v3_agent.responses_api_agents.simple_agent.datasets[0].jsonl_fpath == expected
    for resolved in (config, judge_only):
        resources = resolved.indic_arena_proxy_v3_resources_server.resources_servers.arena_judge
        assert resources.indic_metrics_input_fpath == expected


def test_private_dotenv_configures_and_masks_judge_credentials(tmp_path: Path) -> None:
    private_config = tmp_path / "env.yaml"
    canary = "judge-config-canary"  # pragma: allowlist secret
    OmegaConf.save(
        OmegaConf.create(
            {
                "judge_api_key": canary,
                "judge_base_url": "https://configured.example.invalid/v1",
                "judge_model_name": "configured-judge",
            }
        ),
        private_config,
    )
    private_config.chmod(0o600)
    config = _parse([BENCHMARK / "judge.yaml"], dotenv_path=private_config)
    judge = config.indic_arena_proxy_v3_judge_model.responses_api_models.openai_model
    assert judge.openai_api_key == canary
    assert judge.openai_base_url == "https://configured.example.invalid/v1"
    assert judge.openai_model == "configured-judge"
    masked = hide_secrets_in_container(OmegaConf.to_container(config, resolve=True))
    assert masked["judge_api_key"] == MASKED_VALUE
    assert (
        masked["indic_arena_proxy_v3_judge_model"]["responses_api_models"]["openai_model"]["openai_api_key"]
        == MASKED_VALUE
    )
    assert canary not in json.dumps(masked)


@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("prompt_tokens,budget", [(100, 120000), (14000, 117008)])
async def test_composed_sampling_reaches_the_reference_generation_payload(
    monkeypatch: pytest.MonkeyPatch,
    model_profile: Path,
    tmp_path: Path,
    temperature: float,
    prompt_tokens: int,
    budget: int,
) -> None:
    input_path = tmp_path / "inputs.jsonl"
    messages = [
        {"role": "user", "content": "पहला प्रश्न?\nKeep spacing. "},
        {"role": "assistant", "content": "<think>Earlier reasoning</think>Earlier answer."},
        {"role": "user", "content": "क्यों?"},
    ]
    row = {
        "agent_ref": {"type": "responses_api_agents", "name": "indic_arena_proxy_v3_agent"},
        "responses_create_params": {"input": deepcopy(messages)},
        "uid": "hi:config-parity",
    }
    original_input = json.dumps(row) + "\n"
    input_path.write_text(original_input)
    overrides = {"input_jsonl": str(input_path)}
    if temperature != 1.0:
        overrides["responses_create_params"] = {"temperature": temperature}
    config = _parse([BENCHMARK / "config.yaml", BENCHMARK / "generation.yaml", model_profile], overrides=overrides)
    sampling = OmegaConf.to_container(config.responses_create_params, resolve=True)
    collection = RolloutCollectionConfig(
        input_jsonl_fpath=str(input_path),
        output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
        responses_create_params=sampling,
        num_repeats=1,
        num_repeats_add_seed=False,
    )
    [prepared] = RolloutCollectionHelper()._preprocess_rows_from_config(collection)
    assert prepared["responses_create_params"] == {"input": messages, **sampling}
    assert input_path.read_text() == original_input
    monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", lambda: {})
    model_config = OmegaConf.to_container(config.policy_model.responses_api_models.vllm_model, resolve=True)
    model_config["name"] = "policy_model"
    model = ArenaProxyGenerationModel(
        config=ArenaProxyGenerationConfig.model_validate(model_config),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    endpoint = MagicMock(base_url="http://127.0.0.1:8000/v1")
    endpoint.create_tokenize = AsyncMock(return_value={"count": prompt_tokens})
    endpoint.create_chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-config-parity",
            "object": "chat.completion",
            "created": 0,
            "model": "candidate",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Answer."}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10, "total_tokens": prompt_tokens + 10},
        }
    )
    model._clients = [endpoint]
    response = await model.responses(
        Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "config-parity"}}),
        NeMoGymResponseCreateParamsNonStreaming(**prepared["responses_create_params"]),
    )
    endpoint.create_tokenize.assert_awaited_once_with(
        model="candidate",
        messages=messages,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )
    endpoint.create_chat_completion.assert_awaited_once_with(
        model="candidate",
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        top_k=64,
        seed=42,
        max_tokens=budget,
        chat_template_kwargs={"enable_thinking": True},
    )
    assert response.output[-1].content[0].text == "Answer."
