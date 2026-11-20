# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from pathlib import Path
from runpy import run_path
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from omegaconf import OmegaConf
from starlette.requests import Request

import nemo_gym.client.server_client
from benchmarks.indic.arena_proxy_v3.generation_model import ArenaProxyGenerationConfig, ArenaProxyGenerationModel
from benchmarks.indic.arena_proxy_v3.generation_preflight import validate_preflight
from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime.supervisor.server_supervisor import _configured_servers
from resources_servers.arena_judge.app import ArenaJudgeServer, final_answer_only
from responses_api_models.vllm_model.app import VLLMModel


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks/indic/arena_proxy_v3"


def _model(monkeypatch: pytest.MonkeyPatch, *, context_length: int, prompt_tokens: int) -> tuple:
    monkeypatch.setattr(nemo_gym.client.server_client, "get_global_config_dict", lambda: {})
    config_dict = OmegaConf.to_container(
        OmegaConf.load(BENCHMARK / "generation.yaml").policy_model.responses_api_models.vllm_model,
        resolve=False,
    )
    config_dict.update(
        host="127.0.0.1",
        port=8001,
        name="policy_model",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model="candidate",
        context_length=context_length,
        extra_body={"temperature": 1.0, "top_p": 0.95, "top_k": -1, "seed": 42},
    )
    model = ArenaProxyGenerationModel(
        config=ArenaProxyGenerationConfig.model_validate(config_dict),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    endpoint = MagicMock(base_url="http://127.0.0.1:8000/v1")
    endpoint.create_tokenize = AsyncMock(return_value={"count": prompt_tokens})
    endpoint.create_chat_completion = AsyncMock()
    endpoint.create_response = AsyncMock()
    model._clients = [endpoint]
    return model, endpoint


def _completion(*, content: str, finish_reason: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "candidate",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "reasoning_content": "Private reasoning."},
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


@pytest.mark.parametrize("multi_turn", [False, True])
@pytest.mark.parametrize("temperature,top_p,top_k", [(1.0, 0.95, -1), (0.6, 0.9, "64")])
@pytest.mark.parametrize(
    "context_length,prompt_tokens,budget", [(163840, 1000, 120000), (131072, 21665, 109343), (32768, 2000, 30704)]
)
async def test_generation_matches_reference_chat_payload(
    monkeypatch: pytest.MonkeyPatch,
    multi_turn: bool,
    context_length: int,
    prompt_tokens: int,
    budget: int,
    temperature: float,
    top_p: float,
    top_k: int | str,
) -> None:
    model, endpoint = _model(monkeypatch, context_length=context_length, prompt_tokens=prompt_tokens)
    model.config.extra_body["top_k"] = top_k
    messages = [{"role": "user", "content": "पहला प्रश्न?\nKeep whitespace. "}]
    if multi_turn:
        messages.extend(
            [
                {"role": "assistant", "content": "<think>Historical text</think>Answer."},
                {"role": "user", "content": "क्यों?"},
            ]
        )
    original = deepcopy(messages)
    endpoint.create_chat_completion.return_value = _completion(content="Final answer.", finish_reason="stop")
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "test"}})
    response = await model.responses(
        request,
        NeMoGymResponseCreateParamsNonStreaming(
            input=messages, temperature=temperature, top_p=top_p, max_output_tokens=120000
        ),
    )
    expected_tokenize = {
        "model": "candidate",
        "messages": original,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    expected_generation = {
        "model": "candidate",
        "messages": original,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": int(top_k),
        "seed": 42,
        "max_tokens": budget,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    endpoint.create_tokenize.assert_awaited_once_with(**expected_tokenize)
    endpoint.create_chat_completion.assert_awaited_once_with(**expected_generation)
    assert endpoint.mock_calls[:2] == [
        call.create_tokenize(**expected_tokenize),
        call.create_chat_completion(**expected_generation),
    ]
    endpoint.create_response.assert_not_called()
    assert ArenaJudgeServer._extract_response_output_text(response) == "Final answer."
    assert messages == original
    assert model.config.request_timeout_seconds is None


@pytest.mark.parametrize(
    "content,finish_reason,expected",
    [
        ("", "length", ""),
        ("", "stop", ""),
        ("Partial answer", "length", "Partial answer"),
        ("prefix<think>hidden</think>final", "stop", "final"),
        ("prefix<think>a</think>middle<think>b</think> final ", "stop", " final "),
        ("prefix</think>final", "stop", "final"),
        ("prefix<think>unfinished", "length", ""),
        ("prefix<think>a</think>middle<think>unfinished", "length", ""),
    ],
)
async def test_keeps_length_limited_and_reasoning_only_answers_without_resampling(
    monkeypatch: pytest.MonkeyPatch, content: str, finish_reason: str, expected: str
) -> None:
    model, endpoint = _model(monkeypatch, context_length=163840, prompt_tokens=100)
    endpoint.create_chat_completion.return_value = _completion(content=content, finish_reason=finish_reason)
    response = await model.responses(
        Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "test"}}),
        NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Question"}], max_output_tokens=120000
        ),
    )
    assert final_answer_only(ArenaJudgeServer._extract_response_output_text(response)) == expected
    assert ArenaJudgeServer._extract_response_output_text(response) == expected
    assert any(item.type == "reasoning" for item in response.output)
    assert bool(response.incomplete_details) == (finish_reason == "length")
    endpoint.create_chat_completion.assert_awaited_once()


@pytest.mark.parametrize("count", [None, True, -1, 163800])
async def test_bad_token_count_or_full_context_fails_before_generation(
    monkeypatch: pytest.MonkeyPatch, count: object
) -> None:
    model, endpoint = _model(monkeypatch, context_length=163840, prompt_tokens=100)
    endpoint.create_tokenize.return_value = {"count": count}
    with pytest.raises(ValueError):
        await model.responses(
            Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "test"}}),
            NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Question"}], max_output_tokens=120000
            ),
        )
    endpoint.create_chat_completion.assert_not_called()


def test_gym_resolves_benchmark_entrypoint_inside_vllm_component() -> None:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(BENCHMARK / "generation.yaml")],
                    "policy_base_url": "http://127.0.0.1:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "candidate",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    server = next(_configured_servers(config))
    assert server.dir_path == ROOT / "responses_api_models/vllm_model"
    assert (server.dir_path / server.entrypoint_fpath).resolve() == BENCHMARK / "generation_model.py"
    assert server.config.request_timeout_seconds is None


def test_entrypoint_launches_the_benchmark_model(monkeypatch: pytest.MonkeyPatch) -> None:
    launched = []

    def run_webserver(cls: type[VLLMModel]) -> None:
        launched.append(cls)

    monkeypatch.setattr(VLLMModel, "run_webserver", classmethod(run_webserver))
    run_path(str(BENCHMARK / "generation_model.py"), run_name="__main__")
    assert len(launched) == 1
    assert launched[0].__name__ == "ArenaProxyGenerationModel"
    assert issubclass(launched[0], VLLMModel)


@pytest.mark.parametrize(
    "content,reasoning",
    [("", "Thinking"), ("Answer", ""), ("<think>Leaked</think>Answer", "Thinking"), ("Answer</think>", "Thinking")],
)
def test_preflight_rejects_parser_misconfiguration(content: str, reasoning: str) -> None:
    with pytest.raises(ValueError, match="Thinking preflight failed"):
        validate_preflight(content=content, reasoning=reasoning)


def test_preflight_accepts_separated_reasoning() -> None:
    validate_preflight(content="नई दिल्ली।", reasoning="Thinking separately.")
