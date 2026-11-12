# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LAB sampling defaults must reach the provider, while explicit row values take precedence."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import AnthropicConverter
from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper
from resources_servers.legal_agent_bench.lab_harbor.completion import CompletionState
from responses_api_agents.legal_agent_bench_native.app import (
    LegalAgentBenchNativeAgent,
    LegalAgentBenchNativeAgentConfig,
)
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
RESPONSE = {
    "id": "response",
    "object": "response",
    "created_at": 0,
    "model": "policy",
    "output": [],
    "status": "completed",
    "tools": [],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
}
CHAT_RESPONSE = {
    "id": "chat",
    "object": "chat.completion",
    "created": 0,
    "model": "policy",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
}


def _resolve(harness: str) -> DictConfig:
    config = OmegaConf.load(BENCHMARK_DIR / f"config_{harness}.yaml")
    config.config_paths.append("benchmarks/legal_agent_bench/model_sampling_openai.yaml")
    initial = OmegaConf.merge(
        config,
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "default_host": "127.0.0.1",
            "lab": {"policy_temperature": 0.37, "policy_top_p": 0.73},
        },
    )
    return GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)


def _model_server(config: DictConfig) -> SimpleModelServer:
    settings = OmegaConf.to_container(config.policy_model.responses_api_models.openai_model, resolve=True)
    settings.update(
        name="policy_model", openai_base_url="http://unused/v1", openai_api_key="unused", openai_model="policy"
    )
    server = SimpleModelServer.model_construct(config=SimpleModelServerConfig.model_validate(settings))
    server._semaphore = asyncio.Semaphore(1)
    server._client = SimpleNamespace(
        create_response=AsyncMock(return_value=RESPONSE), create_chat_completion=AsyncMock(return_value=CHAT_RESPONSE)
    )
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("row_sampling", [{}, {"temperature": 0.0, "top_p": 0.81}])
async def test_lab_agent_sampling_reaches_provider(
    row_sampling: dict[str, float], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _resolve("native")
    model = _model_server(config)
    # Collection applies global overrides after row values. The native preset must preserve row sampling.
    source_row = {
        "agent_ref": {"name": "legal_agent_bench_benchmark_agent"},
        "responses_create_params": {"input": [], **row_sampling},
    }
    collection = RolloutCollectionConfig(
        input_jsonl_fpath="unused",
        output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        responses_create_params=dict(config.get("responses_create_params", {})),
    )
    [row] = RolloutCollectionHelper._preprocess_raw_rows([(0, json.dumps(source_row), source_row.copy())], collection)
    params = row["responses_create_params"]
    settings = OmegaConf.to_container(
        next(iter(config.legal_agent_bench_benchmark_agent.responses_api_agents.values())), resolve=True
    )
    from responses_api_agents.legal_agent_bench_native import app

    class Reply:
        cookies = {}

        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def post(**kwargs):
        result = await model.responses(NeMoGymResponseCreateParamsNonStreaming.model_validate(kwargs["json"]))
        return Reply(result.model_dump(mode="json"))

    settings.update(name="agent", max_turns=1)
    runner = LegalAgentBenchNativeAgent.model_construct(
        config=LegalAgentBenchNativeAgentConfig.model_validate(settings),
        server_client=MagicMock(post=AsyncMock(side_effect=post)),
    )
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(side_effect=lambda reply: reply.body))
    state = SimpleNamespace(session=SimpleNamespace(expired=False), completion=CompletionState(), deadline=None)
    await runner._run_loop(state, NeMoGymResponseCreateParamsNonStreaming(**params), "/v1/responses", {})
    upstream = model._client.create_response
    upstream.assert_awaited_once()
    assert upstream.call_args.kwargs["temperature"] == row_sampling.get("temperature", 0.37)
    assert upstream.call_args.kwargs["top_p"] == row_sampling.get("top_p", 0.73)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["hermes", "claude_code", "codex"])
async def test_cli_sampling_defaults_at_model_ingress(harness: str) -> None:
    config = _resolve(harness)
    model = _model_server(config)
    if harness == "hermes":
        # Hermes accepts temperature through its own config; top_p comes from the policy server.
        temperature = config.legal_agent_bench_benchmark_agent.responses_api_agents.hermes_agent.temperature
        await model.chat_completions(
            NeMoGymChatCompletionCreateParamsNonStreaming(
                messages=[{"role": "user", "content": "done"}],
                temperature=temperature,
            )
        )
        upstream = model._client.create_chat_completion
    else:
        if harness == "claude_code":
            body = AnthropicConverter().anthropic_request_to_responses(
                {
                    "model": "policy",
                    "messages": [{"role": "user", "content": "done"}],
                    "max_tokens": 16,
                }
            )
        else:
            body = NeMoGymResponseCreateParamsNonStreaming(input="done")
        await model.responses(body)
        upstream = model._client.create_response
    upstream.assert_awaited_once()
    assert upstream.call_args.kwargs["temperature"] == 0.37
    assert upstream.call_args.kwargs["top_p"] == 0.73
