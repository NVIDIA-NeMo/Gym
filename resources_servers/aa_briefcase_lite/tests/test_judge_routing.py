# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from openai import OpenAI

from nemo_gym.server_utils import ServerClient
from resources_servers.aa_briefcase_lite.app import (
    AABriefcaseLiteResourcesServer,
    AABriefcaseLiteResourcesServerConfig,
)
from resources_servers.gdpval.judge_panel import merge_create_kwargs
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


@pytest.mark.parametrize(
    ("judge_name", "expected_model", "expected_parameters", "model_overrides"),
    [
        ("gpt-5.5", "openai/openai/gpt-5.5", {"reasoning_effort": "high"}, {}),
        ("gemini-3.1-pro", "gcp/google/gemini-3.1-pro-preview", {"reasoning_effort": "high"}, {}),
        (
            "claude-opus-4.8",
            "aws/anthropic/bedrock-claude-opus-4-8",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}, "max_tokens": 16384},
            {},
        ),
        ("gemini-3.1-pro", "legacy-model", {"reasoning_effort": "high"}, {"JUDGE_MODEL_NAME": "legacy-model"}),
        (
            "gemini-3.1-pro",
            "gemini-model",
            {"reasoning_effort": "high"},
            {"JUDGE_MODEL_NAME": "legacy-model", "JUDGE_GEMINI_MODEL": "gemini-model"},
        ),
    ],
)
def test_benchmark_panel_routes_to_matching_upstream_model(
    monkeypatch, tmp_path: Path, judge_name: str, expected_model: str, expected_parameters: dict, model_overrides: dict
) -> None:
    """Exercise the benchmark YAML, panel resolver, SDK, and real model HTTP route.

    Capture the outbound provider request, after the adapter's fixed-model
    override. No provider credentials or network calls are needed.
    """
    for name in ("JUDGE_MODEL_NAME", "JUDGE_GPT_MODEL", "JUDGE_GEMINI_MODEL", "JUDGE_CLAUDE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in model_overrides.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AA_BRIEFCASE_LITE_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("JUDGE_BASE_URL", "http://upstream.invalid/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "dummy")
    benchmark = Path(__file__).resolve().parents[3] / "benchmarks/aa_briefcase_lite/config.yaml"
    config = OmegaConf.load(benchmark)
    model_configs = {}
    for name, section in config.items():
        if "responses_api_models" not in section:
            continue
        model_config = section.responses_api_models.openai_model
        model_config.update(host="127.0.0.1", port=19000 + len(model_configs), name=name)
        model_configs[name] = SimpleModelServerConfig(**OmegaConf.to_container(model_config, resolve=True))
    monkeypatch.setattr("nemo_gym.server_utils.get_global_config_dict", lambda: config)
    resource_config = AABriefcaseLiteResourcesServerConfig(
        **OmegaConf.to_container(
            config.aa_briefcase_lite_resources_server.resources_servers.aa_briefcase_lite, resolve=True
        ),
        host="127.0.0.1",
        port=18000,
    )
    # Resolving the panel does not need dataset loading or Office installation.
    monkeypatch.setattr(AABriefcaseLiteResourcesServer, "model_post_init", lambda self, context: None)
    resource = AABriefcaseLiteResourcesServer(
        config=resource_config, server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )
    judges = resource._resolve_judges()
    judge = next(member for member in judges if member.name == judge_name)
    assert judge.model == expected_model
    model_config = next(
        model for model in model_configs.values() if judge.base_url == f"http://{model.host}:{model.port}/v1"
    )
    server = SimpleModelServer(config=model_config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    server._client = MagicMock()
    server._client.create_chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-panel-routing",
            "object": "chat.completion",
            "created": 0,
            "model": expected_model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        }
    )
    with TestClient(server.setup_webserver()) as transport:
        client = OpenAI(base_url=judge.base_url, api_key=judge.api_key, http_client=transport, max_retries=0)
        client.chat.completions.create(
            **merge_create_kwargs(
                {
                    "model": judge.model,
                    "messages": [{"role": "user", "content": "check"}],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                judge.create_overrides,
            )
        )
    server._client.create_chat_completion.assert_awaited_once()
    forwarded = server._client.create_chat_completion.await_args.kwargs
    assert forwarded["model"] == expected_model
    assert {key: forwarded[key] for key in expected_parameters} == expected_parameters
    assert forwarded["max_tokens"] == expected_parameters.get("max_tokens", 4096)
    assert "temperature" not in forwarded
