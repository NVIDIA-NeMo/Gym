# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise native Gym verification requests and reasoning separation."""

import runpy
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tiktoken

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.server_utils import ServerClient
from resources_servers.genrm_eval.app import GenRMEvalConfig, GenRMEvalResourcesServer, GenRMEvalVerifyRequest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected", "ranking"),
    [
        ('{"overall":{"ranking":2}}', 1, 2),
        ('{"ranking":6}', 0, 6),
        ('{"score_1":4}', 0, None),
        ("Analysis then ranking: 2", 0, None),
        ('{"ranking":-100}', 0, None),
        ('{"ranking":100}', 0, None),
        ('{"ranking":true}', 0, None),
        ('{"ranking":3.5}', 0, None),
    ],
)
async def test_verify(text: str, expected: float, ranking: float | None) -> None:
    server = GenRMEvalResourcesServer(
        config=GenRMEvalConfig(
            host="localhost",
            port=8000,
            name="genrm_eval",
            entrypoint="app.py",
            scoring_module="resources_servers.genrm_eval.preferences",
            metric_key="overall/accuracy",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    body = GenRMEvalVerifyRequest.model_validate(
        {
            "id": "p",
            "verifier_metadata": {"ground_truth_label": 1},
            "responses_create_params": {
                "input": [{"role": "user", "content": "Q"}],
                "metadata": {"response_1": "A", "response_2": "B"},
            },
            "response": {
                "id": "resp_1",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "r",
                        "summary": [{"type": "summary_text", "text": '```json\n{"ranking":6}\n```'}],
                    },
                    {
                        "type": "message",
                        "id": "m",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                ],
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
            },
        }
    )
    result = await server.verify(body)
    assert result.reward == expected
    assert result.predicted_ranking == ranking
    assert not result.request_error
    encoding = tiktoken.encoding_for_model("gpt-4o")
    assert result.response_tokens == len(encoding.encode(text, disallowed_special=()))
    assert result.reasoning_tokens == len(encoding.encode('```json\n{"ranking":6}\n```', disallowed_special=()))
    # Gym drops response.output before aggregation; per-row counts survive that.
    stripped = result.model_dump(exclude={"response", "responses_create_params"})
    aggregate = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[stripped]))
    assert aggregate.key_metrics == {
        "overall/accuracy": expected,
        "median/response_tokens": result.response_tokens,
        "median/reasoning_tokens": result.reasoning_tokens,
    }


def test_tokenizer_is_loaded_at_startup() -> None:
    with patch("resources_servers.genrm_eval.app.tiktoken.encoding_for_model") as load:
        server = GenRMEvalResourcesServer(
            config=GenRMEvalConfig(
                host="localhost",
                port=8000,
                name="genrm_eval",
                entrypoint="app.py",
                scoring_module="resources_servers.genrm_eval.preferences",
                metric_key="overall/accuracy",
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        load.assert_called_once_with("gpt-4o")
        assert server._encoding is load.return_value


def test_configured_coverage_is_in_headline_metrics() -> None:
    with patch("resources_servers.genrm_eval.app.tiktoken.encoding_for_model"):
        server = GenRMEvalResourcesServer(
            config=GenRMEvalConfig(
                host="localhost",
                port=8000,
                name="genrm_eval",
                entrypoint="app.py",
                scoring_module="benchmarks.genrm_rmbench.metrics",
                metric_key="avg",
                additional_metric_keys=["domain_coverage"],
            ),
            server_client=MagicMock(spec=ServerClient),
        )
    assert server.get_key_metrics({"avg": 0.25, "domain_coverage": 0.25, "chat_count": 1}) == {
        "avg": 0.25,
        "domain_coverage": 0.25,
    }


@pytest.mark.parametrize("is_worker", [False, True])
def test_worker_entrypoint(is_worker: bool) -> None:
    with (
        patch("nemo_gym.server_utils.is_nemo_gym_fastapi_entrypoint", return_value=is_worker),
        patch("nemo_gym.base_resources_server.SimpleResourcesServer.run_webserver") as start,
    ):
        module = runpy.run_path(str(Path(__file__).parents[1] / "app.py"), run_name="worker_import")
    assert ("app" in module) == is_worker
    assert start.call_count == int(is_worker)
    if is_worker:
        assert module["app"] is start.return_value
