# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge interfaces must agree on labels, denominators, and slices."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.genrm_eval.app import (
    GenRMEvalConfig,
    GenRMEvalResourcesServer,
    GenRMEvalVerifyRequest,
    parse_ab_ranking,
)
from resources_servers.genrm_eval.mmrb import compute_metrics


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmarks", [("genrm_mmrb1", "genrm_mmrb2"), ("genrm_mmrb2", "genrm_mmrb1")])
async def test_composed_llm_configs_use_independent_verifiers(benchmarks):
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
    for benchmark, text in [("genrm_mmrb1", "[[A]]"), ("genrm_mmrb2", '{"better_response":"A"}')]:
        name = resolved[f"{benchmark}_benchmark_agent"].responses_api_agents.simple_agent.resources_server.name
        settings = OmegaConf.to_container(resolved[name].resources_servers.genrm_eval, resolve=True)
        config = GenRMEvalConfig.model_validate(settings | {"name": name, "host": "localhost", "port": 8000})
        server = GenRMEvalResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        body = GenRMEvalVerifyRequest.model_validate(
            {
                "id": "p",
                "verifier_metadata": {"pair_id": "p", "ground_truth_label": 3},
                "responses_create_params": {"input": [{"role": "user", "content": "Question and candidates"}]},
                "response": {
                    "id": "r",
                    "created_at": 0,
                    "model": "judge",
                    "object": "response",
                    "parallel_tool_calls": False,
                    "tool_choice": "none",
                    "tools": [],
                    "output": [
                        {
                            "type": "message",
                            "id": "m",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": text, "annotations": []}],
                        }
                    ],
                },
            }
        )
        result = await server.verify(body)
        assert result.predicted_ranking == 3
        assert result.reward == 1


@pytest.mark.parametrize(
    ("text", "format", "expected"),
    [
        ("Explanation [[A]]", "brackets", 3),
        ("[[B]]", "brackets", 4),
        ("[[B]] then [[A]]", "brackets", 3),
        ("[[A=B]]", "brackets", None),
        ('{"better_response":"A"}', "json_ab", 3),
        ("{better_response: 'B'}", "json_ab", 4),
        ('[{"better_response":"A"}]', "json_ab", None),
        ('{"better_response":"tie"}', "json_ab", None),
        ('{"better_response":["A"]}', "json_ab", None),
        ("<think>[[A]]</think>[[B]]", "brackets", 4),
        ("<thinking>[[A]]", "brackets", None),
        ('<think>{"better_response":"B"}</think>{"better_response":"A"}', "json_ab", 3),
        ("", "brackets", None),
        ("", "json_ab", None),
    ],
)
def test_verdict_parsers(text, format, expected):
    assert parse_ab_ranking(text, format) == expected


def test_metric_denominators_and_slices():
    rows = [
        {
            "predicted_ranking": ranking,
            "verifier_metadata": {
                "pair_id": "p",
                "ground_truth_label": label,
                "source": "test",
                "order": order,
                "task": "reasoning",
                "response_modality": "text_only",
            },
        }
        for ranking, label, order in [(3, 3, "forward"), (None, 4, "reverse")]
    ]
    result = compute_metrics([rows])
    assert result["overall"] == {
        "accuracy": 0.5,
        "accuracy_valid": 1,
        "invalid_judgment_rate": 0.5,
        "total_pairs": 1,
        "total_judgments": 2,
    }
    assert result["source/test/accuracy"] == 0.5
    assert result["order/forward/accuracy"] == 1
    assert result["order/reverse/accuracy"] == 0
    assert result["response_modality/text_only/accuracy"] == 0.5
    assert compute_metrics([]) == {}
    assert compute_metrics([[rows[1]]])["overall"]["accuracy_valid"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("format", "text"),
    [
        ("genrm", '{"overall":{"ranking":2}}'),
        ("brackets", "[[A]]"),
        ("json_ab", '{"better_response":"A"}'),
    ],
)
async def test_interfaces_score_same_pair(format, text):
    server = GenRMEvalResourcesServer(
        config=GenRMEvalConfig(
            host="localhost",
            port=8000,
            name="genrm_eval",
            entrypoint="app.py",
            scoring_module="resources_servers.genrm_eval.mmrb",
            metric_key="overall/accuracy",
            verdict_format=format,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = GenRMEvalVerifyRequest.model_validate(
        {
            "id": "p",
            "verifier_metadata": {"pair_id": "p", "ground_truth_label": 3},
            "responses_create_params": {"input": [{"role": "user", "content": "Question and candidates"}]},
            "response": {
                "id": "r",
                "created_at": 0,
                "model": "judge",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "output": [
                    {
                        "type": "reasoning",
                        "id": "think",
                        "summary": [{"type": "summary_text", "text": '{"ranking":6}'}],
                    },
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                ],
            },
        }
    )
    result = await server.verify(request)
    assert result.reward == 1
    stripped = result.model_dump(exclude={"response", "responses_create_params"})
    assert server.compute_metrics([[stripped]])["overall/accuracy"] == 1
