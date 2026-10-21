# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.opencode_multimodal_benchmarks.app import (
    VisualBenchmarksConfig,
    VisualBenchmarksServer,
    VisualVerifyRequest,
)
from resources_servers.opencode_multimodal_benchmarks.reporting import benchmark_report


def completion(text, *, mask=False, kind=None):
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="test",
        object="response",
        tools=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        output=[
            {
                "id": "answer",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        metadata={
            "instance_config": json.dumps({"mask_sample": mask}),
            "metrics": json.dumps({"agent_error_kind": kind}),
            "input": "[]",
        },
    )


def task(benchmark="VStarBench", answer="A"):
    return {
        "index": "1",
        "benchmark": benchmark,
        "question": "Which animal?",
        "answer": answer,
        "options": {"A": "cat", "B": "dog"},
        "category": "spatial",
        "l2_category": "detail",
        "cycle_category": "1",
        "split": "test",
    }


@pytest.fixture
def server():
    return VisualBenchmarksServer(
        config=VisualBenchmarksConfig(
            host="127.0.0.1",
            port=12345,
            name="visual",
            entrypoint="app.py",
            judge_model="test-judge",
            judge_endpoint="https://judge.example/v1/chat/completions",
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def request(benchmark, prediction, answer="A"):
    return VisualVerifyRequest(
        responses_create_params={"input": []},
        response=completion(prediction),
        verifier_metadata=task(benchmark, answer),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K"])
@pytest.mark.parametrize("prediction,reward", [(r"\boxed{A}", 1), (r"\boxed{B}", 0), ("", 0)])
async def test_deterministic_scoring(server, benchmark, prediction, reward, monkeypatch):
    judge = AsyncMock(return_value="Z")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request(benchmark, prediction))
    assert result.reward == reward
    if prediction:
        judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_reference_blind_mcq_judge(server, monkeypatch):
    judge = AsyncMock(return_value="A")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VStarBench", "the fuzzy feline on the left"))
    assert result.reward == 1 and result.extraction_method == "llm"
    prompt = judge.call_args.args[0]
    assert "Ground Truth" not in prompt and "correct answer" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("True", 1), ("False", 0)])
async def test_babyvision_semantic_judge(server, monkeypatch, verdict, reward):
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", AsyncMock(return_value=verdict))
    result = await server.verify(request("BabyVision", r"\boxed{four}", answer="4"))
    assert result.reward == reward and result.extracted == "four"


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["BabyVision", "VStarBench"])
async def test_invalid_judge_is_not_wrong(server, monkeypatch, benchmark):
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", AsyncMock(return_value="unparseable"))
    with pytest.raises(JudgeError):
        await server.verify(request(benchmark, "ambiguous answer"))


@pytest.mark.asyncio
async def test_missing_key_is_judge_failure(server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="Missing judge credential"):
        await server._judge("test")


def test_hrbench_cycle_weighting(server):
    rows = [
        task("HRBench8K") | {"hit": 1},
        task("HRBench8K") | {"hit": 1},
        task("HRBench8K") | {"hit": 0, "cycle_category": "2"},
    ]
    results = [[{"verifier_metadata": row | {"index": str(i)}, "reward": row["hit"]}] for i, row in enumerate(rows)]
    metrics = server.compute_metrics(results)
    assert metrics["accuracy"] == 0.5  # row mean would incorrectly be 2/3
    assert metrics["HRBench8K/report"] == benchmark_report("HRBench8K", rows)


def test_labels_are_not_policy_input():
    row = task_row(task(answer="secret-reference") | {"prompt": "Public question", "images": ["/tmp/a.png"]})
    assert "secret-reference" not in json.dumps(row["responses_create_params"])
    assert row["verifier_metadata"]["answer"] == "secret-reference"
    assert "max_output_tokens" not in row["responses_create_params"]


def test_partial_results_are_not_published_as_accuracy(server):
    metrics = server.compute_metrics([[{"verifier_metadata": task() | {"expected_samples": 2}, "reward": 1.0}]])
    assert metrics["VStarBench/completed"] == 1
    assert metrics["VStarBench/expected"] == 2
    assert metrics["VStarBench/partial_accuracy"] == 1.0
    assert "accuracy" not in metrics and not server.get_key_metrics(metrics)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prediction",
    [
        r"<think>It could be B. Cannot determine the answer.</think>\boxed{A}",
        r"<thinking>Maybe B or C.</thinking>\boxed{A}",
    ],
)
async def test_only_visible_answer_is_graded(server, monkeypatch, prediction):
    judge = AsyncMock(return_value="Z")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VStarBench", prediction))
    assert result.reward == 1
    assert result.extracted == "A"
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_unfinished_reasoning_does_not_become_an_answer(server, monkeypatch):
    judge = AsyncMock(return_value="A")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VStarBench", r"<think>The answer might be \boxed{A}"))
    assert result.reward == 0
    judge.assert_not_awaited()
