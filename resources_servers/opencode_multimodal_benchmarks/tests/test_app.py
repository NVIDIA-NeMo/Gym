# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.verification import JudgeError
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
        "expected_samples": 1,
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


def request(benchmark, prediction, answer="A", **metadata):
    return VisualVerifyRequest(
        responses_create_params={"input": []},
        response=completion(prediction),
        verifier_metadata=task(benchmark, answer) | metadata,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K"])
@pytest.mark.parametrize("prediction,reward", [(r"\boxed{A}", 1), (r"\boxed{B}", 0), ("", 0)])
async def test_deterministic_scoring(server, benchmark, prediction, reward, monkeypatch):
    judge = AsyncMock(return_value="Z")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request(benchmark, prediction))
    assert result.reward == reward
    assert result.evaluation_completed is True
    if prediction:
        judge.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K"])
@pytest.mark.parametrize("answer,reward", [("A", 0), ("B", 1)])
async def test_corrected_explicit_answer_supersedes_previous_choice(server, monkeypatch, benchmark, answer, reward):
    judge = AsyncMock()
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(
        request(benchmark, "The answer is A. After rechecking, the answer is B.", answer=answer)
    )
    assert result.reward == reward
    assert result.extracted == "B"
    assert result.scorer == "mcq-final-answer-llm-v4"
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
@pytest.mark.parametrize("benchmark", ["VStarBench"])
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
    results = [
        [{"verifier_metadata": row | {"index": str(i), "expected_samples": 3}, "reward": row["hit"]}]
        for i, row in enumerate(rows)
    ]
    metrics = server.compute_metrics(results)
    assert metrics["accuracy"] == 0.5  # row mean would incorrectly be 2/3
    assert metrics["HRBench8K/report"] == benchmark_report("HRBench8K", rows)


def test_labels_are_not_policy_input(tmp_path):
    image = tmp_path / "question.png"
    image.write_bytes(b"image bytes")
    row = task_row(task(answer="secret-reference") | {"prompt": "Public question", "images": [str(image)]})
    image_input = row["responses_create_params"]["input"][0]["content"][1]
    assert image_input["image_url"] == "data:image/png;base64,aW1hZ2UgYnl0ZXM="
    assert str(tmp_path) not in json.dumps(row["responses_create_params"])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("unfinished", [[], [{"type": "reasoning", "id": "r", "summary": []}]])
async def test_missing_final_message_is_scored_as_incorrect(server, monkeypatch, unfinished):
    judge = AsyncMock(side_effect=AssertionError("No answer must not be judged"))
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    body = request("VStarBench", "A")
    body.response = NeMoGymResponse.model_validate(body.response.model_dump() | {"output": unfinished})
    result = await server.verify(body)
    assert result.reward == 0 and result.extracted == ""
    assert result.evaluation_completed is True and result.mask_sample is False
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_benchmark_is_not_silently_scored(server):
    with pytest.raises(ValueError, match="benchmark"):
        await server.verify(request("NotABenchmark", "A"))


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_type", ["reasoning", "function_call"])
async def test_unfinished_final_turn_does_not_grade_earlier_answer(server, monkeypatch, terminal_type):
    body = request("VStarBench", "A")
    payload = body.model_dump()
    payload["response"]["output"].append(
        {"type": "reasoning", "id": "unfinished", "summary": []}
        if terminal_type == "reasoning"
        else {"type": "function_call", "id": "tool", "call_id": "tool", "name": "bash", "arguments": "{}"}
    )
    judge = AsyncMock(return_value="A")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(VisualVerifyRequest.model_validate(payload))
    assert result.reward == 0
    assert result.scorer == "empty-answer-v1"
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcq_invalid_format_gets_one_changed_reference_blind_prompt(server, monkeypatch):
    judge = AsyncMock(side_effect=["Please provide Example 3", "B"])
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VStarBench", "a visually similar creature"))
    assert result.extracted == "B" and result.reward == 0
    first, second = [call.args[0] for call in judge.await_args_list]
    assert second.startswith(first)
    assert "Do not solve the image question" in second
    assert "correct option is A" not in second
