# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest

from nemo_gym.verification import JudgeError
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request, server  # noqa: F401
from resources_servers.opencode_multimodal_benchmarks.vtcbench import judge_score, rule_score


@pytest.mark.parametrize(
    "answer,prediction,options,expected",
    [
        ("5", "There are 5", {}, 0),
        ("5", "Explanation\nAnswer: 5", {}, 1),
        ("5", "five", {}, 0),
        ("26,000", "26000", {}, 0),
        ("a b c d e", "prefix a b c d e suffix", {}, 0),
        ("New York", "new\n   york", {}, 0),
        ("A", "A. cat", {"A": "cat"}, 1),
        ("A", "A-cat", {"A": "cat"}, 1),
        ("A", "Answer: A", {"A": "cat"}, 1),
        ("A", r"\boxed{A}", {"A": "cat"}, 1),
        ("A", "A)", {"A": "cat"}, 1),
        ("cat", "A", {"A": "cat"}, 1),
        ("cat", "the cat", {"A": "cat"}, 0),
        ("cat", "a cat", {"B": "cat"}, 1),
        ("A", "A", {}, 1),
    ],
)
def test_final_answer_rule_cases(answer, prediction, options, expected):
    assert rule_score({"answer": answer, "options": options}, prediction) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("prediction", ["A", "A. a cat"])
async def test_rule_hit_never_calls_judge(server, monkeypatch, prediction):
    judge = AsyncMock()
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VTCBench", prediction))
    assert result.reward == 1 and result.extraction_method == "rule"
    assert result.scorer == "vtcbench-final-answer-rules-llm-v4"
    judge.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("A", 1), ("B", 0), ("Z", 0)])
async def test_mcq_fallback(server, monkeypatch, verdict, reward):
    judge = AsyncMock(return_value=verdict)
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VTCBench", "the furry animal"))
    assert result.reward == reward and result.extraction_method == "llm"
    assert result.judge_model == server.config.judge_model
    prompt = judge.call_args.args[0]
    assert "Ground Truth" not in prompt
    assert "Options: A. cat B. dog " in prompt and "the furry animal" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("A", 1), ("B", 0)])
async def test_option_text_reference_agrees_between_rule_and_judge(server, monkeypatch, verdict, reward):
    judge = AsyncMock(return_value=verdict)
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    direct = await server.verify(request("VTCBench", "A", answer="cat"))
    assert direct.reward == 1 and direct.extraction_method == "rule"
    judge.assert_not_awaited()
    judged = await server.verify(request("VTCBench", "the feline", answer="cat"))
    assert judged.reward == reward and judged.extraction_method == "llm"
    judge.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict,reward", [("yes", 1), ("No", 0)])
async def test_free_answer_fallback(server, monkeypatch, verdict, reward):
    judge = AsyncMock(return_value=verdict)
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    body = request("VTCBench", "five", answer="5", options={})
    result = await server.verify(body)
    assert result.reward == reward and result.extraction_method == "llm"
    assert "Ground Truth: 5\nPredicted Answer: five" in judge.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("options,temperatures", [({}, [0, 0.5, 1, 1.5, 2]), ({"A": "cat"}, [0, 0.1, 0.2, 0.3, 0.4])])
async def test_invalid_verdict_retries_remain_failure(server, monkeypatch, options, temperatures):
    judge = AsyncMock(return_value="unclear")
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    body = request("VTCBench", "unknown")
    body.verifier_metadata["options"] = options
    with pytest.raises(JudgeError, match="5 attempts"):
        await server.verify(body)
    assert [call.kwargs["temperature"] for call in judge.call_args_list] == pytest.approx(temperatures)


@pytest.mark.asyncio
async def test_transport_failure_is_not_scored_zero(server, monkeypatch):
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", AsyncMock(side_effect=JudgeError("unavailable")))
    with pytest.raises(JudgeError, match="unavailable"):
        await server.verify(request("VTCBench", "unknown"))


@pytest.mark.asyncio
async def test_rule_only_mode(server, monkeypatch):
    server.config.vtc_llm_fallback = False
    judge = AsyncMock()
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VTCBench", "Answer: A"))
    assert result.reward == 1 and result.extraction_method == "rule"
    assert result.scorer == "vtcbench-final-answer-rules-v4"
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_answer_is_zero_without_judge(server, monkeypatch):
    judge = AsyncMock()
    monkeypatch.setattr(VisualBenchmarksServer, "_judge", judge)
    result = await server.verify(request("VTCBench", ""))
    assert result.reward == 0
    judge.assert_not_awaited()


def test_multiselect_and_invalid_mcq_verdicts():
    task = {"options": {"A": "cat", "B": "dog"}, "answer": "B, A"}
    assert rule_score(task, "A, B") == 1
    assert rule_score(task, "B") == 0
    assert rule_score(task, "A, A, B") == 0
    assert judge_score(task, "AB") == 1
    assert judge_score(task, "B") == 0
    assert judge_score(task, "A, B") is None
    assert judge_score(task, "ab") is None


@pytest.mark.parametrize(
    "answer,prediction,options,expected",
    [
        ("3", "maybe 30 or 2. Answer: 7", {}, 0),
        ("3", "maybe 3 or 2.\nAnswer: 7", {}, 0),
        ("no", "I do not know.\nAnswer: yes", {}, 0),
        ("A", "A man is shown.\nAnswer: B", {"A": "man", "B": "woman"}, 0),
        ("B", "A man is shown.\nAnswer: B", {"A": "man", "B": "woman"}, 1),
        ("New York", "Reasoning\nAnswer: new york", {}, 1),
    ],
)
def test_reasoning_cannot_override_final_answer(answer, prediction, options, expected):
    assert rule_score({"answer": answer, "options": options}, prediction) == expected


def test_judge_mcq_verdict_tolerates_surrounding_whitespace():
    assert judge_score({"options": {"A": "cat"}, "answer": "A"}, " A\n") == 1
