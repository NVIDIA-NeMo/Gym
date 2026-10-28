# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.judge import JudgeError
from nemo_gym.server_utils import ServerClient
from resources_servers.opencode_multimodal_benchmarks.agentx import AgentXConfig, AgentXServer
from resources_servers.opencode_multimodal_benchmarks.agentx_rubric import final_response, parse_judge_score
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request


@pytest.fixture
def server():
    return AgentXServer(
        config=AgentXConfig(
            host="127.0.0.1",
            port=12345,
            name="visual",
            entrypoint="agentx.py",
            judge_model="test-judge",
            judge_endpoint="https://judge.example/v1/chat/completions",
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate,score", [("red", 1.0), ("blue", 0.0), ("mostly red", 0.65)])
async def test_full_answer_and_justification(server, monkeypatch, candidate, score):
    judge = AsyncMock(return_value=(score, "raw justification"))
    monkeypatch.setattr(AgentXServer, "_judge_answer", judge)
    body = request("AgentX_GoalAcc", "<think>private</think>" + candidate)
    result = await server.verify(body)
    assert result.reward == score and result.extracted == candidate
    assert result.judge_raw == "raw justification"
    judge.assert_awaited_once_with(body.verifier_metadata, candidate)


@pytest.mark.asyncio
async def test_empty_is_wrong_and_judge_failure_retryable(server, monkeypatch):
    judge = AsyncMock(side_effect=JudgeError("unavailable"))
    monkeypatch.setattr(AgentXServer, "_judge_answer", judge)
    assert (await server.verify(request("AgentX_GoalAcc", ""))).reward == 0
    judge.assert_not_awaited()
    with pytest.raises(JudgeError):
        await server.verify(request("AgentX_GoalAcc", "red"))


def test_weighted_score_full_denominator(server):
    rows = [
        [
            {
                "verifier_metadata": {
                    "index": str(i),
                    "expected_samples": 605,
                    "modality": "Image" if i < 494 else "Video",
                },
                "reward": 1.0 if i < 494 else 0.0,
            }
        ]
        for i in range(605)
    ]
    assert server.compute_metrics(rows)["accuracy"] == 494 / 605
    assert "accuracy" not in server.compute_metrics(rows[:-1])
    with pytest.raises(ValueError):
        server.compute_metrics(rows + rows[:1])


@pytest.mark.parametrize(
    "raw,kind,expected",
    [
        ('{"Score": "1", "Justification": "yes"}', "objective", 1),
        ("```python\n{'Score': 0.4}\n```", "subjective", 0.4),
        ('{"Score": 0.4}', "objective", None),
        ('{"Score": 2}', "subjective", None),
        ("no score", "objective", None),
        ("Score: 10", "objective", None),
        ("Score: 100", "subjective", None),
        ("Score: 1e-3", "objective", None),
        ("Score: 1e-3", "subjective", 0.001),
        ("Score: 1e3", "subjective", None),
        ("Score: -0.5", "subjective", None),
        ("Score: .5", "subjective", 0.5),
        ("Score: 0.5oops", "subjective", None),
        ("Score: 1/2", "objective", None),
        ("Score: 1, Justification: correct", "objective", 1),
        ("Score: '0.5'", "subjective", 0.5),
        ("Score = 0", "objective", 0),
        ("Score 1", "objective", 1),
    ],
)
def test_parser(raw, kind, expected):
    assert parse_judge_score(raw, kind) == expected


def test_reference_never_becomes_policy_input():
    from benchmarks.opencode_multimodal_benchmarks.prepare import task_row

    task = {"prompt": "question", "images": [], "GT_json": "private reference"}
    row = task_row(task)
    assert "private reference" not in json.dumps(row["responses_create_params"])
    assert final_response("<think>unclosed") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,raw,expected", [(200, '{"Score": 0.65}', 0.65), (200, "invalid", None), (503, "", None)]
)
async def test_judge_transport(server, monkeypatch, status, raw, expected):
    from resources_servers.opencode_multimodal_benchmarks import agentx

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value={"choices": [{"message": {"content": raw}}]})
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(agentx, "request", transport)
    server.config.judge_attempts = 1
    task = {"GT_json": "reference", "query_type_raw": "Subjective", "query_type": "subjective"}
    if expected is None:
        with pytest.raises(JudgeError):
            await server._judge_answer(task, "candidate")
    else:
        assert await server._judge_answer(task, "candidate") == (expected, raw)
    payload = transport.call_args.kwargs["json"]
    assert payload["model"] == "test-judge"
    assert payload["max_tokens"] == 2048 and payload["temperature"] == 0
    assert payload["messages"][0]["role"] == "system"
    assert "candidate" in payload["messages"][1]["content"]


def test_preparation_checksum_and_smoke_scope(tmp_path, monkeypatch, agentx_bundle):
    import hashlib

    from benchmarks.opencode_multimodal_benchmarks.agentx_prepare import prepare

    path, media, _, rows = agentx_bundle
    monkeypatch.chdir(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = prepare(str(path), digest, limit="2", media_root=str(media))
    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(prepared) == 2 and prepared[0]["verifier_metadata"]["expected_samples"] == 2
    assert len(prepare(str(path), digest, limit="None", media_root=str(media)).read_text().splitlines()) == 605
    with pytest.raises(ValueError, match="checksum"):
        prepare(str(path), "wrong")


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", ["think", "thinking", "THINKING", 'thinking mode="private"'])
async def test_reasoning_cleanup_keeps_final_answer_and_saved_response(server, monkeypatch, tag):
    closing = tag.split()[0]
    text = f"<{tag}>private analysis</{closing}>Answer: red"
    judge = AsyncMock(return_value=(1.0, "correct"))
    monkeypatch.setattr(AgentXServer, "_judge_answer", judge)
    body = request("AgentX_GoalAcc", text)
    original = body.response.model_dump()
    result = await server.verify(body)
    assert result.extracted == "Answer: red"
    judge.assert_awaited_once_with(body.verifier_metadata, "Answer: red")
    assert body.response.model_dump() == result.response.model_dump() == original


@pytest.mark.asyncio
async def test_unclosed_thinking_is_empty_without_judging(server, monkeypatch):
    judge = AsyncMock()
    monkeypatch.setattr(AgentXServer, "_judge_answer", judge)
    result = await server.verify(request("AgentX_GoalAcc", "<thinking>unfinished private analysis"))
    assert result.reward == 0 and result.extracted == ""
    judge.assert_not_awaited()
