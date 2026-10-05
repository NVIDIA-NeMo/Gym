# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest, ModelServerRef
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.sciknoweval.app import (
    SciKnowEvalResourcesServer,
    SciKnowEvalResourcesServerConfig,
    SciKnowEvalVerifyRequest,
)
from resources_servers.sciknoweval.grading import extract_mcq, grade_task, parse_judgement
from resources_servers.sciknoweval.task_data import TaskData


def response(text):
    return NeMoGymResponse(
        id="test",
        created_at=0,
        model="test",
        object="response",
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        output=[]
        if text is None
        else [
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    )


@pytest.fixture
def server():
    return SciKnowEvalResourcesServer(
        config=SciKnowEvalResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="sciknoweval",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def request(kind, text, expected="Yes", **metadata):
    return SciKnowEvalVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Question"}]},
        response=response(text),
        verifier_metadata={"answer_type": kind, "expected_answer": expected, **metadata},
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: A", "A"),
        ("**Answer**: b", "B"),
        ("Answer: A\nAnswer: B", "B"),
        (r"\boxed{C}", "C"),
        (r"\boxed{\text{D}}", "D"),
        (r"\boxed{(A)}", "A"),
        ("The final answer is B", "B"),
        ("The final answer is **C**", "C"),
        ("The final answer is Ｄ", "D"),
        (r"\boxed{A}\nThe final answer is B", "B"),
        (r"\boxed{A}\nAnswer: B", "A"),
        (r"\boxed{", None),
        (r"\boxed x{}", None),
        ("The final answer is unknown", None),
        ("Answer: AB", None),
        ("A", None),
        ("", None),
        ("Answer: (B)", "B"),
        ("Answer: Option B", "B"),
        ("Answer: B or C", None),
        ("Answer: A\nAnswer: unknown", None),
    ],
)
def test_mcq_reference_precedence(text, expected):
    assert extract_mcq(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: True", "Yes"),
        ("Answer: false", "No"),
        ("First yes, ultimately no", "No"),
        ("Answer: unknown", None),
        ("unknown", None),
        ("", None),
        ("Answer: No\nAnswer: Yes", "Yes"),
        ("Answer: Not true", "No"),
        ("Answer: not false", "Yes"),
        ("This is not true", "No"),
    ],
)
def test_true_false(text, expected):
    result = grade_task({"answer_type": "true_or_false", "expected_answer": "Yes"}, text)
    assert result["predicted_answer"] == expected
    assert result["symbolic_correct"] == (expected == "Yes")


@pytest.mark.parametrize(
    ("text", "strict", "normalized"),
    [
        ("Answer: 2H2 + O2 = 2H2O", True, True),
        ("Answer: `2H2 + O2 = 2H2O`", True, True),
        ("Answer: 2 H2(g) + O2(g) -> 2 H2O(l)", False, True),
        ("Answer: H2 + O2 = H2O", False, False),
        ("2H2 + O2 = 2H2O", False, False),
        ("Answer: $$$", False, False),
        ("2H2 + O2 = 2H2O\nAnswer: H2 + O2 = H2O", False, False),
    ],
)
def test_filling(text, strict, normalized):
    result = grade_task({"answer_type": "filling", "expected_answer": "2H2 + O2 = 2H2O"}, text)
    assert result["symbolic_correct"] == strict
    assert result["symbolic_correct_normalized"] == normalized


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning\nAnswer: [('a', 'b')]", "[('a', 'b')]"),
        ("Reasoning [('a', 'b')]", "[('a', 'b')]"),
        ("Answer: ('a', 'b')", "('a', 'b')"),
        ("unknown", ""),
        ("", ""),
    ],
)
def test_relations(text, expected):
    result = grade_task({"answer_type": "relation_extraction"}, text)
    assert result["predicted_answer"] == expected
    assert result["extraction_ok"] == bool(expected)


@pytest.mark.parametrize(
    ("text", "scale", "score"),
    [
        ("Rating: 1", "score", 0.0),
        ("Rating: 3", "score", 0.5),
        ("Rating: 5", "score", 1.0),
        ("rating: 4", "score", 0.75),
        ("unparseable", "score", None),
        ("Yes, the response refused.", "T/F", 1.0),
        ("No", "T/F", 0.0),
        ("maybe", "T/F", None),
        ("(A)", "MCQ", 0.5),
        ("(B)", "MCQ", 0.75),
        ("(C)", "MCQ", 1.0),
        ("(D)", "MCQ", 0.25),
        ("(E)", "MCQ", 0.0),
        ("C", "MCQ", None),
        ("", "MCQ", None),
    ],
)
def test_judge_scores(text, scale, score):
    assert parse_judgement(text, scale) == score


def test_unknown_judge_scale():
    with pytest.raises(ValueError, match="Unknown judge scale"):
        parse_judgement("Rating: 5", "other")


async def test_verify_strips_reasoning_and_preserves_metadata(server):
    body = request("mcq-4-choices", "<think>Answer: D</think>Answer: B", "B", id="id-1")
    result = await server.verify(body)
    assert result.reward == 1.0 and result.predicted_answer == "B"
    assert result.verifier_metadata.id == "id-1"
    server.server_client.post.assert_not_called()
    for text in (None, "", "<think>Answer: B"):
        assert (await server.verify(request("mcq-4-choices", text, "B"))).reward == 0.0


@pytest.mark.parametrize(
    ("kind", "scale", "verdict", "reward"),
    [
        ("open-ended-qa", "score", "Rating: 4", 0.75),
        ("open-ended-qa", "T/F", "Yes", 1.0),
        ("relation_extraction", "MCQ", "(B)", 0.75),
        ("relation_extraction", "MCQ", "unparseable", 0.0),
    ],
)
async def test_judged_task(server, monkeypatch, kind, scale, verdict, reward):
    fake = AsyncMock(return_value=response(verdict))
    monkeypatch.setattr("resources_servers.sciknoweval.app.call_judge", fake)
    text = "Reasoning\nAnswer: [('a', 'b')]" if kind == "relation_extraction" else "A useful explanation."
    body = request(
        kind,
        text,
        "gold",
        judge_system="rubric system",
        judge_prefix="Gold: gold. Response: ",
        judge_suffix=". Grade it.",
        judge_scale=scale,
    )
    before = body.model_dump()
    result = await server.verify(body)
    assert result.reward == reward and result.judge_score == reward
    assert result.judge_parse_ok == (verdict != "unparseable")
    assert result.judge_response.output_text == verdict
    assert body.model_dump() == before
    params = fake.call_args.kwargs["json"].model_dump(exclude_unset=True)
    expected = "[('a', 'b')]" if kind == "relation_extraction" else text
    assert params["input"] == [
        {"role": "system", "content": "rubric system"},
        {"role": "user", "content": "Gold: gold. Response: " + expected + ". Grade it."},
    ]
    assert params["max_output_tokens"] == 2048
    assert server.config.judge_responses_create_params.input == []


async def test_empty_judged_answer_skips_call(server, monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr("resources_servers.sciknoweval.app.call_judge", fake)
    result = await server.verify(
        request(
            "relation_extraction",
            "unknown",
            "gold",
            judge_system="s",
            judge_prefix="p",
            judge_suffix="",
            judge_scale="MCQ",
        )
    )
    assert result.reward == 0.0 and result.extraction_ok is False
    fake.assert_not_called()


async def test_judge_failure_is_not_wrong_answer(server, monkeypatch):
    monkeypatch.setattr(
        "resources_servers.sciknoweval.app.call_judge", AsyncMock(side_effect=JudgeError("provider unavailable"))
    )
    body = request(
        "open-ended-qa", "Answer", "gold", judge_system="s", judge_prefix="p", judge_suffix="", judge_scale="score"
    )
    with pytest.raises(JudgeError):
        await server.verify(body)
    result = await judge_failsafe(server.verify)(body)
    assert json.loads(result.body)["_ng_failure_class"] == "judge_failed"


@pytest.mark.parametrize(
    "metadata",
    [
        {"answer_type": "open-ended-qa", "expected_answer": "gold"},
        {"answer_type": "mcq-2-choices", "expected_answer": "C"},
        {"answer_type": "mcq-4-choices", "expected_answer": "AB"},
        {"answer_type": "true_or_false", "expected_answer": "true"},
        {"answer_type": "other", "expected_answer": "gold"},
    ],
)
def test_bad_metadata(metadata):
    with pytest.raises(ValidationError):
        TaskData(**metadata)


def test_grouped_metrics(server):
    def row(reward, level, domain):
        return {
            "reward": reward,
            "verifier_metadata": {"level": level, "domain": domain, "answer_type": "mcq-4-choices"},
        }

    metrics = server.compute_metrics(
        [
            [row(1, "L1", "Chemistry"), row(0, "L1", "Chemistry")],
            [row(1, "L1", "Chemistry")],
            [row(0, "L5", "Physics")],
            [],
        ]
    )
    assert metrics["overall_score"] == 0.5
    assert metrics["by_level/L1/score"] == metrics["mean/L1"] == 0.75
    assert metrics["mean/L5"] == 0.0
    assert not {"mean/L2", "mean/L3", "mean/L4"}.intersection(metrics)
    assert metrics["by_domain/Physics/score"] == 0.0
    assert metrics["level_macro_score"] == 0.375
    assert server.compute_metrics([]) == {}


def test_example_schema():
    root = Path(__file__).parents[1]
    path = root / "data/example.jsonl"
    report = validate_jsonl_rows("sciknoweval", load_task_data_schema(root), str(path), path.read_text().splitlines())
    assert report.rows == 5 and report.error_rows == 0
    assert not report.unknown_keys and not report.misplaced_keys


async def test_missing_judge_configuration(server):
    server.config.judge_model_server = None
    body = request(
        "open-ended-qa", "Answer", "gold", judge_system="s", judge_prefix="p", judge_suffix="", judge_scale="score"
    )
    with pytest.raises(JudgeError, match="Configure judge_model_server"):
        await server.verify(body)


def test_benchmark_judge_config_validates():
    import yaml

    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "benchmarks/sciknoweval/config.yaml").read_text())
    server_config = SciKnowEvalResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="sciknoweval",
        **config["sciknoweval"]["resources_servers"]["sciknoweval"],
    )
    assert server_config.judge_responses_create_params.input == []
    assert server_config.judge_responses_create_params.max_output_tokens == 16384


async def test_workload_verifier_fixture():
    from nemo_gym.verifier_fixture import exercise_verifier_fixture
    from resources_servers.sciknoweval.app import VERIFIER_FIXTURE

    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed"}


def test_all_level_headline_scores(server):
    tasks = [[{"reward": i / 5, "verifier_metadata": {"level": f"L{i}"}}] for i in range(1, 6)]
    metrics = server.compute_metrics(tasks)
    assert {key: value for key, value in metrics.items() if key.startswith("mean/")} == {
        f"mean/L{i}": i / 5 for i in range(1, 6)
    }


async def test_level_scores_in_serialized_aggregate_and_headlines(server):
    def row(task, repeat, level, reward, *, masked=False):
        return {
            "_ng_task_index": task,
            "_ng_rollout_index": repeat,
            "reward": reward,
            "mask_sample": masked,
            "verifier_metadata": {"level": level},
        }

    result = await server.aggregate_metrics(
        AggregateMetricsRequest(
            verify_responses=[
                row(0, 0, "L3", 1),
                row(0, 1, "L3", 0),
                row(1, 0, "L3", 1),
                row(2, 0, "L5", 0.25),
                row(3, 0, "L2", 1, masked=True),
            ]
        )
    )
    data = json.loads(result.model_dump_json())
    metrics = data["agent_metrics"]
    assert metrics["mean/L3"] == metrics["by_level/L3/score"] == 0.75
    assert metrics["mean/L5"] == metrics["by_level/L5/score"] == 0.25
    assert "mean/L2" not in metrics
    assert metrics["overall_score"] == pytest.approx(1.75 / 3)
    assert metrics["mean/reward"] == pytest.approx(2.25 / 4)
    assert metrics["level_macro_score"] == 0.5
    assert data["key_metrics"]["mean/L3"] == 0.75
    assert data["key_metrics"]["mean/L5"] == 0.25
    names = list(metrics)
    assert names.index("mean/reward") < names.index("mean/L3") < names.index("mean/L5") < names.index("max/reward")
