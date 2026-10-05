# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, judge_failsafe
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows
from resources_servers.chemeval.app import (
    ChemEvalResourcesServer,
    ChemEvalResourcesServerConfig,
    ChemEvalVerifyRequest,
    finite_json,
)
from resources_servers.chemeval.english_judge import build_judge_messages
from resources_servers.chemeval.loader import load_grader
from resources_servers.chemeval.task_data import TaskData


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
def server(tmp_path):
    # Synthetic scorer isolates the Gym adapter tests from separately licensed reference materials.
    path = tmp_path / "grading.py"
    path.write_text("def grade(sample):\n    sample.update(score=0.75, predicted_answer='4', abs_error=2.5)\n")
    return ChemEvalResourcesServer(
        config=ChemEvalResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="chemeval",
            grader_path=str(path),
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def request(text, family="mcq", **extra):
    return ChemEvalVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "Question"}]},
        response=response(text),
        verifier_metadata={
            "family": family,
            "expected_answer": "B",
            "task": "test_task",
            "level": "L1",
            "dimension": "Test",
            **extra,
        },
    )


def judged_request(text):
    return request(
        text,
        "judged",
        judge_protocol="english_v2",
        judge_question="Question",
        judge_rubric="short_answer",
        judge_scale="1-5",
    )


async def test_deterministic_score_and_metadata(server):
    body = request("<think>private reasoning</think>Answer: B")
    before = body.model_dump()
    original = server._grader.grade

    def grade(sample):
        assert sample["generation"] == "Answer: B"
        assert sample["expected_answer"] == "B"
        original(sample)

    server._grader.grade = grade
    result = await server.verify(body)
    assert result.reward == 0.75 and result.predicted_answer == "4"
    assert result.grading_metrics == {"abs_error": 2.5}
    assert body.model_dump() == before
    server.server_client.post.assert_not_called()


@pytest.mark.parametrize("text", [None, "", "   ", "<think>Answer: B", "<thinking>Answer: B</thinking>"])
async def test_empty_response(server, text):
    assert (await server.verify(request(text))).reward == 0.0
    assert (await server.verify(judged_request(text))).reward == 0.0


@pytest.mark.parametrize("exception", [ValueError, TypeError, OverflowError, ZeroDivisionError, RecursionError])
async def test_malformed_output(server, exception):
    def fail(sample):
        raise exception("bad literal")

    server._grader.grade = fail
    result = await server.verify(request("malformed output"))
    assert result.reward == 0 and result.scoring_error == f"{exception.__name__}: bad literal"


async def test_nonfinite_results(server):
    def grade(sample):
        sample.update(score=float("nan"), predicted_answer=None, abs_error=float("inf"))

    server._grader.grade = grade
    result = await server.verify(request("1e999"))
    assert result.reward == 0 and result.grading_metrics == {"abs_error": None}
    json.loads(result.model_dump_json())
    assert finite_json({"values": (1, float("-inf"))}) == {"values": [1, None]}


@pytest.mark.parametrize(
    "verdict,expected",
    [
        (
            json.dumps(
                {
                    "legacy_score_1_to_5": 5,
                    "outcome": {"score_0_to_1": 1},
                    "evidence": [{"candidate_quote": "Full answer"}],
                }
            ),
            1.0,
        ),
        (json.dumps({"legacy_score_1_to_5": 3, "outcome": {"score_0_to_1": 0.7}}), 0.5),
        ("unknown", 0.0),
    ],
)
async def test_exact_judge_prompt_and_partial_reward(server, monkeypatch, verdict, expected):
    fake = AsyncMock(return_value=response("<think>private</think>" + verdict))
    monkeypatch.setattr("resources_servers.chemeval.app.call_judge", fake)
    body = judged_request("discarded reasoning</thinking>Full answer\nFinal line")
    result = await server.verify(body)
    assert result.reward == expected and result.judge_score == expected
    assert result.judge_parse_ok == (verdict != "unknown")
    assert result.predicted_answer == "Full answer\nFinal line"
    assert result.judgement == verdict and result.judge_response is not None
    params = fake.call_args.kwargs["json"].model_dump(exclude_unset=True)
    assert params["input"] == build_judge_messages(
        rubric="short_answer", question="Question", candidate="Full answer\nFinal line", reference="B"
    )
    assert result.judge_v2 == (json.loads(verdict) if verdict != "unknown" else None)
    assert body.responses_create_params.input[0].content == "Question"
    assert params["temperature"] == 0 and params["max_output_tokens"] == 16384
    assert fake.call_args.kwargs["server_name"] == "judge_model"
    assert server.config.judge_responses_create_params.input == []


async def test_judge_unavailable(server, monkeypatch):
    server.config.judge_model_server = None
    with pytest.raises(JudgeError, match="Configure judge_model_server"):
        await server.verify(judged_request("answer"))
    server.config.judge_model_server = ModelServerRef(type="responses_api_models", name="judge_model")
    monkeypatch.setattr("resources_servers.chemeval.app.call_judge", AsyncMock(side_effect=JudgeError("unavailable")))
    result = await judge_failsafe(server.verify)(judged_request("answer"))
    assert json.loads(result.body)["_ng_failure_class"] == "judge_failed"


@pytest.mark.parametrize(
    "metadata",
    [
        {"family": "unknown"},
        {"family": "regression"},
        {"family": "regression", "gold_span": 0},
        {"family": "regression", "gold_span": float("inf")},
        {"family": "judged"},
        {"family": "judged", "judge_prefix": "", "judge_suffix": "", "judge_scale": "invalid"},
        {"expected_answer": ""},
        {"level": "L5"},
    ],
)
def test_invalid_metadata(metadata):
    with pytest.raises(ValidationError):
        TaskData(
            **(
                {"family": "mcq", "expected_answer": "A", "task": "task", "level": "L1", "dimension": "Test"}
                | metadata
            )
        )


def test_task_macro_metrics(server):
    def row(reward, task, level):
        return {
            "reward": reward,
            "verifier_metadata": {"task": task, "level": level, "dimension": "Test", "family": "mcq"},
        }

    metrics = server.compute_metrics(
        [[row(1, "a", "L1"), row(0, "a", "L1")], [row(1, "a", "L1")], [row(0, "b", "L1")], [row(1, "c", "L4")], []]
    )
    assert metrics["by_task/a/score"] == 0.75
    assert metrics["by_level/L1/score"] == 0.375
    assert metrics["overall_score"] == pytest.approx(1.75 / 3)
    assert metrics["level_macro_score"] == 0.6875
    assert metrics["by_dimension/Test/score"] == metrics["overall_score"]
    assert metrics["num_tasks"] == 3
    assert server.compute_metrics([]) == {}


def test_examples_schema():
    root = Path(__file__).parents[1]
    path = root / "data/example.jsonl"
    report = validate_jsonl_rows("chemeval", load_task_data_schema(root), str(path), path.read_text().splitlines())
    assert report.rows == 5 and report.error_rows == 0
    assert not report.unknown_keys and not report.misplaced_keys


def test_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Restore the bundled grading.py"):
        load_grader(str(tmp_path / "missing.py"))
    invalid = tmp_path / "grader.py"
    invalid.write_text("grade = 0\n")
    with pytest.raises(ValueError, match="must export"):
        load_grader(str(invalid))
    invalid = tmp_path / "grader.txt"
    invalid.write_text("not python")
    with pytest.raises(ValueError, match="Cannot load"):
        load_grader(str(invalid))


def test_judged_empty_gold_is_valid():
    # Two released calculation questions have no gold text; V2 explicitly marks their references as unavailable.
    metadata = judged_request("answer").verifier_metadata.model_dump() | {"expected_answer": ""}
    assert TaskData(**metadata).expected_answer == ""


def test_benchmark_judge_config_validates():
    import yaml

    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "benchmarks/chemeval/config.yaml").read_text())
    server_config = ChemEvalResourcesServerConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="chemeval",
        **config["chemeval"]["resources_servers"]["chemeval"],
    )
    assert server_config.judge_responses_create_params.input == []
    assert server_config.judge_responses_create_params.max_output_tokens == 16384


async def test_workload_verifier_fixture():
    from nemo_gym.verifier_fixture import exercise_verifier_fixture
    from resources_servers.chemeval.app import VERIFIER_FIXTURE

    results = await exercise_verifier_fixture(
        VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="unknown"
    )
    assert {result.kind for result in results} >= {"full_reward", "zero_reward", "malformed"}
