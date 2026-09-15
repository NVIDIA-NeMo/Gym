# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.judge import JudgeError
from resources_servers.job_bench import app


@pytest.mark.parametrize(
    "response",
    ['```json\n{"rubric_passed": true}\n```', 'prefix {"rubric_passed": true} suffix'],
)
def test_parse_judge_json_fallbacks(response: str) -> None:
    parsed, _ = app.judge.parse_judge_json(response)

    assert parsed["rubric_passed"] is True


def test_parse_judge_json_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        app.judge.parse_judge_json("not json")


@pytest.mark.parametrize(("passed", "expected_score"), [(True, 3), (False, 0)])
def test_judge_rubric_scores_model_verdict(monkeypatch, passed: bool, expected_score: int) -> None:
    content = json.dumps(
        {
            "criteria_results": [{"passed": passed, "reasoning": "reason", "evidence": "evidence"}],
            "rubric_passed": passed,
            "overall_reasoning": "reason",
        }
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    completions = SimpleNamespace(create=lambda **_kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(app.judge, "get_openai_client", lambda *_args: client)

    result, debug = app.judge.judge_rubric(
        0,
        {"rubric": "required", "weight": 3, "criterion": ["first", "second"]},
        "answer",
        "judge-model",
        "https://judge.example",
        "test",
        max_retries=1,
    )

    assert result["result"]["score"] == expected_score
    assert result["result"]["criteria_results"][1]["passed"] is False
    assert debug["api_exit_code"] == 0


def test_judge_uses_official_weighted_score(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "answer.txt").write_text("answer", encoding="utf-8")
    rubrics_file = tmp_path / "RUBRICS.json"
    rubrics_file.write_text(
        json.dumps({"rubrics": [{"rubric": "first", "weight": 3}, {"rubric": "second", "weight": 1}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(app.judge, "extract_all_file_contents", lambda _: "answer")
    monkeypatch.setattr(app.judge, "collect_image_attachments", lambda _: [])

    def fake_judge(index, rubric, *_args):
        passed = index == 0
        return {
            "index": index,
            "weight": rubric["weight"],
            "result": {"passed": passed, "score": rubric["weight"] if passed else 0},
        }, {"api_exit_code": 0}

    monkeypatch.setattr(app.judge, "judge_rubric", fake_judge)
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="grok-4.3",
            judge_base_url="https://api.x.ai/v1",
            judge_api_key="test",
            max_judge_workers=2,
        )
    )

    scorecard, results = server._judge(output_dir, rubrics_file)

    assert scorecard["normalized_score"] == 0.75
    assert scorecard["passed_count"] == 1
    assert len(results) == 2


def test_empty_output_fails_without_calling_judge(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    rubrics_file = tmp_path / "RUBRICS.json"
    rubrics_file.write_text(json.dumps({"rubrics": [{"rubric": "required", "weight": 5}]}), encoding="utf-8")
    monkeypatch.setattr(app.judge, "judge_rubric", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(max_judge_workers=1)
    )

    scorecard, _ = server._judge(output_dir, rubrics_file)

    assert scorecard["normalized_score"] == 0


def test_judge_failure_is_not_scored(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "answer.txt").write_text("answer", encoding="utf-8")
    rubrics_file = tmp_path / "RUBRICS.json"
    rubrics_file.write_text(json.dumps({"rubrics": [{"rubric": "required", "weight": 1}]}), encoding="utf-8")
    monkeypatch.setattr(app.judge, "extract_all_file_contents", lambda _: "answer")
    monkeypatch.setattr(app.judge, "collect_image_attachments", lambda _: [])
    monkeypatch.setattr(
        app.judge,
        "judge_rubric",
        lambda *_args: ({}, {"api_exit_code": 2, "error": "judge unavailable"}),
    )
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="grok-4.3",
            judge_base_url="https://api.x.ai/v1",
            judge_api_key="test",
            max_judge_workers=1,
        )
    )

    with pytest.raises(JudgeError, match="judge unavailable"):
        server._judge(output_dir, rubrics_file)


def test_unparseable_judge_response_is_scored_as_failure(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "answer.txt").write_text("answer", encoding="utf-8")
    rubrics_file = tmp_path / "RUBRICS.json"
    rubric = {"rubric": "required", "weight": 1}
    rubrics_file.write_text(json.dumps({"rubrics": [rubric]}), encoding="utf-8")
    monkeypatch.setattr(app.judge, "extract_all_file_contents", lambda _: "answer")
    monkeypatch.setattr(app.judge, "collect_image_attachments", lambda _: [])
    monkeypatch.setattr(
        app.judge,
        "judge_rubric",
        lambda *_args: (
            app.judge.build_failed_rubric_result(0, rubric, "unparseable response"),
            {"api_exit_code": 1, "error": "unparseable response"},
        ),
    )
    server = app.JobBenchResourcesServer.model_construct(
        config=app.JobBenchConfig.model_construct(
            judge_model="grok-4.3",
            judge_base_url="https://api.x.ai/v1",
            judge_api_key="test",
            max_judge_workers=1,
        )
    )

    scorecard, _ = server._judge(output_dir, rubrics_file)

    assert scorecard["normalized_score"] == 0
