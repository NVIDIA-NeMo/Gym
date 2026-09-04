# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.judge import JudgeError
from resources_servers.job_bench import app


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
