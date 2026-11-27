# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from resources_servers.legal_agent_bench import environment_verifier, verifier


@pytest.fixture
def grading(tmp_path, monkeypatch):
    # Keep the actual parser, grader, and report writers. Redirect only the
    # sandbox's /tests and /logs paths, and replace external judge calls.
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "task.json").write_text(
        json.dumps(
            {
                "title": "Save the answer",
                "instructions": "Write answer.txt containing the requested text.",
                "criteria": [{"id": "one", "title": "Answer", "match_criteria": "Contains the requested text."}],
                "metadata": {"lab_source_revision": "pinned-source"},
            }
        )
    )
    logs = tmp_path / "logs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(environment_verifier, "__file__", str(tests / "verifier.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "legal_agent_bench_verify", verifier)
    for name in os.environ:
        if name.startswith("LAB_JUDGE") or name == "LEGAL_AGENT_BENCH_REWARD_MODE":
            monkeypatch.delenv(name)
    monkeypatch.setenv("LAB_JUDGE_PARALLELISM", "1")
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": "test-judge"}]))
    parser = verifier.build_parser()
    invocations = []

    def parse_args(argv):
        args = parser.parse_args(argv)
        assert Path(args.task_json) == tests / "task.json"
        assert args.verifier_dir == "/logs/verifier"
        assert args.reward_json == "/logs/verifier/reward.json"
        args.verifier_dir = str(logs)
        args.reward_json = str(logs / "reward.json")
        invocations.append(args)
        return args

    monkeypatch.setattr(verifier, "build_parser", lambda: SimpleNamespace(parse_args=parse_args))
    prompts = []

    class Judge:
        base_url = "http://judge.test/v1"

        def __init__(self, **kwargs):
            pass

        def evaluate_prompt(self, prompt, *args, **kwargs):
            prompts.append(prompt)
            return {"verdict": "pass" if "requested answer text" in prompt else "fail", "reasoning": "Checked text"}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    return SimpleNamespace(workspace=workspace, logs=logs, prompts=prompts, invocations=invocations)


def test_verifier_copies_deliverables_and_uses_server_metadata(grading):
    output = grading.workspace / "output"
    output.mkdir()
    answer = output / "answer.txt"
    answer.write_text("requested answer text")
    for name in ("config.json", "metrics.json"):
        (grading.workspace / name).write_text(json.dumps({"task": "forged", "gym_commit": "forged", "usage": {}}))
    response = {
        "model": "policy-model",
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "metadata": {
            "lab_finish_reason": "no_tool_calls",
            "lab_harness_provenance": json.dumps(
                {"agent_id": "legal_agent_bench_native", "harness_pin": {"max_turns": 60}, "gym_commit": "trusted"}
            ),
        },
    }
    result = environment_verifier.verify(grading.workspace, {"instance_id": "legal_agent_bench::one"}, response)
    assert result["reward"] == 1 and result["judge_error_count"] == 0
    assert len(grading.prompts) == 1 and "requested answer text" in grading.prompts[0]
    scores = json.loads((grading.logs / "scores.json").read_text())
    assert scores["task"] == "legal_agent_bench::one"
    assert scores["gym_commit"] == "trusted" and scores["harness_pin"] == {"max_turns": 60}
    assert scores["source_revision"] == "pinned-source"
    assert scores["agent"]["model"] == "policy-model"
    assert scores["completion"] == {"status": "completed", "finish_reason": "no_tool_calls"}
    assert scores["usage"] == response["usage"]
    assert answer.read_text() == "requested answer text"
    assert not Path(grading.invocations[0].run_dir).exists()


def test_missing_outputs_are_graded_as_an_empty_submission(grading):
    result = environment_verifier.verify(grading.workspace, {"instance_id": "legal_agent_bench::one"}, None)
    assert result["reward"] == 0 and result["judge_error_count"] == 0
    assert not result.get("verifier_error")
    assert len(grading.prompts) == 1 and "No agent output found" in grading.prompts[0]
    assert (grading.logs / "report.html").is_file()
    assert not Path(grading.invocations[0].run_dir).exists()


def test_verifier_failure_retains_reports_and_returns_failure_marker(grading, monkeypatch):
    def unavailable_judge(**kwargs):
        raise RuntimeError("judge initialization failed")

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", unavailable_judge)
    result = environment_verifier.verify(grading.workspace, {"instance_id": "legal_agent_bench::one"}, None)
    assert result["reward"] == 0 and result["verifier_error"] == 1
    assert result["judge_error_count"] == 1
    error = json.loads((grading.logs / "error.json").read_text())
    assert error == {"error": "judge initialization failed", "error_type": "RuntimeError"}
    assert (grading.logs / "scores.json").is_file() and (grading.logs / "report.html").is_file()
    assert "judge initialization failed" in (grading.logs / "transcript.jsonl").read_text()
    assert not grading.prompts
    assert not Path(grading.invocations[0].run_dir).exists()
