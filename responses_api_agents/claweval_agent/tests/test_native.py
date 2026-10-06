# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional contract tests against the supplied local Claw-Eval fork, without GPU/API calls."""

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.claweval.prepare import prepare_split
from responses_api_agents.claweval_agent.trajectory import trace_to_response
from responses_api_agents.claweval_agent.worker import run_trial


@pytest.fixture
def native_root(monkeypatch):
    value = os.environ.get("CLAW_EVAL_ROOT")
    if not value:
        pytest.skip("Set CLAW_EVAL_ROOT to test the native fork contract")
    root = Path(value)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "src"))
    return root


@pytest.mark.parametrize("split,count", [("general", 161), ("multimodal", 101), ("multi_turn", 38)])
def test_native_export_all_tasks(native_root, split, count, tmp_path):
    output = prepare_split(split, output=tmp_path / f"{split}.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == count
    assert len({row["verifier_metadata"]["task_id"] for row in rows}) == count
    for row in rows:
        assert set(row) == {"agent_ref", "responses_create_params", "verifier_metadata"}
        assert set(row["verifier_metadata"]) == {"task_id", "task_sha256", "split"}
        path = native_root / "tasks" / row["verifier_metadata"]["task_id"] / "task.yaml"
        assert row["verifier_metadata"]["task_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_loop_and_grader_preserve_score_and_grader_isolation(native_root, tmp_path, monkeypatch):
    from claw_eval.models.message import Message
    from claw_eval.models.trace import TokenUsage
    from evaluation import run_multimodal as native

    task_dir = tmp_path / "tasks/T001_test"
    task_dir.mkdir(parents=True)
    task_yaml = task_dir / "task.yaml"
    task_yaml.write_text(
        "task_id: T001_test\ntask_name: Native contract test\nprompt:\n  text: Solve\n"
        "sandbox_files: [input.txt]\nsandbox_grader_files: [secret.txt]\n"
    )
    (task_dir / "input.txt").write_text("public")
    (task_dir / "secret.txt").write_text("private-grader-data")
    (task_dir / "grader.py").write_text(
        "from claw_eval.graders.base import AbstractGrader\n"
        "from claw_eval.models.trace import DimensionScores\n"
        "class TestGrader(AbstractGrader):\n"
        "    def grade(self, messages, dispatches, task, *, audit_data=None, judge=None, **kwargs):\n"
        "        assert messages[-1].message.text == 'Answer'\n"
        "        return DimensionScores(completion=1.0, robustness=0.5, safety=1.0)\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  model_id: test-model\n  api_key: unused\njudge:\n  enabled: false\n")
    provider = SimpleNamespace(model_id="test-model", client=Mock())
    provider.chat = Mock(
        return_value=(Message(role="assistant", content="Answer"), TokenUsage(input_tokens=4, output_tokens=2))
    )
    monkeypatch.setattr(native, "build_provider", lambda cfg: provider)
    workspaces = []

    def sandbox(**kwargs):
        workspace = kwargs["workspace"]
        assert (workspace / "input.txt").read_text() == "public"
        assert not (workspace / "secret.txt").exists()
        workspaces.append(workspace)
        return nullcontext(SimpleNamespace(url="http://unused", instance_id="test-sandbox"))

    def snapshot(*args, **kwargs):
        assert (workspaces[0] / "secret.txt").read_text() == "private-grader-data"
        return {}

    monkeypatch.setattr(native, "RemoteSandbox", sandbox)
    monkeypatch.setattr(native, "_collect_env_snapshot", snapshot)
    result = run_trial(
        {
            "verifier_metadata": {
                "task_id": "T001_test",
                "task_sha256": hashlib.sha256(task_yaml.read_bytes()).hexdigest(),
            },
            "prompt": "Solve",
            "claweval_config": str(config),
            "seed": 1001,
            "fixture_root": str(tmp_path / "tasks"),
            "no_judge": True,
            "output_dir": str(tmp_path / "out"),
            "sandbox_image": "test.sqsh",
            "sandbox_dependencies": str(tmp_path),
            "sandbox_port": 18080,
            "sandbox_ready_timeout": 5,
        },
        tmp_path,
    )
    assert result["task_score"] == 0.9
    assert result["passed"] is True
    response = trace_to_response(Path(result["trace"]), "T001_test", "test-model", expected_score=0.9)
    assert response.output[-1].content[0].text == "Answer"
    assert response.usage.total_tokens == 6
    provider.client.close.assert_called_once()


def test_native_task_selection_and_unknown_id_are_atomic(native_root, tmp_path):
    from responses_api_agents.claweval_agent.worker import prepare

    output = tmp_path / "selected.jsonl"
    payload = {"split": "general", "task_ids": ["T002_email_triage"], "agent_name": "native", "output": str(output)}
    prepare(payload, native_root)
    original = output.read_bytes()
    row = json.loads(original)
    assert row["verifier_metadata"]["task_id"] == "T002_email_triage"
    assert row["agent_ref"]["name"] == "native"
    with pytest.raises(ValueError, match="Unknown task IDs"):
        prepare({**payload, "task_ids": ["missing_task"]}, native_root)
    assert output.read_bytes() == original


def test_worker_dispatches_preparation_and_rejects_unknown_operation(native_root, tmp_path, monkeypatch):
    import io
    import signal
    import sys

    from responses_api_agents.claweval_agent.worker import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    payload = {
        "operation": "prepare",
        "claweval_root": str(native_root),
        "split": "multi_turn",
        "agent_name": "native",
        "output": str(tmp_path / "tasks.jsonl"),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    main()
    assert len((tmp_path / "tasks.jsonl").read_text().splitlines()) == 38
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({**payload, "operation": "unknown"})))
    with pytest.raises(ValueError, match="Unknown operation"):
        main()


@pytest.mark.parametrize("problem", ["prompt", "missing_config", "missing_asset", "user_agent_key"])
def test_native_preflight_fails_before_any_model_request(native_root, tmp_path, problem):
    path = tmp_path / "tasks/T001_test/task.yaml"
    path.parent.mkdir(parents=True)
    task = "task_id: T001_test\ntask_name: Test\nprompt:\n  text: Solve\n"
    if problem == "missing_asset":
        task += "sandbox_files: [missing.txt]\n"
    if problem == "user_agent_key":
        task += "user_agent:\n  enabled: true\n  goal: Test\n"
    path.write_text(task)
    config = tmp_path / "config.yaml"
    if problem != "missing_config":
        config.write_text("model:\n  model_id: unused\njudge:\n  enabled: false\n")
    payload = {
        "verifier_metadata": {"task_id": "T001_test", "task_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "prompt": "Changed" if problem == "prompt" else "Solve",
        "claweval_config": "config.yaml",
        "seed": 1001,
    }
    expected = {
        "prompt": "does not match the native task",
        "missing_config": "config does not exist",
        "missing_asset": "Missing Claw-Eval assets",
        "user_agent_key": "user-agent model API key",
    }
    with pytest.raises((ValueError, FileNotFoundError), match=expected[problem]):
        run_trial(payload, tmp_path)
