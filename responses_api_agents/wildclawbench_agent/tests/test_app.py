# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise source binding, reward admission and the actual subprocess boundary offline."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.client.server_client import ServerClient
from responses_api_agents.wildclawbench_agent import data
from responses_api_agents.wildclawbench_agent.app import WildClawAgent, WildClawConfig, WildClawRunRequest
from responses_api_agents.wildclawbench_agent.metrics import benchmark_metrics
from responses_api_agents.wildclawbench_agent.worker import grader_failures, read_events
from responses_api_models.vllm_model.wildclawbench import restore_reasoning


@pytest.fixture
def fixture_source(tmp_path, monkeypatch):
    task = tmp_path / "tasks/category/task.md"
    task.parent.mkdir(parents=True)
    task.write_text("""---
id: task
modality: pure-text
timeout_seconds: 300
---
## Prompt
Preserve this exact user instruction.
## Automated Checks
SECRET GRADING RUBRIC
## Workspace Path
```
workspace/category/task
```
""")
    workspace = tmp_path / "workspace/category/task"
    (workspace / "exec").mkdir(parents=True)
    (workspace / "exec/input.txt").write_text("input data")
    (workspace / "gt").mkdir()
    (workspace / "gt/answer.txt").write_text("SECRET ANSWER")
    pin = tmp_path / "source-pin.json"
    pin.write_text(
        json.dumps(
            dict(
                revision="a" * 40,
                tasks={"task": "tasks/category/task.md"},
                files={"tasks/category/task.md": data.digest(task)},
            )
        )
    )
    monkeypatch.setattr(data, "PIN_PATH", pin)
    return tmp_path


def test_preparation_retains_prompt_but_hides_rubric_and_answer(fixture_source):
    (row,) = data.prepare_rows(fixture_source, ["task"])
    visible = json.dumps(row["responses_create_params"])
    assert row["responses_create_params"]["input"][0]["content"] == "Preserve this exact user instruction."
    assert "SECRET" not in json.dumps(row)
    assert "input data" not in visible
    original = row["verifier_metadata"]["workspace_sha256"]
    (fixture_source / "workspace/category/task/exec/input.txt").write_text("changed input")
    assert data.prepare_rows(fixture_source, ["task"])[0]["verifier_metadata"]["workspace_sha256"] != original


def test_preparation_rejects_source_drift_and_missing_assets(fixture_source):
    source = fixture_source / "tasks/category/task.md"
    source.write_text(source.read_text() + "modified grader")
    with pytest.raises(ValueError, match="source pin mismatch"):
        data.prepare_rows(fixture_source, ["task"])
    with pytest.raises(FileNotFoundError, match="Missing prepared"):
        data.workspace_digest(fixture_source, "workspace/missing")


def test_workspace_symlink_cannot_import_external_assets(fixture_source, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (fixture_source / "workspace/category/task/exec/escape").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        data.workspace_digest(fixture_source, "workspace/category/task")


def test_shards_cover_all_60_once_and_reject_invalid_selections():
    shards = [data.selected_tasks("all", shard_count=16, shard_index=i) for i in range(16)]
    tasks = [task for shard in shards for task in shard]
    assert len(tasks) == len(set(tasks)) == 60
    assert set(tasks) == data.source_pin()["tasks"].keys()
    with pytest.raises(ValueError):
        data.selected_tasks(tasks[0] + "," + tasks[0])
    with pytest.raises(ValueError):
        data.selected_tasks("../untrusted")


def scored_rows():
    pin = data.source_pin()
    return [
        dict(
            task_id=task,
            source_revision=pin["revision"],
            task_sha256=pin["files"][name],
            reward=0.0,
            native_reward=0.0,
            mask_sample=False,
        )
        for task, name in pin["tasks"].items()
    ]


def test_valid_zero_is_scored_but_incomplete_or_masked_suite_has_no_headline():
    rows = scored_rows()
    assert benchmark_metrics(rows)["wildclawbench/overall_score"] == 0
    assert "wildclawbench/overall_score" not in benchmark_metrics(rows[:-1])
    rows[-1]["mask_sample"] = True
    assert "wildclawbench/overall_score" not in benchmark_metrics(rows)
    assert benchmark_metrics(rows)["wildclawbench/num_masked"] == 1


def test_metrics_reject_duplicates_reward_changes_and_mixed_sources():
    rows = scored_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark_metrics([*rows, rows[0]])
    rows[0]["reward"] = 0.5
    with pytest.raises(ValueError, match="original grader"):
        benchmark_metrics(rows)
    rows[0]["reward"] = 0
    rows[0]["source_revision"] = "b" * 40
    with pytest.raises(ValueError, match="Mixed"):
        benchmark_metrics(rows)


def test_task_mean_is_weighted_by_tasks_not_shard_means():
    rows = scored_rows()
    rows[0].update(reward=1.0, native_reward=1.0)
    assert benchmark_metrics(rows)["wildclawbench/overall_score"] == pytest.approx(1 / 60)


def test_grader_failures_keep_missing_deliverable_zero_and_mask_nested_judge_errors():
    assert not grader_failures(dict(overall_score=0, error="results.md not found"))
    assert grader_failures(dict(overall_score=0, llm_error="HTTP 429")) == ["llm_error"]
    assert grader_failures(dict(section=dict(judge_error="timeout"))) == ["section.judge_error"]
    assert not grader_failures(dict(overall_score=0, trap_errors=3))
    assert not grader_failures(dict(overall_score=0, error="screenshot.png too small, likely invalid"))
    assert not grader_failures(dict(overall_score=0, error="invalid json on line 2: malformed output"))
    assert grader_failures(dict(overall_score=0, error="ground_truth.json not found: /tmp_workspace/gt")) == ["error"]
    assert grader_failures(dict(llm_items=dict(answer=dict(reason="llm_judge_failed")))) == ["llm_items.answer.reason"]


def test_reasoning_fields_keep_tools_and_allow_openclaw_replay():
    payload = dict(
        choices=[dict(message=dict(content="<think>\nanalysis\n</think>\nanswer", tool_calls=[{"id": "x"}]))]
    )
    message = restore_reasoning(payload)["choices"][0]["message"]
    assert message == dict(
        reasoning="\nanalysis\n", reasoning_content="\nanalysis\n", content="\nanswer", tool_calls=[{"id": "x"}]
    )
    plain = dict(choices=[dict(message=dict(content="ordinary answer"))])
    assert restore_reasoning(plain) == plain


def test_jsonl_unicode_tool_content_roundtrips_and_corruption_is_not_dropped(tmp_path):
    path = tmp_path / "chat.jsonl"
    event = dict(type="message", message=dict(role="toolResult", content="one\u2028two\u0085three"))
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n")
    assert read_events(path) == [event]
    path.write_text(path.read_text() + "broken\n")
    with pytest.raises(json.JSONDecodeError):
        read_events(path)


@pytest.mark.parametrize("daemon_error,expected_code", [(False, 0), (True, 1)])
def test_real_cleanup_accepts_already_removed_container_but_not_daemon_failure(tmp_path, daemon_error, expected_code):
    import os

    docker = tmp_path / "docker"
    message = (
        "Cannot connect to the Docker daemon"
        if daemon_error
        else "Error response from daemon: No such container: episode"
    )
    docker.write_text(f"#!/bin/sh\nprintf '%s\\n' '{message}' >&2\nexit 1\n")
    docker.chmod(0o700)
    config = tmp_path / "worker.json"
    config.write_text(json.dumps(dict(output=str(tmp_path / "episode"))))
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [sys.executable, "-m", "responses_api_agents.wildclawbench_agent.worker", str(config), "--cleanup"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == expected_code
    if daemon_error:
        assert "Sandbox cleanup failed" in result.stderr


def make_agent(tmp_path, worker_python):
    config = WildClawConfig(
        name="wildclaw_agent",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        source_root=str(tmp_path),
        worker_python=worker_python,
        output_root=str(tmp_path / "episodes"),
        model_name="policy",
        model_server=dict(type="responses_api_models", name="policy_model"),
        sandbox_runtime="docker",
    )
    return WildClawAgent(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.asyncio
async def test_real_subprocess_retains_artifacts_and_retry_does_not_repeat_execution(tmp_path, monkeypatch):
    script = tmp_path / "worker-python"
    script.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys
from pathlib import Path
config=json.loads(Path(sys.argv[3]).read_text())
output=Path(config['output'])
if '--cleanup' in sys.argv:
    (output/'cleaned').write_text('yes')
else:
    (output/'executed').open('x').write('one execution only')
    result=dict(**config['metadata'],reward=0.0,native_reward=0.0,mask_sample=False,errors=[],artifact_dir=str(output),events=[
        {'type':'message','id':'a','message':{'role':'assistant','content':[{'type':'text','text':'No deliverable'}]}}
    ])
    (output/'result.json').write_text(json.dumps(result))
"""
    )
    script.chmod(0o700)
    agent = make_agent(tmp_path, str(script))
    monkeypatch.setattr(
        WildClawAgent, "resolve_model_base_url", lambda self, name, rollout: f"http://localhost/rollouts/{rollout}/v1"
    )
    request = WildClawRunRequest.model_validate(
        dict(
            _ng_rollout_id="rollout-a",
            responses_create_params=dict(input=[dict(role="user", content="task")]),
            verifier_metadata=dict(
                task_id="task", task_sha256="a" * 64, source_revision="b" * 40, timeout_seconds=300
            ),
        )
    )
    first, duplicate = await asyncio.gather(agent.run(request), agent.run(request))
    assert first.reward == duplicate.reward == 0
    assert not first.mask_sample
    output = Path(first.artifact_dir)
    assert (output / "cleaned").read_text() == "yes"
    assert (output / "executed").read_text() == "one execution only"
    saved = json.loads((output / "worker.json").read_text())
    assert saved["policy_base_url"] == "http://localhost/rollouts/rollout-a/v1"
    assert (await agent.run(request)).reward == 0
    changed = request.model_copy(deep=True)
    changed.verifier_metadata["task_id"] = "other-task"
    with pytest.raises(HTTPException) as exc:
        await agent.run(changed)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_missing_rollout_id_is_rejected_before_execution(tmp_path):
    agent = make_agent(tmp_path, sys.executable)
    body = WildClawRunRequest.model_validate(dict(responses_create_params=dict(input="test"), verifier_metadata={}))
    with pytest.raises(HTTPException) as exc:
        await agent.run(body)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_cancellation_stops_the_worker_and_runs_scoped_cleanup(tmp_path, monkeypatch):
    script = tmp_path / "slow-worker-python"
    script.write_text(
        f"#!{sys.executable}\n"
        + """import json,os,sys,time
from pathlib import Path
config=json.loads(Path(sys.argv[3]).read_text())
output=Path(config['output'])
if '--cleanup' in sys.argv:
    (output/'cleaned').write_text('yes')
else:
    (output/'started').write_text(str(os.getpid()))
    time.sleep(120)
"""
    )
    script.chmod(0o700)
    agent = make_agent(tmp_path, str(script))
    monkeypatch.setattr(WildClawAgent, "resolve_model_base_url", lambda *args: "http://localhost/v1")
    body = WildClawRunRequest.model_validate(
        dict(
            _ng_rollout_id="cancelled",
            responses_create_params=dict(input=[dict(role="user", content="task")]),
            verifier_metadata=dict(task_id="task", timeout_seconds=300),
        )
    )
    running = asyncio.create_task(agent.run(body))
    started = None
    for _ in range(100):
        matches = list((tmp_path / "episodes").glob("*/started"))
        if matches:
            started = matches[0]
            break
        await asyncio.sleep(0.02)
    assert started is not None
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert (started.parent / "cleaned").read_text() == "yes"
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(int(started.read_text()), 0)
