# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise source binding, reward admission and the actual subprocess boundary offline."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from omegaconf import OmegaConf

from benchmarks.wildclawbench.merge import merge
from nemo_gym.client import ServerClient
from responses_api_agents.wildclawbench_agent import data
from responses_api_agents.wildclawbench_agent.app import WildClawAgent, WildClawConfig, WildClawRunRequest
from responses_api_agents.wildclawbench_agent.enroot_docker import (
    TrackedSubprocess,
    process_start,
    remove,
    stop_owned_groups,
)
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
            evaluation_protocol=dict(model_name="policy", context_window=131072, max_tokens=8192, thinking="high"),
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


def test_mixed_models_or_budgets_cannot_be_merged_into_one_score():
    rows = scored_rows()
    rows[0]["evaluation_protocol"]["model_name"] = "another-model"
    with pytest.raises(ValueError, match="Mixed evaluated models"):
        benchmark_metrics(rows)


def test_merger_checks_original_receipts_before_publishing(tmp_path):
    rows = scored_rows()
    for row in rows:
        artifact = tmp_path / row["task_id"]
        artifact.mkdir()
        row.update(artifact_dir=str(artifact), workspace_sha256="c" * 64)
        (artifact / "result.json").write_text(json.dumps(row))
    paths = [tmp_path / "shard-0.jsonl", tmp_path / "shard-1.jsonl"]
    for index, path in enumerate(paths):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows[index::2]))
    output = tmp_path / "metrics.json"
    assert merge(paths, output)["wildclawbench/overall_score"] == 0
    receipt = Path(rows[0]["artifact_dir"]) / "result.json"
    changed = json.loads(receipt.read_text())
    changed["reward"] = 0.75
    receipt.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Native/Gym receipt mismatch"):
        merge(paths, tmp_path / "must-not-publish.json")
    assert not (tmp_path / "must-not-publish.json").exists()


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
    with (
        patch(
            "responses_api_agents.wildclawbench_agent.app.get_global_config_dict",
            return_value=OmegaConf.create(
                {
                    "policy_model": {
                        "responses_api_models": {
                            "vllm_model": {
                                "entrypoint": "wildclawbench.py",
                                "uses_reasoning_parser": True,
                                "propagate_context_overflow_errors": True,
                            }
                        }
                    }
                }
            ),
        ),
        patch(
            "responses_api_agents.wildclawbench_agent.app.subprocess.run",
            return_value=SimpleNamespace(stdout="sha256:" + "a" * 64),
        ),
    ):
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
    result=dict(**config['metadata'],evaluation_protocol=config['evaluation_protocol'],reward=0.0,native_reward=0.0,mask_sample=False,errors=[],artifact_dir=str(output),events=[
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
            _ng_task_index=3,
            _ng_rollout_index=2,
            _ng_attempt_index=0,
            _ng_run_id="run-one",
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
    assert saved["policy_base_url"] == "http://localhost/rollouts/3-2/v1"
    assert (await agent.run(request)).reward == 0
    for changes in ({"_ng_run_id": "run-two"}, {"_ng_attempt_index": 1}):
        retried = request.model_copy(update=changes)
        result = await agent.run(retried)
        assert result.artifact_dir != first.artifact_dir
        assert Path(result.artifact_dir, "executed").exists()
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


def test_enroot_detached_warmups_are_owned_and_cleanup_is_idempotent(tmp_path, monkeypatch):
    rootfs = tmp_path / "data/episode"
    rootfs.mkdir(parents=True)
    state = tmp_path / "episode.json"
    state.write_text("{}")
    records = state.with_suffix(".processes.json")
    proxy = TrackedSubprocess(records)
    service = proxy.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    removed = []

    def native_remove(names):
        removed.extend(names)
        state.unlink()
        rootfs.rmdir()
        return 0

    shim = SimpleNamespace(state_path=lambda name: state, remove_container=native_remove)
    monkeypatch.setenv("ENROOT_DATA_PATH", str(tmp_path / "data"))
    try:
        assert json.loads(records.read_text())[0]["start"] == process_start(service.pid)
        assert remove(shim, "episode") == 0
        assert service.wait(timeout=5) < 0
        assert not records.exists()
        assert remove(shim, "episode") == 0
        assert removed == ["episode"]
    finally:
        if service.poll() is None:
            import os
            import signal

            os.killpg(service.pid, signal.SIGKILL)
            service.wait(timeout=5)


def test_enroot_cleanup_does_not_signal_a_reused_or_unowned_pid(tmp_path):
    service = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    records = tmp_path / "groups.json"
    records.write_text(json.dumps([dict(pid=service.pid, start="not-the-recorded-process")]))
    try:
        stop_owned_groups(records)
        assert service.poll() is None
    finally:
        service.kill()
        service.wait(timeout=5)


def test_enroot_warmup_cannot_outlive_a_failed_ownership_receipt(tmp_path, monkeypatch):
    import responses_api_agents.wildclawbench_agent.enroot_docker as bridge

    started = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(bridge.subprocess, "Popen", start)
    proxy = TrackedSubprocess(tmp_path / "absent-directory/groups.json")
    with pytest.raises(FileNotFoundError):
        proxy.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    assert started[0].poll() is not None


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


def test_prepare_requires_output_before_hashing(monkeypatch):
    from benchmarks.wildclawbench import prepare

    monkeypatch.delenv("WILDCLAW_DATASET_PATH", raising=False)
    hashed = MagicMock()
    monkeypatch.setattr(prepare, "prepare_rows", hashed)
    with pytest.raises(KeyError, match="WILDCLAW_DATASET_PATH"):
        prepare.prepare()
    hashed.assert_not_called()


def test_explicit_output_path_is_honored(fixture_source, monkeypatch):
    from benchmarks.wildclawbench.prepare import prepare

    monkeypatch.setenv("WILDCLAW_ROOT", str(fixture_source))
    monkeypatch.setenv("WILDCLAW_TASKS", "task")
    monkeypatch.delenv("WILDCLAW_DATASET_PATH", raising=False)
    target = fixture_source / "explicit.jsonl"
    assert prepare(output_fpath=target) == target
    assert json.loads(target.read_text())["verifier_metadata"]["task_id"] == "task"


def test_worker_environment_blocks_hidden_native_overrides(tmp_path, monkeypatch):
    from responses_api_agents.wildclawbench_agent.app import worker_environment

    agent = make_agent(tmp_path, sys.executable)
    for name in ("WILDCLAW_OPENCLAW_MODEL_PARAMS", "WILDCLAW_OPENCLAW_THINKING", "OPENCLAW_IMAGE_MODEL"):
        monkeypatch.setenv(name, "unrecorded override")
    monkeypatch.setenv("JUDGE_MODEL", "different-judge")
    monkeypatch.setenv("DOCKER_NETWORK_MODE", "host")
    env = worker_environment(agent.config, tmp_path)
    assert all(
        env[name] == ""
        for name in ("WILDCLAW_OPENCLAW_MODEL_PARAMS", "WILDCLAW_OPENCLAW_THINKING", "OPENCLAW_IMAGE_MODEL")
    )
    assert env["JUDGE_MODEL"] == agent.config.judge_model
    assert env["DOCKER_NETWORK_MODE"] == ""


@pytest.mark.parametrize("changed", ["judge_model", "sandbox_image"])
def test_merge_rejects_different_judge_or_image(changed):
    rows = scored_rows()
    for row in rows:
        row["evaluation_protocol"].update(judge_model="judge-a", sandbox_image="sha256:a")
    rows[-1]["evaluation_protocol"][changed] = "different"
    with pytest.raises(ValueError, match="protocol"):
        benchmark_metrics(rows)


def test_copy_guards_nested_symlinks_in_both_directions(tmp_path, monkeypatch):
    from responses_api_agents.wildclawbench_agent.enroot_docker import guard_copy_path

    root = tmp_path / "data/episode"
    root.mkdir(parents=True)
    outside = tmp_path / "host-secret"
    outside.write_text("unchanged")
    (root / "escape").symlink_to(outside)
    (root / "safe").mkdir()
    (root / "safe/nested").symlink_to(outside)
    shim = SimpleNamespace(
        state_path=lambda name: tmp_path / (name + ".json"),
        container_path=lambda name, value: root / value.lstrip("/"),
    )
    monkeypatch.setenv("ENROOT_DATA_PATH", str(tmp_path / "data"))
    for value in ("/escape", "/safe", "/escape/child"):
        with pytest.raises(ValueError, match="escapes"):
            guard_copy_path(shim, "episode", value)
    guard_copy_path(shim, "episode", "/normal/new")
    assert outside.read_text() == "unchanged"


@pytest.mark.parametrize("unavailable_task", [False, True])
def test_process_marker_cleans_setsid_descendant_but_not_other_process(tmp_path, monkeypatch, unavailable_task):
    import errno
    import os

    from responses_api_agents.wildclawbench_agent.enroot_docker import stop_marked_processes

    marker = "wildclaw-test-" + tmp_path.name
    marked = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        env=dict(os.environ, WILDCLAW_CONTAINER=marker),
    )
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    if unavailable_task:
        pidfd_open = os.pidfd_open
        iterdir = Path.iterdir
        invalid_pid = 2147483647

        def open_task(pid):
            if pid == invalid_pid:
                raise OSError(errno.EINVAL, "Task cannot be opened as a PID handle")
            return pidfd_open(pid)

        def entries(path):
            if path == Path("/proc"):
                return iter(Path(f"/proc/{pid}") for pid in (invalid_pid, marked.pid, other.pid))
            return iterdir(path)

        monkeypatch.setattr(os, "pidfd_open", open_task)
        monkeypatch.setattr(Path, "iterdir", entries)
    try:
        stop_marked_processes(marker)
        assert marked.wait(timeout=5) < 0
        assert other.poll() is None
    finally:
        for process in (marked, other):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_process_marker_does_not_hide_pidfd_resource_exhaustion(monkeypatch):
    import errno
    import os

    from responses_api_agents.wildclawbench_agent.enroot_docker import stop_marked_processes

    def exhausted(pid):
        raise OSError(errno.EMFILE, "No PID handles available")

    monkeypatch.setattr(Path, "iterdir", lambda path: iter([Path("/proc/2147483647")]))
    monkeypatch.setattr(os, "pidfd_open", exhausted)
    with pytest.raises(OSError) as exc:
        stop_marked_processes("unused")
    assert exc.value.errno == errno.EMFILE


@pytest.mark.parametrize("existing,contents", [(False, False), (True, False), (True, True)])
def test_container_copy_preserves_native_directory_layout_and_executable_modes(tmp_path, existing, contents):
    from responses_api_agents.wildclawbench_agent.enroot_docker import copy_path

    source = tmp_path / "source"
    source.mkdir()
    executable = source / "run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    destination = tmp_path / "destination"
    if existing:
        destination.mkdir()
    assert copy_path(source, destination, contents) == 0
    copied = destination / ("source/run.sh" if existing and not contents else "run.sh")
    assert copied.read_bytes() == executable.read_bytes()
    assert copied.stat().st_mode & 0o777 == 0o755


def test_container_copy_works_when_destination_cannot_store_host_acls(tmp_path, monkeypatch):
    import errno
    import os

    from responses_api_agents.wildclawbench_agent.enroot_docker import copy_path

    source = tmp_path / "source"
    source.mkdir()
    (source / "input.txt").write_text("unchanged task bytes")
    attempts = []
    monkeypatch.setattr(os, "listxattr", lambda *args, **kwargs: ["system.posix_acl_access"])
    monkeypatch.setattr(os, "getxattr", lambda *args, **kwargs: b"host-acl")

    def unsupported(*args, **kwargs):
        attempts.append(args[1])
        raise OSError(errno.ENOTSUP, "ACLs unsupported")

    monkeypatch.setattr(os, "setxattr", unsupported)
    destination = tmp_path / "destination"
    assert copy_path(source, destination, False) == 0
    assert (destination / "input.txt").read_text() == "unchanged task bytes"
    assert attempts == ["system.posix_acl_access"]


def test_container_copy_keeps_symlink_and_replaces_existing_file(tmp_path):
    from responses_api_agents.wildclawbench_agent.enroot_docker import copy_path

    source = tmp_path / "link"
    source.symlink_to("relative-target")
    destination = tmp_path / "destination"
    destination.write_text("previous")
    assert copy_path(source, destination, False) == 0
    assert destination.is_symlink()
    assert destination.readlink() == Path("relative-target")
