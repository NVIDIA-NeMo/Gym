# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import fcntl
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.visualwebarena import run as runner
from nemo_gym.web import LocalSiteLockPool, WebTask
from resources_servers.webarena_browser import shared_state as state
from resources_servers.webarena_browser.tests.test_shared_state import INTENTS


class _Control:
    def __init__(self):
        self.resets = 0
        self.busy = False
        self.timeout_seconds = 0.02
        self.poll_seconds = 0.001

    def reset(self):
        self.resets += 1

    def require_idle(self):
        if self.busy:
            raise getattr(runner, "_WorkersNotIdleError", RuntimeError)("worker still active")
        return [{"status": "ok", "sessions": 0}]


@pytest.fixture
def collection_boundary(monkeypatch, tmp_path):
    rows = {
        i: {
            "web_task": {
                "benchmark": "visualwebarena",
                "task_id": i,
                "intent": intent,
                "sites": ["classifieds" if i in {"678", "751"} else "shopping"],
            }
        }
        for i, intent in {"0": "Read a page", "1": "Write a page", "2": "Read another page", **INTENTS}.items()
    }
    # Only dataset loading and the collector process are controlled. Epoch
    # ownership, resource-side evidence, files and result reconciliation run.
    monkeypatch.setattr(runner, "_population", lambda values: (copy.deepcopy(rows), {"0", "2"}))
    monkeypatch.setattr(
        state, "_observe_account_state", lambda page, task, urls: (task.sites[0], {"baseline": task.sites[0]})
    )
    root = tmp_path / "state"
    calls = []

    def collect(command, **kwargs):
        phase_input = command[command.index("--input") + 1]
        phase_output = command[command.index("--output") + 1]
        inputs = runner._read_jsonl(runner.Path(phase_input))
        calls.append(inputs)
        output = []
        for row in inputs:
            task = WebTask.model_validate(row["web_task"])
            receipt = state.verify_initial_state(task, None, {}, state_root=str(root), required=True)
            output.append(
                {
                    "benchmark": "visualwebarena",
                    "task_id": task.task_id,
                    "reward": 1,
                    "mask_sample": False,
                    "failure_kind": None,
                    "verifier_result": {"valid_sample": True, "metadata": {"shared_state": receipt}},
                }
            )
        runner.Path(phase_output).write_text("".join(json.dumps(row) + "\n" for row in output))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", collect)
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    (tmp_path / "config.yaml").write_text("agent_name: visualwebarena_benchmark_agent\n")
    return (
        {
            "config": tmp_path / "config.yaml",
            "input_path": source,
            "output_dir": tmp_path / "output",
            "state_root": root,
        },
        calls,
        rows,
    )


def test_normal_waves_and_six_reset_epochs_reconcile_every_task(collection_boundary):
    arguments, calls, rows = collection_boundary
    control = _Control()
    path = runner.run(**arguments, control=control)
    assert [[r["web_task"]["task_id"] for r in phase] for phase in calls] == [
        ["0", "2"],
        ["1"],
        ["346", "678"],
        ["345", "751"],
        ["336"],
        ["335"],
        ["340"],
        ["92"],
    ]
    assert control.resets == 7
    output = runner._read_jsonl(path)
    assert {r["task_id"] for r in output} == set(rows) and len(output) == len(rows)
    assert len(list(arguments["state_root"].glob("*/epoch-*-task-*.json"))) == 8
    assert json.loads((arguments["output_dir"] / "summary.json").read_text())["complete"]


def test_subset_retains_reference_epoch_identity_and_requires_fresh_output(collection_boundary):
    arguments, calls, rows = collection_boundary
    control = _Control()
    runner.run(**arguments, control=control, task_ids={"345", "751"})
    assert control.resets == 1 and len(calls) == 1
    assert all(row["web_task"]["task_kwargs"]["shared_state"]["epoch"] == 2 for row in calls[0])
    with pytest.raises(FileExistsError):
        runner.run(**arguments, control=control, task_ids={"345", "751"})


@pytest.mark.parametrize("kind", ["missing", "duplicate", "masked", "cleanup", "receipt", "nonfinite"])
def test_phase_results_cannot_silently_change_the_denominator(tmp_path, kind):
    row = {
        "task_id": "345",
        "benchmark": "visualwebarena",
        "reward": 1,
        "mask_sample": False,
        "verifier_result": {
            "valid_sample": True,
            "metadata": {"shared_state": {"run_id": "run", "epoch": 2, "task_id": "345"}},
        },
    }
    rows = [row]
    if kind == "missing":
        rows = []
    elif kind == "duplicate":
        rows.append(copy.deepcopy(row))
    elif kind == "masked":
        row["mask_sample"] = True
    elif kind == "cleanup":
        row["cleanup_failure_kind"] = "web:cleanup_pending"
    elif kind == "receipt":
        row["verifier_result"]["metadata"]["shared_state"] = {}
    else:
        row["reward"] = float("nan")
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RuntimeError):
        runner._check_results(path, {"345"}, run_id="run", epoch=2)


def test_failed_collector_with_uncertain_cleanup_prevents_later_resets(collection_boundary, monkeypatch):
    arguments, calls, rows = collection_boundary
    control = _Control()

    def failed(*args, **kwargs):
        control.busy = True
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(runner.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run(**arguments, control=control)
    assert len(list(arguments["state_root"].glob("requires-recovery-*.json"))) == 1
    assert control.resets == 1
    control.busy = False
    arguments["output_dir"] = arguments["output_dir"].with_name("retry")
    with pytest.raises(RuntimeError, match="unresolved cleanup"):
        runner.run(**arguments, control=control)
    assert control.resets == 1


def test_population_rejects_missing_or_changed_task_identities():
    with pytest.raises(ValueError, match="complete maintained"):
        runner._population([])
    rows = [
        {"web_task": {"benchmark": "visualwebarena", "task_id": str(i), "intent": INTENTS.get(str(i), "synthetic")}}
        for i in range(908)
    ]
    with pytest.raises(ValueError, match="population intent identity"):
        runner._population(rows)
    rows[345]["web_task"]["intent"] += " changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._population(rows)


def test_normal_stage_resets_exclusively_then_keeps_shared_lease_and_parallel_collection(
    collection_boundary, monkeypatch
):
    arguments, calls, rows = collection_boundary
    root = arguments["state_root"]
    root.mkdir()
    control = _Control()
    reset = control.reset
    collect = runner.subprocess.run

    def reset_exclusively():
        with (root / "site.lease").open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
        reset()

    def collect_under_shared_lease(*args, **kwargs):
        with (root / "site.lease").open("a+") as other:
            fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return collect(*args, **kwargs)

    monkeypatch.setattr(control, "reset", reset_exclusively)
    monkeypatch.setattr(runner.subprocess, "run", collect_under_shared_lease)
    path = runner.run(**arguments, control=control, stage="read-only", concurrency=2)
    assert control.resets == 1
    assert {r["task_id"] for r in runner._read_jsonl(path)} == {"0", "2"}
    command = json.loads((arguments["output_dir"] / "read-only/command.json").read_text())
    assert command[command.index("--concurrency") + 1] == "2"
    assert all(row["web_task"]["mutation_class"] == "read_only" for row in calls[0])


def test_phase_annotations_preserve_real_site_pool_reader_writer_concurrency(collection_boundary):
    arguments, calls, rows = collection_boundary
    runner.run(**arguments, control=_Control(), stage="read-only", concurrency=2)

    async def exercise():
        pool = LocalSiteLockPool()
        tasks = [WebTask.model_validate(row["web_task"]) for row in calls[0]]
        leases = await asyncio.wait_for(
            asyncio.gather(*(pool.acquire(f"reader-{i}", task) for i, task in enumerate(tasks))), 1
        )
        assert (await pool.health())["reader_leases_by_site"] == {"shopping": 2}
        writer = asyncio.create_task(
            pool.acquire("writer", WebTask(benchmark="visualwebarena", task_id="1", sites=["shopping"]))
        )
        try:
            await asyncio.sleep(0)
            assert not writer.done()
            for lease in leases:
                await pool.release(lease, healthy=True)
            lease = await asyncio.wait_for(writer, 1)
            assert (await pool.health())["writer_sites"] == ["shopping"]
            await pool.release(lease, healthy=True)
        finally:
            if not writer.done():
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)

    asyncio.run(exercise())


def test_nonatomic_lock_downgrade_cannot_admit_another_runs_state(collection_boundary, monkeypatch):
    arguments, calls, rows = collection_boundary
    control = _Control()
    flock = runner.fcntl.flock

    def intervening_reset(lease, mode):
        flock(lease, mode)
        if mode == fcntl.LOCK_SH:
            lease.seek(0)
            lease.truncate()
            json.dump({"run_id": "intervening-reset-owner"}, lease)
            lease.flush()

    monkeypatch.setattr(runner.fcntl, "flock", intervening_reset)
    with pytest.raises(RuntimeError, match="ownership changed"):
        runner.run(**arguments, control=control, task_ids={"0"})
    assert control.resets == 1 and not calls


@pytest.mark.parametrize("task_id", ["0", "1", "345"])
def test_initial_reset_waits_for_every_other_normal_shared_lease(collection_boundary, task_id):
    arguments, calls, rows = collection_boundary
    root = arguments["state_root"]
    root.mkdir()
    control = _Control()
    with (root / "site.lease").open("a+") as other, ThreadPoolExecutor(max_workers=1) as pool:
        fcntl.flock(other, fcntl.LOCK_SH)
        result = pool.submit(runner.run, **arguments, control=control, task_ids={task_id})
        try:
            time.sleep(0.05)
            assert not result.done() and control.resets == 0
        finally:
            fcntl.flock(other, fcntl.LOCK_UN)
        assert result.result(timeout=3).is_file()
    assert control.resets == 1


@pytest.mark.parametrize("failed", [False, True])
def test_next_normal_run_resets_state_left_by_success_or_failed_protected_epoch(
    collection_boundary, monkeypatch, failed
):
    arguments, calls, rows = collection_boundary
    control = _Control()
    collect = runner.subprocess.run
    dirty = False

    def reset():
        nonlocal dirty
        dirty = False
        control.resets += 1

    def collect_and_mutate(*args, **kwargs):
        nonlocal dirty
        assert not dirty, "a previous run's mutation survived the initial reset"
        dirty = True
        if failed and control.resets == 1:
            raise subprocess.CalledProcessError(1, args[0])
        return collect(*args, **kwargs)

    monkeypatch.setattr(control, "reset", reset)
    monkeypatch.setattr(runner.subprocess, "run", collect_and_mutate)
    if failed:
        with pytest.raises(subprocess.CalledProcessError):
            runner.run(**arguments, control=control, task_ids={"345"})
    else:
        runner.run(**arguments, control=control, task_ids={"345"})
    assert dirty
    arguments["output_dir"] = arguments["output_dir"].with_name("second-run")
    runner.run(**arguments, control=control, task_ids={"0"})
    assert control.resets == 2


@pytest.mark.parametrize("task_ids", [{"0", "2"}, {"346", "678"}])
def test_incomplete_collection_resumes_gaps_without_resetting_or_replaying_valid_zeroes(
    collection_boundary, monkeypatch, task_ids
):
    arguments, calls, _ = collection_boundary
    control = _Control()
    policy_calls = []
    commands = []
    timeouts = []
    retained_line = None
    waited_for_cleanup = False
    require_idle = control.require_idle

    def idle_after_cleanup():
        nonlocal waited_for_cleanup
        if len(commands) == 1 and not waited_for_cleanup:
            waited_for_cleanup = True
            raise getattr(runner, "_WorkersNotIdleError", RuntimeError)("previous attempt is still closing")
        return require_idle()

    monkeypatch.setattr(control, "require_idle", idle_after_cleanup)

    def collect(command, **kwargs):
        nonlocal retained_line
        commands.append(command)
        timeouts.append(kwargs["timeout"])
        inputs = runner._read_jsonl(runner.Path(command[command.index("--input") + 1]))
        calls.append(inputs)
        output = runner.Path(command[command.index("--output") + 1])
        existing = runner._read_jsonl(output) if output.exists() else []
        cached = {row["task_id"] for row in existing}
        if cached:
            assert "--resume" in command and "+resubmit=true" in command
            assert output.read_text() == retained_line
        else:
            assert "--resume" not in command
        assert control.resets == 1
        pending = [row for row in inputs if row["web_task"]["task_id"] not in cached]
        for item in pending[:1]:
            task = WebTask.model_validate(item["web_task"])
            policy_calls.append(task.task_id)
            receipt = state.verify_initial_state(
                task, None, {}, state_root=str(arguments["state_root"]), required=True
            )
            result = {
                "benchmark": "visualwebarena",
                "task_id": task.task_id,
                "reward": 0 if not cached else 1,
                "mask_sample": False,
                "verifier_result": {"valid_sample": True, "metadata": {"shared_state": receipt}},
            }
            with output.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        if not cached:
            retained_line = output.read_text()
        output.with_name("rollouts_manifest.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", collect)
    output = runner.run(**arguments, control=control, task_ids=task_ids)
    assert len(commands) == 2 and control.resets == 1
    assert waited_for_cleanup and timeouts[1] < timeouts[0]
    assert set(policy_calls) == task_ids and len(policy_calls) == len(task_ids)
    assert [row["reward"] for row in runner._read_jsonl(output)] == [0, 1]
    summary = json.loads((arguments["output_dir"] / "summary.json").read_text())
    assert summary["tasks"] == 2 and summary["raw_score"] == summary["full_successes"] == 1


def test_missing_phase_outputs_have_a_bounded_retry_budget(collection_boundary, monkeypatch):
    arguments, calls, _ = collection_boundary
    control = _Control()

    def collect(command, **kwargs):
        calls.append(command)
        output = runner.Path(command[command.index("--output") + 1])
        output.write_text("")
        output.with_name("rollouts_manifest.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", collect)
    with pytest.raises(RuntimeError, match="missing or extra task IDs"):
        runner.run(**arguments, control=control, task_ids={"0"})
    assert len(calls) == 3 and control.resets == 1
    assert not (arguments["output_dir"] / "summary.json").exists()


@pytest.mark.parametrize("problem", ["busy", "no_manifest", "changed_cached"])
def test_phase_resume_keeps_cleanup_manifest_and_cached_result_guards(collection_boundary, monkeypatch, problem):
    arguments, calls, _ = collection_boundary
    control = _Control()
    count = 0

    def collect(command, **kwargs):
        nonlocal count
        count += 1
        output = runner.Path(command[command.index("--output") + 1])
        rows = [
            {
                "task_id": "0",
                "benchmark": "visualwebarena",
                "reward": 0,
                "mask_sample": False,
                "verifier_result": {"valid_sample": True},
            }
        ]
        if count == 2:
            assert "--resume" in command
            rows[0]["reward"] = 1
            rows.append({**rows[0], "task_id": "2"})
        output.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if problem != "no_manifest":
            output.with_name("rollouts_manifest.json").write_text("{}")
        control.busy = problem == "busy"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", collect)
    message = {
        "busy": "worker still active",
        "no_manifest": "native Gym manifest",
        "changed_cached": "previously valid",
    }[problem]
    with pytest.raises(RuntimeError, match=message):
        runner.run(**arguments, control=control, task_ids={"0", "2"})
    assert count == (2 if problem == "changed_cached" else 1)
    assert control.resets == 1 and not (arguments["output_dir"] / "summary.json").exists()
    assert bool(list(arguments["state_root"].glob("requires-recovery-*.json"))) == (problem == "busy")
