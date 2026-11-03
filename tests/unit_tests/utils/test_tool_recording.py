# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nemo_gym.utils import tool_recording
from nemo_gym.utils.tool_recording import ClockContext, ToolCallRecorder, tool_recording_files


def recorder(path: Path, *, producer_id: str = "producer-1") -> ToolCallRecorder:
    return ToolCallRecorder(
        path=path,
        producer_id=producer_id,
        invocation_id="invocation-1",
        agent_instance="assistant",
        clock=ClockContext(location="sandbox", owner="sandbox-1"),
        boundary="executor",
        timing_source="test.subprocess",
    )


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_start_is_readable_before_execution_and_finish_preserves_explicit_outcome(tmp_path):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    with capture.call(native_call_id="call-1", tool_name="bash", model_call_ref={"model_call_id": "model-1"}) as call:
        start = events(path)
        assert len(start) == 1
        assert start[0]["kind"] == "start"
        assert start[0]["status"] == "incomplete"
        assert start[0]["span_id"] == call.span_id
        call.set_outcome("failed", failure_id="failure-1")
    start, finish = events(path)
    assert finish["span_id"] == start["span_id"]
    assert finish["status"] == "failed"
    assert finish["failure_id"] == "failure-1"
    assert finish["model_call_ref"] == {"model_call_id": "model-1"}
    assert finish["duration_source"] == "monotonic"
    assert finish["duration_ms"] >= 0
    assert finish["sequence"] == start["sequence"] + 1


def test_normal_return_does_not_infer_success_and_final_submission_does_not_invent_end(tmp_path):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    with capture.call(native_call_id="call-1", tool_name="bash"):
        pass
    assert events(path)[1]["status"] == "unknown"
    with pytest.raises(SystemExit):
        with capture.call(native_call_id="submit", tool_name="submit"):
            raise SystemExit(0)
    assert events(path)[-1]["kind"] == "start"


def test_tool_exception_is_preserved_and_cancellation_is_local(tmp_path):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    error = RuntimeError("tool failed")
    with pytest.raises(RuntimeError) as caught:
        with capture.call(native_call_id="call-1", tool_name="bash"):
            raise error
    assert caught.value is error
    assert events(path)[-1]["status"] == "failed"
    with pytest.raises(asyncio.CancelledError):
        with capture.call(native_call_id="call-2", tool_name="bash"):
            raise asyncio.CancelledError
    assert events(path)[-1]["status"] == "cancelled"


def test_capture_write_failure_does_not_replace_tool_result_or_error(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    real_append = tool_recording.append_jsonl
    monkeypatch.setattr(tool_recording, "append_jsonl", lambda *_, **__: (_ for _ in ()).throw(OSError("full")))
    handle = capture.start(native_call_id="call-1", tool_name="bash")
    assert capture.failed_spans == (handle.span_id,)
    monkeypatch.setattr(tool_recording, "append_jsonl", real_append)
    capture.finish(handle, status="completed")
    assert events(path)[0]["capture_failed"] is True
    assert events(path)[0]["status"] == "completed"


def test_repeated_callbacks_and_parent_handles(tmp_path):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    parent = capture.start(native_call_id="call", tool_name="agent")
    child = capture.start(native_call_id="call", tool_name="bash", parent=parent)
    capture.finish(child, status="completed")
    capture.finish(child, status="failed")
    capture.finish(parent, status="completed")
    assert len(events(path)) == 4
    assert child.span_id != parent.span_id
    assert events(path)[1]["parent_span_id"] == parent.span_id
    assert events(path)[2]["status"] == "completed"


def test_threads_keep_call_local_outcomes_and_sequences(tmp_path):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)

    def run(index: int) -> None:
        with capture.call(native_call_id="reused", tool_name="bash") as handle:
            handle.set_outcome("failed" if index % 2 else "completed")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(32)))
    rows = events(path)
    assert [row["sequence"] for row in rows] == list(range(64))
    assert len({row["span_id"] for row in rows}) == 32
    assert sum(row["kind"] == "finish" and row["status"] == "failed" for row in rows) == 16


def test_clock_discontinuity_rotates_domain_but_monotonic_duration_survives(tmp_path, monkeypatch):
    wall, monotonic = iter([100.0, 99.0]), iter([10.0, 12.0])
    monkeypatch.setattr(tool_recording.time, "time", lambda: next(wall))
    monkeypatch.setattr(tool_recording.time, "monotonic", lambda: next(monotonic))
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    handle = capture.start(native_call_id="call", tool_name="bash")
    capture.finish(handle, status="completed")
    start, finish = events(path)
    assert start["clock_domain"] != finish["clock_domain"]
    assert finish["duration_ms"] == 2000.0


def test_standalone_staged_modules_need_no_gym_or_third_party_packages(tmp_path):
    for path, source in tool_recording_files(tmp_path).items():
        Path(path).write_text(source)
    script = """from tool_recording import ClockContext, ToolCallRecorder
r = ToolCallRecorder(path="events.jsonl", producer_id="child", invocation_id="root", agent_instance=None,
                    clock=ClockContext(location="producer", owner="child"), boundary="lifecycle", timing_source="hook")
with r.call(native_call_id="call", tool_name="tool") as call:
    call.set_outcome("completed")
"""
    subprocess.run([sys.executable, "-S", "-c", script], cwd=tmp_path, check=True)
    assert events(tmp_path / "events.jsonl")[-1]["status"] == "completed"


def test_killed_process_leaves_start_and_concurrent_processes_do_not_corrupt_stream(tmp_path):
    for path, source in tool_recording_files(tmp_path).items():
        Path(path).write_text(source)
    script = """import os, sys
from tool_recording import ClockContext, ToolCallRecorder
producer = sys.argv[1]
r = ToolCallRecorder(path="events.jsonl", producer_id=producer, invocation_id="root", agent_instance=None,
                    clock=ClockContext(location="producer", owner=producer), boundary="executor", timing_source="hook")
for i in range(10):
    h = r.start(native_call_id="reused", tool_name="bash")
    if producer == "killed":
        os._exit(7)
    r.finish(h, status="completed")
"""
    processes = [
        subprocess.Popen([sys.executable, "-S", "-c", script, name], cwd=tmp_path)
        for name in ("one", "two", "three", "killed")
    ]
    assert [p.wait(timeout=10) for p in processes] == [0, 0, 0, 7]
    rows = events(tmp_path / "events.jsonl")
    assert len(rows) == 61
    assert len({(row["producer_id"], row["sequence"]) for row in rows}) == 61
    assert [row["kind"] for row in rows if row["producer_id"] == "killed"] == ["start"]
    for name in ("one", "two", "three"):
        assert [row["sequence"] for row in rows if row["producer_id"] == name] == list(range(20))


def test_finish_can_attach_failure_details_without_overwriting_the_outcome(tmp_path):
    path = tmp_path / "events.jsonl"
    capture = recorder(path)
    call = capture.start(native_call_id="call", tool_name="bash")
    call.set_outcome("failed")
    capture.finish(call, failure_id="failure-1")
    assert events(path)[-1]["status"] == "failed"
    assert events(path)[-1]["failure_id"] == "failure-1"
