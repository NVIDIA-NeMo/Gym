# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for `check_correctness` worker and direct IPC lifecycle."""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from unittest.mock import MagicMock

import pytest
from lcb_integration import compute_code_generation_metrics
from lcb_integration.compute_code_generation_metrics import check_correctness


_SAMPLE: dict[str, str] = {"input_output": json.dumps({"inputs": ["1", "2"], "outputs": ["1", "4"]})}


@pytest.fixture
def patched_mp(monkeypatch):
    """Replace the pipe and process so lifecycle ordering is observable."""
    result_connection: MagicMock = MagicMock(name="result_connection")
    child_connection: MagicMock = MagicMock(name="child_connection")
    pipe_factory: MagicMock = MagicMock(return_value=(result_connection, child_connection))
    process_instance: MagicMock = MagicMock(name="Process")
    process_factory: MagicMock = MagicMock(return_value=process_instance)

    monkeypatch.setattr(compute_code_generation_metrics.multiprocessing, "Pipe", pipe_factory)
    monkeypatch.setattr(compute_code_generation_metrics.multiprocessing, "Process", process_factory)

    return result_connection, child_connection, pipe_factory, process_instance, process_factory


class TestCheckCorrectnessReap:
    """check_correctness must close IPC and reap its worker on every exit path."""

    def test_kill_is_followed_by_reap_join(self, patched_mp):
        result_connection, child_connection, _pipe_factory, process, _process_factory = patched_mp
        result_connection.poll.return_value = False
        process.is_alive.return_value = True

        result, metadata = check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        process.start.assert_called_once()
        process.kill.assert_called_once()
        process.join.assert_called_once_with(timeout=5)
        result_connection.close.assert_called_once()
        assert child_connection.close.call_count == 2
        assert result == [-1, -1]
        assert metadata is None

    def test_connections_close_when_process_start_raises(self, patched_mp):
        result_connection, child_connection, _pipe_factory, process, _process_factory = patched_mp
        process.start.side_effect = RuntimeError("boom")
        process.is_alive.return_value = False

        with pytest.raises(RuntimeError, match="boom"):
            check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        result_connection.close.assert_called_once()
        child_connection.close.assert_called_once()

    def test_happy_path_receives_result_before_reaping(self, patched_mp):
        result_connection, child_connection, _pipe_factory, process, _process_factory = patched_mp
        result_connection.poll.return_value = True
        result_connection.recv_bytes.return_value = pickle.dumps(
            {
                "version": compute_code_generation_metrics._WORKER_RESULT_VERSION,
                "result": [1, 1],
                "metadata": {"ok": True},
            }
        )
        process.is_alive.return_value = False

        result, metadata = check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        assert result == [1, 1]
        assert metadata == {"ok": True}
        result_connection.recv_bytes.assert_called_once_with(
            maxlength=compute_code_generation_metrics._DEFAULT_RESULT_MAX_BYTES
        )
        process.join.assert_called_once_with(timeout=5)
        assert child_connection.close.call_count == 2
        result_connection.close.assert_called_once()

    @pytest.mark.parametrize("sample", [{}, {"input_output": "not-json"}, {"input_output": {"outputs": []}}])
    def test_invalid_input_output_short_circuits(self, patched_mp, sample):
        _result_connection, _child_connection, pipe_factory, process, _process_factory = patched_mp

        result, metadata = check_correctness(sample, generation="g", timeout=1, debug=False)

        assert result == [-1]
        assert metadata is None
        pipe_factory.assert_not_called()
        process.start.assert_not_called()

    @pytest.mark.parametrize(
        "side_effect",
        [EOFError(), OSError("oversized"), pickle.UnpicklingError("malformed")],
        ids=["eof", "oversized", "malformed"],
    )
    def test_invalid_child_payload_fails_closed(self, patched_mp, side_effect):
        result_connection, _child_connection, _pipe_factory, process, _process_factory = patched_mp
        result_connection.poll.return_value = True
        if isinstance(side_effect, pickle.UnpicklingError):
            result_connection.recv_bytes.return_value = b"not-pickle"
        else:
            result_connection.recv_bytes.side_effect = side_effect
        process.is_alive.return_value = False

        result, metadata = check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        assert result == [-1, -1]
        assert metadata is None
        process.join.assert_called_once_with(timeout=5)

    def test_global_timeout_caps_input_scaled_backstop(self, patched_mp):
        result_connection, _child_connection, _pipe_factory, process, _process_factory = patched_mp
        result_connection.poll.return_value = False
        process.is_alive.return_value = True
        sample = {"input_output": {"inputs": ["1"] * 50, "outputs": ["1"] * 50}}

        check_correctness(
            sample,
            generation="ignored",
            timeout=10,
            debug=False,
            global_timeout_seconds=20,
        )

        result_connection.poll.assert_called_once_with(20)

    def test_worker_interruption_still_kills_and_reaps_child(self, patched_mp):
        result_connection, _child_connection, _pipe_factory, process, _process_factory = patched_mp
        result_connection.poll.side_effect = KeyboardInterrupt
        process.is_alive.return_value = True

        with pytest.raises(KeyboardInterrupt):
            check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)

        process.kill.assert_called_once()
        process.join.assert_called_once_with(timeout=5)
        result_connection.close.assert_called_once()


@pytest.mark.skipif(sys.platform != "linux", reason="requires fork and Linux procfs")
def test_direct_ipc_preserves_result_parity_and_leaves_no_descendants(monkeypatch):
    children_path = f"/proc/{os.getpid()}/task/{os.getpid()}/children"

    def child_pids():
        with open(children_path) as stream:
            return set(stream.read().split())

    baseline_children = child_pids()
    baseline_fd_count = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(
        compute_code_generation_metrics,
        "run_test",
        lambda *args, **kwargs: ([True, False], {"error_code": 0, "detail": "complete"}),
    )

    for sample in (_SAMPLE, {"input_output": json.loads(_SAMPLE["input_output"])}):
        result, metadata = check_correctness(sample, generation="ignored", timeout=1, debug=False)
        assert result == [True, False]
        assert metadata == {"error_code": 0, "detail": "complete"}

    time.sleep(0.05)
    assert child_pids() == baseline_children
    assert len(os.listdir("/proc/self/fd")) <= baseline_fd_count


@pytest.mark.skipif(sys.platform != "linux", reason="requires fork and Linux procfs")
def test_repeated_child_crashes_leave_no_descendants_or_file_descriptors(monkeypatch):
    children_path = f"/proc/{os.getpid()}/task/{os.getpid()}/children"

    def child_pids():
        with open(children_path) as stream:
            return set(stream.read().split())

    def crash(*args, **kwargs):
        os._exit(7)

    baseline_children = child_pids()
    baseline_fd_count = len(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(compute_code_generation_metrics, "_temp_run", crash)

    for _ in range(16):
        result, metadata = check_correctness(_SAMPLE, generation="ignored", timeout=1, debug=False)
        assert result == [-1, -1]
        assert metadata is None

    time.sleep(0.05)
    assert child_pids() == baseline_children
    assert len(os.listdir("/proc/self/fd")) <= baseline_fd_count
