# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import time

import pytest

from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.looping_assessment import assess_looping_record
from nemo_gym.rollouts.looping_runtime import LoopingExecutor, _assess_capture, _capture_input
from tests.unit_tests.rollouts.test_rollout_health import _call, _looping_record


@pytest.fixture
def completed_rollout(tmp_path):
    row = _looping_record()
    row.pop("agent_ref")
    row.update(task_id={"taskset": "environment:tasks", "task_id": "typed"}, task_input={"input": "prompt"})
    CaptureStore(tmp_path).record(
        "0-0",
        _call(
            response=row["response"], dialect="responses", model_ref={"type": "responses_api_models", "name": "policy"}
        ),
    )
    config = {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
        "agent": {"responses_api_agents": {"impl": {"model_server": {"name": "policy"}}}},
        "environment": {"environment_servers": {"impl": {"agent_server": {"name": "agent"}}}},
    }
    executor = LoopingExecutor.from_config(config, config={"enabled": True, "max_pending": 1})
    try:
        yield executor, row, tmp_path
    finally:
        executor.close(timeout_s=1)


def test_live_worker_matches_saved_assessment(completed_rollout):
    executor, row, directory = completed_rollout
    source = _capture_input(row, row, (directory,), executor._policies["environment"])
    assert source.record["trajectory_identity"] == {"task_id": "typed", "rollout_id": "0-0"}
    assert _assess_capture(source).input_digest == assess_looping_record(row).input_digest
    assert executor.submit_completed(row, row)
    executor.close(timeout_s=10)
    assert executor.metrics_snapshot()["looping/severe_union/flagged_total"] == 1
    assert LoopingExecutor.from_config({}, config={"enabled": False}) is None


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX process suspension")
@pytest.mark.parametrize("kill_worker", [False, True])
def test_full_queue_and_failed_worker_remain_unknown_with_bounded_cleanup(completed_rollout, kill_worker):
    executor, row, _ = completed_rollout
    worker = executor._processes[0]
    os.kill(worker.pid, signal.SIGSTOP)
    assert executor.submit_completed(row, row)
    assert not executor.submit_completed(row, row)
    if kill_worker:
        worker.kill()
        worker.join(timeout=1)
    started = time.monotonic()
    executor.close(timeout_s=0.2)
    assert time.monotonic() - started < 1 and not worker.is_alive()
    snapshot = executor.metrics_snapshot()
    assert snapshot["looping/severe_union/assessed_total"] == 0
    assert snapshot["looping/severe_union/unknown_total"] == 2
