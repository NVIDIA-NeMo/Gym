# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The evidence collector reads a saved row in bounded memory: a row whose ``ng_trajectory`` holds every model call's
whole request reaches gigabytes, and recovery must not need that much memory to read its identities and statuses."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.orchestration.collect_evidence import COLLECT_SCRIPT


resource = pytest.importorskip("resource")

MIB = 1 << 20


def _trajectory_head(task: int) -> str:
    return (
        '{"schema_version": "5", "task_id": "t%d", "rollout_id": "r0", "invocations": '
        '[{"invocation_id": "i0", "status": "completed", "in_session": true}], "verifier_invocations": [], '
        '"turns": [], "tool_calls": [], "gaps": [], "model_calls": [' % task
    )


def _model_call(index: int, items: int, item_bytes: int) -> str:
    # Each call carries the whole history so far, as a harness's captured request does.
    history = ",".join(
        '{"type": "function_call_output", "call_id": "c%d", "output": "%s"}' % (n, "x" * item_bytes)
        for n in range(items)
    )
    return '{"model_call_id": "m%d", "started_at": 1.0, "request": {"input": [%s]}, "response": {"output": []}}' % (
        index,
        history,
    )


def _write_store(job: Path, *, big_row_calls: int = 0, small_rows: int = 2) -> None:
    """Two or more scored rows, and with ``big_row_calls`` one more whose trajectory is that many requests of ~8 MiB."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True)
    count = small_rows + (1 if big_row_calls else 0)
    inputs = [{"_ng_task_index": task, "_ng_rollout_index": 0} for task in range(count)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
    (artifacts / "rollouts_failures.jsonl").write_text("")
    with (artifacts / "rollouts.jsonl").open("w") as out:
        for task in range(small_rows):
            row: dict[str, Any] = {
                "_ng_task_index": task,
                "_ng_rollout_index": 0,
                "_ng_attempt_index": 0,
                "reward": 1.0,
                "response": {"id": "resp", "object": "response", "status": "completed", "output": []},
                "ng_trajectory": json.loads(_trajectory_head(task) + _model_call(0, 3, 4096) + "]}"),
            }
            out.write(json.dumps(row) + "\n")
        if big_row_calls:
            task = small_rows
            out.write('{"_ng_task_index": %d, "_ng_rollout_index": 0, "_ng_attempt_index": 0, "reward": 0.0, ' % task)
            out.write('"response": {"id": "resp", "object": "response", "status": "completed", "output": []}, ')
            out.write('"ng_trajectory": ' + _trajectory_head(task))
            for index in range(big_row_calls):
                out.write(("," if index else "") + _model_call(index, 8, MIB))
            out.write("]}}\n")


def _collect(job: Path, *, script: str = COLLECT_SCRIPT, address_limit: int | None = None) -> dict[str, Any]:
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))

    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(job)],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=limit if address_limit else None,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


def test_a_row_far_above_the_memory_limit_is_read_in_bounded_memory(tmp_path: Path) -> None:
    _write_store(tmp_path, big_row_calls=32)  # one 256 MiB row: parsed whole it needs several times the limit
    row_bytes = (tmp_path / "artifacts" / "rollouts.jsonl").stat().st_size
    assert row_bytes > 250 * MIB
    collected = _collect(tmp_path, address_limit=512 * MIB)
    by_task = {row["_ng_task_index"]: row for row in collected["results"]}
    assert sorted(by_task) == [0, 1, 2]
    big = by_task[2]
    assert big["reward"] == 0.0
    # What the collector reads of the trajectory survives the pruning: identities, call ids and item kinds.
    assert [call["model_call_id"] for call in big["ng_trial_observations"]["model_calls"]] == [
        "m%d" % n for n in range(32)
    ]
    assert big["ng_trial_observations"]["model_calls"][5]["input_items"] == [
        {"type": "function_call_output", "call_id": "c%d" % n} for n in range(8)
    ]


def test_the_row_limit_does_not_change_what_the_collector_reports(tmp_path: Path) -> None:
    _write_store(tmp_path, big_row_calls=3)
    whole = _collect(tmp_path, script=COLLECT_SCRIPT.replace("BIG_LINE = 16 << 20", "BIG_LINE = 1 << 40"))
    pruned = _collect(tmp_path, script=COLLECT_SCRIPT.replace("BIG_LINE = 16 << 20", "BIG_LINE = 1024"))
    assert pruned == whole


def test_a_truncated_last_row_is_still_recognised_in_bounded_memory(tmp_path: Path) -> None:
    _write_store(tmp_path, big_row_calls=3)
    path = tmp_path / "artifacts" / "rollouts.jsonl"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 5 * MIB])  # cut inside the last row's trajectory, newline lost
    collected = _collect(tmp_path, address_limit=512 * MIB)
    assert sorted(row["_ng_task_index"] for row in collected["results"]) == [0, 1]
