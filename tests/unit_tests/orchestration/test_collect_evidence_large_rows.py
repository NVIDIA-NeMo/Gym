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


def _run(job: Path, *, script: str = COLLECT_SCRIPT, address_limit: int | None = None):
    def limit() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))

    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(job)],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=limit if address_limit else None,
    )


def _collect(job: Path, *, script: str = COLLECT_SCRIPT, address_limit: int | None = None) -> dict[str, Any]:
    done = _run(job, script=script, address_limit=address_limit)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout)


WHOLE = ("BIG_LINE = 16 << 20", "BIG_LINE = 1 << 40")
PRUNED = ("BIG_LINE = 16 << 20", "BIG_LINE = 1024")


def _script(*swaps: tuple[str, str]) -> str:
    script = COLLECT_SCRIPT
    for old, new in swaps:
        assert old in script or old.startswith("VALIDATE"), old
        script = script.replace(old, new)
    return script


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


def _write_rows(job: Path, trajectories: list[bytes]) -> None:
    """One scored row per trajectory (raw JSON bytes), each trajectory the row's ``ng_trajectory``."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True)
    inputs = [{"_ng_task_index": task, "_ng_rollout_index": 0} for task in range(len(trajectories))]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
    (artifacts / "rollouts_failures.jsonl").write_text("")
    with (artifacts / "rollouts.jsonl").open("wb") as out:
        for task, trajectory in enumerate(trajectories):
            out.write(b'{"_ng_task_index": %d, "_ng_rollout_index": 0, "_ng_attempt_index": 0, "reward": 1.0, ' % task)
            out.write(b'"response": {"id": "resp", "object": "response", "status": "completed", "output": []}, ')
            out.write(b'"ng_trajectory": ' + trajectory + b"}\n")


def _trajectory(task_id: str, padding: bytes = b"", *, call_id: str = "c0", text: bytes | None = None) -> bytes:
    """A trajectory with one model call whose request carries ``padding`` (raw JSON string body) as tool output."""
    content = b'"content": "' + text + b'", ' if text is not None else b""
    return (
        b'{"schema_version": "5", "task_id": "' + task_id.encode() + b'", "rollout_id": "r0", "invocations": [], '
        b'"verifier_invocations": [], "turns": [], "tool_calls": [], "gaps": [], "model_calls": [{"model_call_id": "'
        + call_id.encode()
        + b'", "started_at": 1.0, "request": {"input": [{"type": "function_call_output", "call_id": "'
        + call_id.encode()
        + b'", "output": "'
        + padding
        + b'"}]}, "response": {"output": [{"type": "message", "role": "assistant", '
        + content
        + b'"status": "completed"}]}}]}'
    )


LONG_TASK = "t" * 1025


def test_a_long_identity_survives_pruning_and_distinct_long_identities_stay_distinct(tmp_path: Path) -> None:
    ids = [LONG_TASK, "u" * 1025, "t" * 1024 + "v", "short"]
    # The materialized input of each task carries the same identity as its trajectory, as the producer writes it.
    _write_rows(tmp_path, [_trajectory(task_id, b"x" * 4000, call_id=task_id) for task_id in ids])
    whole = _collect(tmp_path, script=_script(WHOLE))
    pruned = _collect(tmp_path, script=_script(PRUNED))
    assert pruned == whole
    # Pruning reaches only ng_trajectory, so what is hashed from the saved answer and score is unchanged.
    assert all(len(row["_ng_collected_grading_binding"]) == 64 for row in pruned["results"])
    observed = [row["ng_trial_observations"] for row in sorted(pruned["results"], key=lambda r: r["_ng_task_index"])]
    assert [item["task_id"] for item in observed] == ids
    assert [item["model_calls"][0]["model_call_id"] for item in observed] == ids
    assert [item["model_calls"][0]["input_items"][0]["call_id"] for item in observed] == ids
    assert len({item["task_id"] for item in observed}) == len(ids)


INVALID_STRINGS = {
    "bad_escape": b"x" * 2000 + b"\\q" + b"x" * 10,
    "bad_utf8": b"x" * 2000 + b"\xff\xfe" + b"x" * 10,
    "truncated_utf8": b"x" * 2000 + b"\xe2\x82" + b"x" * 10,
    "raw_control_byte": b"x" * 2000 + b"\x01" + b"x" * 10,
    "raw_newline_in_string": b"x" * 2000 + b"\t" + b"x" * 10,
    "short_unicode_escape": b"x" * 2000 + b"\\u12" + b"x" * 10,
}


@pytest.mark.parametrize("kind", INVALID_STRINGS)
@pytest.mark.parametrize("chunk", [4 << 20, 7], ids=["chunked", "tiny_chunks"])
def test_pruning_does_not_hide_an_invalid_string(tmp_path: Path, kind: str, chunk: int) -> None:
    _write_rows(tmp_path, [_trajectory("a", b"y" * 10), _trajectory("b", INVALID_STRINGS[kind])])
    whole = _run(tmp_path, script=_script(WHOLE))
    pruned = _run(
        tmp_path,
        script=_script(PRUNED, ("CHUNK = 4 << 20", f"CHUNK = {chunk}"), ("VALIDATE = 1 << 20", "VALIDATE = 5")),
    )
    assert whole.returncode != 0
    assert pruned.returncode != 0
    assert pruned.stdout == whole.stdout == ""


@pytest.mark.parametrize("chunk", [1, 3, 7, 64])
def test_valid_escapes_and_characters_split_across_chunks_are_accepted_alike(tmp_path: Path, chunk: int) -> None:
    padding = (
        b"x" * 1500
        + b'\\"quoted\\" \\\\ \\/ \\b\\f\\n\\r\\t \\u00e9 \\ud83d\\ude00 \\ud800 \xe2\x82\xac \xf0\x9f\x98\x80 '
    ) * 4
    _write_rows(tmp_path, [_trajectory("a", padding), _trajectory("b", b"z" * 3000)])
    whole = _collect(tmp_path, script=_script(WHOLE))
    pruned = _collect(
        tmp_path,
        script=_script(PRUNED, ("CHUNK = 4 << 20", f"CHUNK = {chunk}"), ("VALIDATE = 1 << 20", "VALIDATE = 9")),
    )
    assert pruned == whole
    assert sorted(row["_ng_task_index"] for row in pruned["results"]) == [0, 1]


def test_a_truncated_last_row_inside_an_elided_string_is_still_ignored(tmp_path: Path) -> None:
    _write_rows(tmp_path, [_trajectory("a", b"y" * 10), _trajectory("b", b"x" * 5000)])
    path = tmp_path / "artifacts" / "rollouts.jsonl"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 2000])
    for swap in (WHOLE, PRUNED):
        collected = _collect(tmp_path, script=_script(swap))
        assert [row["_ng_task_index"] for row in collected["results"]] == [0]
