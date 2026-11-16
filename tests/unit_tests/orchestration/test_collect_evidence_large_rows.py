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
    "raw_line_feed_in_string": b"x" * 2000 + b"\n" + b"x" * 10,
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


def _write_raw_rows(job: Path, rows: list[tuple[bytes, int]]) -> None:
    """Scored rows from raw ``(trajectory prefix, filler MiB)`` pairs, the filler written in pieces."""
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True)
    inputs = [{"_ng_task_index": task, "_ng_rollout_index": 0} for task in range(len(rows))]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs))
    (artifacts / "rollouts_failures.jsonl").write_text("")
    with (artifacts / "rollouts.jsonl").open("wb") as out:
        for task, (field, size) in enumerate(rows):
            out.write(b'{"_ng_task_index": %d, "_ng_rollout_index": 0, "_ng_attempt_index": 0, "reward": 1.0, ' % task)
            out.write(b'"response": {"id": "resp", "object": "response", "status": "completed", "output": []}, ')
            out.write(b'"' + field + b'": {"calls": [{"started_at": 1.0, "blob": "')
            for _ in range(size):
                out.write(b"x" * MIB)
            out.write(b'"}]}}\n')


def test_an_oversized_row_outside_the_pruned_values_is_read_in_the_memory_the_whole_row_needs(tmp_path: Path) -> None:
    # The bulk is in ng_model_call_capture, which pruning does not touch: the rebuilt row is the whole record, and the
    # reader must not keep the pruner's copy of it alive while the record is parsed.
    _write_raw_rows(tmp_path, [(b"ng_model_call_capture", 128)])
    collected = _collect(tmp_path, address_limit=512 * MIB)
    assert [row["_ng_task_index"] for row in collected["results"]] == [0]


def test_an_escape_dense_string_is_scanned_in_bounded_memory(tmp_path: Path) -> None:
    # 9 MiB of newlines is an ~18 MiB record: every escape must cost no state beyond the chunk itself.
    _write_rows(tmp_path, [_trajectory("a", b"\\n" * (9 * MIB))])
    collected = _collect(tmp_path, address_limit=512 * MIB)
    assert [row["ng_trial_observations"]["task_id"] for row in collected["results"]] == ["a"]


@pytest.mark.parametrize("key", ["code", "name", "status", "task_id", "model_call_id"])
def test_a_payload_field_named_like_an_identity_is_not_capped_when_pruning_is_triggered(
    tmp_path: Path, key: str
) -> None:
    # A tool output is uninterpreted payload: a field in it that shares a name with an identity the collector reads
    # elsewhere is pruned like any other long string, however long.
    big = b"x" * (2 * MIB)
    trajectory = _trajectory("a", b"y").replace(
        b'"output": "y"', b'"output": {"' + key.encode() + b'": "' + big + b'"}'
    )
    _write_rows(tmp_path, [trajectory, _trajectory("b", b"z")])
    whole = _collect(tmp_path, script=_script(WHOLE))
    pruned = _collect(tmp_path, script=_script(PRUNED))
    assert pruned == whole


def test_an_identity_above_the_cap_still_fails_loudly(tmp_path: Path) -> None:
    _write_rows(tmp_path, [_trajectory("a", b"y" * 10), _trajectory("t" * (2 * MIB), b"z" * 3000)])
    assert _run(tmp_path, script=_script(PRUNED)).returncode != 0


def _tool_call_trajectory(output: Any) -> bytes:
    """A trajectory whose one tool call holds ``output``, serialized with ``orjson`` as ``RolloutStore`` writes it."""
    import orjson

    from nemo_gym.trajectory.trajectory_record import TrajectoryToolCall

    call = TrajectoryToolCall(
        invocation_id="i0", tool_call_id="tc0", tool_name="read", status="completed", output=output
    )
    assert TrajectoryToolCall.model_validate(call.model_dump(mode="json")).output == output
    trajectory = json.loads(_trajectory("a", b"y"))
    trajectory["tool_calls"] = [call.model_dump(mode="json")]
    return orjson.dumps(trajectory)


@pytest.mark.parametrize(
    "key",
    ["C:\\work\\file.py", 'say "hi"', "tab\there", "\u00e9\u4e2d", "task_id\\u0000"],
)
def test_a_payload_under_an_escaped_key_is_pruned_not_capped(tmp_path: Path, key: str) -> None:
    # A tool output may hold any key, one that needs an escape in JSON included: the 2 MiB string under it is payload.
    output = {key: {"payload": "x" * (2 * MIB)}}
    _write_rows(tmp_path, [_tool_call_trajectory(output), _trajectory("b", b"z")])
    whole = _collect(tmp_path, script=_script(WHOLE))
    pruned = _collect(tmp_path, script=_script(PRUNED))
    assert pruned == whole


def test_an_identity_written_with_escapes_is_still_kept_when_pruning(tmp_path: Path) -> None:
    # \u0069 is "i": the key "invocation_id" written with escapes is the same key, so its long string is still capped.
    trajectory = _trajectory("a", b"y").replace(
        b'"invocations": []', b'"invocations": [{"\\u0069nvocation_id": "%s"}]' % (b"t" * (2 * MIB))
    )
    _write_rows(tmp_path, [trajectory, _trajectory("b", b"z")])
    assert _run(tmp_path, script=_script(PRUNED)).returncode != 0


def test_an_extra_closing_brace_in_a_corrupt_last_row_is_ignored_like_any_incomplete_tail(tmp_path: Path) -> None:
    _write_rows(tmp_path, [_trajectory("a", b"y" * 10)])
    path = tmp_path / "artifacts" / "rollouts.jsonl"
    with path.open("ab") as out:
        out.write(
            b'{"_ng_task_index": 1, "_ng_rollout_index": 0, "ng_trajectory": {"x": "' + b"x" * (17 * MIB) + b'"}}}'
        )
    for swap in (WHOLE, PRUNED):
        collected = _collect(tmp_path, script=_script(swap))
        assert [row["_ng_task_index"] for row in collected["results"]] == [0]
