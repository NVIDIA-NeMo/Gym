# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export of accepted finish answers as Artificial Analysis grading batches, on synthetic rollouts."""

import hashlib
import json

import pytest

from resources_servers.critpt_custom_grader.export_aa_batches import main


PROBLEMS = ["synthetic_a", "synthetic_b", "synthetic_c"]


def _finished(answer, *, call_id="finish-1"):
    arguments = {"reason": "done", "derivation": "worked it out", "answer": answer}
    acknowledgement = {"protocol": "critpt-finish-v1", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest()}
    return {
        "status": "completed",
        "metadata": None,
        "output": [
            {"type": "function_call", "name": "finish", "arguments": json.dumps(arguments), "call_id": call_id},
            {"type": "function_call_output", "call_id": call_id, "output": json.dumps(acknowledgement)},
        ],
    }


def _row(task, repeat, response, *, category="passed", reward=1.0):
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "problem_id": PROBLEMS[task],
        "response": response,
        "reward": reward,
        "category": category,
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(tmp_path, rows, *extra):
    rollouts = _write(tmp_path / "rollouts.jsonl", rows)
    out_dir = tmp_path / "aa_batches"
    code = main(["--rollouts", str(rollouts), "--out-dir", str(out_dir), "--batch-size", "3", *extra])
    return code, out_dir


def _complete_repeat(repeat):
    return [
        _row(0, repeat, _finished("Notes first.\n```python\ndef answer():\n    return 1\n```")),
        # Grading of this answer failed locally; the export still sends its answer.
        _row(1, repeat, _finished("def answer():\n    return 2"), category="reference_failed", reward=0.0),
        # A completed episode without an accepted finish scores zero: exported as an empty answer.
        _row(2, repeat, {"status": "completed", "metadata": None, "output": []}, category="candidate_no_answer"),
    ]


def test_full_repeats_are_written_in_the_replay_format(tmp_path, capsys):
    code, out_dir = _run(tmp_path, _complete_repeat(0) + _complete_repeat(1), "--model", "model-x")

    assert code == 0
    for repeat in (0, 1):
        submissions = _read_jsonl(out_dir / f"repeat-{repeat}" / "submissions.jsonl")
        assert [s["submission_id"] for s in submissions] == [0, 1, 2]
        assert all(isinstance(s["ts"], float) for s in submissions)
        # The submission object is what the CritPt resources server builds and replay posts unchanged.
        assert [s["submission"] for s in submissions] == [
            {
                "problem_id": "synthetic_a",
                "generated_code": "```python\ndef answer():\n    return 1\n```",
                "model": "model-x",
                "generation_config": {},
            },
            {
                "problem_id": "synthetic_b",
                "generated_code": "```python\ndef answer():\n    return 2\n```",
                "model": "model-x",
                "generation_config": {},
            },
            {
                "problem_id": "synthetic_c",
                "generated_code": "```python\n```",
                "model": "model-x",
                "generation_config": {},
            },
        ]
        sources = _read_jsonl(out_dir / f"repeat-{repeat}" / "sources.jsonl")
        assert [(s["submission_id"], s["task_index"], s["rollout_index"]) for s in sources] == [
            (0, 0, repeat),
            (1, 1, repeat),
            (2, 2, repeat),
        ]
        assert [s["category"] for s in sources] == ["candidate_ready", "candidate_ready", "candidate_no_answer"]
        assert sources[1]["local_category"] == "reference_failed"
        assert sources[1]["local_reward"] == 0.0

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["model"] == "model-x"
    record = manifest["repeats"]["0"]
    assert (record["written"], record["submissions"], record["no_answer"], record["problems"]) == (True, 3, 1, [])
    digest = hashlib.sha256((out_dir / "repeat-0" / "submissions.jsonl").read_bytes()).hexdigest()
    assert record["submissions_sha256"] == digest
    assert "repeat 1: 3 submissions, 1 without an answer" in capsys.readouterr().out


def test_unscorable_missing_or_failed_rollouts_leave_their_repeat_unwritten(tmp_path, capsys):
    unscorable = _complete_repeat(1)
    unscorable[0] = _row(0, 1, {"status": "failed", "error": {"message": "provider down"}, "output": []})
    missing = _complete_repeat(2)[1:]
    no_response = _complete_repeat(4)
    no_response[2] = {key: value for key, value in no_response[2].items() if key != "response"}

    # A newer attempt of task 2 in repeat 3 failed: the failure stands over the older answer.
    failed = {"_ng_task_index": 2, "_ng_rollout_index": 3, "_ng_attempt_index": 1, "error": "synthetic"}
    _write(tmp_path / "rollouts_failures.jsonl", [failed])

    code, out_dir = _run(tmp_path, _complete_repeat(0) + unscorable + missing + _complete_repeat(3) + no_response)

    assert code == 3
    assert (out_dir / "repeat-0" / "submissions.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["repeats"]["0"]["written"] is True
    expected = {
        "1": [{"task_index": 0, "reason": "response_incomplete"}],
        "2": [{"task_index": 0, "reason": "0 rollouts"}],
        "3": [{"task_index": 2, "reason": "latest_attempt_failed"}],
        "4": [{"task_index": 2, "reason": "response_missing"}],
    }
    for repeat, problems in expected.items():
        assert manifest["repeats"][repeat] == {"written": False, "problems": problems}
        assert not (out_dir / f"repeat-{repeat}").exists()
    assert "repeat 1: not written ({'response_incomplete': 1})" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{**_row(0, 0, {}), "problem_id": ""}], "has no problem_id"),
        ([_row(0, 0, {}), {**_row(0, 1, {}), "problem_id": "other"}], "two problem ids"),
        ([_row(0, 0, {}), {**_row(1, 0, {}), "problem_id": "synthetic_a"}], "share a problem id"),
        ([_row(0, 0, {}), _row(1, 0, {})], "cover 2 tasks; a batch needs exactly 3"),
        ([{key: value for key, value in _row(0, 0, {}).items() if key != "_ng_rollout_index"}], "_ng_rollout_index"),
    ],
)
def test_inconsistent_rollouts_are_refused_before_writing(tmp_path, capsys, rows, message):
    code, out_dir = _run(tmp_path, rows)

    assert code == 2
    assert message in capsys.readouterr().err
    assert not out_dir.exists()


def test_the_latest_attempt_of_a_rollout_is_exported(tmp_path):
    # A recovery run appends a newer attempt after the masked one it replaces.
    masked = {
        **_row(2, 0, {"status": "failed", "error": {"message": "provider down"}, "output": []}),
        "_ng_attempt_index": 0,
    }
    rerun = {**_row(2, 0, _finished("def answer():\n    return 3")), "_ng_attempt_index": 1}

    code, out_dir = _run(tmp_path, _complete_repeat(0)[:2] + [masked, rerun])

    assert code == 0
    submissions = _read_jsonl(out_dir / "repeat-0" / "submissions.jsonl")
    assert submissions[2]["submission"]["generated_code"] == "```python\ndef answer():\n    return 3\n```"


def test_a_rerun_after_a_sidecar_failure_is_exported_and_non_superseding_rows_are_ignored(tmp_path):
    # Attempt 0 failed into the sidecar; recovery's attempt 1 landed in the rollouts file.
    failed = {"_ng_task_index": 2, "_ng_rollout_index": 0, "_ng_attempt_index": 0, "error": "synthetic"}
    # A row the dispatch budget never started is no execution and supersedes nothing.
    unstarted = {
        "_ng_task_index": 1,
        "_ng_rollout_index": 0,
        "_ng_attempt_index": 5,
        "_ng_omitted_rollout": True,
        "_ng_dispatch_drained": True,
    }
    _write(tmp_path / "rollouts_failures.jsonl", [failed, unstarted])
    rows = _complete_repeat(0)
    rows[2] = {**_row(2, 0, _finished("def answer():\n    return 3")), "_ng_attempt_index": 1}

    code, out_dir = _run(tmp_path, rows)

    assert code == 0
    assert len(_read_jsonl(out_dir / "repeat-0" / "submissions.jsonl")) == 3


def test_a_task_whose_every_rollout_failed_leaves_its_repeats_unwritten(tmp_path):
    failed = [{"_ng_task_index": 2, "_ng_rollout_index": r, "_ng_attempt_index": 0, "error": "x"} for r in (0, 1)]
    _write(tmp_path / "rollouts_failures.jsonl", failed)

    code, out_dir = _run(tmp_path, _complete_repeat(0)[:2] + _complete_repeat(1)[:2])

    assert code == 3
    manifest = json.loads((out_dir / "manifest.json").read_text())
    for repeat in ("0", "1"):
        assert manifest["repeats"][repeat]["problems"] == [{"task_index": 2, "reason": "latest_attempt_failed"}]


def test_an_invalid_submission_is_not_exported(tmp_path):
    rows = _complete_repeat(0) + _complete_repeat(1)
    two_finishes = _finished("def answer():\n    return 1")
    second = _finished("def answer():\n    return 2", call_id="finish-2")
    two_finishes["output"] += second["output"]
    rows[3] = _row(0, 1, two_finishes)

    code, out_dir = _run(tmp_path, rows)

    assert code == 3
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["repeats"]["1"]["problems"] == [{"task_index": 0, "reason": "submission_invalid"}]


def test_rollouts_without_any_accepted_finish_are_refused(tmp_path, capsys):
    no_finish = {"status": "completed", "metadata": None, "output": []}
    code, out_dir = _run(tmp_path, [_row(task, 0, no_finish) for task in range(3)])

    assert code == 2
    assert "not the agentic recipe" in capsys.readouterr().err
    assert not out_dir.exists()


def test_a_line_that_is_not_an_object_is_refused(tmp_path, capsys):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("\n[1, 2]\n")

    assert main(["--rollouts", str(rollouts), "--out-dir", str(tmp_path / "out")]) == 2
    assert "rollouts.jsonl:2 is not a JSON object" in capsys.readouterr().err
    rollouts.write_text("{not json\n")
    assert main(["--rollouts", str(rollouts), "--out-dir", str(tmp_path / "out")]) == 2
    assert "rollouts.jsonl:1 is not valid JSON" in capsys.readouterr().err
    assert main(["--rollouts", str(tmp_path / "absent.jsonl"), "--out-dir", str(tmp_path / "out")]) == 2
    assert "absent.jsonl" in capsys.readouterr().err


def test_an_earlier_export_is_never_overwritten(tmp_path, capsys):
    rows = _complete_repeat(0)
    assert _run(tmp_path, rows)[0] == 0
    before = (tmp_path / "aa_batches" / "repeat-0" / "submissions.jsonl").read_bytes()

    code, out_dir = _run(tmp_path, rows)

    assert code == 2
    assert "exists; export into a new directory" in capsys.readouterr().err
    assert (out_dir / "repeat-0" / "submissions.jsonl").read_bytes() == before


def _managed_run(tmp_path):
    """A run Gym's Store manages: three tasks, one repeat, each answered once."""
    from unittest.mock import Mock

    from nemo_gym.rollouts import RolloutStore, RunManifest

    inputs = [
        {"_ng_task_index": task, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}, "problem_id": PROBLEMS[task]}
        for task in range(3)
    ]
    source = _write(tmp_path / "source.jsonl", inputs)
    prepare = Mock(
        side_effect=lambda: (inputs, RunManifest.create(source, inputs, {}, {"agent": {"responses_api_agents": {}}}))
    )
    output = tmp_path / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row in store.pending(3):
            store.record_dispatch(row)
            answer = f"def answer():\n    return {row['_ng_task_index']}"
            store.record_outcome(row | {"reward": 0.0, "response": _finished(answer)})
        first = next(row for row in store.selected("success") if row["_ng_task_index"] == 0)
        store.allocate_attempt(first)  # Recovery reserved a newer attempt, interrupted before it saved an outcome.
    return output, prepare


def test_a_managed_run_never_exports_an_answer_a_reserved_attempt_superseded(tmp_path, capsys):
    from nemo_gym.rollouts import RolloutStore

    output, prepare = _managed_run(tmp_path)

    code = main(["--rollouts", str(output), "--out-dir", str(tmp_path / "stale"), "--batch-size", "3"])

    assert code == 3
    manifest = json.loads((tmp_path / "stale" / "manifest.json").read_text())
    assert manifest["repeats"]["0"]["problems"] == [{"task_index": 0, "reason": "latest_attempt_missing"}]

    # Once that attempt saves its answer, a fresh export sends the new one.
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        (row,) = [row for row in resumed.pending(3) if row["_ng_task_index"] == 0]
        resumed.record_dispatch(row)
        resumed.record_outcome(row | {"reward": 0.0, "response": _finished("def answer():\n    return 9")})

    assert main(["--rollouts", str(output), "--out-dir", str(tmp_path / "fresh"), "--batch-size", "3"]) == 0
    submissions = _read_jsonl(tmp_path / "fresh" / "repeat-0" / "submissions.jsonl")
    assert submissions[0]["submission"]["generated_code"] == "```python\ndef answer():\n    return 9\n```"


@pytest.mark.parametrize("key", ["_ng_task_index", "_ng_rollout_index"])
def test_a_boolean_rollout_identity_is_refused(tmp_path, capsys, key):
    rows = _complete_repeat(0)
    rows[0] = {**rows[0], key: True}

    code, out_dir = _run(tmp_path, rows)

    assert code == 2
    assert f"no integer {key}" in capsys.readouterr().err
    assert not out_dir.exists()
