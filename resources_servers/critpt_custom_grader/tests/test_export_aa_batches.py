# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export of accepted finish answers as Artificial Analysis grading batches, on synthetic rollouts."""

import hashlib
import json

import pytest

from resources_servers.critpt_custom_grader.export_aa_batches import generated_code, main


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


def test_unscorable_missing_or_duplicate_rollouts_leave_their_repeat_unwritten(tmp_path, capsys):
    unscorable = _complete_repeat(1)
    unscorable[0] = _row(0, 1, {"status": "failed", "error": {"message": "provider down"}, "output": []})
    missing = _complete_repeat(2)[1:]
    duplicate = _complete_repeat(3) + [_row(2, 3, _finished("def answer():\n    return 3"))]
    no_response = _complete_repeat(4)
    no_response[2] = {key: value for key, value in no_response[2].items() if key != "response"}

    code, out_dir = _run(tmp_path, _complete_repeat(0) + unscorable + missing + duplicate + no_response)

    assert code == 3
    assert (out_dir / "repeat-0" / "submissions.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["repeats"]["0"]["written"] is True
    expected = {
        "1": [{"task_index": 0, "reason": "response_incomplete"}],
        "2": [{"task_index": 0, "reason": "0 rollouts"}],
        "3": [{"task_index": 2, "reason": "2 rollouts"}],
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


def test_a_line_that_is_not_an_object_is_refused(tmp_path, capsys):
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("\n[1, 2]\n")

    assert main(["--rollouts", str(rollouts), "--out-dir", str(tmp_path / "out")]) == 2
    assert "rollouts.jsonl:2 is not a JSON object" in capsys.readouterr().err


def test_an_earlier_export_is_never_overwritten(tmp_path, capsys):
    rows = _complete_repeat(0)
    assert _run(tmp_path, rows)[0] == 0
    before = (tmp_path / "aa_batches" / "repeat-0" / "submissions.jsonl").read_bytes()

    code, out_dir = _run(tmp_path, rows)

    assert code == 2
    assert "exists; export into a new directory" in capsys.readouterr().err
    assert (out_dir / "repeat-0" / "submissions.jsonl").read_bytes() == before


def test_generated_code_wraps_sources_like_the_critpt_server():
    assert generated_code("x = 1") == "```python\nx = 1\n```"
    assert generated_code(None) == "```python\n```"
