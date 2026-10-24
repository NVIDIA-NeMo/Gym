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

import json
from pathlib import Path

from pytest import mark, raises

from benchmarks.swebench.pro.prepare import OUTPUT_FPATH as PRO_OUTPUT_FPATH
from benchmarks.swebench.pro_multiturn.prepare import (
    DECOMPOSITIONS_FPATH,
    build_row,
    prepare,
    spec_text,
)


REQUIREMENTS = "- Add `parse`.\n\n- `parse` rejects empty input.\n  - including whitespace\n\n- Add `render`."
INTERFACE = (
    "The golden patch introduces the following new public interfaces:\n\n"
    "Function: parse\nPath: a.py\n\n---\n\nFunction: render\nPath: b.py"
)
TURNS = [
    {"request": "Please add a parser.", "requirements": [[1, 4]], "interface": [[3, 4]]},
    {"request": "Now add rendering.", "requirements": [[6, 6]], "interface": [[8, 9]]},
]


def pro_row(instance_id: str = "instance_example") -> dict:
    return {
        "instance_id": instance_id,
        "requirements": REQUIREMENTS,
        "interface": INTERFACE,
        "responses_create_params": {"input": [{"role": "user", "content": "single-turn prompt"}]},
        "run_script": "#!/bin/bash\n",
    }


def test_build_row_puts_the_first_turn_in_the_prompt_and_the_rest_in_followups() -> None:
    row = build_row(pro_row(), TURNS)

    (first_message,) = row["responses_create_params"]["input"]
    assert first_message["role"] == "user"
    assert first_message["content"].startswith("<uploaded_files>\n/app\n</uploaded_files>")
    assert (
        "<request>\nPlease add a parser.\n\nRequirements:\n- Add `parse`.\n\n- `parse` rejects empty input.\n"
        "  - including whitespace\n\nNew interfaces introduced:\nFunction: parse\nPath: a.py\n</request>"
    ) in first_message["content"]
    assert "Add `render`" not in first_message["content"]
    assert row["followup_user_messages"] == [
        "Now add rendering.\n\nRequirements:\n- Add `render`.\n\nNew interfaces introduced:\nFunction: render\nPath: b.py"
    ]
    assert row["run_script"] == "#!/bin/bash\n"


@mark.parametrize(
    "turns, message",
    [
        (TURNS[:1], "at least two turns"),
        ([TURNS[0], TURNS[1] | {"requirements": []}], r"requirements lines missing=\[6\] repeated=\[\]"),
        ([TURNS[0], TURNS[1] | {"requirements": [[4, 6]]}], r"requirements lines missing=\[\] repeated=\[4\]"),
        ([TURNS[0], TURNS[1] | {"interface": [[8, 12]]}], r"interface range \[8, 12\] is outside 1..9"),
    ],
)
def test_build_row_rejects_turns_that_do_not_cover_the_spec_exactly_once(turns: list[dict], message: str) -> None:
    with raises(ValueError, match=message):
        build_row(pro_row(), turns)


@mark.parametrize(
    "request_text, message",
    [
        ("Please add a parser so the tests pass.", "turn 1 request mentions tests or grading: 'tests'"),
        ("Add a parser in `parser_utils.py`.", "turn 1 request introduces 'parser_utils.py'"),
        ("Add `parse` and then `render`.", "turn 1 request introduces 'render'"),
        ("Please add a parser; we'll also need rendering.", "turn 1 request previews later requests"),
    ],
)
def test_build_row_rejects_requests_that_leak_or_steer(request_text: str, message: str) -> None:
    turns = [TURNS[0] | {"request": request_text}, TURNS[1]]
    with raises(ValueError, match=message):
        build_row(pro_row(), turns)


def test_build_row_rejects_feedback_on_earlier_turns() -> None:
    turns = [TURNS[0], TURNS[1] | {"request": "That works. Now add `render`."}]
    with raises(ValueError, match="turn 2 request gives feedback on earlier work: 'That works'"):
        build_row(pro_row(), turns)


def test_build_row_rejects_requests_that_copy_the_problem_statement_or_repeat_their_lines() -> None:
    row = pro_row() | {
        "problem_statement": "Calling the parser on an empty string crashes the whole import job.",
        "requirements": REQUIREMENTS.replace(
            "rejects empty input.", "returns an error that names the offending field."
        ),
    }

    copied = [
        TURNS[0] | {"request": "The parser on an empty string crashes the whole import job, please fix."},
        TURNS[1],
    ]
    with raises(
        ValueError,
        match="turn 1 request copies the problem statement: 'the parser on an empty string crashes the'",
    ):
        build_row(row, copied)

    repeated = [
        TURNS[0],
        TURNS[1] | {"request": "Now make sure `parse` returns an error that names the offending field."},
    ]
    with raises(
        ValueError,
        match="turn 2 request repeats requirement or interface text: 'parse returns an error that names the offending'",
    ):
        build_row(row, repeated)

    own_words = [TURNS[0] | {"request": "Empty input breaks imports, so let's start with a parser."}, TURNS[1]]
    assert build_row(row, own_words)["followup_user_messages"][0].startswith("Now add rendering.")


def test_prepare_keeps_only_decomposed_instances(tmp_path: Path) -> None:
    source_fpath = tmp_path / "pro.jsonl"
    source_fpath.write_text("".join(json.dumps(pro_row(name)) + "\n" for name in ("a", "b")), encoding="utf-8")
    decompositions_fpath = tmp_path / "decompositions.json"
    decompositions_fpath.write_text(json.dumps({"b": TURNS}), encoding="utf-8")

    output_fpath = prepare(source_fpath, decompositions_fpath, tmp_path / "out.jsonl")

    rows = [json.loads(line) for line in output_fpath.read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in rows] == ["b"]
    decompositions_fpath.write_text(json.dumps({"c": TURNS}), encoding="utf-8")
    with raises(ValueError, match=r"missing from .*: \['c'\]"):
        prepare(source_fpath, decompositions_fpath, tmp_path / "out.jsonl")


@mark.skipif(not PRO_OUTPUT_FPATH.is_file(), reason="prepared SWE-bench Pro data is not available")
def test_shipped_decompositions_cover_their_tasks(tmp_path: Path) -> None:
    output_fpath = prepare(output_fpath=tmp_path / "out.jsonl")

    decompositions = json.loads(DECOMPOSITIONS_FPATH.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in output_fpath.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["instance_id"] for row in rows) == sorted(decompositions)
    assert all(len(row["followup_user_messages"]) == len(decompositions[row["instance_id"]]) - 1 for row in rows)


@mark.parametrize(
    "raw, expected",
    [
        ('"- Add `parse`.\\n\\n- Add `render`."', "- Add `parse`.\n\n- Add `render`."),
        ('"- Say \\"hi\\".\n- Keep lines."', '"- Say \\"hi\\".\n- Keep lines."'),
        ("- Plain text.\n- Two lines.", "- Plain text.\n- Two lines."),
    ],
)
def test_spec_text_decodes_json_quoted_fields_only_when_they_have_no_real_newlines(raw: str, expected: str) -> None:
    assert spec_text({"requirements": raw}, "requirements") == expected
