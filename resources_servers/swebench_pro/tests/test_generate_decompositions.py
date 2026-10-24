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
from argparse import Namespace
from pathlib import Path

from pytest import MonkeyPatch, raises

from benchmarks.swebench.pro_multiturn import generate_decompositions as module
from benchmarks.swebench.pro_multiturn.generate_decompositions import decompose, judge, merge, parse_json, split_issues


ROW = {
    "instance_id": "instance_example",
    "problem_statement": "Add parsing and rendering.",
    "requirements": "- Add `parse`.\n  - including whitespace\n\n- Add `render`.",
    "interface": "Function: parse\n\nFunction: render",
}


def test_split_issues_flags_ranges_that_cut_through_a_bullet() -> None:
    whole = [
        {"request": "a", "requirements": [[1, 2]], "interface": [[1, 1]]},
        {"request": "b", "requirements": [[4, 4]], "interface": [[3, 3]]},
    ]
    assert split_issues(ROW, whole) == []

    cut = [
        {"request": "a", "requirements": [[1, 1]], "interface": [[1, 1]]},
        {"request": "b", "requirements": [[2, 4]], "interface": [[3, 3]]},
    ]
    assert split_issues(ROW, cut) == [
        "turn 1 requirements range [1, 1] ends inside a bullet",
        "turn 2 requirements range [2, 4] starts inside a bullet",
    ]


def test_parse_json_ignores_reasoning_and_surrounding_text() -> None:
    assert parse_json('<think>{"no": 1}</think>Here it is: {"decomposable": false} done') == {"decomposable": False}
    with raises(ValueError, match="no JSON object"):
        parse_json("nothing here")


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies

    async def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        return self.replies.pop(0)


async def test_judge_flags_only_what_a_majority_of_valid_votes_flag() -> None:
    turns = [
        {"request": "Add parsing.", "requirements": [[1, 2]], "interface": [[1, 1]]},
        {"request": "Now rendering.", "requirements": [[4, 4]], "interface": [[3, 3]]},
    ]
    clean = json.dumps({flag: False for flag in module.JUDGE_FLAGS} | {"evidence": ""})
    leaky = json.dumps({flag: flag == "adds_information" for flag in module.JUDGE_FLAGS} | {"evidence": "x"})

    verdict = await judge(FakeClient([leaky, clean, "not json"]), ROW, turns, 3)
    assert verdict["flagged"] == [] and verdict["valid"] == 2

    verdict = await judge(FakeClient([leaky, leaky, clean]), ROW, turns, 3)
    assert verdict["flagged"] == ["adds_information"]


async def test_rewrite_replaces_only_the_requests_and_keeps_the_ranges() -> None:
    current = [
        {
            "request": "Add parsing and rendering, starting with parsing.",
            "requirements": [[1, 2]],
            "interface": [[1, 1]],
        },
        {"request": "Now rendering.", "requirements": [[4, 4]], "interface": [[3, 3]]},
    ]
    clean = json.dumps({flag: False for flag in module.JUDGE_FLAGS} | {"evidence": ""})
    client = FakeClient(
        [
            json.dumps({"requests": ["Let's start with parsing."]}),
            json.dumps({"requests": ["Let's start with parsing.", "Next, the output side."]}),
            clean,
            clean,
            clean,
        ]
    )

    result = await decompose(client, ROW, "", Namespace(attempts=2, judge_samples=3), current)

    assert result["status"] == "accepted"
    assert result["history"][0]["issues"] == ["return exactly 2 requests, not 1"]
    assert result["turns"] == [
        current[0] | {"request": "Let's start with parsing."},
        current[1] | {"request": "Next, the output side."},
    ]


def test_merge_adds_accepted_allowed_records_and_replaces_only_when_asked(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    decompositions_fpath = tmp_path / "decompositions.json"
    decompositions_fpath.write_text(json.dumps({"kept": ["hand-written"]}), encoding="utf-8")
    monkeypatch.setattr(module, "DECOMPOSITIONS_FPATH", decompositions_fpath)
    output = tmp_path / "generated.jsonl"
    records = [
        {"instance_id": "kept", "status": "accepted", "turns": ["generated"]},
        {"instance_id": "new", "status": "accepted", "turns": ["t"]},
        {"instance_id": "rejected", "status": "rejected"},
        {"instance_id": "not_golden", "status": "accepted", "turns": ["t"]},
    ]
    output.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    allowed = tmp_path / "ids.txt"
    allowed.write_text("kept\nnew\nrejected\n", encoding="utf-8")

    merge(Namespace(output=output, instance_ids=allowed, replace=False))

    assert json.loads(decompositions_fpath.read_text(encoding="utf-8")) == {"kept": ["hand-written"], "new": ["t"]}

    merge(Namespace(output=output, instance_ids=allowed, replace=True))

    assert json.loads(decompositions_fpath.read_text(encoding="utf-8")) == {"kept": ["generated"], "new": ["t"]}
