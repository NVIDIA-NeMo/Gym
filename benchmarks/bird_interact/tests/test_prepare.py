# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``benchmarks/bird_interact/prepare.py``: the ground truth merge and the rows kept, without a download."""

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.bird_interact import prepare as bird_prepare


CUSTOM_TEST = "def test_case(pred_sqls, sol_sqls, db_name, conn): pass"


def _row(instance_id: str) -> dict[str, Any]:
    """A public row: its ground truth, and its follow-up's, are withheld."""
    return {
        "instance_id": instance_id,
        "selected_database": "shop",
        "amb_user_query": "Show the best sellers.",
        "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
        "knowledge_ambiguity": [],
        "sol_sql": [],
        "external_knowledge": [],
        "test_cases": [],
        "preprocess_sql": [],
        "category": "Query",
        "follow_up": {"query": "And last year?"},
    }


def _truth(
    instance_id: str, *, test_cases: tuple[str, ...] = (), follow_up_test_cases: tuple[str, ...] = (), follow_up=True
) -> dict[str, Any]:
    truth: dict[str, Any] = {
        "instance_id": instance_id,
        "sol_sql": ["SELECT 1"],
        "external_knowledge": [1],
        "test_cases": list(test_cases),
    }
    if follow_up:
        truth["follow_up"] = {
            "sol_sql": ["SELECT 2"],
            "external_knowledge": [],
            "test_cases": list(follow_up_test_cases),
        }
    return truth


class TestSelectTasks:
    def test_the_ground_truth_of_the_task_and_its_follow_up_is_merged_in(self) -> None:
        kept, left_out = bird_prepare.select_tasks([_row("q1")], {"q1": _truth("q1")})

        (row,) = kept
        assert (row["sol_sql"], row["external_knowledge"]) == (["SELECT 1"], [1])
        assert row["follow_up"] == {
            "query": "And last year?",
            "sol_sql": ["SELECT 2"],
            "external_knowledge": [],
            "test_cases": [],
        }
        assert not left_out

    def test_rows_the_server_cannot_grade_are_left_out_with_their_reason(self) -> None:
        rows = [_row("no_truth"), _row("custom"), _row("custom_follow_up"), _row("follow_up_without_truth")]
        truth = {
            "custom": _truth("custom", test_cases=(CUSTOM_TEST,)),
            "custom_follow_up": _truth("custom_follow_up", follow_up_test_cases=(CUSTOM_TEST,)),
            "follow_up_without_truth": _truth("follow_up_without_truth", follow_up=False),
        }

        kept, left_out = bird_prepare.select_tasks(rows, truth)

        assert kept == []
        assert left_out == {"no ground truth": 2, "custom test cases": 2}


class TestPrepare:
    def test_writes_the_mode_file_from_the_download_and_the_ground_truth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        download = tmp_path / "download"
        download.mkdir()
        (download / bird_prepare.ROWS_FILE).write_text("".join(json.dumps(_row(i)) + "\n" for i in ("q1", "q2")))
        ground_truth = tmp_path / "ground_truth.jsonl"
        ground_truth.write_text(json.dumps(_truth("q1")) + "\n")
        monkeypatch.setattr(bird_prepare, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(bird_prepare, "download_release", lambda release: download)

        output = bird_prepare.prepare("full", "a_interact", ground_truth_jsonl=str(ground_truth))

        assert output == tmp_path / "data" / "full" / "a_interact.jsonl"
        assert [json.loads(line)["instance_id"] for line in output.read_text().splitlines()] == ["q1"]
        assert "kept 1 of 2 tasks; left out: 1 with no ground truth" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"release": "mini", "ground_truth_jsonl": "gt.jsonl"}, "release must be"),
            ({"mode": "both", "ground_truth_jsonl": "gt.jsonl"}, "mode must be"),
            ({}, "withholds its ground truth"),
        ],
    )
    def test_bad_arguments_and_a_missing_ground_truth_are_refused(self, arguments: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            bird_prepare.prepare(**arguments)
