# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``benchmarks/bird_interact/prepare.py``: the ground truth merge and the rows kept, without a download."""

import json
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
from pydantic import ValidationError

from benchmarks.bird_interact import prepare as bird_prepare


CUSTOM_TEST = "def test_case(pred_sqls, sol_sqls, db_name, conn): pass"


def _row(instance_id: str, category: str = "Query") -> dict[str, Any]:
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
        "category": category,
        "follow_up": {"query": "And last year?", "category": "Query"},
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
            "category": "Query",
            "sol_sql": ["SELECT 2"],
            "external_knowledge": [],
            "test_cases": [],
        }
        assert not left_out

    def test_a_solution_written_as_one_string_becomes_a_list_of_it(self) -> None:
        truth = _truth("q1")
        truth["follow_up"]["sol_sql"] = "SELECT 2"

        (row,), _ = bird_prepare.select_tasks([_row("q1")], {"q1": truth})

        assert row["follow_up"]["sol_sql"] == ["SELECT 2"]

    def test_rows_the_server_cannot_grade_are_left_out_with_their_reason(self) -> None:
        management_follow_up = _row("management_follow_up")
        management_follow_up["follow_up"]["category"] = "Management"
        rows = [
            _row("no_truth"),
            _row("management", "Management"),
            management_follow_up,
            _row("follow_up_without_truth"),
        ]
        truth = {
            "management": _truth("management", test_cases=(CUSTOM_TEST,)),
            "management_follow_up": _truth("management_follow_up", follow_up_test_cases=(CUSTOM_TEST,)),
            "follow_up_without_truth": _truth("follow_up_without_truth", follow_up=False),
        }

        kept, left_out = bird_prepare.select_tasks(rows, truth)

        assert kept == []
        assert left_out == {"no ground truth": 2, "management task": 2}

    @pytest.mark.parametrize("phase", ["task", "follow_up"])
    def test_a_management_task_is_left_out_by_its_category_even_without_test_cases(self, phase: str) -> None:
        row = _row("m1", "Management" if phase == "task" else "Query")
        if phase == "follow_up":
            row["follow_up"]["category"] = "Management"

        kept, left_out = bird_prepare.select_tasks([row], {"m1": _truth("m1")})

        assert (kept, left_out) == ([], {"management task": 1})

    def test_a_phase_without_a_category_is_a_query_as_in_the_adk(self) -> None:
        row = _row("q1")
        del row["category"]

        kept, _ = bird_prepare.select_tasks([row], {"q1": _truth("q1")})

        assert [task["instance_id"] for task in kept] == ["q1"]

    def test_a_row_the_server_could_not_read_is_refused(self) -> None:
        row = _row("q1")
        del row["selected_database"]

        with pytest.raises(ValidationError):
            bird_prepare.select_tasks([row], {"q1": _truth("q1")})

    def test_a_query_task_is_kept_whatever_its_test_cases(self) -> None:
        # BIRD-Interact grades a query by comparing its result with the solution's, so its test cases do not count.
        truth = {"query": _truth("query", test_cases=(CUSTOM_TEST,), follow_up_test_cases=(CUSTOM_TEST,))}

        kept, left_out = bird_prepare.select_tasks([_row("query")], truth)

        assert [row["instance_id"] for row in kept] == ["query"]
        assert not left_out


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
        (task,) = [json.loads(line) for line in output.read_text().splitlines()]
        assert task["task_id"] == {"taskset": "bird_interact:a_interact", "task_id": "q1"}
        assert task["task_input"]["instance_id"] == "q1" and task["task_input"]["sol_sql"] == ["SELECT 1"]
        assert "kept 1 of 2 tasks; left out: 1 with no ground truth" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"release": "mini", "ground_truth_jsonl": "gt.jsonl"}, "release must be"),
            ({"mode": "both", "ground_truth_jsonl": "gt.jsonl"}, "mode must be"),
            ({}, "withholds its ground truth"),
        ],
    )
    def test_bad_arguments_and_a_missing_ground_truth_are_refused(
        self, arguments: dict, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An empty data/: the ground truth the authors sent is not there.
        monkeypatch.setattr(bird_prepare, "DATA_DIR", tmp_path)
        with pytest.raises(ValueError, match=message):
            bird_prepare.prepare(**arguments)

    def _download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        download = tmp_path / "download"
        download.mkdir()
        (download / bird_prepare.ROWS_FILE).write_text(json.dumps(_row("q1")) + "\n")
        monkeypatch.setattr(bird_prepare, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(bird_prepare, "download_release", lambda release: download)

    @pytest.mark.parametrize("release", ["lite", "full"])
    def test_the_ground_truth_in_data_is_read_by_default(
        self, release: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._download(tmp_path, monkeypatch)
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / bird_prepare.GROUND_TRUTH_FILES[release]).write_text(json.dumps(_truth("q1")) + "\n")

        output = bird_prepare.prepare(release, "c_interact")

        assert [json.loads(line)["task_input"]["sol_sql"] for line in output.read_text().splitlines()] == [
            ["SELECT 1"]
        ]

    def test_each_release_reads_its_own_download(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bird_prepare, "DATA_DIR", tmp_path / "data")
        for release in ("lite", "full"):
            bird_prepare.database_dir(release).mkdir(parents=True)
            row = _row(f"{release}_q1")
            (bird_prepare.database_dir(release) / bird_prepare.ROWS_FILE).write_text(json.dumps(row) + "\n")
        monkeypatch.setattr(bird_prepare, "download_release", bird_prepare.database_dir)
        ground_truth = tmp_path / "ground_truth.jsonl"
        ground_truth.write_text("".join(json.dumps(_truth(f"{release}_q1")) + "\n" for release in ("lite", "full")))

        written = {
            release: bird_prepare.prepare(release, "c_interact", ground_truth_jsonl=str(ground_truth))
            for release in ("lite", "full")
        }

        for release, output in written.items():
            (task,) = [json.loads(line) for line in output.read_text().splitlines()]
            assert task["task_input"]["instance_id"] == f"{release}_q1"

    def test_the_ground_truth_of_another_release_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._download(tmp_path, monkeypatch)
        ground_truth = tmp_path / "other_release.jsonl"
        ground_truth.write_text(json.dumps(_truth("full_q1")) + "\n")

        with pytest.raises(ValueError, match="none of lite's tasks"):
            bird_prepare.prepare("lite", "c_interact", ground_truth_jsonl=str(ground_truth))

    def test_an_empty_ground_truth_file_downloads_and_keeps_the_rows_already_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._download(tmp_path, monkeypatch)
        ground_truth = tmp_path / "ground_truth.jsonl"
        ground_truth.write_text(json.dumps(_truth("q1")) + "\n")
        output = bird_prepare.prepare("lite", "c_interact", ground_truth_jsonl=str(ground_truth))
        written = output.read_text()
        ground_truth.write_text("")

        assert bird_prepare.prepare("lite", "c_interact", ground_truth_jsonl=str(ground_truth)) == output

        assert output.read_text() == written
        assert "is empty, so no rows were written" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (
                json.dumps({"sol_sql": ["SELECT 1"]}) + "\n",
                r"ground_truth.jsonl, line 1: the entry has no instance_id",
            ),
            ("\n" + json.dumps(_truth("q1"))[:-1] + "\n", r"ground_truth.jsonl, line 2: not JSON"),
            (None, r"No ground truth file at .*ground_truth.jsonl"),
        ],
    )
    def test_a_bad_ground_truth_file_is_refused_by_name_before_the_download(
        self, content: str | None, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bird_prepare, "DATA_DIR", tmp_path / "data")
        downloads: list[str] = []
        monkeypatch.setattr(bird_prepare, "download_release", downloads.append)
        ground_truth = tmp_path / "ground_truth.jsonl"
        if content is not None:
            ground_truth.write_text(content)

        with pytest.raises(ValueError, match=message):
            bird_prepare.prepare("lite", "c_interact", ground_truth_jsonl=str(ground_truth))

        assert downloads == []

    def test_a_null_ground_truth_field_is_an_empty_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._download(tmp_path, monkeypatch)
        truth = _truth("q1") | {"external_knowledge": None}
        ground_truth = tmp_path / "ground_truth.jsonl"
        ground_truth.write_text(json.dumps(truth) + "\n")

        output = bird_prepare.prepare("lite", "c_interact", ground_truth_jsonl=str(ground_truth))

        (task,) = [json.loads(line) for line in output.read_text().splitlines()]
        assert task["task_input"]["external_knowledge"] == []


@pytest.mark.parametrize("release", ["lite", "full"])
def test_download_release_is_pinned_to_the_ground_truth_revision(
    release: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(bird_prepare, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs) or kwargs["local_dir"]
    )

    assert bird_prepare.download_release(release) == tmp_path / release / "databases"
    assert calls == [
        {
            "repo_id": f"birdsql/bird-interact-{release}",
            "repo_type": "dataset",
            "revision": bird_prepare.RELEASE_REVISIONS[release],
            "local_dir": tmp_path / release / "databases",
        }
    ]
