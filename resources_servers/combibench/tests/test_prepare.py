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

"""Preparation is the module that produces a silently wrong dataset rather than a crash; test it with the fetch stubbed."""

import json
import sys
from pathlib import Path

import pytest

from benchmarks.combibench import prepare
from benchmarks.combibench_with_solution import prepare as prepare_with_solution


FIXTURES = Path(__file__).resolve().parent / "fixtures"
SYNTHETIC = json.loads((FIXTURES / "synthetic_problems.json").read_text(encoding="utf-8"))


def _hundred(split: str = "test") -> list[dict]:
    """A complete synthetic corpus: the five fixtures cycled to the expected size."""
    rows = []
    for index in range(prepare.EXPECTED_ROWS[split]):
        base = dict(SYNTHETIC[index % len(SYNTHETIC)])
        base["theorem_name"] = f"{base['theorem_name']}_{index}"
        base["formal_statement"] = base["formal_statement"].replace(
            SYNTHETIC[index % len(SYNTHETIC)]["theorem_name"], base["theorem_name"]
        )
        rows.append(base)
    return rows


def _must_not_fetch(*_args, **_kwargs):
    raise AssertionError("no upstream fetch may happen in this test")


_REAL_LOAD_GITHUB_ROWS = prepare.load_github_rows


@pytest.fixture(autouse=True)
def _stub_default_source(request, monkeypatch):
    """The default source is GitHub; serve it from the same synthetic corpus the HF stubs use.

    The GitHub parser tests exercise the real loader against a local tree.
    """
    if request.cls is not None and request.cls.__name__ == "TestGithubSource":
        return
    monkeypatch.setattr(prepare, "load_github_rows", lambda split, cache_dir: prepare.load_hf_rows(split))


class TestRowShape:
    def test_default_source_is_github(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: _hundred(split))
        output = prepare.prepare(output=tmp_path / "out.jsonl")
        row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        assert row["dataset_source"] == "github" and row["dataset_revision"] == prepare.GITHUB_REVISION

    def test_row_carries_task_fields_and_provenance(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: _hundred(split))
        output = prepare.prepare(split="test", source="hf", output=tmp_path / "out.jsonl")
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 100
        row = rows[0]
        assert set(row) == {
            "theorem_name",
            "formal_statement",
            "answers",
            "natural_language",
            "tag",
            "source",
            "split",
            "dataset_source",
            "dataset_revision",
        }
        assert row["answers"] == ["10"] and row["split"] == "test"
        assert row["dataset_source"] == "hf" and row["dataset_revision"] == prepare.HF_REVISION
        assert "responses_create_params" not in row, "prompts are applied at rollout time, not baked in"

    def test_with_solution_entry_point_selects_its_split(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: _hundred(split))
        output = prepare_with_solution.prepare(output=tmp_path / "out.jsonl")
        row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
        assert row["split"] == "test_with_solution"


class TestFailClosed:
    def test_short_corpus_is_rejected_and_nothing_is_written(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: _hundred(split)[:99])
        output = tmp_path / "out.jsonl"
        with pytest.raises(SystemExit) as excinfo:
            prepare.prepare(output=output)
        assert "loaded 99, expected 100" in str(excinfo.value)
        assert not output.exists()

    def test_answer_count_must_match_solution_abbrevs(self, tmp_path, monkeypatch) -> None:
        rows = _hundred()
        rows[3]["answer"] = ["10", "11"]
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: rows)
        with pytest.raises(SystemExit) as excinfo:
            prepare.prepare(output=tmp_path / "out.jsonl")
        assert "1 solution abbrev(s) but 2 answer(s)" in str(excinfo.value)

    def test_statement_without_sorry_is_rejected(self, tmp_path, monkeypatch) -> None:
        rows = _hundred()
        rows[0]["formal_statement"] = rows[0]["formal_statement"].replace("sorry", "10")
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: rows)
        with pytest.raises(SystemExit, match="no sorry to fill"):
            prepare.prepare(output=tmp_path / "out.jsonl")

    def test_limit_skips_the_manifest_but_still_validates(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", lambda split: _hundred(split)[:7])
        output = prepare.prepare(limit=5, output=tmp_path / "out.jsonl")
        assert len(output.read_text(encoding="utf-8").splitlines()) == 5


class TestArguments:
    @pytest.mark.parametrize("limit", [0, -1])
    def test_non_positive_limit_is_rejected_before_fetching(self, limit, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", _must_not_fetch)
        monkeypatch.setattr(prepare, "load_github_rows", _must_not_fetch)
        with pytest.raises(ValueError, match="positive"):
            prepare.prepare(limit=limit, output=tmp_path / "out.jsonl")

    def test_unknown_split_is_rejected_before_fetching(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", _must_not_fetch)
        with pytest.raises(ValueError, match="split"):
            prepare.prepare(split="train", output=tmp_path / "out.jsonl")

    def test_cli_rejects_a_non_positive_limit_at_parse_time(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", _must_not_fetch)
        monkeypatch.setattr(sys, "argv", ["prepare", "--limit", "0", "--output", str(tmp_path / "o.jsonl")])
        with pytest.raises(SystemExit):
            prepare.main()

    def test_cli_source_file_path(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(prepare, "load_hf_rows", _must_not_fetch)
        output = tmp_path / "example.jsonl"
        monkeypatch.setattr(
            sys,
            "argv",
            ["prepare", "--source-file", str(FIXTURES / "synthetic_problems.json"), "--output", str(output)],
        )
        prepare.main()
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 5
        assert rows[0]["dataset_source"] == "synthetic" and rows[0]["dataset_revision"] is None


class TestGithubSource:
    def test_lean_files_and_metadata_are_joined(self, tmp_path, monkeypatch) -> None:
        lean_dir = tmp_path / "lean" / "CombiBench"
        (lean_dir / "with_solution").mkdir(parents=True)
        (lean_dir / "metadata.csv").write_text(
            "theorem_name,natural_language,answer,source,tag,formal_statement_existence,comment\n"
            'p_1,"Two coins.","[3 / 4, 3 / 4, 1 / 2]",,hackmath,,\n'
            'p_2,"Prove it.",,,imo,,\n'
            'p_3,"Closed form.","fun n => (∑ i : Fin k, (n i + 1))",,brualdi,,\n',
            encoding="utf-8",
        )
        (lean_dir / "p_1.lean").write_text(
            "import Mathlib\n\nabbrev p_1_1_solution : ℕ := sorry\n\nabbrev p_1_2_solution : ℕ := sorry\n\n"
            "abbrev p_1_3_solution : ℕ := sorry\n\n/--\nTwo coins.\n-/\ntheorem p_1 : True := by sorry\n",
            encoding="utf-8",
        )
        (lean_dir / "p_2.lean").write_text(
            "import Mathlib\n\n-- a note\ntheorem p_2 : True := by sorry\n", encoding="utf-8"
        )
        (lean_dir / "p_3.lean").write_text(
            "import Mathlib\n\nabbrev p_3_solution {k} : (Fin k → ℕ) → ℕ := sorry\n\ntheorem p_3 : True := by sorry\n",
            encoding="utf-8",
        )
        for name in ("p_1", "p_2", "p_3"):
            (lean_dir / "with_solution" / f"{name}_sol.lean").write_text(
                f"import Mathlib\n\ntheorem {name} : True := by sorry\n", encoding="utf-8"
            )
        monkeypatch.setattr(prepare, "fetch_github_tree", lambda cache_dir: lean_dir)
        monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"test": 3, "test_with_solution": 3})

        output = prepare.prepare(source="github", output=tmp_path / "gh.jsonl", cache_dir=tmp_path)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert [r["answers"] for r in rows] == [
            ["3 / 4", "3 / 4", "1 / 2"],
            None,
            ["fun n => (∑ i : Fin k, (n i + 1))"],
        ]
        assert "/--" not in rows[0]["formal_statement"] and "-- a note" not in rows[1]["formal_statement"]
        assert rows[0]["dataset_revision"] == prepare.GITHUB_REVISION

        solution = prepare.prepare(
            source="github", split="test_with_solution", output=tmp_path / "s.jsonl", cache_dir=tmp_path
        )
        assert len(solution.read_text(encoding="utf-8").splitlines()) == 3

    def test_missing_lean_file_fails_closed(self, tmp_path, monkeypatch) -> None:
        lean_dir = tmp_path / "lean" / "CombiBench"
        lean_dir.mkdir(parents=True)
        (lean_dir / "metadata.csv").write_text(
            "theorem_name,natural_language,answer,source,tag,formal_statement_existence,comment\np_1,x,,,imo,,\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(prepare, "fetch_github_tree", lambda cache_dir: lean_dir)
        with pytest.raises(SystemExit, match="missing"):
            prepare.prepare(source="github", output=tmp_path / "gh.jsonl", cache_dir=tmp_path)
