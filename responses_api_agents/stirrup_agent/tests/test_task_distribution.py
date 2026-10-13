# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import random
from pathlib import Path

import pytest

from responses_api_agents.stirrup_agent import task_distribution as td
from responses_api_agents.stirrup_agent.task_distribution import (
    MISSING_VALUE,
    build_distribution,
    build_distribution_from_dataset,
    compose_key,
    extract_metadata,
    iter_dataset_rows,
    main,
    resolve_default_dataset,
    sample_task_ids,
)


def _row(task_id: str, **metadata) -> dict:
    return {"responses_create_params": {"input": "", "metadata": {"task_id": task_id, **metadata}}}


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class TestExtractMetadata:
    def test_responses_create_params_metadata(self) -> None:
        row = _row("t1", sector="Legal")
        assert extract_metadata(row) == {"task_id": "t1", "sector": "Legal"}

    def test_top_level_metadata_fallback(self) -> None:
        row = {"metadata": {"task_id": "t1", "sector": "Legal"}}
        assert extract_metadata(row) == {"task_id": "t1", "sector": "Legal"}

    def test_row_itself_fallback(self) -> None:
        row = {"task_id": "t1", "sector": "Legal"}
        assert extract_metadata(row) == {"task_id": "t1", "sector": "Legal"}

    def test_non_mapping_params_falls_through(self) -> None:
        row = {"responses_create_params": "oops", "metadata": {"task_id": "t1"}}
        assert extract_metadata(row) == {"task_id": "t1"}


class TestComposeKey:
    def test_single_column(self) -> None:
        assert compose_key({"sector": "Legal"}, ["sector"]) == "Legal"

    def test_composite_key(self) -> None:
        meta = {"sector": "Legal", "occupation": "Lawyer"}
        assert compose_key(meta, ["sector", "occupation"]) == "Legal | Lawyer"

    def test_missing_value_placeholder(self) -> None:
        assert compose_key({}, ["sector"]) == MISSING_VALUE

    def test_custom_separator(self) -> None:
        meta = {"a": "x", "b": "y"}
        assert compose_key(meta, ["a", "b"], separator="::") == "x::y"

    def test_non_string_value_is_stringified(self) -> None:
        assert compose_key({"n": 5}, ["n"]) == "5"


class TestBuildDistribution:
    def test_percentages_and_task_ids(self) -> None:
        rows = [
            _row("a", sector="Legal"),
            _row("b", sector="Legal"),
            _row("c", sector="Healthcare"),
            _row("d", sector="Finance"),
        ]
        dist = build_distribution(rows, ["sector"])
        assert dist["Legal"]["percentage"] == 0.5
        assert dist["Legal"]["task_ids"] == ["a", "b"]
        assert dist["Healthcare"]["percentage"] == 0.25
        assert dist["Finance"]["task_ids"] == ["d"]

    def test_ordering_is_descending_by_share(self) -> None:
        rows = [
            _row("a", sector="Legal"),
            _row("b", sector="Legal"),
            _row("c", sector="Healthcare"),
        ]
        assert list(build_distribution(rows, ["sector"]).keys()) == ["Legal", "Healthcare"]

    def test_percentages_sum_to_one_unrounded(self) -> None:
        rows = [_row(str(i), sector=s) for i, s in enumerate(["a", "a", "b", "c", "c", "c", "d"])]
        dist = build_distribution(rows, ["sector"], precision=None)
        assert pytest.approx(sum(e["percentage"] for e in dist.values())) == 1.0

    def test_composite_columns(self) -> None:
        rows = [
            _row("a", sector="Legal", occupation="Lawyer"),
            _row("b", sector="Legal", occupation="Paralegal"),
        ]
        dist = build_distribution(rows, ["sector", "occupation"])
        assert set(dist.keys()) == {"Legal | Lawyer", "Legal | Paralegal"}

    def test_empty_rows_yields_empty(self) -> None:
        assert build_distribution([], ["sector"]) == {}

    def test_missing_column_grouped_under_placeholder(self) -> None:
        rows = [_row("a"), _row("b", sector="Legal")]
        dist = build_distribution(rows, ["sector"])
        assert MISSING_VALUE in dist
        assert dist[MISSING_VALUE]["task_ids"] == ["a"]

    def test_missing_task_id_uses_positional_fallback(self) -> None:
        rows = [{"responses_create_params": {"metadata": {"sector": "Legal"}}}]
        dist = build_distribution(rows, ["sector"])
        assert dist["Legal"]["task_ids"] == ["task_id_index_0"]

    def test_requires_columns(self) -> None:
        with pytest.raises(ValueError):
            build_distribution([_row("a", sector="Legal")], [])

    def test_precision_rounding(self) -> None:
        rows = [_row(str(i), sector="a" if i == 0 else "b") for i in range(3)]
        dist = build_distribution(rows, ["sector"], precision=2)
        assert dist["b"]["percentage"] == 0.67


class TestIterAndDatasetWrapper:
    def test_iter_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps(_row("a", sector="Legal")) + "\n\n", encoding="utf-8")
        assert len(list(iter_dataset_rows(path))) == 1

    def test_iter_raises_on_bad_json(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            list(iter_dataset_rows(path))

    def test_build_from_dataset(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "d.jsonl", [_row("a", sector="Legal"), _row("b", sector="Legal")])
        dist = build_distribution_from_dataset(path, ["sector"])
        assert dist["Legal"]["percentage"] == 1.0


class TestSampleTaskIds:
    def _dist(self):
        return {
            "Legal": {"percentage": 0.5, "task_ids": ["a", "b"]},
            "Healthcare": {"percentage": 0.5, "task_ids": ["c", "d"]},
        }

    def test_zero_or_negative_returns_empty(self) -> None:
        assert sample_task_ids(self._dist(), 0) == []
        assert sample_task_ids(self._dist(), -3) == []

    def test_without_replacement_no_duplicates(self) -> None:
        rng = random.Random(0)
        sampled = sample_task_ids(self._dist(), 3, rng=rng)
        assert len(sampled) == 3
        assert len(set(sampled)) == 3

    def test_without_replacement_capped_at_total(self) -> None:
        sampled = sample_task_ids(self._dist(), 100, rng=random.Random(1))
        assert sorted(sampled) == ["a", "b", "c", "d"]

    def test_with_replacement_allows_more_than_total(self) -> None:
        sampled = sample_task_ids(self._dist(), 10, rng=random.Random(2), replace=True)
        assert len(sampled) == 10

    def test_empty_distribution_returns_empty(self) -> None:
        assert sample_task_ids({}, 5) == []

    def test_zero_weight_distribution_returns_empty(self) -> None:
        dist = {"x": {"percentage": 0.0, "task_ids": ["a"]}}
        assert sample_task_ids(dist, 5) == []
        assert sample_task_ids(dist, 5, replace=True) == []

    def test_with_replacement_skips_empty_groups(self) -> None:
        dist = {"x": {"percentage": 1.0, "task_ids": []}}
        assert sample_task_ids(dist, 3, rng=random.Random(3), replace=True) == []


class TestResolveDefaultDataset:
    def test_returns_first_existing(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.jsonl"
        present = _write_jsonl(tmp_path / "present.jsonl", [_row("a", sector="Legal")])
        assert resolve_default_dataset([missing, present]) == present

    def test_priority_order(self, tmp_path: Path) -> None:
        first = _write_jsonl(tmp_path / "first.jsonl", [_row("a", sector="Legal")])
        second = _write_jsonl(tmp_path / "second.jsonl", [_row("b", sector="Legal")])
        assert resolve_default_dataset([first, second]) == first

    def test_returns_none_when_nothing_exists(self, tmp_path: Path) -> None:
        assert resolve_default_dataset([tmp_path / "a.jsonl", tmp_path / "b.jsonl"]) is None


class TestMain:
    def test_uses_default_dataset_when_omitted(self, tmp_path: Path, capsys, monkeypatch) -> None:
        default_ds = _write_jsonl(tmp_path / "gdpval.jsonl", [_row("a", sector="Legal")])
        monkeypatch.setattr(td, "DEFAULT_DATASET_CANDIDATES", (tmp_path / "missing.jsonl", default_ds))
        rc = main(["--column", "sector"])
        assert rc == 0
        captured = capsys.readouterr()
        assert str(default_ds) in captured.err
        assert json.loads(captured.out)["Legal"]["percentage"] == 1.0

    def test_errors_when_no_default_and_none_specified(self, tmp_path: Path, capsys, monkeypatch) -> None:
        monkeypatch.setattr(td, "DEFAULT_DATASET_CANDIDATES", (tmp_path / "missing.jsonl",))
        rc = main(["--column", "sector"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "no default gdpval dataset was found" in err.lower()
        assert "gym eval prepare --benchmark gdpval" in err
        assert "--dataset" in err

    def test_defaults_to_occupation_column(self, tmp_path: Path, capsys) -> None:
        dataset = _write_jsonl(
            tmp_path / "d.jsonl",
            [_row("a", occupation="Lawyer"), _row("b", occupation="Lawyer"), _row("c", occupation="Nurse")],
        )
        rc = main(["--dataset", str(dataset)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "defaulting to 'occupation'" in captured.err
        data = json.loads(captured.out)
        assert data["Lawyer"]["task_ids"] == ["a", "b"]
        assert data["Nurse"]["percentage"] == pytest.approx(1 / 3)

    def test_errors_when_specified_dataset_missing(self, tmp_path: Path, capsys) -> None:
        rc = main(["--dataset", str(tmp_path / "nope.jsonl"), "--column", "sector"])
        assert rc == 2
        assert "Dataset not found" in capsys.readouterr().err

    def test_writes_output_file(self, tmp_path: Path, capsys) -> None:
        dataset = _write_jsonl(
            tmp_path / "d.jsonl",
            [_row("a", sector="Legal"), _row("b", sector="Legal"), _row("c", sector="Healthcare")],
        )
        out = tmp_path / "dist.json"
        rc = main(["--dataset", str(dataset), "--column", "sector", "--output", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["Legal"]["task_ids"] == ["a", "b"]
        assert "3 tasks" in capsys.readouterr().err

    def test_stdout_when_no_output(self, tmp_path: Path, capsys) -> None:
        dataset = _write_jsonl(tmp_path / "d.jsonl", [_row("a", sector="Legal")])
        rc = main(["--dataset", str(dataset), "--column", "sector"])
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["Legal"]["percentage"] == 1.0

    def test_no_rounding_and_compact(self, tmp_path: Path, capsys) -> None:
        dataset = _write_jsonl(
            tmp_path / "d.jsonl", [_row("a", sector="x"), _row("b", sector="y"), _row("c", sector="y")]
        )
        rc = main(["--dataset", str(dataset), "--column", "sector", "--precision", "-1", "--indent", "-1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "\n  " not in out  # compact (no indentation)
        assert json.loads(out)["y"]["percentage"] == pytest.approx(2 / 3)

    def test_composite_columns_cli(self, tmp_path: Path, capsys) -> None:
        dataset = _write_jsonl(
            tmp_path / "d.jsonl",
            [_row("a", sector="Legal", occupation="Lawyer")],
        )
        rc = main(["--dataset", str(dataset), "--column", "sector", "--column", "occupation"])
        assert rc == 0
        assert "Legal | Lawyer" in json.loads(capsys.readouterr().out)
