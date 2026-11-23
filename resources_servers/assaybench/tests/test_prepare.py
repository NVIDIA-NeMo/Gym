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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from benchmarks.assaybench import prepare
from benchmarks.assaybench.prepare import (
    BIOGRID_RANKING_PROMPT,
    METADATA_FIELDS,
    build_rows,
    render_question,
    to_gym_row,
)
from resources_servers.assaybench import create_examples
from resources_servers.assaybench.create_examples import build_example_rows, example_records
from resources_servers.assaybench.task_data import TaskData


DATA_DIR = Path(__file__).absolute().parent.parent / "data"


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestTemplate:
    def test_template_matches_the_installed_package(self) -> None:
        # The transcription is checked against the source itself, not a committed copy.
        from assaybench.utils.prompt_loaders import load_objective_prompt

        assert BIOGRID_RANKING_PROMPT == load_objective_prompt("biogrid_ranking_prompt")

    def test_render_strips_a_trailing_period_from_phenotype_only(self) -> None:
        with_period, without_period = example_records()[0], example_records()[4]
        assert with_period["phenotype"].endswith(".") and not without_period["phenotype"].endswith(".")
        question = render_question(with_period)
        assert "each of which increases drug resistance as measured by increased cell proliferation." in question
        assert "proliferation.." not in question
        # The upstream condition_clause already carries its leading space; nothing is added or trimmed.
        assert "conducted over 12 Days under Etoposide treatment (130.0 nM)." in question
        assert "each of which decreases ISRE reporter activity." in render_question(without_period)


class TestRows:
    def test_gym_row_shape(self) -> None:
        row = to_gym_row(example_records()[2], split="test")
        assert row["split"] == "test"
        assert row["dataset_name"] == "example_3_inc"
        assert row["num_genes"] == len(row["relevance_genes"]) == len(row["relevance_scores"]) == 40
        assert min(row["relevance_scores"]) < 0 < max(row["relevance_scores"])
        assert set(METADATA_FIELDS) <= set(row)
        assert "responses_create_params" not in row  # benchmark rows: the prompt is applied at run time

    def test_build_rows_filters_the_split_column_and_checks_the_count(self, tmp_path: Path, monkeypatch) -> None:
        records = example_records()
        splits = ["test", "train", "test", "validation", "test"]
        pq.write_table(
            pa.Table.from_pylist([{**r, "yearfold0": s} for r, s in zip(records, splits)]), tmp_path / "b.parquet"
        )
        monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"train": 1, "validation": 1, "test": 3, "LaTest": 19})

        rows = build_rows("test", parquet_path=tmp_path / "b.parquet")
        assert [row["dataset_name"] for row in rows] == ["example_1", "example_3_inc", "example_5"]
        assert all(row["split"] == "test" for row in rows)
        assert rows[0]["question"] == render_question(records[0])

        monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"train": 1, "validation": 1, "test": 1, "LaTest": 19})
        with pytest.raises(ValueError, match="Expected 1 rows"):
            build_rows("test", parquet_path=tmp_path / "b.parquet")
        with pytest.raises(ValueError, match="Unknown split"):
            build_rows("dev")

    def test_split_argument_selects_the_cohort(self, monkeypatch, tmp_path: Path) -> None:
        seen = []
        monkeypatch.setattr(prepare, "build_rows", lambda split: seen.append(split) or [{"split": split}])
        monkeypatch.setattr(prepare, "OUTPUT_FPATH", tmp_path / "assaybench_benchmark.jsonl")
        assert prepare.prepare() == tmp_path / "assaybench_benchmark.jsonl"
        assert prepare.prepare(split="LaTest") == tmp_path / "assaybench_benchmark.jsonl"
        monkeypatch.setattr("sys.argv", ["prepare.py", "--split", "validation"])
        prepare.main()
        assert seen == ["test", "LaTest", "validation"]
        assert _read_jsonl(tmp_path / "assaybench_benchmark.jsonl") == [{"split": "validation"}]


class TestExampleFixture:
    def test_committed_fixture_is_current_and_scores_perfectly(self, monkeypatch, tmp_path: Path) -> None:
        # `gym dataset collate` stamps the declaring instance onto the fixture as `task_source`.
        committed = [
            {k: v for k, v in row.items() if k != "task_source"} for row in _read_jsonl(DATA_DIR / "example.jsonl")
        ]
        assert committed == build_example_rows()
        assert len(committed) == 5 and len({row["dataset_name"] for row in committed}) == 5
        # Pre-materialized with the paper prompt (the example-data gate does not apply prompt_config).
        assert all(row["responses_create_params"]["input"][1]["role"] == "user" for row in committed)

        monkeypatch.setattr("sys.argv", ["create_examples.py", "--output-dir", str(tmp_path), "--rollouts"])
        create_examples.main()
        assert _read_jsonl(tmp_path / "example.jsonl") == build_example_rows()
        rollouts = _read_jsonl(tmp_path / "example_rollouts.jsonl")
        assert [r["reward"] for r in rollouts] == [1.0] * 5
        assert all(r["extraction_mode"] == "dspy_answer" for r in rollouts)
        assert [r["_ng_task_index"] for r in rollouts] == list(range(5))
        assert [r["dataset_name"] for r in _read_jsonl(DATA_DIR / "example_rollouts.jsonl")] == [
            r["dataset_name"] for r in rollouts
        ]

    def test_task_data_schema_accepts_the_fixture_and_requires_ground_truth(self) -> None:
        for row in _read_jsonl(DATA_DIR / "example.jsonl"):
            task = TaskData.model_validate(row)
            assert len(task.relevance_genes) == len(task.relevance_scores) == task.num_genes
        with pytest.raises(ValidationError):
            TaskData.model_validate({"question": "no ground truth"})
