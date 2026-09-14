# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from resources_servers.assaybench import prepare
from resources_servers.assaybench.prepare import (
    BIOGRID_RANKING_PROMPT,
    EXPECTED_ROWS,
    HF_REVISION,
    METADATA_FIELDS,
    PROMPT_FIELDS,
    build_example_rows,
    build_rows,
    example_records,
    render_question,
    to_gym_row,
)


DATA_DIR = Path(__file__).absolute().parent.parent / "data"

PHENOTYPE_CLASSES = {
    "Fitness / Proliferation / Viability",
    "Drug / Chemical / Environmental Response",
    "Host-Pathogen / Infection Response",
    "Molecular Output / Reporter / Pathway Activity",
    "Trafficking / Localization / Structural Phenotypes",
}


class TestTemplate:
    def test_template_matches_the_installed_package(self) -> None:
        # The transcription is checked against the source itself, not a committed copy.
        from assaybench.utils.prompt_loaders import load_objective_prompt

        assert BIOGRID_RANKING_PROMPT == load_objective_prompt("biogrid_ranking_prompt")

    def test_template_reads_exactly_the_declared_fields(self) -> None:
        placeholders = set(re.findall(r"{(\w+)}", BIOGRID_RANKING_PROMPT))
        assert placeholders == set(PROMPT_FIELDS)

    def test_render_strips_the_trailing_period_of_phenotype(self) -> None:
        record = example_records()[0]
        assert record["phenotype"].endswith(".")
        question = render_question(record)
        assert "each of which increases drug resistance as measured by increased cell proliferation." in question
        assert "proliferation.." not in question
        # The upstream column already carries its leading space; nothing is added or trimmed.
        assert "conducted over 12 Days under Etoposide treatment (130.0 nM)." in question

    def test_render_leaves_phenotype_without_period_alone(self) -> None:
        record = dict(example_records()[4])
        assert not record["phenotype"].endswith(".")
        assert "each of which decreases ISRE reporter activity." in render_question(record)


class TestRows:
    def test_gym_row_shape(self) -> None:
        row = to_gym_row(example_records()[2], split="test")
        assert row["split"] == "test"
        assert row["dataset_name"] == "example_3_inc"
        assert row["num_genes"] == len(row["relevance_genes"]) == len(row["relevance_scores"]) == 40
        assert min(row["relevance_scores"]) < 0 < max(row["relevance_scores"])
        assert set(METADATA_FIELDS) <= set(row)
        assert "responses_create_params" not in row  # benchmark rows: the prompt is applied at run time

    def test_committed_examples_are_current(self) -> None:
        committed = [
            json.loads(line) for line in (DATA_DIR / "example.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert committed == build_example_rows()
        assert len(committed) == 5
        # The fixture is pre-materialized (the example-data gate does not apply prompt_config).
        assert all(row["responses_create_params"]["input"][1]["role"] == "user" for row in committed)
        assert {row["cleaned_phenotype"] for row in committed} <= PHENOTYPE_CLASSES
        assert len({row["dataset_name"] for row in committed}) == 5

    def test_expected_counts_are_the_papers_table_1(self) -> None:
        assert EXPECTED_ROWS == {"train": 1349, "validation": 218, "test": 334, "LaTest": 19}
        assert re.fullmatch(r"[0-9a-f]{40}", HF_REVISION)

    def test_build_rows_filters_the_split_column(self, tmp_path: Path, monkeypatch) -> None:
        records = example_records()
        table = pa.Table.from_pylist(
            [
                {**record, "yearfold0": split}
                for record, split in zip(records, ["test", "train", "test", "validation", "test"])
            ]
        )
        path = tmp_path / "biogrid.parquet"
        pq.write_table(table, path)
        monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"train": 1, "validation": 1, "test": 3, "LaTest": 19})

        rows = build_rows("test", parquet_path=path)
        assert [row["dataset_name"] for row in rows] == ["example_1", "example_3_inc", "example_5"]
        assert all(row["split"] == "test" for row in rows)
        assert rows[0]["question"] == render_question(records[0])

        with pytest.raises(ValueError, match="Expected 1 rows"):
            monkeypatch.setattr(prepare, "EXPECTED_ROWS", {"train": 1, "validation": 1, "test": 1, "LaTest": 19})
            build_rows("test", parquet_path=path)

    def test_build_rows_rejects_unknown_split(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            build_rows("dev")
