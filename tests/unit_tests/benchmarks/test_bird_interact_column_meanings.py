# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The column descriptions a catalog built with --column-meanings holds, as auto-ontology-eval writes them."""

import json
from pathlib import Path

import pytest

from benchmarks.bird_interact.auto_ontology import column_meanings


SCHEMA = """CREATE TABLE "scans" (
scanregistry text NOT NULL,
cloud_metrics jsonb NULL,
    PRIMARY KEY (scanregistry)
);

First 3 rows:
scanregistry    cloud_metrics
--------------  -------------------------------------------------------------
SC1             {'Noise_Db': 1.318, 'Cover': {'Pct': '91.2%'}, 'Tags': [1, 2], 'Total_Pts': 4656
SC2             {'Noise_Db': None, 'Cover': {'Pct': '88.3%'}}
...
"""


def test_a_jsonb_cell_gives_each_field_it_holds_before_the_cut() -> None:
    fields = column_meanings.stored_fields(
        "{'Noise_Db': 1.318, 'Cover': {'Pct': '91.2%'}, 'Tags': [1, 2], 'Total_Pts': 4656"
    )

    # Lists are skipped, nested keys join with dots, and the field the cut reaches is left out.
    assert fields == {"Noise_Db": "1.318", "Cover.Pct": "'91.2%'"}


def test_the_sample_rows_are_read_by_the_rule_s_column_spans() -> None:
    cells = column_meanings.sample_cells(SCHEMA)

    assert cells["scans"]["scanregistry"] == ["SC1", "SC2"]
    assert cells["scans"]["cloud_metrics"][1] == "{'Noise_Db': None, 'Cover': {'Pct': '88.3%'}}"


def test_a_jsonb_column_s_description_has_a_line_per_field_with_its_stored_values(tmp_path: Path) -> None:
    database = tmp_path / "archeology"
    database.mkdir()
    (database / "archeology_schema.txt").write_text(SCHEMA)
    meanings = {
        "archeology|scans|scanregistry": "TEXT. The scan's code. Example: SC1.",
        "archeology|scans|cloud_metrics": {
            "column_meaning": "JSONB. The point cloud's metrics.",
            "fields_meaning": {"Noise_Db": "Noise, in dB.", "Cover": {"Pct": "Coverage, as a share."}},
        },
        "not-a-key": "skipped",
    }
    (database / "archeology_column_meaning_base.json").write_text(json.dumps(meanings))

    descriptions = column_meanings.column_descriptions("archeology", tmp_path)

    assert descriptions == [
        ("scans", "scanregistry", "TEXT. The scan's code. Example: SC1."),
        (
            "scans",
            "cloud_metrics",
            "JSONB. The point cloud's metrics.\n"
            "Noise_Db: Noise, in dB. Stored values in sample rows: 1.318.\n"
            "Cover.Pct: Coverage, as a share. Stored values in sample rows: '91.2%', '88.3%'.",
        ),
    ]
    assert column_meanings.column_descriptions("missing", tmp_path) == []


@pytest.mark.parametrize(
    ("table", "column", "found"),
    [
        ("Scans", "ScanRegistry", ("scans", "scanregistry")),
        # A table the catalog lacks: the one catalog table with that column.
        ("scan_log", "noise", ("readings", "noise")),
        # A table the catalog has is never matched to another table's column.
        ("scans", "noise", None),
        # A column two catalog tables have is matched to neither.
        ("scan_log", "id", None),
    ],
)
def test_a_meanings_entry_names_the_catalog_s_column_without_regard_to_case(
    table: str, column: str, found: tuple[str, str] | None
) -> None:
    catalog = [("scans", "scanregistry"), ("scans", "id"), ("readings", "noise"), ("readings", "id")]

    assert column_meanings.catalog_column(table, column, catalog) == found
