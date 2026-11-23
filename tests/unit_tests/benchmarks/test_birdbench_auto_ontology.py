# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BIRD catalog image's inputs: Gym's pin of Auto Ontology and BIRD's descriptions."""

import re
from pathlib import Path

from benchmarks.birdbench.auto_ontology.build_catalog import column_descriptions


AUTO_ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "birdbench" / "auto_ontology"


def test_auto_ontology_is_pinned_to_a_full_commit():
    build_image = (AUTO_ONTOLOGY_DIR / "build_image.sh").read_text()
    repository = re.search(r"^AUTO_ONTOLOGY_REPOSITORY=(\S+)$", build_image, re.MULTILINE)
    commit = re.search(r"^AUTO_ONTOLOGY_COMMIT=(\S+)$", build_image, re.MULTILINE)
    assert repository is not None and repository.group(1).startswith("https://github.com/")
    assert commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit.group(1))


def test_bird_descriptions_join_the_column_and_value_descriptions(tmp_path):
    (tmp_path / "frpm.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "CDSCode,CDS code,California school code,text,not useful\n"
        'Charter School (Y/N), ,Charter school,integer,"0: no\n1: yes"\n'
        ",,orphan,,\n"
    )
    (tmp_path / "schools.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "City,,Ciudad — city,text,\n".encode("cp1252")
    )

    assert column_descriptions(tmp_path) == [
        {"table_name": "frpm", "column_name": "CDSCode", "description": "California school code"},
        {"table_name": "frpm", "column_name": "Charter School (Y/N)", "description": "Charter school, 0: no 1: yes"},
        {"table_name": "schools", "column_name": "City", "description": "Ciudad — city"},
    ]
