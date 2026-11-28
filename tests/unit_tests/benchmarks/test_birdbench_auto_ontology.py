# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BIRD catalog image's inputs: Gym's pin of Auto Ontology, BIRD's databases and descriptions, and the model
settings that key the catalog build's cache."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.birdbench.auto_ontology.build_catalog import column_descriptions, missing_databases


AUTO_ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "birdbench" / "auto_ontology"


def test_missing_databases_names_each_split_database_with_no_sqlite_file(tmp_path):
    split_file = tmp_path / "dev.json"
    split_file.write_text(json.dumps([{"db_id": db_id} for db_id in ("financial", "superhero", "toxicology")]))
    databases = tmp_path / "dev_databases"
    (databases / "financial").mkdir(parents=True)
    (databases / "financial" / "financial.sqlite").write_bytes(b"")
    # An interrupted extraction: a directory with no database, or only a hidden file.
    (databases / "superhero").mkdir()
    (databases / "toxicology").mkdir()
    (databases / "toxicology" / "._toxicology.sqlite").write_bytes(b"")

    assert missing_databases(databases, split_file) == ["superhero", "toxicology"]
    (databases / "superhero" / "superhero.sqlite").write_bytes(b"")
    (databases / "toxicology" / "toxicology.sqlite").write_bytes(b"")
    assert missing_databases(databases, split_file) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="build_image.sh is a bash script")
def test_the_catalog_build_is_keyed_by_the_model_settings_but_not_the_keys(tmp_path):
    """BuildKit leaves the secret's contents out of the cache key, so build_image.sh passes a digest of them."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "docker_calls"
    # python stands in for the databases check; docker records its arguments, one per line.
    (bin_dir / "python").write_text(f"#!/bin/sh\ncat >/dev/null\necho {tmp_path}\n")
    (bin_dir / "docker").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {calls}\n')
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    def digest(models_env: str) -> str:
        (tmp_path / "models.env").write_text(models_env)
        subprocess.run(
            ["bash", str(AUTO_ONTOLOGY_DIR / "build_image.sh"), str(tmp_path / "models.env")], env=env, check=True
        )
        arguments = calls.read_text()
        assert "sk-first" not in arguments and "sk-second" not in arguments and "nvapi-first" not in arguments
        (value,) = re.findall(r"^MODELS_SETTINGS_DIGEST=([0-9a-f]{64})$", arguments, re.MULTILINE)
        return value

    built = digest("DEFAULT_MODELS_API_KEY=sk-first\nEMBED_MODEL=embed-a\n")
    assert digest("# a comment\nEMBED_MODEL=embed-a\n\nDEFAULT_MODELS_API_KEY=sk-second\n") == built
    assert digest("DEFAULT_MODELS_API_KEY=sk-first\nEMBED_MODEL=embed-b\n") != built
    assert (
        digest("DEFAULT_MODELS_API_KEY=sk-first\nEMBED_MODEL=embed-a\nEMBED_ENDPOINT=https://e.invalid/v1\n") != built
    )
    # A key's type picks Auto Ontology's default models.
    assert digest("DEFAULT_MODELS_API_KEY=nvapi-first\nEMBED_MODEL=embed-a\n") != built


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


def test_the_pairing_s_rows_are_bird_sql_s_so_its_task_data_checks_them():
    """bird_sql's instance declares the dataset: collation then checks each row against bird_sql's TaskData."""
    import yaml

    config = yaml.safe_load((AUTO_ONTOLOGY_DIR.parent / "auto_ontology_agent.yaml").read_text())
    [dataset] = config["birdbench_bird_sql_resources_server"]["resources_servers"]["bird_sql"]["datasets"]
    assert dataset["taskset"] == "birdbench"
    assert dataset["jsonl_fpath"].endswith("birdbench_sqlite_dev_auto_ontology_agent.jsonl")
    assert "auto_ontology_agent" not in config
    agent = config["birdbench_bird_sql_environment_server"]["environment_servers"]["single_agent_turn"]["agent_server"]
    assert agent["name"] == "auto_ontology_agent"
