# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts between files nothing imports across at runtime: the example rows match the schema,
the fixture task parses like a real one, and the base Dockerfile still carries upstream's layer."""

import json
import sys
from pathlib import Path

from resources_servers.oragentbench.app import load_task
from resources_servers.oragentbench.task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
import make_example_data  # noqa: E402
import prepare_oragentbench as prep  # noqa: E402


def test_example_rows_are_the_collated_form_of_the_five_synthetic_rows():
    """The tracked file is the ``gym dataset collate`` output: ``agent_ref`` becomes ``task_source``."""
    rows = [json.loads(line) for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()]
    expected = []
    for row in make_example_data.rows():
        agent_ref = row.pop("agent_ref")
        expected.append(row | {"task_source": agent_ref["name"]})
    assert len(rows) == 5 and rows == expected
    for row in rows:
        TaskData.model_validate(row)
        assert row["task_name"].startswith("synthetic/") and row["docker_image"] != prep.BASE_IMAGE_TAG
    assert {row["difficulty"] for row in rows} == {"easy", "medium", "hard"}


def test_fixture_task_parses_and_its_reference_is_present():
    task = load_task(SERVER_DIR / "tests" / "fixtures" / "toy_assignment")
    assert task.steps[0].solution_dir is not None
    assert (SERVER_DIR / "tests" / "fixtures" / "toy_assignment" / "tests" / "reference_metrics.json").exists()


def test_base_dockerfile_reproduces_upstream_base_then_adds_tmux():
    text = (SERVER_DIR / "docker" / "Dockerfile").read_text()
    upstream = [
        "FROM python:3.11-slim",
        "ENV ORCLAW_SOLVE_TIME_LIMIT_SECONDS=300",
        "ENV ORCLAW_SCIP_GAP=0.0005",
        "pyscipopt",
        "pyomo",
        "networkx",
        "WORKDIR /app",
    ]
    for line in upstream:
        assert line in text, line
    assert text.index("WORKDIR /app") < text.index(
        "RUN apt-get update && apt-get install -y --no-install-recommends tmux"
    )
