# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


RUBRICS = {
    "rubrics": [
        {
            "rubric": "Does the report state the eligible population count?",
            "weight": 10,
            "criterion": [
                "The report gives a population count",
                "The count is 1,108",
            ],
        },
        {
            "rubric": "Does the deliverable include a plot of the age distribution?",
            "weight": 5,
            "criterion": ["An age distribution figure is present"],
        },
    ]
}


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A minimal prepared cache with one main-split task."""
    task_dir = tmp_path / "tasks" / "main" / "biostatisticians" / "task1"
    task_folder = task_dir / "task_folder"
    search_files = task_dir / "files_required_to_search"
    task_folder.mkdir(parents=True)
    search_files.mkdir(parents=True)

    (task_folder / "TASK_INSTRUCTIONS.txt").write_text("Analyze the cohort.\n", encoding="utf-8")
    (task_folder / "cohort.csv").write_text("id,age\n1,39\n", encoding="utf-8")
    (search_files / "guidance.txt").write_text("FDA E9 guidance.\n", encoding="utf-8")
    (task_dir / "RUBRICS.json").write_text(json.dumps(RUBRICS), encoding="utf-8")
    (task_dir / "task_card.md").write_text("# Brief\n", encoding="utf-8")

    return tmp_path / "tasks"


@pytest.fixture
def task_id() -> str:
    return "main/biostatisticians/task1"
