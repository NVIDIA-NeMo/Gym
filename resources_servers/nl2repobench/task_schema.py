# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin schema for the pinned NL2RepoBench task format.

Unlike DeepSWE, NL2RepoBench ships no structured task config: each upstream task directory
(``test_files/<proName>/``) is exactly four flat files (``start.md``, ``test_case_count.txt``,
``test_commands.json``, ``test_files.json``). Note the two JSON files are bare arrays upstream
(e.g. ``["pip install -e .", "pytest tests"]``), not ``{"commands": [...]}``/``{"files": [...]}``
objects; ``TestCommands``/``TestFiles`` accept either shape so a future upstream format change to
an object wrapper keeps working.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class TestCommands(BaseModel):
    commands: list[str]

    @classmethod
    def model_validate_json_flexible(cls, raw: str) -> "TestCommands":
        data = json.loads(raw)
        if isinstance(data, list):
            return cls(commands=data)
        return cls.model_validate(data)


class TestFiles(BaseModel):
    files: list[str]

    @classmethod
    def model_validate_json_flexible(cls, raw: str) -> "TestFiles":
        data = json.loads(raw)
        if isinstance(data, list):
            return cls(files=data)
        return cls.model_validate(data)


class Task:
    """A single validated NL2RepoBench task directory (``proName`` = ``name``)."""

    def __init__(self, task_dir: Path | str) -> None:
        self.task_dir = Path(task_dir).resolve()
        self.name = self.task_dir.name
        self.start_md = (self.task_dir / "start.md").read_text(encoding="utf-8")
        self.test_case_count = _parse_test_case_count(
            (self.task_dir / "test_case_count.txt").read_text(encoding="utf-8")
        )
        self.test_commands = TestCommands.model_validate_json_flexible(
            (self.task_dir / "test_commands.json").read_text(encoding="utf-8")
        )
        self.test_files = TestFiles.model_validate_json_flexible(
            (self.task_dir / "test_files.json").read_text(encoding="utf-8")
        )


def _parse_test_case_count(raw: str) -> int:
    text = raw.strip()
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"Invalid test_case_count.txt content: {raw!r}") from error
