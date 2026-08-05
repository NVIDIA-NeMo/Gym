# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both judging paths must skip the same run-state files, from one definition."""

import ast
from pathlib import Path


COMPARISON = Path(__file__).resolve().parents[1] / "comparison.py"


def test_comparison_does_not_keep_its_own_copy():
    """A second literal set is how the two paths drifted in the first place."""
    tree = ast.parse(COMPARISON.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IGNORE_FILES":
                    raise AssertionError("comparison.py redefines IGNORE_FILES; import it instead")


def test_the_import_is_lazy():
    """A module-level import drags Ray and FastAPI into every importer.

    ``multistage_elo`` imports this module at module scope, so the cost would be
    paid on any import of the comparison path.
    """
    tree = ast.parse(COMPARISON.read_text())
    for node in tree.body:  # module scope only
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("responses_api_agents"):
            raise AssertionError(f"module-level import of {node.module}; keep it inside _ignore_files()")


def test_accessor_returns_the_agents_definition():
    """The values must come from file_reader, not be re-listed here."""
    src = COMPARISON.read_text()
    assert "_ignore_files" in src
    assert "from responses_api_agents.stirrup_agent.file_reader import IGNORE_FILES" in src
