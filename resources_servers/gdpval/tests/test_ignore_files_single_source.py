# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both judging paths must skip the same run-state files, from one definition."""

import ast
from pathlib import Path

from nemo_gym.deliverables import IGNORE_FILES


GDPVAL = Path(__file__).resolve().parents[1]
COMPARISON = GDPVAL / "comparison.py"
INSPECT_CONVERSION = GDPVAL / "inspect_conversion.py"


def _modules_importing_ignore_files(path: Path) -> set[str]:
    return {
        node.module or ""
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and any(alias.name == "IGNORE_FILES" for alias in node.names)
    }


def test_comparison_does_not_keep_its_own_copy():
    """A second literal set is how the two paths drifted in the first place."""
    tree = ast.parse(COMPARISON.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IGNORE_FILES":
                    raise AssertionError("comparison.py redefines IGNORE_FILES; import it instead")


def test_comparison_uses_the_shared_definition():
    """The judge and Stirrup's file reader both read ``nemo_gym.deliverables``."""
    from resources_servers.gdpval import comparison

    assert _modules_importing_ignore_files(COMPARISON) == {"nemo_gym.deliverables"}
    assert comparison.IGNORE_FILES is IGNORE_FILES


def test_comparison_does_not_import_the_agent_at_module_scope():
    """A module-level agent import drags Ray and FastAPI into every importer.

    ``multistage_elo`` imports this module at module scope, so the cost would be
    paid on any import of the comparison path.
    """
    tree = ast.parse(COMPARISON.read_text())
    for node in tree.body:  # module scope only
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("responses_api_agents"):
            raise AssertionError(f"module-level import of {node.module}")


def test_inspect_conversion_uses_the_shared_definition():
    assert _modules_importing_ignore_files(INSPECT_CONVERSION) == {"nemo_gym.deliverables"}
