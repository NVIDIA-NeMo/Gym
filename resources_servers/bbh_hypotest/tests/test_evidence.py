# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from resources_servers.bbh_hypotest.evidence import (
    EvidenceError,
    build_hypotest_prompt,
    load_evidence_cells,
    safe_artifact_path,
)


def test_prompt_preserves_observations_and_adds_artifact_contract():
    prompt = build_hypotest_prompt([{"role": "user", "content": "Test H0"}])
    assert "[USER]\nTest H0" in prompt
    assert "./analysis.py" in prompt


def test_safe_artifact_path_rejects_workspace_escape(tmp_path: Path):
    with pytest.raises(EvidenceError, match="escapes workspace"):
        safe_artifact_path(tmp_path, "../outside.py")


def test_python_markers_become_ordered_cells(tmp_path: Path):
    (tmp_path / "analysis.py").write_text("x = 1\n# %% second\nprint(x)\n")
    artifact, cells = load_evidence_cells(tmp_path)
    assert artifact.name == "analysis.py"
    assert cells == ["x = 1", "print(x)"]


def test_notebook_returns_only_nonempty_code_cells(tmp_path: Path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["ignore"]},
            {"cell_type": "code", "source": ["x = 2\n", "print(x)"]},
            {"cell_type": "code", "source": []},
        ]
    }
    (tmp_path / "analysis.ipynb").write_text(json.dumps(notebook))
    artifact, cells = load_evidence_cells(tmp_path)
    assert artifact.name == "analysis.ipynb"
    assert cells == ["x = 2\nprint(x)"]
