# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "check_runtime_dependencies.py"
SPEC = importlib.util.spec_from_file_location("check_runtime_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runtime_dependencies = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime_dependencies
SPEC.loader.exec_module(runtime_dependencies)


def test_committed_manifest_is_valid_and_capability_scoped() -> None:
    manifest = Path(__file__).resolve().parents[1] / "runtime_dependencies.toml"

    dependencies = runtime_dependencies.load_manifest(manifest)

    assert [dependency.id for dependency in dependencies] == ["pointer-gym-sandbox-runtime-hardening"]
    assert dependencies[0].commit == "50f7634ee22cfc9184d44d7668a4036dffe7a532"  # pragma: allowlist secret
    assert runtime_dependencies.select_dependencies(dependencies, ["pointer-gym-sandbox"]) == dependencies
    assert runtime_dependencies.select_dependencies(dependencies, ["nano-omni-training"]) == []


def test_dependency_status_requires_exact_commit_ancestry(tmp_path: Path) -> None:
    dependency = runtime_dependencies.load_manifest(Path(__file__).resolve().parents[1] / "runtime_dependencies.toml")[
        0
    ]

    with patch.object(
        runtime_dependencies,
        "_git",
        side_effect=[
            runtime_dependencies.subprocess.CompletedProcess([], 0, "", ""),
            runtime_dependencies.subprocess.CompletedProcess([], 1, "", ""),
        ],
    ):
        assert runtime_dependencies.dependency_status(tmp_path, dependency) == (
            False,
            "commit exists locally but is not an ancestor",
        )


def test_cli_rejects_unknown_capability(capsys) -> None:
    manifest = Path(__file__).resolve().parents[1] / "runtime_dependencies.toml"

    exit_code = runtime_dependencies.main(["--manifest", str(manifest), "--capability", "not-declared"])

    assert exit_code == 2
    assert "undeclared capability: not-declared" in capsys.readouterr().err
