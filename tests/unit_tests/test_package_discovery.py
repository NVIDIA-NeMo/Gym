# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
from fnmatch import fnmatchcase
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIND = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"]["find"]

# Runtime trees and rollout outputs that agent servers create inside their package directories.
GENERATED_TREES = [
    "responses_api_agents/anyswe_agent/anyswe_claude_code_agent_deps/lib/node_modules",
    "responses_api_agents/anyswe_agent/anyswe_results_1700000000000_abcd1234/astropy__astropy-12907_1",
    "responses_api_agents/anyterminal_agent/deps/anyterminal_codex_agent_deps/lib/node_modules",
    "responses_api_agents/anyterminal_agent/results/anyterminal_results_1700000000000_abcd1234/task_1",
]


def _excluded(package: str) -> bool:
    # setuptools filters discovered package names with fnmatchcase against each exclude pattern.
    return any(fnmatchcase(package, pattern) for pattern in FIND["exclude"])


@pytest.mark.parametrize("tree", GENERATED_TREES)
def test_generated_runtime_trees_are_excluded(tree: str) -> None:
    parts = tree.split("/")
    # Every package level from the generated directory down must be excluded.
    for depth in range(3, len(parts) + 1):
        assert _excluded(".".join(parts[:depth])), ".".join(parts[:depth])


@pytest.mark.parametrize(
    "package",
    [
        "responses_api_agents.anyswe_agent",
        "responses_api_agents.anyswe_agent.tests",
        "responses_api_agents.anyterminal_agent",
        "responses_api_agents.anyterminal_agent.tests",
    ],
)
def test_agent_packages_still_ship(package: str) -> None:
    assert not _excluded(package)


def test_setuptools_discovery_skips_generated_trees(tmp_path: Path) -> None:
    setuptools = pytest.importorskip("setuptools")
    for tree in GENERATED_TREES + ["responses_api_agents/anyswe_agent/tests"]:
        (tmp_path / tree).mkdir(parents=True)
        (tmp_path / tree / "module.py").write_text("")

    found = setuptools.find_namespace_packages(where=str(tmp_path), include=FIND["include"], exclude=FIND["exclude"])

    assert "responses_api_agents.anyswe_agent.tests" in found
    assert not [package for package in found if "_deps" in package or "results" in package]
