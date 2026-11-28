# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pre-commit hook that requires selection fields in every manifest."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK_PATH = REPO_ROOT / "scripts" / "check_manifest_selection_fields.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("gym_check_manifest_selection_fields", HOOK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hook = _load_hook()

COMPLETE = {
    "maintainers": ["alice-nv"],
    "capabilities": ["reasoning", "reasoning/deductive_reasoning"],
    "domains": ["mathematics"],
    "requirements": {"gpu": False, "network": "unknown"},
    "primary_metric": "pass@1",
}


def _write(relative: str, **fields: object) -> Path:
    """A manifest at ``relative`` under the current directory."""
    kind = "benchmark" if relative.startswith("benchmarks/") else "environment"
    path = Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": path.parent.name, "kind": kind, **fields}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _in_tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_a_manifest_must_declare_every_selection_field() -> None:
    path = _write("benchmarks/fresh/manifest.yaml")

    [problem] = hook.check([path])

    assert "set maintainers, capabilities, domains, requirements.gpu, requirements.network, primary_metric." in problem


def test_unknown_is_accepted_and_environments_need_no_primary_metric() -> None:
    fields = {key: value for key, value in COMPLETE.items() if key != "primary_metric"}
    paths = [
        _write(
            "benchmarks/fresh/manifest.yaml",
            **(COMPLETE | {"primary_metric": "unknown", "capabilities": "unknown", "domains": "unknown"}),
        ),
        _write("environments/fresh/manifest.yaml", **fields),
    ]

    assert hook.check(paths) == []


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"maintainers": ["TODO"]}, "maintainers"),
        ({"maintainers": ["alice-nv", " todo "]}, "maintainers"),
        ({"primary_metric": "TODO"}, "primary_metric"),
        ({"primary_metric": "TODO: name the headline metric"}, "primary_metric"),
        ({"capabilities": "TODO"}, "capabilities"),
        ({"domains": ["TODO"]}, "domains"),
        ({"requirements": {"gpu": False}}, "requirements.network"),
    ],
)
def test_a_scaffold_placeholder_or_a_missing_requirement_is_not_a_value(changes: dict, field: str) -> None:
    path = _write("benchmarks/fresh/manifest.yaml", **(COMPLETE | changes))

    [problem] = hook.check([path])

    assert f"set {field}." in problem


def test_an_empty_label_list_means_none_applies() -> None:
    path = _write("benchmarks/fresh/manifest.yaml", **(COMPLETE | {"capabilities": []}))

    assert hook.check([path]) == []


def test_a_value_that_only_starts_with_todo_is_real() -> None:
    path = _write("benchmarks/fresh/manifest.yaml", **(COMPLETE | {"maintainers": ["todorov"]}))

    assert hook.check([path]) == []


def test_task_data_and_other_files_are_not_catalog_manifests() -> None:
    dataset = Path("environments/harbor/dataset")
    dataset.mkdir(parents=True)
    (dataset / "manifest.toml").write_text("[dataset]\n", encoding="utf-8")
    paths = [
        _write("environments/harbor/dataset/task/manifest.yaml"),
        _write("environments/.cache/stale/manifest.yaml"),
        _write("resources_servers/other/manifest.yaml"),
    ]

    assert hook.check(paths) == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_every_manifest_in_the_repository_declares_the_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    tracked = subprocess.run(
        ["git", "ls-files", "benchmarks/**/manifest.yaml", "environments/**/manifest.yaml"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert tracked
    assert hook.check([Path(path) for path in tracked]) == []
