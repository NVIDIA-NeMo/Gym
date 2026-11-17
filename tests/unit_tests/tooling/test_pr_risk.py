# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/autobot/pr_risk.py"
_spec = importlib.util.spec_from_file_location("pr_risk", SCRIPT)
pr_risk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_risk)

API = {
    "nemo_gym/__init__.py": '_LAZY_EXPORTS: dict[str, str] = {"helper": "core.shared"}\n',
    "nemo_gym/core/__init__.py": '_PUBLIC_API = {"Thing": "things"}\n',
    "nemo_gym/core/things.py": "class Thing:\n    pass\n",
    "nemo_gym/core/shared.py": "def helper():\n    pass\n",
    "nemo_gym/core/helpers.py": "def internal():\n    pass\n",
}
PROTECTED = ["pyproject.toml", ".github/workflows/*"]


def lines(count: int) -> str:
    return "".join(f"line {index}\n" for index in range(count))


def commit(repo: Path, files: dict[str, str | None]) -> None:
    for relative, content in files.items():
        path = repo / relative
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "-C", repo, *identity, "commit", "-q", "--no-verify", "-m", "x"], check=True)


def worktrees(tmp_path: Path, before: dict[str, str | None], after: dict[str, str | None]) -> tuple[Path, Path]:
    head = tmp_path / "head"
    subprocess.run(["git", "init", "-q", head], check=True)
    commit(head, {**API, **before})
    subprocess.run(["git", "-C", head, "worktree", "add", "-q", "--detach", tmp_path / "base"], check=True)
    commit(head, after)
    return tmp_path / "base", head


def grade(tmp_path: Path, before: dict[str, str | None], after: dict[str, str | None]) -> tuple[str, list[str]]:
    return pr_risk.grade(*worktrees(tmp_path, before, after), PROTECTED)


@pytest.mark.parametrize(("count", "tier"), [(100, "low"), (101, "mid"), (400, "mid"), (401, "high")])
def test_tier_follows_changed_line_count(tmp_path: Path, count: int, tier: str) -> None:
    assert grade(tmp_path, {}, {"scripts/a.py": lines(count)})[0] == tier


def test_added_and_removed_lines_count(tmp_path: Path) -> None:
    assert grade(tmp_path, {"scripts/a.py": lines(60)}, {"scripts/a.py": None, "scripts/b.py": lines(60)}) == (
        "mid",
        ["120 changed lines in 2 non-doc files"],
    )


def test_non_utf8_filename_is_counted(tmp_path: Path) -> None:
    assert grade(tmp_path, {}, {os.fsdecode(b"scripts/bad\xff.py"): lines(401)})[0] == "high"


def test_docs_lockfiles_and_data_are_not_counted(tmp_path: Path) -> None:
    after = {"docs/guide.md": lines(500), "README.md": lines(500), "a.lock": lines(500), "s/data/x.jsonl": lines(500)}
    assert grade(tmp_path, {}, after) == ("low", ["0 changed lines in 2 non-doc files"])


@pytest.mark.parametrize("path", ["resources_servers/s/app.py", "nemo_gym/verification/judge.py"])
def test_scoring_change_is_mid(tmp_path: Path, path: str) -> None:
    assert grade(tmp_path, {}, {path: "x\n"}) == ("mid", ["1 changed lines in 1 non-doc files", f"scoring: {path}"])


@pytest.mark.parametrize("path", ["resources_servers/s/tests/test_app.py", "resources_servers/s/data/x.json"])
def test_scoring_tests_and_data_are_low(tmp_path: Path, path: str) -> None:
    assert grade(tmp_path, {}, {path: "x\n"})[0] == "low"


@pytest.mark.parametrize(
    "path", ["nemo_gym/__init__.py", "nemo_gym/core/__init__.py", "nemo_gym/core/things.py", "nemo_gym/core/shared.py"]
)
def test_public_surface_change_is_high(tmp_path: Path, path: str) -> None:
    tier, evidence = grade(tmp_path, {}, {path: API[path] + "# edit\n"})
    assert tier == "high"
    assert f"public surface: {path}" in evidence


def test_unlisted_module_is_not_public_surface(tmp_path: Path) -> None:
    assert grade(tmp_path, {}, {"nemo_gym/core/helpers.py": API["nemo_gym/core/helpers.py"] + "# edit\n"})[0] == "low"


@pytest.mark.parametrize("path", ["pyproject.toml", ".github/workflows/ci/test.yml"])
def test_protected_file_change_is_high(tmp_path: Path, path: str) -> None:
    tier, evidence = grade(tmp_path, {}, {path: "x\n"})
    assert tier == "high"
    assert f"protected: {path}" in evidence


def test_protected_patterns(tmp_path: Path) -> None:
    listed = tmp_path / "protected_files.yaml"
    listed.write_text("# comment\n- a.toml\n- 'b/*'\n")
    assert pr_risk.protected_patterns(listed) == ["a.toml", "b/*"]
    assert pr_risk.protected_patterns(tmp_path / "missing.yaml") == []
    assert {"pyproject.toml", "uv.lock"} <= set(pr_risk.protected_patterns())


def test_cli_prints_the_tier_on_stdout(tmp_path: Path) -> None:
    base, head = worktrees(tmp_path, {}, {"scripts/a.py": lines(150)})
    result = subprocess.run(
        [sys.executable, SCRIPT, "--base", base, "--head", head], capture_output=True, text=True, check=True
    )
    assert result.stdout == "mid\n"
    assert result.stderr == "150 changed lines in 1 non-doc files\n"


def test_repository_public_surface_is_discovered() -> None:
    assert "nemo_gym/core/__init__.py" in pr_risk.public_surface(ROOT)
