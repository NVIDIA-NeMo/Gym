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
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/autobot/pr_risk.py"
_spec = importlib.util.spec_from_file_location("pr_risk", SCRIPT)
pr_risk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_risk)

API = {
    "nemo_gym/core/__init__.py": '_PUBLIC_API = {"Thing": "things", "helper": "reexports"}\n',
    "nemo_gym/core/things.py": "class Thing:\n    size: int = 1\n\n    def run(self, x: int) -> int:\n        return x\n",
    "nemo_gym/core/reexports.py": "from nemo_gym.core.helpers import helper\n",
    "nemo_gym/core/helpers.py": "def helper(a: str) -> str:\n    return a\n",
    "pyproject.toml": '[project]\nname = "nemo-gym"\ndependencies = ["fastapi"]\n\n[tool.ruff]\nline-length = 119\n',
}


def lines(count: int, prefix: str = "line") -> str:
    return "".join(f"{prefix} {index}\n" for index in range(count))


def make_tree(root: Path, files: dict[str, str | bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    return root


def grade(
    tmp_path: Path, before: dict[str, str | bytes], after: dict[str, str | bytes], protected: tuple[str, ...] = ()
) -> tuple[str, list[str]]:
    base = make_tree(tmp_path / "base", {**API, **before})
    head = make_tree(tmp_path / "head", {**API, **after})
    return pr_risk.grade(base, head, protected)


@pytest.mark.parametrize(
    ("changed", "tier"),
    [(0, "low"), (100, "low"), (101, "mid"), (400, "mid"), (401, "high")],
)
def test_tier_follows_changed_line_count(tmp_path: Path, changed: int, tier: str) -> None:
    assert grade(tmp_path, {}, {"responses_api_agents/a/app.py": lines(changed)})[0] == tier


def test_docs_only_change_is_low_at_any_size(tmp_path: Path) -> None:
    after = {"docs/guide.md": lines(5000), "fern/docs.yml": lines(500), "README.md": "x\n"}
    assert grade(tmp_path, {}, after) == ("low", ["no changes outside docs (3 doc files)"])


def test_lockfiles_data_and_binaries_are_not_counted(tmp_path: Path) -> None:
    after = {"uv.lock": lines(5000), "resources_servers/s/data/example.jsonl": lines(500), "logo.bin": b"\0" * 900}
    tier, evidence = grade(tmp_path, {}, after)
    assert (tier, evidence) == ("low", ["0 changed lines in 3 non-doc files"])


def test_removed_and_modified_lines_both_count(tmp_path: Path) -> None:
    before = {"scripts/a.py": lines(60), "scripts/gone.py": lines(30)}
    after = {"scripts/a.py": lines(60, "edited")}
    assert grade(tmp_path, before, after) == ("mid", ["150 changed lines in 2 non-doc files"])


@pytest.mark.parametrize(
    "path",
    [
        "resources_servers/s/app.py",
        "resources_servers/s/configs/s.yaml",
        "nemo_gym/verification/judge.py",
        "nemo_gym/metrics/pass_at_k.py",
    ],
)
def test_small_scoring_change_is_mid(tmp_path: Path, path: str) -> None:
    tier, evidence = grade(tmp_path, {}, {path: lines(3)})
    assert tier == "mid"
    assert f"scoring: {path}" in evidence


@pytest.mark.parametrize(
    "path",
    [
        "resources_servers/s/tests/test_app.py",
        "resources_servers/s/data/example.json",
        "resources_servers/s/README.md",
    ],
)
def test_scoring_tests_data_and_docs_stay_low(tmp_path: Path, path: str) -> None:
    assert grade(tmp_path, {}, {path: lines(3)})[0] == "low"


@pytest.mark.parametrize(
    ("after", "reason"),
    [
        ({"nemo_gym/core/__init__.py": '_PUBLIC_API = {"Thing": "things"}\n'}, "export removed: nemo_gym.core.helper"),
        (
            {"nemo_gym/core/__init__.py": '_PUBLIC_API = {"Thing": "things", "helper": "reexports", "New": "new"}\n'},
            "export added: nemo_gym.core.New",
        ),
        (
            {
                "nemo_gym/core/things.py": "class Thing:\n    size: int = 1\n\n    def run(self, x: int, y: int) -> int:\n        return x\n"
            },
            "export changed: nemo_gym.core.Thing",
        ),
        (
            {
                "nemo_gym/core/things.py": "class Thing:\n    size: str = 1\n\n    def run(self, x: int) -> int:\n        return x\n"
            },
            "export changed: nemo_gym.core.Thing",
        ),
        (
            {"nemo_gym/core/helpers.py": "def helper(a: str, b: str = '') -> str:\n    return a\n"},
            "export changed: nemo_gym.core.helper",
        ),
    ],
)
def test_public_surface_change_is_high_at_any_size(tmp_path: Path, after: dict[str, str], reason: str) -> None:
    tier, evidence = grade(tmp_path, {}, after)
    assert tier == "high"
    assert reason in evidence


def test_private_and_body_changes_keep_the_public_surface(tmp_path: Path) -> None:
    after = {
        "nemo_gym/core/things.py": (
            "class Thing:\n    size: int = 1\n    _cache: dict = {}\n\n"
            "    def run(self, x: int) -> int:\n        return self._twice(x) // 2\n\n"
            "    def _twice(self, x: int) -> int:\n        return 2 * x\n"
        ),
        "nemo_gym/core/helpers.py": "def helper(a: str) -> str:\n    return a.strip()\n\n\ndef _unused() -> None:\n    pass\n",
    }
    tier, evidence = grade(tmp_path, {}, after)
    assert tier == "low"
    assert not [reason for reason in evidence if reason.startswith("export")]


PROTECTED = ("pyproject.toml", "uv.lock", ".github/workflows/*", "SECURITY.md")


@pytest.mark.parametrize(
    ("before", "after", "path"),
    [
        ({}, {"pyproject.toml": API["pyproject.toml"] + "# note\n"}, "pyproject.toml"),
        ({"uv.lock": "a\n"}, {}, "uv.lock"),
        ({}, {".github/workflows/ci/test.yml": "x\n"}, ".github/workflows/ci/test.yml"),
        ({}, {"SECURITY.md": "x\n"}, "SECURITY.md"),
    ],
)
def test_protected_file_change_is_high_at_any_size(
    tmp_path: Path, before: dict[str, str], after: dict[str, str], path: str
) -> None:
    tier, evidence = grade(tmp_path, before, after, PROTECTED)
    assert tier == "high"
    assert f"protected: {path}" in evidence


def test_protected_globs_match_the_whole_path(tmp_path: Path) -> None:
    after = {"resources_servers/s/tests/pyproject.toml": "x\n", "scripts/uv.lock": "x\n"}
    assert grade(tmp_path, {}, after, PROTECTED)[0] == "low"


def test_protected_patterns_load_from_the_companion_file(tmp_path: Path) -> None:
    assert pr_risk.protected_patterns(tmp_path / "missing.yaml") == ()
    companion = tmp_path / "protected_files.yaml"
    companion.write_text("# header\n- a.toml  # inline\n\n- 'b/*'\n- \"*.lock\"\n")
    assert pr_risk.protected_patterns(companion) == ("a.toml", "b/*", "*.lock")
    assert yaml.safe_load(companion.read_text()) == list(pr_risk.protected_patterns(companion))
    for invalid in ("files:\n  - a\n", "-a\n", "- ''\n"):
        companion.write_text(invalid)
        with pytest.raises(ValueError, match="expected a YAML list entry"):
            pr_risk.protected_patterns(companion)


def test_repository_protects_packaging_files() -> None:
    assert pr_risk.PROTECTED_FILES == SCRIPT.with_name("protected_files.yaml")
    assert {"pyproject.toml", "uv.lock"} <= set(pr_risk.protected_patterns())
    assert yaml.safe_load(pr_risk.PROTECTED_FILES.read_text()) == list(pr_risk.protected_patterns())


def test_cli_prints_only_the_tier_on_stdout_and_ignores_the_worktree_git_file(tmp_path: Path) -> None:
    base = make_tree(tmp_path / "base", {**API, ".git": "gitdir: /a\n"})
    head = make_tree(tmp_path / "head", {**API, ".git": "gitdir: /b\n", "scripts/x.py": lines(150)})
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--base", str(base), "--head", str(head)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert result.stdout == "mid\n"
    assert result.stderr == "150 changed lines in 1 non-doc files\n"


def test_cli_rejects_a_missing_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pr_risk.main(["--base", str(tmp_path), "--head", str(tmp_path / "missing")])
    assert "is not a directory" in capsys.readouterr().err


def test_repository_public_api_is_discovered() -> None:
    exports = pr_risk.public_api(ROOT)
    assert ("nemo_gym.core", "classify_failure") in exports
    assert all(
        not pr_risk.export_signature(ROOT, f"{pkg}.{mod}", name).startswith("undefined")
        for (pkg, name), mod in exports.items()
    )
