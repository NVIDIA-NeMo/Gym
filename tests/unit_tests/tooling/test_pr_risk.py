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


@pytest.mark.parametrize("inserted", [True, False])
def test_repeated_unchanged_lines_are_not_counted(tmp_path: Path, inserted: bool) -> None:
    short, long = {"scripts/t.txt": "x\n" * 500}, {"scripts/t.txt": "marker\n" + "x\n" * 500}
    before, after = (short, long) if inserted else (long, short)
    assert grade(tmp_path, before, after) == ("low", ["1 changed lines in 1 non-doc files"])


@pytest.mark.parametrize("linked_in_head", [True, False])
def test_files_beneath_a_directory_symlink_do_not_exist(tmp_path: Path, linked_in_head: bool) -> None:
    real = make_tree(tmp_path / "real", {**API, "shared/x.py": lines(401), "scripts/x.py": lines(401)})
    linked = make_tree(tmp_path / "linked", {**API, "shared/x.py": lines(401)})
    (linked / "scripts").symlink_to("shared")
    base, head = (real, linked) if linked_in_head else (linked, real)
    assert pr_risk.grade(base, head) == ("high", ["402 changed lines in 2 non-doc files"])


def test_a_symlink_counts_its_target(tmp_path: Path) -> None:
    base, head = make_tree(tmp_path / "base", API), make_tree(tmp_path / "head", API)
    (base / "link").symlink_to("a")
    (head / "link").symlink_to("b")
    assert pr_risk.grade(base, head) == ("low", ["2 changed lines in 1 non-doc files"])


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
    "path", ["nemo_gym/core/__init__.py", "nemo_gym/core/things.py", "nemo_gym/core/reexports.py"]
)
def test_changing_a_public_api_file_is_high_at_any_size(tmp_path: Path, path: str) -> None:
    tier, evidence = grade(tmp_path, {}, {path: API[path] + "# edit\n"})
    assert tier == "high"
    assert f"public surface: {path}" in evidence


def test_changing_a_module_outside_the_public_api_is_not_public_surface(tmp_path: Path) -> None:
    assert grade(tmp_path, {}, {"nemo_gym/core/helpers.py": API["nemo_gym/core/helpers.py"] + "# edit\n"})[0] == "low"


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
    companion.write_text(
        "# header\n- a.toml  # inline\n\n- 'b/*'\n- \"*.lock\"\n  # indented comment\n- .github/workflows/*\n"
        "- 'scripts/release #*.py'  # quoted hash\n- 'it''s'\n- \"say \\\"hi\\\" \\\\ #x\"\n"
    )
    expected = ("a.toml", "b/*", "*.lock", ".github/workflows/*", "scripts/release #*.py", "it's", 'say "hi" \\ #x')
    assert pr_risk.protected_patterns(companion) == expected
    assert yaml.safe_load(companion.read_text()) == list(pr_risk.protected_patterns(companion))
    unsupported = ("- *.lock\n", "- true\n", "- 1.0\n", "- a: b\n", "- a#b\n", "- a\tb\n", "- a\n  - b\n", '- "\\t"\n')
    malformed = ("files:\n  - a\n", "-a\n", "- ''\n", "- 'open\n", "- 'a' b\n", "- 'a'# c\n")
    for invalid in unsupported + malformed:
        companion.write_text(invalid)
        with pytest.raises(ValueError, match="expected a YAML list entry"):
            pr_risk.protected_patterns(companion)


def test_quoted_protected_glob_keeps_its_hash(tmp_path: Path) -> None:
    companion = tmp_path / "protected_files.yaml"
    companion.write_text("- 'scripts/release #*.py'\n")
    protected = pr_risk.protected_patterns(companion)
    assert protected == ("scripts/release #*.py",)
    tier, evidence = grade(tmp_path, {"scripts/release #1.py": "a\n"}, {"scripts/release #1.py": "b\n"}, protected)
    assert tier == "high"
    assert "protected: scripts/release #1.py" in evidence


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


def test_repository_public_surface_is_discovered() -> None:
    surface = pr_risk.public_surface(ROOT)
    assert "nemo_gym/core/__init__.py" in surface
    assert all((ROOT / path).is_file() for path in surface)
