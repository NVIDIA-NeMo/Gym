# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The port is checked against recorded results of upstream's standalone ``apply_patch`` binary.

To compare against a live upstream build, or to re-record the fixture after changing cases::

    cargo build -p codex-apply-patch --bin apply_patch --release   # in codex-rs
    CODEX_APPLY_PATCH_BIN=/path/to/apply_patch pytest resources_servers/codex_tools/tests/test_apply_patch.py
    python resources_servers/codex_tools/tests/test_apply_patch.py /path/to/apply_patch   # re-record
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from resources_servers.codex_tools.apply_patch import run_apply_patch_tool


PORT = Path(__file__).parents[1] / "apply_patch.py"
FIXTURE = Path(__file__).parent / "fixtures" / "apply_patch_upstream.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8", "surrogateescape"))


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_bytes().decode("utf-8", "surrogateescape")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_case(command: list[str], case: dict, root: Path) -> dict:
    """Run an apply_patch command on a fresh copy of the case's files."""
    root.mkdir()
    _write_files(root, case["files"])
    result = subprocess.run([*command, case["patch"]], cwd=root, capture_output=True, text=True)

    def normalize(text: str) -> str:
        return text.replace(os.path.realpath(root), "<ROOT>").replace(str(root), "<ROOT>")

    return {
        "exit": result.returncode,
        "stdout": normalize(result.stdout),
        "stderr": normalize(result.stderr),
        "files": _snapshot(root),
    }


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_port_matches_recorded_upstream(case: dict, tmp_path: Path) -> None:
    assert run_case([sys.executable, str(PORT)], case, tmp_path / "case") == case["expected"]


@pytest.mark.skipif(
    not os.environ.get("CODEX_APPLY_PATCH_BIN"), reason="set CODEX_APPLY_PATCH_BIN to an upstream build"
)
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_recorded_fixture_matches_live_upstream(case: dict, tmp_path: Path) -> None:
    assert run_case([os.environ["CODEX_APPLY_PATCH_BIN"]], case, tmp_path / "case") == case["expected"]


class TestApplyPatchTool:
    """The model-visible behaviour of Codex's apply_patch tool handler."""

    def test_success_output(self, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("a\nb\n")
        patch = "*** Begin Patch\n*** Update File: f.py\n@@\n-b\n+c\n*** Add File: g.py\n+new\n*** End Patch"

        output, success = run_apply_patch_tool(patch, str(tmp_path))

        assert success
        assert output == (
            "Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess. Updated the following files:\nA g.py\nM f.py\n"
        )
        assert (tmp_path / "f.py").read_text() == "a\nc\n"

    def test_verification_failure_writes_nothing(self, tmp_path: Path) -> None:
        # The standalone binary would already have written new.txt (see the partial_failure case);
        # the tool handler verifies every hunk first.
        (tmp_path / "u.txt").write_text("1\n")
        patch = "*** Begin Patch\n*** Add File: new.txt\n+x\n*** Update File: u.txt\n@@\n-nomatch\n+2\n*** End Patch"

        output, success = run_apply_patch_tool(patch, str(tmp_path))

        assert not success
        assert (
            output == f"apply_patch verification failed: Failed to find expected lines in {tmp_path}/u.txt:\nnomatch"
        )
        assert _snapshot(tmp_path) == {"u.txt": "1\n"}

    @pytest.mark.parametrize(
        "patch, expected",
        [
            ("nonsense", "invalid patch: The first line of the patch must be '*** Begin Patch'"),
            (
                "*** Begin Patch\n*** Update File: f\n*** End Patch",
                "invalid hunk at line 2, Update file hunk for path 'f' is empty",
            ),
            (
                "*** Begin Patch\n*** Environment ID: remote\n*** Add File: a\n+x\n*** End Patch",
                "apply_patch environment selection is unavailable for this turn",
            ),
        ],
    )
    def test_parse_errors(self, patch: str, expected: str, tmp_path: Path) -> None:
        assert run_apply_patch_tool(patch, str(tmp_path)) == (f"apply_patch verification failed: {expected}", False)

    def test_empty_patch_reports_no_files_modified(self, tmp_path: Path) -> None:
        output, success = run_apply_patch_tool("*** Begin Patch\n*** End Patch", str(tmp_path))
        assert (output, success) == ("Exit code: 1\nWall time: 0 seconds\nOutput:\nNo files were modified.\n", False)

    @pytest.mark.parametrize("target", ["../outside.txt", "/tmp/codex_tools_outside.txt", "link/escape.txt"])
    def test_paths_outside_workspace_are_rejected(self, target: str, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "link").symlink_to(tmp_path)
        patch = f"*** Begin Patch\n*** Add File: {target}\n+x\n*** End Patch"

        output, success = run_apply_patch_tool(patch, str(workspace), workspace_root=str(workspace))

        assert not success
        assert output.startswith("apply_patch verification failed: path ") and "is outside the workspace" in output
        assert not (tmp_path / "outside.txt").exists() and not (tmp_path / "escape.txt").exists()
        assert not Path("/tmp/codex_tools_outside.txt").exists()

    def test_absolute_path_inside_workspace(self, tmp_path: Path) -> None:
        patch = f"*** Begin Patch\n*** Add File: {tmp_path}/abs.txt\n+x\n*** End Patch"
        output, success = run_apply_patch_tool(patch, str(tmp_path), workspace_root=str(tmp_path))
        assert success and output.endswith(f"A {tmp_path}/abs.txt\n")


if __name__ == "__main__":
    # Re-record expectations from an upstream binary: python test_apply_patch.py /path/to/apply_patch
    import tempfile

    fixture = json.loads(FIXTURE.read_text())
    for case in fixture["cases"]:
        with tempfile.TemporaryDirectory() as directory:
            case["expected"] = run_case([sys.argv[1]], case, Path(directory) / "case")
    FIXTURE.write_text(json.dumps(fixture, indent=1, ensure_ascii=True) + "\n")
