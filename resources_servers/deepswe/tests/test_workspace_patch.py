# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from resources_servers.deepswe.workspace_patch import WorkspacePatchError, capture_workspace_tree, create_model_patch


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path, files: dict[str, str | bytes]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "DeepSWE Test")
    _git(repo, "config", "user.email", "deepswe-test@example.com")
    for relative_path, contents in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, bytes):
            path.write_bytes(contents)
        else:
            path.write_text(contents, encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _apply_at_base(repo: Path, base_commit: str, target: Path, patch_path: Path) -> None:
    _git(repo, "worktree", "add", "--detach", str(target), base_commit)
    _git(target, "apply", "--binary", str(patch_path))


def test_cli_isolated_mode_ignores_neighboring_module_shadow(tmp_path: Path) -> None:
    repo, _ = _repository(tmp_path, {"file.txt": "unchanged\n"})
    helper_dir = tmp_path / "helper"
    helper_dir.mkdir()
    helper_path = helper_dir / "workspace_patch.py"
    shutil.copy(Path(__file__).parents[1] / "workspace_patch.py", helper_path)
    (helper_dir / "bisect.py").write_text('raise RuntimeError("agent module was imported")\n', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-I", str(helper_path), "snapshot", "--repo", str(repo)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["initial_tree"]) in {40, 64}


def test_patch_captures_committed_and_uncommitted_changes_from_abbreviated_base(tmp_path: Path) -> None:
    repo, base_commit = _repository(
        tmp_path,
        {
            ".gitignore": "ignored.log\n",
            "committed.txt": "base committed\n",
            "unstaged.txt": "base unstaged\n",
            "deleted.txt": "delete me\n",
            "binary.bin": b"before\x00bytes",
        },
    )
    initial_tree = capture_workspace_tree(repo, anchor_ref="refs/nemo-gym/test-initial")

    (repo / "committed.txt").write_text("committed change\n", encoding="utf-8")
    (repo / "committed-new.txt").write_text("committed addition\n", encoding="utf-8")
    _git(repo, "add", "committed.txt", "committed-new.txt")
    _git(repo, "commit", "-m", "agent commit")
    (repo / "unstaged.txt").write_text("unstaged change\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged addition\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    (repo / "nested" / "untracked file.txt").parent.mkdir()
    (repo / "nested" / "untracked file.txt").write_text("untracked addition\n", encoding="utf-8")
    (repo / "deleted.txt").unlink()
    (repo / "binary.bin").write_bytes(b"after\x00binary\xffbytes")
    (repo / "ignored.log").write_text("must not be captured\n", encoding="utf-8")

    patch_path = tmp_path / "model.patch"
    result = create_model_patch(
        repo,
        initial_tree=initial_tree,
        base_commit=_git(repo, "rev-parse", "--short=7", base_commit),
        output_path=patch_path,
    )

    assert result["changed_paths"] == 7
    assert result["patch_bytes"] == patch_path.stat().st_size
    verification = tmp_path / "verification"
    _apply_at_base(repo, base_commit, verification, patch_path)
    assert (verification / "committed.txt").read_text(encoding="utf-8") == "committed change\n"
    assert (verification / "committed-new.txt").read_text(encoding="utf-8") == "committed addition\n"
    assert (verification / "unstaged.txt").read_text(encoding="utf-8") == "unstaged change\n"
    assert (verification / "staged.txt").read_text(encoding="utf-8") == "staged addition\n"
    assert (verification / "nested" / "untracked file.txt").read_text(encoding="utf-8") == "untracked addition\n"
    assert not (verification / "deleted.txt").exists()
    assert (verification / "binary.bin").read_bytes() == b"after\x00binary\xffbytes"
    assert not (verification / "ignored.log").exists()
    assert capture_workspace_tree(verification) == result["synthetic_tree"]


@pytest.mark.parametrize("base_commit", ["HEAD", "-deadbeef", "deadbeef^{tree}", "abc"])
def test_patch_rejects_non_hexadecimal_or_too_short_base_commit(tmp_path: Path, base_commit: str) -> None:
    repo, _ = _repository(tmp_path, {"file.txt": "unchanged\n"})
    initial_tree = capture_workspace_tree(repo)

    with pytest.raises(WorkspacePatchError, match="Invalid base commit"):
        create_model_patch(
            repo,
            initial_tree=initial_tree,
            base_commit=base_commit,
            output_path=tmp_path / "model.patch",
        )


def test_patch_excludes_untouched_preexisting_workspace_dirt(tmp_path: Path) -> None:
    repo, base_commit = _repository(
        tmp_path,
        {
            ".gitignore": "ignored.txt\n",
            "dirty.txt": "base dirty\n",
            "staged.txt": "base staged\n",
        },
    )
    (repo / "dirty.txt").write_text("preexisting unstaged\n", encoding="utf-8")
    (repo / "staged.txt").write_text("preexisting staged\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    (repo / "preexisting.txt").write_text("preexisting untracked\n", encoding="utf-8")
    (repo / "ignored.txt").write_text("preexisting ignored\n", encoding="utf-8")
    real_index_before = _git(repo, "diff", "--cached")

    initial_tree = capture_workspace_tree(repo)
    assert _git(repo, "diff", "--cached") == real_index_before
    (repo / "agent.txt").write_text("agent change\n", encoding="utf-8")

    patch_path = tmp_path / "model.patch"
    result = create_model_patch(
        repo,
        initial_tree=initial_tree,
        base_commit=base_commit,
        output_path=patch_path,
    )

    assert result["changed_paths"] == 1
    verification = tmp_path / "verification"
    _apply_at_base(repo, base_commit, verification, patch_path)
    assert (verification / "agent.txt").read_text(encoding="utf-8") == "agent change\n"
    assert (verification / "dirty.txt").read_text(encoding="utf-8") == "base dirty\n"
    assert (verification / "staged.txt").read_text(encoding="utf-8") == "base staged\n"
    assert not (verification / "preexisting.txt").exists()
    assert not (verification / "ignored.txt").exists()


def test_patch_excludes_configured_harness_artifacts(tmp_path: Path) -> None:
    repo, base_commit = _repository(tmp_path, {"source.py": "before\n"})
    initial_tree = capture_workspace_tree(repo)
    (repo / "source.py").write_text("after\n", encoding="utf-8")
    (repo / "export.json").write_text('{"session": "harness-owned"}\n', encoding="utf-8")

    patch_path = tmp_path / "model.patch"
    result = create_model_patch(
        repo,
        initial_tree=initial_tree,
        base_commit=base_commit,
        output_path=patch_path,
        excluded_paths=("export.json",),
    )

    assert result["changed_paths"] == 1
    verification = tmp_path / "verification"
    _apply_at_base(repo, base_commit, verification, patch_path)
    assert (verification / "source.py").read_text(encoding="utf-8") == "after\n"
    assert not (verification / "export.json").exists()


@pytest.mark.parametrize("excluded_path", ["", ".", "/tmp/export.json", "../export.json", "state/../../secret"])
def test_patch_rejects_unsafe_excluded_paths(tmp_path: Path, excluded_path: str) -> None:
    repo, base_commit = _repository(tmp_path, {"source.py": "before\n"})
    initial_tree = capture_workspace_tree(repo)

    with pytest.raises(WorkspacePatchError, match="Excluded workspace path"):
        create_model_patch(
            repo,
            initial_tree=initial_tree,
            base_commit=base_commit,
            output_path=tmp_path / "model.patch",
            excluded_paths=(excluded_path,),
        )


def test_patch_handles_file_directory_transitions(tmp_path: Path) -> None:
    repo, base_commit = _repository(
        tmp_path,
        {
            "directory/child.txt": "child\n",
            "file": "file\n",
        },
    )
    initial_tree = capture_workspace_tree(repo)
    shutil.rmtree(repo / "directory")
    (repo / "directory").write_text("now a file\n", encoding="utf-8")
    (repo / "file").unlink()
    (repo / "file" / "child.txt").parent.mkdir()
    (repo / "file" / "child.txt").write_text("now a directory\n", encoding="utf-8")

    patch_path = tmp_path / "model.patch"
    create_model_patch(repo, initial_tree=initial_tree, base_commit=base_commit, output_path=patch_path)

    verification = tmp_path / "verification"
    _apply_at_base(repo, base_commit, verification, patch_path)
    assert (verification / "directory").read_text(encoding="utf-8") == "now a file\n"
    assert (verification / "file" / "child.txt").read_text(encoding="utf-8") == "now a directory\n"


def test_unchanged_workspace_produces_empty_patch(tmp_path: Path) -> None:
    repo, base_commit = _repository(tmp_path, {"file.txt": "unchanged\n"})
    initial_tree = capture_workspace_tree(repo)
    patch_path = tmp_path / "model.patch"

    result = create_model_patch(
        repo,
        initial_tree=initial_tree,
        base_commit=base_commit,
        output_path=patch_path,
    )

    assert result["changed_paths"] == 0
    assert result["patch_bytes"] == 0
    assert patch_path.read_bytes() == b""
