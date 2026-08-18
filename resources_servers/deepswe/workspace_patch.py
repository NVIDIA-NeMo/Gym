# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture a complete Git workspace delta without mutating the real index."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import IO, Sequence


OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
ABBREVIATED_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{4,64}")


class WorkspacePatchError(RuntimeError):
    """A Git workspace could not be captured safely."""


def _git_command(repo: Path, arguments: Sequence[str]) -> list[str]:
    return ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *arguments]


def _run_git(
    repo: Path,
    arguments: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    stdout: int | IO[bytes] = subprocess.PIPE,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        _git_command(repo, arguments),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        command = "git " + " ".join(arguments[:2])
        raise WorkspacePatchError(f"{command} failed with exit code {result.returncode}: {stderr}")
    return result


def _validate_object_id(value: str, *, name: str) -> None:
    if not OBJECT_ID_PATTERN.fullmatch(value):
        raise WorkspacePatchError(f"Invalid {name}: {value!r}")


def _resolve_commit(repo: Path, value: str) -> str:
    """Resolve a full or abbreviated hexadecimal commit ID safely."""

    if not ABBREVIATED_OBJECT_ID_PATTERN.fullmatch(value):
        raise WorkspacePatchError(f"Invalid base commit: {value!r}")
    commit = (
        _run_git(repo, ["rev-parse", "--verify", "--end-of-options", f"{value}^{{commit}}"])
        .stdout.decode("ascii")
        .strip()
    )
    _validate_object_id(commit, name="resolved base commit")
    return commit


def _temporary_index_environment(index_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_INDEX_FILE"] = str(index_path)
    return environment


def capture_workspace_tree(repo: str | Path, *, anchor_ref: str | None = None) -> str:
    """Write the current committed and uncommitted workspace state to a Git tree."""

    repo_path = Path(repo).resolve()
    with tempfile.TemporaryDirectory(prefix="deepswe-index-") as temporary_dir:
        index_path = Path(temporary_dir) / "index"
        environment = _temporary_index_environment(index_path)
        _run_git(repo_path, ["read-tree", "HEAD"], env=environment)
        _run_git(repo_path, ["add", "--all", "--", "."], env=environment)
        tree = _run_git(repo_path, ["write-tree"], env=environment).stdout.decode("ascii").strip()

    _validate_object_id(tree, name="workspace tree")
    if anchor_ref is not None:
        if not anchor_ref.startswith("refs/nemo-gym/"):
            raise WorkspacePatchError(f"Snapshot ref must be below refs/nemo-gym/: {anchor_ref!r}")
        _run_git(repo_path, ["update-ref", anchor_ref, tree])
    return tree


def _changed_paths(repo: Path, initial_tree: str, final_tree: str) -> list[str]:
    output = _run_git(
        repo,
        ["diff", "--name-only", "--no-ext-diff", "--no-renames", "-z", initial_tree, final_tree, "--"],
    ).stdout
    return [os.fsdecode(path) for path in output.split(b"\0") if path]


def _normalize_excluded_paths(paths: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for value in paths:
        path = PurePosixPath(value)
        if not value or "\0" in value or path.is_absolute() or ".." in path.parts:
            raise WorkspacePatchError(f"Excluded workspace path must be repository-relative: {value!r}")
        normalized_path = path.as_posix()
        if normalized_path == ".":
            raise WorkspacePatchError("Excluded workspace path cannot be the repository root")
        normalized.append(normalized_path)
    return tuple(dict.fromkeys(normalized))


def _is_excluded(path: str, excluded_paths: Sequence[str]) -> bool:
    return any(path == excluded or path.startswith(excluded + "/") for excluded in excluded_paths)


def _index_paths(repo: Path, environment: dict[str, str]) -> list[str]:
    output = _run_git(repo, ["ls-files", "--cached", "-z"], env=environment).stdout
    return [os.fsdecode(path) for path in output.split(b"\0") if path]


def _tree_entry(repo: Path, tree: str, path: str) -> tuple[str, str] | None:
    output = _run_git(repo, ["ls-tree", "--full-tree", "-z", tree, "--", path]).stdout
    if not output:
        return None
    entries = [entry for entry in output.split(b"\0") if entry]
    if len(entries) != 1:
        raise WorkspacePatchError(f"Expected one final-tree entry for {path!r}, found {len(entries)}")
    metadata, separator, returned_path = entries[0].partition(b"\t")
    if not separator or os.fsdecode(returned_path) != path:
        raise WorkspacePatchError(f"Git returned an unexpected tree entry for {path!r}")
    fields = metadata.decode("ascii").split()
    if len(fields) != 3:
        raise WorkspacePatchError(f"Git returned malformed tree metadata for {path!r}")
    mode, _object_type, object_id = fields
    _validate_object_id(object_id, name=f"object id for {path!r}")
    return mode, object_id


def _build_synthetic_tree(repo: Path, base_commit: str, final_tree: str, changed_paths: Sequence[str]) -> str:
    with tempfile.TemporaryDirectory(prefix="deepswe-index-") as temporary_dir:
        index_path = Path(temporary_dir) / "index"
        environment = _temporary_index_environment(index_path)
        _run_git(repo, ["read-tree", base_commit], env=environment)

        base_paths = _index_paths(repo, environment)
        paths_to_remove = {
            base_path
            for base_path in base_paths
            for changed_path in changed_paths
            if base_path == changed_path
            or base_path.startswith(changed_path + "/")
            or changed_path.startswith(base_path + "/")
        }
        for path in sorted(paths_to_remove, key=lambda value: (-value.count("/"), value)):
            _run_git(repo, ["update-index", "--force-remove", "--", path], env=environment)

        for path in changed_paths:
            entry = _tree_entry(repo, final_tree, path)
            if entry is None:
                continue
            mode, object_id = entry
            _run_git(repo, ["update-index", "--add", "--cacheinfo", mode, object_id, path], env=environment)

        synthetic_tree = _run_git(repo, ["write-tree"], env=environment).stdout.decode("ascii").strip()

    _validate_object_id(synthetic_tree, name="synthetic tree")
    return synthetic_tree


def create_model_patch(
    repo: str | Path,
    *,
    initial_tree: str,
    base_commit: str,
    output_path: str | Path,
    excluded_paths: Sequence[str] = (),
) -> dict[str, str | int]:
    """Create a patch containing every path changed by the agent since seeding."""

    _validate_object_id(initial_tree, name="initial tree")
    repo_path = Path(repo).resolve()
    output = Path(output_path).resolve()
    resolved_base_commit = _resolve_commit(repo_path, base_commit)

    _run_git(repo_path, ["cat-file", "-e", f"{initial_tree}^{{tree}}"])
    final_tree = capture_workspace_tree(repo_path)
    normalized_excluded_paths = _normalize_excluded_paths(excluded_paths)
    changed_paths = [
        path
        for path in _changed_paths(repo_path, initial_tree, final_tree)
        if not _is_excluded(path, normalized_excluded_paths)
    ]
    synthetic_tree = _build_synthetic_tree(repo_path, resolved_base_commit, final_tree, changed_paths)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        _run_git(
            repo_path,
            [
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                resolved_base_commit,
                synthetic_tree,
                "--",
            ],
            stdout=stream,
        )

    return {
        "initial_tree": initial_tree,
        "final_tree": final_tree,
        "synthetic_tree": synthetic_tree,
        "changed_paths": len(changed_paths),
        "patch_bytes": output.stat().st_size,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--repo", required=True)
    snapshot.add_argument("--anchor-ref", default="refs/nemo-gym/initial-workspace")

    patch = subparsers.add_parser("patch")
    patch.add_argument("--repo", required=True)
    patch.add_argument("--initial-tree", required=True)
    patch.add_argument("--base-commit", required=True)
    patch.add_argument("--output", required=True)
    patch.add_argument("--exclude-path", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "snapshot":
            result: dict[str, str | int] = {
                "initial_tree": capture_workspace_tree(arguments.repo, anchor_ref=arguments.anchor_ref)
            }
        else:
            result = create_model_patch(
                arguments.repo,
                initial_tree=arguments.initial_tree,
                base_commit=arguments.base_commit,
                output_path=arguments.output,
                excluded_paths=arguments.exclude_path,
            )
    except WorkspacePatchError as error:
        raise SystemExit(f"DeepSWE workspace capture failed: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
