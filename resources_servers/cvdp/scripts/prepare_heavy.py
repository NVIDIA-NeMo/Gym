# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare public CVDP heavy bundles as immutable project snapshots and Gym rows.

Reads the published task JSONL and manifest without applying reference patches.
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


def git(*args: str) -> str:
    """Run local Git operations with actionable errors and decoded diagnostics."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise ValueError(f"Git preparation failed ({' '.join(args)}): {result.stderr}")
    return result.stdout.strip()


def prepare(dataset: Path, bundles: Path, workspaces: Path, output: Path, cache: Path) -> Path:
    """Write heavy Gym rows and content-addressed archives, preserving all project assets."""
    workspaces.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    mappings = {}
    current = {}
    for line in (bundles / "manifest.txt").read_text().splitlines():
        key, separator, value = line.strip().partition(": ")
        if key == "id":
            current = {}
            mappings[value] = current
        elif separator and key in {"original_commit", "sanitized_commit", "ref"}:
            current[key] = value
    prepared = []
    receipts = []
    imported = set()
    for line in dataset.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row["id"]
        repo = row["context"]["repo"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo) or repo in {".", ".."}:
            raise ValueError(f"Unsafe repository name: {repo}")
        mapping = mappings[task]
        commit = row["context"]["commit"]
        if commit not in {mapping["sanitized_commit"], mapping["original_commit"]}:
            raise ValueError(f"Task commit does not match public manifest: {task}")
        commit = mapping["sanitized_commit"]
        mirror = cache / f"{repo}.git"
        if repo not in imported:
            if not mirror.exists():
                git("init", "--bare", str(mirror))
            git(
                "--git-dir",
                str(mirror),
                "fetch",
                str((bundles / f"{repo}.bundle").resolve()),
                "+refs/heads/context-*:refs/heads/context-*",
            )
            imported.add(repo)
        # Upstream extracts external/ from CVDP wrapper repositories; _ext_ repositories are already project roots.
        tree = commit if "_ext_" in repo else f"{commit}:external"
        with tempfile.NamedTemporaryFile(dir=workspaces, suffix=".tar.gz", delete=False) as stream:
            archive = Path(stream.name)
        try:
            git("--git-dir", str(mirror), "archive", "--format=tar.gz", f"--output={archive}", tree)
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            archive.replace(workspaces / f"{digest}.tar.gz")
        finally:
            archive.unlink(missing_ok=True)
        messages = []
        if row.get("system_message"):
            messages.append({"role": "system", "content": row["system_message"]})
        messages.append({"role": "user", "content": row["prompt"]})
        prepared.append(
            {
                "responses_create_params": {"input": messages},
                "verifier_metadata": {
                    "task_id": task,
                    "categories": row.get("categories", []),
                    "target_files": list(row.get("patch", {})),
                    "harness_files": row["harness"],
                    "repository_workspace_sha256": digest,
                },
            }
        )
        receipts.append({"task_id": task, "repo": repo, "commit": commit, "tree": tree, "sha256": digest})
        print(f"Prepared {len(prepared)}: {task}", flush=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in prepared))
    temporary.replace(output)
    output.with_suffix(".repositories.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return output


def main() -> None:
    """Prepare downloaded heavy tasks without changing any existing dataset config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--bundles", required=True, type=Path)
    parser.add_argument("--workspaces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.dataset, args.bundles, args.workspaces, args.output, args.cache)


if __name__ == "__main__":
    main()
