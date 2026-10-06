# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-side snapshot helper: alternate indexes leave the candidate index alone."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOTS = ["/workspace", "/opt", "/home", "/app", "/repo", "/tmp", "/entire-cli", "/entireio-cli", "/no-magic"]


def discover():
    roots = ROOTS + os.environ.get("HARBOR_REPO_PATHS", "").split(":")
    repos = set()
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for base, dirs, files in os.walk(root):
            if ".git" in dirs or ".git" in files:
                repos.add(base)
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", ".cache"}]
            if len(Path(base).relative_to(root).parts) >= 3:
                dirs[:] = []
    return sorted(repos)


def git(repo, *args, env=None):
    result = subprocess.run(
        ["git", "-c", "safe.directory=*", "-c", "core.hooksPath=/dev/null", "-C", repo, *args],
        env=env,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed in {repo}: " + result.stderr.decode(errors="replace"))
    return result.stdout.decode(errors="replace")


def snapshot(repo):
    with tempfile.TemporaryDirectory(prefix="gym-swet-index-") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=temp + "/index")
        git(repo, "read-tree", "HEAD", env=env)
        git(repo, "add", "-A", env=env)
        return git(repo, "write-tree", env=env).strip()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    base = request.get("baseline", {})
    previous = request.get("previous", base)
    repos = discover() if not base else sorted(base)
    if not repos:
        raise RuntimeError("No task repositories found")
    trees = {repo: snapshot(repo) for repo in repos}
    namespace = request["namespace"]
    if not namespace.startswith("refs/nemo-gym/") or ".." in namespace:
        raise ValueError("Invalid snapshot ref namespace")
    output = {"trees": trees, "repositories": {}}
    for repo, tree in trees.items():
        baseline = base.get(repo, tree)
        # Preserve trees during ordinary candidate git gc without touching HEAD,
        # branches, or the candidate's index. Full patches remain on the host.
        git(repo, "update-ref", namespace + "/baseline", baseline)
        git(repo, "update-ref", namespace + "/previous", tree)
        prior = previous.get(repo, baseline)
        output["repositories"][repo] = {
            "cumulative": git(repo, "diff", baseline, tree),
            "incremental": git(repo, "diff", prior, tree),
            "binary": git(repo, "diff", "--binary", baseline, tree),
        }
    Path(sys.argv[2]).write_text(json.dumps(output))


if __name__ == "__main__":
    main()
