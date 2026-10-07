# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-side snapshot helper: alternate indexes leave the candidate index alone."""

import json
import os
import subprocess
import sys
import tempfile
from hashlib import sha256
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


def run_git(repo, *args, env=None, git_dir=None):
    command = ["git", "-c", "safe.directory=*", "-c", "core.hooksPath=/dev/null", "-C", repo]
    if git_dir is not None:
        command += ["--git-dir=" + str(git_dir), "--work-tree=" + repo]
    return subprocess.run(
        [*command, *args],
        env=env,
        capture_output=True,
    )


def git(repo, *args, env=None, git_dir=None):
    result = run_git(repo, *args, env=env, git_dir=git_dir)
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed in {repo}: " + result.stderr.decode(errors="replace"))
    return result.stdout.decode(errors="replace")


def snapshot_store(repo, namespace):
    """Keep unborn repositories' objects and refs outside their prepared .git.

    Some official images contain a root-owned, empty parent repository around a
    committed nested repository. Do not initialize a candidate commit or change
    ownership just to capture the prepared files. Reuse this private store even
    if the candidate subsequently creates its own first commit.
    """
    store = Path(tempfile.gettempdir()) / namespace.replace("/", "-") / sha256(repo.encode()).hexdigest()
    if not store.is_dir():
        head = run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
        if head.returncode == 0:
            return None
        reference = git(repo, "symbolic-ref", "-q", "HEAD").strip()
        if run_git(repo, "show-ref", "--verify", "--quiet", reference).returncode != 1:
            raise RuntimeError(f"Cannot snapshot invalid HEAD in {repo}")
        object_format = git(repo, "rev-parse", "--show-object-format").strip()
        git(repo, "init", "--bare", "--quiet", "--object-format=" + object_format, str(store))
    exclude = Path(git(repo, "rev-parse", "--absolute-git-dir").strip()) / "info" / "exclude"
    (store / "info" / "exclude").write_bytes(exclude.read_bytes() if exclude.is_file() else b"")
    return store


def snapshot(repo, git_dir=None):
    with tempfile.TemporaryDirectory(prefix="gym-swet-index-") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=temp + "/index")
        git(repo, "read-tree", "HEAD" if git_dir is None else "--empty", env=env, git_dir=git_dir)
        git(repo, "add", "-A", env=env, git_dir=git_dir)
        return git(repo, "write-tree", env=env, git_dir=git_dir).strip()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    base = request.get("baseline", {})
    previous = request.get("previous", base)
    repos = discover() if not base else sorted(base)
    if not repos:
        raise RuntimeError("No task repositories found")
    namespace = request["namespace"]
    if not namespace.startswith("refs/nemo-gym/") or ".." in namespace:
        raise ValueError("Invalid snapshot ref namespace")
    stores = {repo: snapshot_store(repo, namespace) for repo in repos}
    trees = {repo: snapshot(repo, stores[repo]) for repo in repos}
    output = {"trees": trees, "repositories": {}}
    for repo, tree in trees.items():
        git_dir = stores[repo]
        baseline = base.get(repo, tree)
        # Preserve trees during ordinary candidate git gc without touching HEAD,
        # branches, or the candidate's index. Full patches remain on the host.
        git(repo, "update-ref", namespace + "/baseline", baseline, git_dir=git_dir)
        git(repo, "update-ref", namespace + "/previous", tree, git_dir=git_dir)
        prior = previous.get(repo, baseline)
        output["repositories"][repo] = {
            "cumulative": git(repo, "diff", baseline, tree, git_dir=git_dir),
            "incremental": git(repo, "diff", prior, tree, git_dir=git_dir),
            "binary": git(repo, "diff", "--binary", baseline, tree, git_dir=git_dir),
        }
    Path(sys.argv[2]).write_text(json.dumps(output))


if __name__ == "__main__":
    main()
