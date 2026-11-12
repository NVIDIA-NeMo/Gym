#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the Pants pilot and compare a bounded sample of real main commits.

Invoked by an external scheduler; this script does not schedule itself or publish
Git changes. Results and state belong outside the checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scorecard import write_scorecards


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def change_kind(paths: list[str]) -> str:
    """Classify real source changes for sampling, independent of pilot metadata."""
    if any(Path(path).suffix in {".yaml", ".yml", ".toml", ".lock", ".json", ".jsonl"} for path in paths):
        return "configuration"
    if any(path.startswith(("nemo_gym/", "tests/unit_tests/")) for path in paths):
        return "core"
    if any(
        path.startswith(("resources_servers/", "responses_api_agents/", "responses_api_models/")) for path in paths
    ):
        return "server"
    return "other"


def select_samples(candidates: list[dict], *, limit: int, completed: set[str]) -> list[dict]:
    """Prefer one recent commit of each relevant kind, then fill by recency."""
    available = [item for item in candidates if item["sha"] not in completed and item["kind"] != "other"]
    selected = []
    for kind in ("core", "server", "configuration"):
        match = next((item for item in available if item["kind"] == kind), None)
        if match is not None and len(selected) < limit:
            selected.append(match)
    for item in available:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def candidates_from_git(root: Path, *, lookback: int) -> list[dict]:
    candidates = []
    for sha in git(root, "rev-list", "--first-parent", f"--max-count={lookback}", "origin/main").splitlines():
        parents = git(root, "rev-list", "--parents", "-n", "1", sha).split()
        if len(parents) < 2:
            continue
        parent = parents[1]
        paths = git(root, "diff", "--name-only", parent, sha).splitlines()
        candidates.append({"sha": sha, "parent": parent, "kind": change_kind(paths), "paths": paths})
    return candidates


def refresh(root: Path, *, fetch: bool, rebase: bool) -> None:
    if git(root, "status", "--porcelain"):
        raise RuntimeError("Pilot checkout has changes; commit or resolve them before monitoring.")
    if fetch:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "fetch",
                "https://github.com/NVIDIA-NeMo/Gym.git",
                "main:refs/remotes/origin/main",
            ],
            check=True,
        )
    if rebase:
        if git(root, "branch", "--show-current") != "glovisotto/pants-pilot":
            raise RuntimeError("Automatic rebase requires branch glovisotto/pants-pilot.")
        before = git(root, "rev-parse", "HEAD")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        git(root, "update-ref", f"refs/pants-pilot/checkpoints/{stamp}", before)
        process = subprocess.run(["git", "-C", str(root), "rebase", "origin/main"], check=False)
        if process.returncode:
            subprocess.run(["git", "-C", str(root), "rebase", "--abort"], check=False)
            raise RuntimeError(
                f"Rebase failed; previous pilot preserved at {before}. Resolve manually before retrying."
            )


def run_sample(root: Path, *, sample: dict, patch: bytes, destination: Path, repeats: int) -> dict:
    destination.mkdir(parents=True)
    checkout = destination / "checkout"
    outcome = {**sample, "status": "failed"}
    added = False
    try:
        with (destination / "prepare.log").open("w") as log:
            subprocess.run(
                ["git", "-C", str(root), "worktree", "add", "--detach", str(checkout), sample["sha"]],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            added = True
            subprocess.run(
                ["git", "-C", str(checkout), "apply", "--index", "--binary", "-"],
                input=patch,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        command = [
            sys.executable,
            "scripts/pants/compare.py",
            "--base",
            sample["sha"],
            "--parent",
            sample["parent"],
            "--output",
            str(destination / "comparison"),
            "--repeats",
            str(repeats),
        ]
        launcher = os.environ.get("PANTS_BIN", shutil.which("pants") or str(root / ".pants-bin/pants"))
        launcher = shutil.which(launcher) or str((root / launcher).resolve())
        with (destination / "compare.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=checkout,
                env={**os.environ, "PANTS_BIN": launcher},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outcome.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
    except subprocess.CalledProcessError as error:
        outcome["error"] = str(error)
    finally:
        if added:
            # Only this invocation's disposable checkout; all measurements live beside it.
            result = subprocess.run(
                ["git", "-C", str(root), "worktree", "remove", "--force", str(checkout)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                outcome["cleanup_error"] = result.stderr
        write_json(destination / "sample.json", outcome)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one batch (the only supported mode).")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=60, help="Maximum recent main commits to inspect.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--no-rebase", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List candidate samples without fetching or rebasing.")
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 3 or args.repeats < 1 or args.lookback < 1:
        parser.error("Use 1–3 samples and positive repeats/lookback.")
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the pilot checkout.")
    if args.dry_run:
        print(
            json.dumps(
                select_samples(
                    candidates_from_git(root, lookback=args.lookback), limit=args.max_samples, completed=set()
                ),
                indent=2,
            )
        )
        return 0
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".monitor.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        parser.error(f"Another batch may be active: {lock}. Inspect before removing a stale lock.")
    try:
        refresh(root, fetch=not args.no_fetch, rebase=not args.no_rebase)
        pilot_head = git(root, "rev-parse", "HEAD")
        main_head = git(root, "rev-parse", "origin/main")
        fork_point = git(root, "merge-base", pilot_head, main_head)
        patch = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", fork_point, pilot_head])
        if not patch:
            raise RuntimeError("No pilot overlay found relative to main.")
        overlay_hash = hashlib.sha256(patch).hexdigest()
        state_path = output / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"completed": {}}
        completed = set(state["completed"].get(overlay_hash, []))
        samples = select_samples(
            candidates_from_git(root, lookback=args.lookback), limit=args.max_samples, completed=completed
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        batch = output / "batches" / stamp
        batch.mkdir(parents=True)
        metadata = {"pilot_head": pilot_head, "main_head": main_head, "overlay_sha256": overlay_hash, "samples": []}
        write_json(batch / "batch.json", metadata)
        for sample in samples:
            outcome = run_sample(
                root, sample=sample, patch=patch, destination=batch / sample["sha"], repeats=args.repeats
            )
            metadata["samples"].append(outcome)
            if outcome["status"] == "completed":
                completed.add(sample["sha"])
            state["completed"][overlay_hash] = sorted(completed)
            write_json(state_path, state)
            write_json(batch / "batch.json", metadata)
        lines = [
            "# Pants pilot comparison batch",
            "",
            f"Main: `{main_head}`; pilot: `{pilot_head}`.",
            "",
            "Measurements cover the declared pilot subset, not the entire production CI suite.",
            "",
        ]
        for outcome in metadata["samples"]:
            sha = outcome["sha"]
            lines.append(
                f"- `{sha[:12]}` ({outcome['kind']}): {outcome['status']} — "
                f"[comparison]({sha}/comparison/summary.md), [logs]({sha}/compare.log)."
            )
        if not samples:
            lines.append("No new eligible samples in the configured lookback.")
        (batch / "summary.md").write_text("\n".join(lines) + "\n")
        write_scorecards(output)
        print(batch / "summary.md")
        return int(any(item["status"] != "completed" for item in metadata["samples"]))
    finally:
        lock.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
