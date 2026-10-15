# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the CI-rendered table and splice it into the main Fern page.

Uses only Python's standard library and gh. Periodic publications select the
triggering run explicitly; ordinary docs publications tolerate missing results.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .documentation import BEGIN, END, PAGE, atomic_write, replace_block


ROOT = Path(__file__).resolve().parents[2]
UNAVAILABLE = BEGIN + "\n\n<Note>Harness measurements are currently unavailable.</Note>\n\n" + END


def eligible(run: dict, *, repo: str) -> bool:
    """Accept only completed, uncancelled periodic/on-demand upstream main runs."""
    return (
        isinstance(run, dict)
        and run.get("status") == "completed"
        and run.get("conclusion") != "cancelled"
        and run.get("head_branch") == "main"
        and run.get("event") in ("schedule", "workflow_dispatch")
        and isinstance(run.get("head_repository"), dict)
        and run["head_repository"].get("full_name") == repo
    )


def source_run(*, repo: str, run_id: int | None, run_attempt: int | None = None) -> dict:
    """Use the triggering run, or look up a main run for an ordinary docs publish."""
    if run_id is not None:
        endpoint = f"repos/{repo}/actions/runs/{run_id}"
        if run_attempt is not None:
            endpoint += f"/attempts/{run_attempt}"
        run = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))
        if not eligible(run, repo=repo):
            raise ValueError("the requested run is not a completed periodic/manual upstream main run")
        return run
    runs = []
    for event in ("schedule", "workflow_dispatch"):
        response = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{repo}/actions/workflows/cicd-main.yml/runs?branch=main&event={event}&status=completed&per_page=20",
            ],
            text=True,
        )
        runs.extend(json.loads(response)["workflow_runs"])
    candidates = [run for run in runs if eligible(run, repo=repo)]
    if not candidates:
        raise ValueError("no completed periodic/manual main run is available")
    return max(candidates, key=lambda run: (run["run_started_at"], run["id"]))


def publish(*, page: Path, run: dict, repo: str) -> None:
    """Download one exact-attempt rendered artifact; no probes or checker imports."""
    original = page.read_text()
    with tempfile.TemporaryDirectory(prefix="conformance-docs-") as directory:
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run["id"]),
                "--repo",
                repo,
                "--name",
                f"harness-conformance-docs-{run['id']}-{run['run_attempt']}",
                "--dir",
                directory,
            ],
            check=True,
        )
        block = (Path(directory) / "table.mdx").read_text()
    atomic_write(page, replace_block(original, block))
    print(f"Inserted harness measurements from run {run['id']}, attempt {run['run_attempt']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="NVIDIA-NeMo/Gym")
    parser.add_argument("--page", type=Path, default=ROOT / PAGE)
    parser.add_argument("--run-id", type=int, help="exact workflow_run event id")
    parser.add_argument("--run-attempt", type=int, help="exact workflow_run attempt; requires --run-id")
    parser.add_argument(
        "--strict", action="store_true", help="stop periodic publication if its artifact is unavailable"
    )
    args = parser.parse_args(argv)
    if args.run_attempt is not None and args.run_id is None:
        parser.error("--run-attempt requires --run-id")
    try:
        run = source_run(repo=args.repo, run_id=args.run_id, run_attempt=args.run_attempt)
        publish(page=args.page, run=run, repo=args.repo)
    except Exception as exc:
        # Ordinary docs publication must tolerate unexpected API/data failures as
        # well as transport errors. Preserve programming errors in strict mode.
        if args.strict and not isinstance(exc, (OSError, ValueError, subprocess.CalledProcessError)):
            raise
        severity = "error" if args.strict else "warning"
        message = str(exc).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
        print(f"::{severity}::Harness measurements unavailable: {message}", file=sys.stderr)
        if args.strict:
            return 2
        # Older release refs may have no generated block. Keep their page intact.
        try:
            atomic_write(args.page, replace_block(args.page.read_text(), UNAVAILABLE))
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
