# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze first-parent main snapshots without checking out or changing a branch."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

from native_core.lock_assets import uv_version_from_text


DEPENDENCY_INPUTS = ("uv.lock", "pyproject.toml", ".python-version")


class Snapshot(TypedDict):
    source_sha: str
    commit_timestamp: str
    buckets: list[str]
    previous_source_sha: str | None
    diff_paths: list[str]
    dependency_inputs: dict[str, str]
    uv_version: str
    anchor: bool


class ReplayPlan(TypedDict):
    schema_version: int
    main_anchor: str
    end_utc: str
    start_utc: str
    lookback_days: int
    interval_hours: int
    unfilled_buckets: list[str]
    snapshots: list[Snapshot]
    plan_id: str


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo)


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_plan(
    repo: Path,
    *,
    main_anchor: str,
    end_utc: str,
    lookback_days: int = 7,
    interval_hours: int = 6,
) -> ReplayPlan:
    """Select nearest eligible commit times, then replay in first-parent ancestry order.

    Git commit clocks can go backwards. Scan the complete chain rather than using
    --since or rev-list --before, choose the greatest timestamp at each cutoff,
    and break timestamp ties in favor of the descendant. Bucket membership remains
    explicit even when clock inversions reorder snapshots for ancestry-safe replay.
    """
    end = datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
    if end.utcoffset() != timedelta(0) or end.microsecond:
        raise ValueError("end_utc must use UTC and whole seconds")
    if lookback_days <= 0 or interval_hours <= 0 or lookback_days * 24 % interval_hours:
        raise ValueError("Positive lookback days must contain an exact number of interval hours")
    anchor = _git(repo, "rev-parse", "--verify", f"{main_anchor}^{{commit}}").decode().strip()
    history = [
        (sha, int(timestamp))
        for sha, timestamp in (
            line.split("\0")
            for line in _git(repo, "log", "--first-parent", "--format=%H%x00%ct", anchor).decode().splitlines()
        )
    ]
    # Reversing this single-parent chain gives ancestry order independently of dates.
    ordered = list(reversed(history))
    start = end - timedelta(days=lookback_days)
    memberships: dict[str, list[str]] = {}
    unfilled = []
    bucket = start
    while bucket <= end:
        eligible = [
            (timestamp, index, sha)
            for index, (sha, timestamp) in enumerate(ordered)
            if timestamp <= bucket.timestamp()
        ]
        if eligible:
            memberships.setdefault(max(eligible)[2], []).append(_utc(bucket))
        else:
            unfilled.append(_utc(bucket))
        bucket += timedelta(hours=interval_hours)
    memberships.setdefault(anchor, [])
    snapshots: list[Snapshot] = []
    previous = None
    for sha, timestamp in ordered:
        if sha not in memberships:
            continue
        snapshots.append(
            {
                "source_sha": sha,
                "commit_timestamp": _utc(datetime.fromtimestamp(timestamp, UTC)),
                "buckets": memberships[sha],
                "previous_source_sha": previous,
                "diff_paths": (
                    sorted(
                        filter(
                            None,
                            _git(repo, "diff", "--name-only", "--no-renames", "-z", previous, sha, "--")
                            .decode()
                            .split("\0"),
                        )
                    )
                    if previous
                    else []
                ),
                "dependency_inputs": {
                    name: hashlib.sha256(_git(repo, "show", f"{sha}:{name}")).hexdigest() for name in DEPENDENCY_INPUTS
                },
                "uv_version": uv_version_from_text(_git(repo, "show", f"{sha}:scripts/ci/setup_dev.sh").decode()),
                "anchor": sha == anchor,
            }
        )
        previous = sha
    plan: ReplayPlan = {
        "schema_version": 1,
        "main_anchor": anchor,
        "end_utc": _utc(end),
        "start_utc": _utc(start),
        "lookback_days": lookback_days,
        "interval_hours": interval_hours,
        "unfilled_buckets": unfilled,
        "snapshots": snapshots,
        "plan_id": "",
    }
    payload = {key: value for key, value in plan.items() if key != "plan_id"}
    plan["plan_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def write_plan(plan: ReplayPlan, *, repo: Path, output: Path) -> None:
    """Create an immutable external manifest; never replace existing evidence."""
    root = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if output.resolve().is_relative_to(root):
        raise ValueError("Replay manifests must be written outside the checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--main-anchor", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--interval-hours", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        args.repo,
        main_anchor=args.main_anchor,
        end_utc=args.end_utc,
        lookback_days=args.lookback_days,
        interval_hours=args.interval_hours,
    )
    write_plan(plan, repo=args.repo, output=args.output)
    print(json.dumps({"plan_id": plan["plan_id"], "snapshots": len(plan["snapshots"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
