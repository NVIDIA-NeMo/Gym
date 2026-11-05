# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess

import pytest
from replay_plan import build_plan, write_plan


def git(repo, *args, date=None):
    environment = dict(os.environ)
    if date:
        environment.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.check_output(["git", *args], cwd=repo, env=environment).decode().strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for key, value in (("user.name", "Test"), ("user.email", "test@example.org"), ("commit.gpgsign", "false")):
        git(repo, "config", key, value)
    for name, value in {
        "uv.lock": "lock1\n",
        "pyproject.toml": "[project]\nname='example'\n",
        ".python-version": "3.13.14\n",
        "scripts/ci/setup_dev.sh": "curl -LsSf https://astral.sh/uv/0.13.4/install.sh | sh\n",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    return repo


def commit(repo, *, hour, name):
    (repo / f"{name}.txt").write_text(name)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", name, date=f"2026-11-{4 + hour // 24:02d}T{hour % 24:02d}:00:00Z")
    return git(repo, "rev-parse", "HEAD")


def plan(repo, anchor="HEAD"):
    return build_plan(repo, main_anchor=anchor, end_utc="2026-11-05T00:00:00Z", lookback_days=1, interval_hours=6)


def test_nonmonotonic_dates_choose_nearest_time_but_replay_in_ancestry_order(repository):
    a = commit(repository, hour=0, name="a")
    (repository / "uv.lock").write_text("lock2\n")
    b = commit(repository, hour=12, name="b")
    c = commit(repository, hour=6, name="c")
    d = commit(repository, hour=18, name="d")
    anchor = commit(repository, hour=30, name="anchor")
    result = plan(repository)
    snapshots = result["snapshots"]
    assert [s["source_sha"] for s in snapshots] == [a, b, c, d, anchor]
    assert snapshots[1]["buckets"] == ["2026-11-04T12:00:00Z"]
    assert snapshots[2]["buckets"] == ["2026-11-04T06:00:00Z"]
    assert snapshots[3]["buckets"] == ["2026-11-04T18:00:00Z", "2026-11-05T00:00:00Z"]
    assert snapshots[-1]["anchor"] and snapshots[-1]["buckets"] == []
    assert snapshots[0]["previous_source_sha"] is None and snapshots[0]["diff_paths"] == []
    assert snapshots[1]["previous_source_sha"] == a
    assert snapshots[1]["diff_paths"] == ["b.txt", "uv.lock"]
    assert snapshots[0]["dependency_inputs"]["uv.lock"] == hashlib.sha256(b"lock1\n").hexdigest()
    assert snapshots[1]["dependency_inputs"]["uv.lock"] == hashlib.sha256(b"lock2\n").hexdigest()
    assert all(s["uv_version"] == "0.13.4" for s in snapshots)
    assert result == plan(repository, anchor)


def test_empty_slots_and_equal_timestamp_ties(repository):
    commit(repository, hour=7, name="a")
    chosen = commit(repository, hour=7, name="b")
    result = plan(repository)
    assert result["unfilled_buckets"] == ["2026-11-04T00:00:00Z", "2026-11-04T06:00:00Z"]
    assert len(result["snapshots"]) == 1
    assert result["snapshots"][0]["source_sha"] == chosen
    assert len(result["snapshots"][0]["buckets"]) == 3


def test_first_parent_ignores_side_branch_even_when_its_date_is_closer(repository):
    initial = commit(repository, hour=0, name="a")
    git(repository, "checkout", "-qb", "side")
    side = commit(repository, hour=12, name="side")
    git(repository, "checkout", "-q", "main")
    main = commit(repository, hour=18, name="main")
    git(repository, "merge", "--no-ff", "-qm", "merge", "side", date="2026-11-04T20:00:00Z")
    result = plan(repository)
    assert side not in [s["source_sha"] for s in result["snapshots"]]
    assert result["snapshots"][0]["source_sha"] == initial
    assert "2026-11-04T12:00:00Z" in result["snapshots"][0]["buckets"]
    assert result["snapshots"][1]["source_sha"] == main


def test_immutable_external_write_and_stable_content_id(repository, tmp_path):
    commit(repository, hour=0, name="a")
    result = plan(repository)
    output = tmp_path / "evidence" / "plan.json"
    write_plan(result, repo=repository, output=output)
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        write_plan(result, repo=repository, output=output)
    with pytest.raises(ValueError, match="outside"):
        write_plan(result, repo=repository, output=repository / "results/plan.json")
    changed = build_plan(
        repository, main_anchor="HEAD", end_utc="2026-11-05T00:00:00Z", lookback_days=1, interval_hours=12
    )
    assert result["plan_id"] != changed["plan_id"]


@pytest.mark.parametrize(
    "end,days,hours",
    [
        ("2026-11-05T00:00:00", 1, 6),
        ("2026-11-05T00:00:00+01:00", 1, 6),
        ("2026-11-05T00:00:00Z", 0, 6),
        ("2026-11-05T00:00:00Z", 1, 5),
    ],
)
def test_invalid_windows_fail_before_planning(repository, end, days, hours):
    with pytest.raises(ValueError):
        build_plan(repository, main_anchor="HEAD", end_utc=end, lookback_days=days, interval_hours=hours)
