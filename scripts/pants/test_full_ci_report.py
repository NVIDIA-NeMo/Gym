# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

from full_ci_report import routing, timing


ROOT = Path(__file__).resolve().parents[2]


def commit(root, path):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("sample\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "sample",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in (
        ".github/actions/classify-changes/action.yml",
        ".github/workflows/unit-tests.yml",
        ".github/workflows/code-linting.yml",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    commit(tmp_path, "initial.txt")
    return tmp_path


def test_docs_only_routes_lint_on_both_arms(tmp_path):
    root = repository(tmp_path)
    commit(root, "docs/example.md")
    result = routing(root, "HEAD^", False)
    assert result["docs_only"] == "true"
    assert [item["job"] for item in result["matrix"]["include"]] == ["lint", "lint"]


def test_server_change_uses_same_four_core_shards_and_changed_servers(tmp_path):
    root = repository(tmp_path)
    commit(root, "resources_servers/example/app.py")
    result = routing(root, "HEAD^", False)
    jobs = [item["job"] for item in result["matrix"]["include"] if item["engine"] == "baseline"]
    assert result["run_servers"] == "true"
    assert jobs.count("core") == 4
    assert jobs.count("sandbox") == 1
    assert jobs.count("changed-servers") == 1
    assert "servers" not in jobs


def test_full_run_preserves_eight_server_shards_and_session(tmp_path):
    root = repository(tmp_path)
    commit(root, "docs/example.md")
    result = routing(root, "HEAD^", True)
    jobs = [item["job"] for item in result["matrix"]["include"] if item["engine"] == "pants"]
    assert jobs.count("core") == 4
    assert jobs.count("servers") == 8
    assert jobs.count("session-routing") == 1


def test_session_path_uses_source_workflow_patterns(tmp_path):
    root = repository(tmp_path)
    commit(root, "nemo_gym/client/example.py")
    result = routing(root, "HEAD^", False)
    assert result["session_routing"]
    assert result["run_full"] == "true"


def test_timing_includes_queue_gaps_and_keeps_failures():
    jobs = [
        {
            "name": "baseline / core / 0",
            "started_at": "2026-11-04T10:00:00Z",
            "completed_at": "2026-11-04T10:02:00Z",
            "conclusion": "success",
        },
        {
            "name": "baseline / combined coverage",
            "started_at": "2026-11-04T10:03:00Z",
            "completed_at": "2026-11-04T10:04:00Z",
            "conclusion": "failure",
        },
        {
            "name": "pants / core / 0",
            "started_at": "2026-11-04T10:00:00Z",
            "completed_at": "2026-11-04T10:09:00Z",
            "conclusion": "success",
        },
    ]
    result = timing(jobs, "baseline")
    assert result["elapsed_seconds"] == 240
    assert result["job_seconds"] == 180
    assert not result["successful"]


def test_report_fails_closed_when_job_disappears(tmp_path):
    import argparse
    import json

    import pytest
    from full_ci_report import report

    route = tmp_path / "routing.json"
    route.write_text(
        json.dumps(
            {
                "source_sha": "abc",
                "experiment_sha": "def",
                "run_full": "false",
                "run_servers": "false",
                "matrix": {
                    "include": [{"engine": engine, "job": "lint", "shard": 0} for engine in ("baseline", "pants")]
                },
            }
        )
    )
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": []}))
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"id": 42, "run_attempt": 1, "created_at": "2026-11-04T10:00:00Z"}))
    output = tmp_path / "summary"
    with pytest.raises(SystemExit):
        report(argparse.Namespace(route=route, jobs=jobs, run=run, artifacts=tmp_path, output=output))
    results = json.loads((output / "results.json").read_text())
    assert results["run"]["created_at"] == "2026-11-04T10:00:00Z"
    assert not results["valid"]
    assert any("Missing execution metadata" in error for error in results["errors"])
    assert any("Missing or additional measured jobs" in error for error in results["errors"])


def test_hierarchical_arm_jobs_exclude_other_arm_and_skipped_routes():
    jobs = [
        {
            "name": "baseline / lint / 0",
            "started_at": "2026-11-04T10:00:00Z",
            "completed_at": "2026-11-04T10:01:00Z",
            "conclusion": "success",
        },
        {"name": "baseline / combined coverage", "conclusion": "skipped"},
        {"name": "baseline / core", "conclusion": "skipped"},
        {
            "name": "pants / lint / 0",
            "started_at": "2026-11-04T10:00:00Z",
            "completed_at": "2026-11-04T10:03:00Z",
            "conclusion": "success",
        },
        {"name": "Compare outcomes and complete CI timing", "conclusion": None},
    ]
    baseline = timing(jobs, "baseline")
    pants = timing(jobs, "pants")
    assert baseline["job_count"] == 1
    assert baseline["elapsed_seconds"] == 60
    assert pants["elapsed_seconds"] == 180
    assert baseline["successful"] and pants["successful"]


def test_coverage_dependency_is_private_to_each_reusable_arm():
    workflow = (ROOT / ".github/workflows/pants-full-ci.yml").read_text()
    arm = (ROOT / ".github/workflows/pants-full-ci-arm.yml").read_text()
    assert workflow.count("uses: ./.github/workflows/pants-full-ci-arm.yml") == 2
    coverage = arm.split("  coverage:\n", 1)[1]
    assert "    needs: core\n" in coverage
    assert "needs: [" not in coverage
    assert "matrix.engine" not in arm


def test_feedback_time_includes_shared_preparation_and_initial_queue():
    jobs = [
        {
            "name": "baseline / core / 0",
            "started_at": "2026-11-04T10:05:00Z",
            "completed_at": "2026-11-04T10:07:00Z",
            "conclusion": "success",
        },
        {
            "name": "baseline / combined coverage",
            "started_at": "2026-11-04T10:08:00Z",
            "completed_at": "2026-11-04T10:09:00Z",
            "conclusion": "success",
        },
        {
            "name": "pants / core / 0",
            "started_at": "2026-11-04T10:03:00Z",
            "completed_at": "2026-11-04T10:06:00Z",
            "conclusion": "success",
        },
    ]
    baseline = timing(jobs, "baseline", "2026-11-04T10:00:00Z")
    pants = timing(jobs, "pants", "2026-11-04T10:00:00Z")
    assert baseline["feedback_seconds"] == 540
    assert baseline["elapsed_seconds"] == 240
    assert baseline["job_seconds"] == 180
    assert pants["feedback_seconds"] == 360
    assert baseline["jobs"][0]["feedback_seconds"] == 420
    assert baseline["jobs"][1]["job_seconds"] == 60
