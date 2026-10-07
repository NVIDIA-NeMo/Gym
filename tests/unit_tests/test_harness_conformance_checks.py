# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise check publication from probe evidence, including failed and partial runs."""

import json
import subprocess
from pathlib import Path

import pytest
from scripts.harness_conformance import checks
from scripts.harness_conformance.ci import report


def write_probe_result(root, harness, exit_code):
    output = root / harness / "probes"
    output.mkdir(parents=True)
    (output / "conformance_summary.json").write_text(
        json.dumps({"runner_status": "completed", "full_suite": True, "harnesses": {harness: {}}})
    )
    report(harness, output=output, exit_code=exit_code)


def test_mixed_probe_results_fail_p0_with_per_harness_explanation(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "tested-sha")
    for harness, code in (("pi", 0), ("codex", 1), ("hermes", 2)):
        write_probe_result(tmp_path, harness, code)
    check = checks.p0_check(["pi", "codex", "hermes", "opencode"], artifacts=tmp_path, sha="tested-sha")
    assert check["name"] == "Harness conformance P0"
    assert check["head_sha"] == "tested-sha"
    assert check["conclusion"] == "failure"
    assert check["output"]["title"] == "1 passed, 1 failed, 2 unavailable"
    for harness, status in (("pi", "passed"), ("codex", "failed"), ("hermes", "unavailable")):
        assert f"| {harness} | {status} |" in check["output"]["summary"]


def test_successful_probe_passes_p0(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "tested-sha")
    write_probe_result(tmp_path, "pi", 0)
    assert checks.p0_check(["pi"], artifacts=tmp_path, sha="tested-sha")["conclusion"] == "success"


@pytest.mark.parametrize("selection,conclusion", [(None, "failure"), ([], "success")])
def test_failed_selection_is_distinct_from_no_affected_harnesses(tmp_path, selection, conclusion):
    assert checks.p0_check(selection, artifacts=tmp_path, sha="tested-sha")["conclusion"] == conclusion


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        "broken json",
        "null",
        "[]",
        "{}",
        '{"harness": "pi", "sha": "old-sha", "status": "passed"}',
        '{"harness": "codex", "sha": "tested-sha", "status": "passed"}',
        '{"harness": "pi", "sha": "tested-sha", "status": "unknown"}',
    ],
)
def test_missing_invalid_or_stale_evidence_never_passes(tmp_path, evidence):
    if evidence is not None:
        path = tmp_path / "pi" / "probes" / "ci-result.json"
        path.parent.mkdir(parents=True)
        path.write_text(evidence)
    check = checks.p0_check(["pi"], artifacts=tmp_path, sha="tested-sha")
    assert check["conclusion"] == "failure"
    assert "1 unavailable" in check["output"]["title"]


@pytest.mark.parametrize("download_failure", [False, True])
def test_publisher_downloads_current_attempt_and_posts_check(tmp_path, monkeypatch, download_failure):
    env = {
        "GITHUB_REPOSITORY": "example/Gym",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "tested-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_TEMP": str(tmp_path),
        "SELECTED_HARNESSES": '["pi"]',
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    posted = []

    def run(args, **kwargs):
        if args[:3] == ["gh", "run", "download"]:
            assert args[3:9] == ["123", "--repo", "example/Gym", "--name", "harness-conformance-pi-123-2", "--dir"]
            write_probe_result(Path(args[9]).parent, "pi", 0)
            if download_failure:
                raise subprocess.CalledProcessError(1, args)
        else:
            assert args == ["gh", "api", "--method", "POST", "repos/example/Gym/check-runs", "--input", "-"]
            posted.append(json.loads(kwargs["input"]))

    monkeypatch.setattr(checks.subprocess, "run", run)
    checks.main()
    assert len(posted) == 1
    assert posted[0]["head_sha"] == "tested-sha"
    assert posted[0]["conclusion"] == ("failure" if download_failure else "success")
    assert posted[0]["details_url"] == "https://github.com/example/Gym/actions/runs/123/attempts/2"
    assert posted[0]["external_id"] == "harness-conformance-p0-123-2"
    assert "Harness conformance P0" in (tmp_path / "summary.md").read_text()
