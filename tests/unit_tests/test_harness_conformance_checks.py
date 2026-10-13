# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise check publication from probe evidence, including failed and partial runs."""

import json
import subprocess
from pathlib import Path

import pytest
from scripts.harness_conformance import checks
from scripts.harness_conformance.ci import report
from scripts.harness_conformance.feedback import failure_details


def write_probe_result(root, harness, exit_code):
    output = root / harness / "probes"
    output.mkdir(parents=True)
    (output / "conformance_summary.json").write_text(
        json.dumps(
            {
                "runner_status": "completed",
                "full_suite": True,
                "harnesses": {
                    harness: {
                        "scenarios": [
                            {
                                "scenario": "tool_success",
                                "verdict": "fulfilled" if exit_code == 0 else "not_fulfilled",
                                "issues": [] if exit_code == 0 else ["tool status differs from the witness"],
                            }
                        ]
                    }
                },
            }
        )
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
@pytest.mark.parametrize("listing_failure", [False, True])
@pytest.mark.parametrize(
    "probe_exit_code,probe_job_result",
    [(0, "success"), (1, "success"), (2, "failure"), (0, "failure"), (0, "cancelled"), (0, "skipped")],
)
def test_publisher_downloads_current_attempt_and_posts_check(
    tmp_path, monkeypatch, download_failure, listing_failure, probe_exit_code, probe_job_result
):
    env = {
        "GITHUB_REPOSITORY": "example/Gym",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_SHA": "tested-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "RUNNER_TEMP": str(tmp_path),
        "SELECTED_HARNESSES": '["pi"]',
        "PROBE_JOB_RESULT": probe_job_result,
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    posted = []

    def run(args, **kwargs):
        if args[:3] == ["gh", "run", "download"]:
            assert args[3:9] == ["123", "--repo", "example/Gym", "--name", "harness-conformance-pi-123-2", "--dir"]
            write_probe_result(Path(args[9]).parent, "pi", probe_exit_code)
            if download_failure:
                raise subprocess.CalledProcessError(1, args)
        else:
            assert args == ["gh", "api", "--method", "POST", "repos/example/Gym/check-runs", "--input", "-"]
            posted.append(json.loads(kwargs["input"]))

    monkeypatch.setattr(checks.subprocess, "run", run)

    def artifact_listing(*args, **kwargs):
        if listing_failure:
            raise subprocess.CalledProcessError(1, args[0])
        return "harness-conformance-pi-123-2\t456\n"

    monkeypatch.setattr(checks.subprocess, "check_output", artifact_listing)
    failed_conformance = download_failure or probe_exit_code != 0
    assert checks.main() == int(failed_conformance or probe_job_result not in ("success", "skipped"))
    assert len(posted) == 1
    assert posted[0]["head_sha"] == "tested-sha"
    assert posted[0]["conclusion"] == ("failure" if failed_conformance else "success")
    assert posted[0]["details_url"] == "https://github.com/example/Gym/actions/runs/123/attempts/2"
    assert posted[0]["external_id"] == "harness-conformance-p0-123-2"
    if not listing_failure:
        assert "https://github.com/example/Gym/actions/runs/123/artifacts/456" in posted[0]["output"]["summary"]
    assert "Harness conformance P0" in (tmp_path / "summary.md").read_text()


def test_actionable_summary_counts_scenarios_and_includes_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "tested-sha")
    write_probe_result(tmp_path, "codex", 1)
    summary = checks.p0_check(["codex"], artifacts=tmp_path, sha="tested-sha")["output"]["summary"]
    assert "| codex | failed | 1 / 1 | tool_success: tool status differs from the witness |" in summary
    assert 'python scripts/run_harness_conformance.py --harness codex --output "$(mktemp -d)/probes"' in summary
    assert "probes/conformance_report.md" in summary


def test_failure_count_respects_scenario_verdict_and_explains_execution_error(tmp_path):
    # TE-8 / TE-9 are alternatives: failing TE-9 alone must not inflate the count.
    scenarios = [
        {
            "scenario": "tool_success",
            "verdict": "fulfilled",
            "evidence": {"TE-8": {"verdict": "fulfilled"}, "TE-9": {"verdict": "not_fulfilled"}},
        },
        {
            "scenario": "retry_429",
            "verdict": "not_fulfilled",
            "evidence": {"TE-8": {"verdict": "not_fulfilled"}, "TE-9": {"verdict": "not_fulfilled"}},
        },
        {
            "scenario": "model_error",
            "verdict": "not_fulfilled",
            "execution": {"returncode": 1},
            "issues": ["episode process failed; see episode.log", "expected exactly one collected rollout"],
        },
    ]
    (tmp_path / "conformance_summary.json").write_text(json.dumps({"harnesses": {"hermes": {"scenarios": scenarios}}}))
    count, first = failure_details("hermes", tmp_path)
    assert count == "2 / 3"
    assert first.startswith("retry_429: TE-8 / TE-9:")
    assert "execution also failed in model_error: episode process failed; see episode.log" in first


@pytest.mark.parametrize("contents", [None, "{}", "null", '{"harnesses": {"pi": {"scenarios": []}}}'])
def test_missing_diagnostics_are_unknown_not_zero(tmp_path, contents):
    if contents is not None:
        (tmp_path / "conformance_summary.json").write_text(contents)
    count, first = failure_details("pi", tmp_path)
    assert count == "Unknown"
    assert "setup/test logs" in first


def test_first_failure_includes_checker_assertion_and_reason(tmp_path):
    scenarios = [
        {
            "scenario": "tool_success",
            "verdict": "not_fulfilled",
            "artifact_report": "evidence/hash/evidence_summary.json",
            "evidence": {"TE-3": {"verdict": "not_fulfilled"}},
        }
    ]
    (tmp_path / "conformance_summary.json").write_text(
        json.dumps({"harnesses": {"opencode": {"scenarios": scenarios}}})
    )
    results = tmp_path / "opencode/tool_success/evidence/hash/evidence_results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text(
        json.dumps(
            {
                "findings": [
                    {
                        "evidence": "TE-3",
                        "assertion": "turn.question",
                        "reason": "non-null model-visible prompt is required",
                    }
                ]
            }
        )
        + "\n"
    )
    count, first = failure_details("opencode", tmp_path)
    assert count == "1 / 1"
    assert first == "tool_success: TE-3 / turn.question: non-null model-visible prompt is required"
