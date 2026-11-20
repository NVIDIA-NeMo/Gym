# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier counts retain completed checks even when the probe reports execution errors."""

import json
import subprocess
import sys

import pytest
from scripts.harness_conformance import ci_table
from scripts.harness_conformance.scenarios import Scenario


COMMIT = "a" * 40


@pytest.fixture
def ci_evidence(tmp_path):
    artifacts = tmp_path / "artifacts"
    for harness in ci_table.HARNESSES:
        probes = artifacts / harness / "probes"
        probes.mkdir(parents=True)
        (probes / "ci-result.json").write_text(
            json.dumps(
                {
                    "harness": harness,
                    "sha": COMMIT,
                    "status": "unavailable",
                }
            )
        )
    probes = artifacts / "pi" / "probes"
    (probes / "ci-result.json").write_text(json.dumps({"harness": "pi", "sha": COMMIT, "status": "failed"}))
    summary = {
        "runner_status": "completed",
        "full_suite": True,
        "harnesses": {
            "pi": {
                "verdict": "not_fulfilled",
                "scenarios": [
                    {
                        "scenario": s.name,
                        "verdict": "not_fulfilled",
                        "checks": [{"id": "calls.outcome", "kind": "schema", "tier": "P0", "status": "fail"}],
                    }
                    for s in ci_table.SCENARIOS
                ],
            }
        },
    }
    (probes / "conformance_summary.json").write_text(json.dumps(summary))
    hermes = artifacts / "hermes" / "probes"
    (hermes / "ci-result.json").write_text(json.dumps({"harness": "hermes", "sha": COMMIT, "status": "passed"}))
    passed = json.loads(json.dumps(summary))
    passed["harnesses"]["hermes"] = passed["harnesses"].pop("pi")
    row = passed["harnesses"]["hermes"]
    row["verdict"] = "fulfilled"
    for scenario in row["scenarios"]:
        scenario["verdict"] = "fulfilled"
        scenario["checks"][0]["status"] = "pass"
    (hermes / "conformance_summary.json").write_text(json.dumps(passed))
    return artifacts, probes, summary, tmp_path / "rendered" / "table.mdx"


def test_ci_renders_same_table_without_retaining_json(ci_evidence):
    artifacts, _, _, output = ci_evidence

    ci_table.render_ci(
        artifacts=artifacts,
        commit=COMMIT,
        output=output,
        run_url="https://github.com/NVIDIA-NeMo/Gym/actions/runs/12/attempts/2",
    )
    assert [p.name for p in output.parent.iterdir()] == ["table.mdx"]
    text = output.read_text()
    assert (
        f'`pi` | <Badge intent="error" minimal>0/{len(ci_table.SCENARIOS)}</Badge> | <Badge intent="error" minimal>{len(ci_table.SCENARIOS)}</Badge> | 0 | 0'
        in text
        and f'`hermes` | <Badge intent="launch" minimal>{len(ci_table.SCENARIOS)}/{len(ci_table.SCENARIOS)}</Badge> | 0 | 0 | 0'
        in text
    )
    assert "`calls.outcome` | ✗" in text and "<sub>Updated" in text
    assert "Generation report" not in text and "Unit-test gates" not in text


@pytest.mark.parametrize(
    "problem",
    [
        "sha",
        "missing",
        "partial",
        "scenarios",
        "identifier",
        "missing_verdict",
        "invalid_verdict",
        "missing_tier",
        "tier",
    ],
)
def test_ci_invalid_evidence_renders_dashes_without_discarding_other_harnesses(ci_evidence, problem):
    artifacts, probes, summary, output = ci_evidence
    if problem == "sha":
        (probes / "ci-result.json").write_text(json.dumps({"harness": "pi", "sha": "b" * 40, "status": "failed"}))
    elif problem == "missing":
        (probes / "ci-result.json").unlink()
    elif problem == "partial":
        summary["full_suite"] = False
    elif problem == "scenarios":
        summary["harnesses"]["pi"]["scenarios"].pop()
    elif problem == "missing_verdict":
        del summary["harnesses"]["pi"]["scenarios"][0]["verdict"]
    elif problem == "invalid_verdict":
        summary["harnesses"]["pi"]["scenarios"][0]["verdict"] = "invalid"
    elif problem == "missing_tier":
        del summary["harnesses"]["pi"]["scenarios"][0]["checks"][0]["tier"]
    elif problem == "tier":
        summary["harnesses"]["pi"]["scenarios"][0]["checks"][0]["tier"] = "P3"
    else:
        summary["harnesses"]["pi"]["scenarios"][0]["checks"][0]["id"] = "<script>"
    (probes / "conformance_summary.json").write_text(json.dumps(summary))
    ci_table.render_ci(
        artifacts=artifacts,
        commit=COMMIT,
        output=output,
        run_url="https://github.com/NVIDIA-NeMo/Gym/actions/runs/12/attempts/2",
    )
    assert "`pi` | — | — | — | —" in output.read_text()
    assert (
        f'`hermes` | <Badge intent="launch" minimal>{len(ci_table.SCENARIOS)}/{len(ci_table.SCENARIOS)}</Badge> | 0 | 0 | 0'
        in output.read_text()
    )
    assert "### `pi`" not in output.read_text()


def test_ci_execution_errors_do_not_hide_completed_check_results(ci_evidence):
    artifacts, probes, summary, output = ci_evidence
    (probes / "ci-result.json").write_text(json.dumps({"harness": "pi", "sha": COMMIT, "status": "unavailable"}))
    for scenario in summary["harnesses"]["pi"]["scenarios"][:5]:
        scenario["verdict"] = "fulfilled"
        scenario["checks"][0]["status"] = "pass"
    summary["harnesses"]["pi"]["scenarios"][-1]["execution"] = {"returncode": 1, "timed_out": False}
    (probes / "conformance_summary.json").write_text(json.dumps(summary))

    ci_table.render_ci(artifacts=artifacts, commit=COMMIT, output=output, run_url="https://example.com/run")

    text = output.read_text()
    assert (
        f'`pi` | <Badge intent="error" minimal>5/{len(ci_table.SCENARIOS)}</Badge> | <Badge intent="error" minimal>{len(ci_table.SCENARIOS) - 5}</Badge> | 0 | 0'
        in text
    )
    assert (
        f'`hermes` | <Badge intent="launch" minimal>{len(ci_table.SCENARIOS)}/{len(ci_table.SCENARIOS)}</Badge> | 0 | 0 | 0'
        in text
    )
    expected_checks = " | ".join(["✓"] * 5 + ["✗"] * (len(ci_table.SCENARIOS) - 5))
    assert "### `pi`" in text and f"`calls.outcome` | {expected_checks} |" in text
    assert "Unavailable" not in text and "Passed" not in text and "Failed" not in text


def test_ci_renderer_does_not_import_probe_runtime_modules():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scripts.harness_conformance import ci_table; "
            "assert 'scripts.harness_conformance.episode' not in sys.modules; "
            "assert 'scripts.harness_conformance.runner' not in sys.modules",
        ],
        check=True,
    )


def test_ci_table_counts_higher_tier_failure_as_failed_scenario(ci_evidence, monkeypatch):
    artifacts, _, _, output = ci_evidence
    # The support PR has no catalog P1 scenario yet. Exercise a real P1
    # compaction check instead of relabeling a failed P0 calls.outcome check.
    if not any(s.name == "context_compaction" for s in ci_table.SCENARIOS):
        scenario = Scenario("context_compaction", "Automatic compaction", tier="P1", compaction=True)
        monkeypatch.setattr(ci_table, "SCENARIOS", (*ci_table.SCENARIOS, scenario))
        for summary_path in artifacts.glob("*/probes/conformance_summary.json"):
            fixture = json.loads(summary_path.read_text())
            for fixture_row in fixture["harnesses"].values():
                fixture_row["scenarios"].append(
                    {
                        "scenario": scenario.name,
                        "verdict": "fulfilled",
                        "checks": [{"id": "calls.outcome", "tier": "P0", "kind": "schema", "status": "pass"}],
                    }
                )
            summary_path.write_text(json.dumps(fixture))
    path = artifacts / "hermes" / "probes" / "conformance_summary.json"
    summary = json.loads(path.read_text())
    row = summary["harnesses"]["hermes"]
    row["scenarios"][-1]["verdict"] = "not_fulfilled"
    row["scenarios"][-1]["tier"] = "P1"
    row["scenarios"][-1]["checks"].append(
        {"id": "compaction.event", "tier": "P1", "kind": "behavioral", "status": "fail"}
    )
    path.write_text(json.dumps(summary))
    ci_table.render_ci(artifacts=artifacts, commit=COMMIT, output=output, run_url="https://example.com/run")
    count = len(ci_table.SCENARIOS)
    text = output.read_text()
    assert (
        f'`hermes` | <Badge intent="error" minimal>{count - 1}/{count}</Badge> | 0 | <Badge intent="error" minimal>1</Badge> | 0'
        in text
    )
    assert "| Harness | Scenarios passed | First failure: P0 | First failure: P1 | First failure: P2 |" in text


def test_ci_command_downloads_its_own_attempt_and_has_unambiguous_output(tmp_path, monkeypatch):
    calls = []

    def download(harnesses, **kwargs):
        calls.append((harnesses, kwargs))

    monkeypatch.setattr(ci_table, "download_artifacts", download)
    output = tmp_path / "table.mdx"
    assert (
        ci_table.main(
            [
                "--repo",
                "NVIDIA-NeMo/Gym",
                "--run-id",
                "12",
                "--run-attempt",
                "2",
                "--commit",
                COMMIT,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0][1]["run_id"] == "12" and calls[0][1]["attempt"] == "2"
    assert "`pi` | — | — | — | —" in output.read_text()
