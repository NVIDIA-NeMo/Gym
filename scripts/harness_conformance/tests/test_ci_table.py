# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing CI evidence produces an unavailable row without discarding other measurements."""

import json
import subprocess
import sys

import pytest
from scripts.harness_conformance import ci_table


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
                        "checks": [{"id": "calls.outcome", "kind": "schema", "status": "fail"}],
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
        f"`pi` | Failed | 0/{len(ci_table.SCENARIOS)}" in text
        and f"`hermes` | Passed | {len(ci_table.SCENARIOS)}/{len(ci_table.SCENARIOS)}" in text
    )
    assert "`calls.outcome` | ✗" in text and "<sub>Updated" in text
    assert "Generation report" not in text and "Unit-test gates" not in text


@pytest.mark.parametrize(
    "problem", ["sha", "missing", "partial", "scenarios", "identifier", "missing_verdict", "invalid_verdict"]
)
def test_ci_invalid_evidence_is_reported_as_unavailable(ci_evidence, problem):
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
    else:
        summary["harnesses"]["pi"]["scenarios"][0]["checks"][0]["id"] = "<script>"
    (probes / "conformance_summary.json").write_text(json.dumps(summary))
    ci_table.render_ci(
        artifacts=artifacts,
        commit=COMMIT,
        output=output,
        run_url="https://github.com/NVIDIA-NeMo/Gym/actions/runs/12/attempts/2",
    )
    assert "`pi` | Unavailable | —" in output.read_text()
    assert f"`hermes` | Passed | {len(ci_table.SCENARIOS)}/{len(ci_table.SCENARIOS)}" in output.read_text()
    assert "### `pi`" not in output.read_text()


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
    assert "Unavailable" in output.read_text()
