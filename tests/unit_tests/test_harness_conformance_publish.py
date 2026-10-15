# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep periodic updates tied to their source run without blocking unrelated docs."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.harness_conformance import publish


RUN = {
    "id": 12,
    "run_attempt": 2,
    "status": "completed",
    "conclusion": "failure",
    "head_branch": "main",
    "event": "schedule",
    "head_repository": {"full_name": "NVIDIA-NeMo/Gym"},
    "run_started_at": "2026-10-13T08:00:00Z",
}


@pytest.fixture
def page(tmp_path):
    page = tmp_path / "reference.mdx"
    page.write_text(f"Introduction\n{publish.BEGIN}\nold measurement\n{publish.END}\nATIF reference\n")
    return page


def test_publish_downloads_one_rendered_artifact_from_exact_attempt(page, monkeypatch):
    commands = []
    block = publish.BEGIN + "\n| pi | Failed |\n" + publish.END

    def download(command, *, check):
        assert check
        commands.append(command)
        (Path(command[command.index("--dir") + 1]) / "table.mdx").write_text(block)

    monkeypatch.setattr(publish.subprocess, "run", download)
    publish.publish(page=page, run=RUN, repo="NVIDIA-NeMo/Gym")
    assert len(commands) == 1 and "harness-conformance-docs-12-2" in commands[0]
    assert page.read_text() == f"Introduction\n{block}\nATIF reference\n"


def test_workflow_run_uses_triggering_run_without_latest_lookup(monkeypatch):
    commands = []

    def api(command, **kwargs):
        commands.append(command)
        return json.dumps(RUN)

    monkeypatch.setattr(publish.subprocess, "check_output", api)
    assert publish.source_run(repo="NVIDIA-NeMo/Gym", run_id=12) == RUN
    assert commands == [["gh", "api", "repos/NVIDIA-NeMo/Gym/actions/runs/12"]]


def test_triggering_attempt_remains_pinned_if_the_run_is_rerun(monkeypatch):
    commands = []

    def api(command, **kwargs):
        commands.append(command)
        return json.dumps(RUN)

    monkeypatch.setattr(publish.subprocess, "check_output", api)
    result = publish.source_run(repo="NVIDIA-NeMo/Gym", run_id=12, run_attempt=2)
    assert result["run_attempt"] == 2
    assert commands == [["gh", "api", "repos/NVIDIA-NeMo/Gym/actions/runs/12/attempts/2"]]


@pytest.mark.parametrize(
    "change",
    [
        {"head_branch": "pull-request/123"},
        {"conclusion": "cancelled"},
        {"status": "in_progress"},
        {"head_repository": {"full_name": "elsewhere/Gym"}},
        {"event": "push"},
    ],
)
def test_untrusted_or_cancelled_source_runs_are_rejected(monkeypatch, change):
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *a, **kw: json.dumps({**RUN, **change}))
    with pytest.raises(ValueError, match="upstream main"):
        publish.source_run(repo="NVIDIA-NeMo/Gym", run_id=12)


def test_latest_lookup_filters_main_server_side_and_skips_cancelled_runs(monkeypatch):
    commands = []

    def api(command, **kwargs):
        commands.append(command)
        return json.dumps({"workflow_runs": [RUN, {**RUN, "id": 13, "conclusion": "cancelled"}]})

    monkeypatch.setattr(publish.subprocess, "check_output", api)
    assert publish.source_run(repo="NVIDIA-NeMo/Gym", run_id=None)["id"] == 12
    assert all("branch=main" in command[-1] and "status=completed" in command[-1] for command in commands)


@pytest.mark.parametrize("strict", [True, False])
def test_artifact_outage_stops_only_periodic_publication(page, monkeypatch, capsys, strict):
    original = page.read_text()
    monkeypatch.setattr(publish, "source_run", lambda **kw: RUN)

    def missing(**kwargs):
        raise subprocess.CalledProcessError(1, ["gh", "run", "download"])

    monkeypatch.setattr(publish, "publish", missing)
    args = ["--page", str(page), "--run-id", "12"] + (["--strict"] if strict else [])
    assert publish.main(args) == (2 if strict else 0)
    if strict:
        assert page.read_text() == original and "::error::" in capsys.readouterr().err
    else:
        assert "currently unavailable" in page.read_text()
        assert page.read_text().startswith("Introduction") and page.read_text().endswith("ATIF reference\n")
        assert "::warning::" in capsys.readouterr().err


def test_old_release_without_generated_block_can_still_publish(page, monkeypatch):
    page.write_text("Older release documentation\n")

    def missing(**kwargs):
        raise ValueError("no measurement")

    monkeypatch.setattr(publish, "source_run", missing)
    assert publish.main(["--page", str(page)]) == 0
    assert page.read_text() == "Older release documentation\n"


@pytest.mark.parametrize("response", [{**RUN, "head_repository": None}, {}, None])
def test_unexpected_api_responses_do_not_block_ordinary_publication(page, monkeypatch, response):
    monkeypatch.setattr(publish.subprocess, "check_output", lambda *a, **kw: json.dumps(response))
    assert publish.main(["--page", str(page), "--run-id", "12"]) == 0
    assert "currently unavailable" in page.read_text()


@pytest.mark.parametrize("error", [TypeError("unexpected API data"), RuntimeError("unexpected API failure")])
def test_non_strict_boundary_tolerates_unexpected_errors(page, monkeypatch, error):
    def unexpected(**kwargs):
        raise error

    monkeypatch.setattr(publish, "source_run", unexpected)
    assert publish.main(["--page", str(page)]) == 0
    assert "currently unavailable" in page.read_text()
    with pytest.raises(type(error)):
        publish.main(["--page", str(page), "--strict"])


def test_invalid_rendered_artifact_leaves_page_untouched(page, monkeypatch):
    original = page.read_text()

    def invalid(command, **kwargs):
        (Path(command[command.index("--dir") + 1]) / "table.mdx").write_text("content without markers")

    monkeypatch.setattr(publish.subprocess, "run", invalid)
    with pytest.raises(ValueError, match="markers"):
        publish.publish(page=page, run=RUN, repo="NVIDIA-NeMo/Gym")
    assert page.read_text() == original


def test_workflow_renders_in_ci_and_publishes_from_trusted_helper():
    root = Path(__file__).resolve().parents[2]
    text = (root / ".github/workflows/publish-fern-docs.yml").read_text()
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["publish"]
    assert "head_repository.full_name == github.repository" in job["if"] and "conclusion" not in job["if"]
    assert "concurrency" not in workflow and job["concurrency"]["cancel-in-progress"] is False
    helper = next(s for s in job["steps"] if s.get("name") == "Checkout current table publishing helper")
    assert helper["with"]["ref"] == "main" and helper["with"]["persist-credentials"] is False
    insert = next(s for s in job["steps"] if s.get("name") == "Insert rendered harness measurements")
    assert insert["working-directory"] == "./publisher-source"
    assert "--run-id" in insert["run"] and "--strict" in insert["run"]
    assert insert["env"]["SOURCE_RUN_ID"] == "${{ github.event.workflow_run.id }}"
    assert insert["env"]["SOURCE_RUN_ATTEMPT"] == "${{ github.event.workflow_run.run_attempt }}"
    assert "--run-attempt" in insert["run"]
    probes = yaml.safe_load((root / ".github/workflows/harness-conformance.yml").read_text())
    assert probes["permissions"] == {"contents": "read"}
    report = probes["jobs"]["report"]
    assert report["timeout-minutes"] == 5
    assert report["permissions"] == {"contents": "read", "actions": "read", "checks": "write"}
    assert len(report["steps"]) == 2
    assert "setup_dev" not in str(report["steps"]) and "ci_table" not in str(report["steps"])
    docs = probes["jobs"]["docs_table"]
    assert docs["needs"] == ["select", "probe"]
    assert docs["permissions"] == {"contents": "read", "actions": "read"}
    assert "always()" in docs["if"] and "schedule" in docs["if"] and "workflow_dispatch" in docs["if"]
    render = next(s for s in docs["steps"] if s.get("name") == "Download and render this attempt's measurements")
    assert "scripts.harness_conformance.ci_table" in render["run"] and "--run-attempt" in render["run"]
    assert "conformance-evidence" not in render["run"]
    upload = docs["steps"][-1]
    assert upload["with"]["name"] == "harness-conformance-docs-${{ github.run_id }}-${{ github.run_attempt }}"
