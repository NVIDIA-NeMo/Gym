# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
HEAD = "a" * 40
MERGE = "b" * 40
QUEUE = "c" * 40


def _workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def _resolve(tmp_path, monkeypatch, event="push", ref="pull-request/42", **overrides):
    steps = _workflow("cicd-main.yml")["jobs"]["classify_changes"]["steps"]
    step = next(step for step in steps if step.get("id") == "revision")
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/bash\nset -eu\nprintf "%s\\n" "$2" >> "$API_CALLS"\n'
        'case "$2" in\n'
        "  repos/NVIDIA-NeMo/Gym/pulls/42)\n"
        '    [[ "$PR_API_EXIT" == 0 ]] || exit "$PR_API_EXIT"\n'
        '    printf "%s %s\\n" "$PR_HEAD" "$PR_MERGE";;\n'
        "  repos/NVIDIA-NeMo/Gym/git/commits/*)\n"
        '    [[ "$COMMIT_API_EXIT" == 0 ]] || exit "$COMMIT_API_EXIT"\n'
        '    printf "%s\\n" "$MERGE_HEAD";;\n'
        "  *) exit 99;;\nesac\n"
    )
    gh.chmod(0o755)
    env = {
        "EVENT_NAME": event,
        "REF_NAME": ref,
        "EVENT_SHA": HEAD,
        "MERGE_GROUP_SHA": QUEUE,
        "GITHUB_REPOSITORY": "NVIDIA-NeMo/Gym",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "API_CALLS": str(tmp_path / "calls"),
        "PR_HEAD": HEAD,
        "PR_MERGE": MERGE,
        "MERGE_HEAD": HEAD,
        "PR_API_EXIT": "0",
        "COMMIT_API_EXIT": "0",
        **overrides,
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    result = subprocess.run(["bash", "-e", "-u", "-o", "pipefail", "-c", step["run"]], capture_output=True, text=True)
    output = tmp_path / "output"
    calls = tmp_path / "calls"
    return (
        result,
        output.read_text() if output.exists() else "",
        calls.read_text().splitlines() if calls.exists() else [],
    )


def test_pr_mirror_push_uses_verified_merge_commit(tmp_path, monkeypatch):
    result, output, calls = _resolve(tmp_path, monkeypatch)
    assert result.returncode == 0, result.stderr
    assert output == f"sha={MERGE}\n"
    assert calls == ["repos/NVIDIA-NeMo/Gym/pulls/42", f"repos/NVIDIA-NeMo/Gym/git/commits/{MERGE}"]


@pytest.mark.parametrize("event,ref", [("push", "main"), ("schedule", "main"), ("workflow_dispatch", "main")])
def test_non_pr_events_keep_event_revision(tmp_path, monkeypatch, event, ref):
    result, output, calls = _resolve(tmp_path, monkeypatch, event, ref)
    assert result.returncode == 0, result.stderr
    assert output == f"sha={HEAD}\n"
    assert calls == []


def test_merge_group_uses_queue_revision(tmp_path, monkeypatch):
    result, output, calls = _resolve(tmp_path, monkeypatch, "merge_group", "gh-readonly-queue/main/pr-42")
    assert result.returncode == 0, result.stderr
    assert output == f"sha={QUEUE}\n"
    assert calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"PR_HEAD": QUEUE},
        {"PR_MERGE": ""},
        {"PR_MERGE": "null"},
        {"PR_MERGE": "short"},
        {"MERGE_HEAD": QUEUE},
        {"PR_API_EXIT": "1"},
        {"COMMIT_API_EXIT": "1"},
    ],
)
def test_unavailable_or_stale_merge_fails_without_head_fallback(tmp_path, monkeypatch, overrides):
    result, output, _ = _resolve(tmp_path, monkeypatch, **overrides)
    assert result.returncode != 0
    assert output == ""


def test_missing_queue_revision_fails(tmp_path, monkeypatch):
    result, output, calls = _resolve(tmp_path, monkeypatch, "merge_group", MERGE_GROUP_SHA="")
    assert result.returncode != 0
    assert output == ""
    assert calls == []


def test_build_and_test_checkouts_share_resolved_revision():
    jobs = _workflow("cicd-main.yml")["jobs"]
    shared = "${{ needs.classify_changes.outputs.sha }}"
    assert jobs["classify_changes"]["outputs"]["sha"] == "${{ steps.revision.outputs.sha }}"
    for name, job in jobs.items():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                expected = "${{ steps.revision.outputs.sha }}" if name == "classify_changes" else shared
                assert step["with"]["ref"] == expected, name
        if job.get("uses", "").startswith("./.github/workflows/"):
            assert "classify_changes" in job["needs"]
            assert job["with"]["checkout-ref"] == shared
            workflow = _workflow(job["uses"].split("/")[-1])
            inputs = workflow["on"]["workflow_call"]["inputs"]
            assert inputs["checkout-ref"]["default"] == ""
            assert inputs["checkout-ref"]["required"] == "false"
            for child in workflow["jobs"].values():
                for step in child.get("steps", []):
                    if step.get("uses", "").startswith("actions/checkout@"):
                        assert step["with"]["ref"] == "${{ inputs.checkout-ref }}"


def test_change_detection_uses_checked_out_revision():
    jobs = _workflow("cicd-main.yml")["jobs"]
    classify = next(step for step in jobs["classify_changes"]["steps"] if step.get("id") == "changes")
    assert classify["with"]["head-sha"] == "${{ steps.revision.outputs.sha }}"
    detect = _workflow("unit-tests.yml")["jobs"]["detect"]
    classify = next(step for step in detect["steps"] if step.get("id") == "changes")
    assert classify["with"]["head-sha"] == "${{ inputs.checkout-ref }}"
    routing = next(step for step in detect["steps"] if step.get("id") == "session-routing-changes")
    assert routing["with"]["sha"] == "${{ inputs.checkout-ref || github.sha }}"
    action = yaml.load((ROOT / ".github/actions/classify-changes/action.yml").read_text(), Loader=yaml.BaseLoader)
    changed = next(step for step in action["runs"]["steps"] if step.get("id") == "changed-files")
    assert changed["with"]["sha"] == "${{ inputs.head-sha || github.sha }}"
