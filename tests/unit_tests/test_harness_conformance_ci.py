# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check CI routing and distinguish evidence gaps from failed measurements."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from scripts.harness_conformance.ci import changed_harnesses, report, select_harnesses
from scripts.harness_conformance.registry import HARNESSES


@pytest.mark.parametrize("harness", HARNESSES)
def test_adapter_changes_select_only_that_harness(harness):
    assert select_harnesses([f"responses_api_agents/{harness}_agent/requirements.txt", "fern/docs.yml"]) == [harness]


@pytest.mark.parametrize(
    "path",
    [
        "nemo_gym/rollout_collection.py",
        "nemo_gym/prompts/system.md",
        "responses_api_agents/shared/prompts.md",
        "environment_servers/legacy_agent/app.py",
        "responses_api_agents/opencode_sandboxed_agent/app.py",
        "resources_servers/example_mcp_weather/app.py",
        "scripts/harness_conformance/registry.py",
        "tests/unit_tests/harness_capabilities/test_checker.py",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".github/workflows/harness-conformance.yml",
        "new_shared_module.py",
    ],
)
def test_shared_or_unknown_dependencies_run_all(path):
    assert select_harnesses(["responses_api_agents/pi_agent/app.py", path]) == list(HARNESSES)


def test_documentation_only_and_multiple_adapters():
    assert select_harnesses(["README.md", "fern/assets/example.png", "LICENSE"]) == []
    assert set(
        select_harnesses(["responses_api_agents/pi_agent/app.py", "responses_api_agents/codex_agent/app.py"])
    ) == {
        "pi",
        "codex",
    }


def test_full_branch_diff_includes_earlier_commits_and_deleted_paths(tmp_path):
    # A disposable Git repo models GitHub's checkout; no user repository is touched.
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    adapter = tmp_path / "responses_api_agents/pi_agent/app.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("original\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    adapter.unlink()
    git("add", "-u")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "remove adapter")
    (tmp_path / "README.md").write_text("later docs commit\n")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "commit", "-qm", "docs")
    assert changed_harnesses(base, root=tmp_path) == ["pi"]
    # Renaming out of an adapter must retain the source path in change selection.
    git("mv", "README.md", "responses_api_agents/pi_agent/README.md")
    git("-c", "commit.gpgsign=false", "commit", "-qam", "move docs")
    before_move = git("rev-parse", "HEAD")
    git("mv", "responses_api_agents/pi_agent/README.md", "README.md")
    git("-c", "commit.gpgsign=false", "commit", "-qam", "move docs out")
    assert changed_harnesses(before_move, root=tmp_path) == ["pi"]


def test_missing_base_runs_all(tmp_path, capsys):
    assert changed_harnesses("", root=tmp_path) == list(HARNESSES)
    assert changed_harnesses("missing-base", root=tmp_path) == list(HARNESSES)
    assert "::warning::" in capsys.readouterr().out


@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_completed_report_distinguishes_failure_classes(tmp_path, capsys, exit_code):
    (tmp_path / "conformance_summary.json").write_text(
        json.dumps({"runner_status": "completed", "full_suite": True, "harnesses": {"pi": {}}})
    )
    (tmp_path / "conformance_report.md").write_text("| pi | FAIL |\n")
    text = report("pi", output=tmp_path, exit_code=exit_code)
    annotations = capsys.readouterr().out
    assert "| pi | FAIL |" in text
    if exit_code == 0:
        assert "::warning" not in annotations
        assert "are fulfilled" in text
    elif exit_code == 1:
        assert "::warning title=Harness conformance::" in annotations
        assert "not fulfilled" in text
    else:
        assert "::warning title=Harness conformance unavailable::" in annotations
        assert "could not be evaluated" in text


@pytest.mark.parametrize(
    "summary",
    [
        None,
        "broken json",
        '{"runner_status": "running"}',
        '{"runner_status": "completed", "full_suite": false, "harnesses": {"pi": {}}}',
        '{"runner_status": "completed", "full_suite": true, "harnesses": {"codex": {}}}',
    ],
)
def test_missing_or_partial_results_never_pass(tmp_path, capsys, summary):
    if summary is not None:
        (tmp_path / "conformance_summary.json").write_text(summary)
    assert "could not be evaluated" in report("pi", output=tmp_path, exit_code=0)
    assert "::warning title=Harness conformance unavailable::" in capsys.readouterr().out


def test_custom_check_is_separate_from_required_ci_and_probe_permissions():
    root = Path(__file__).resolve().parents[2]
    main = yaml.safe_load((root / ".github/workflows/cicd-main.yml").read_text())["jobs"]
    workflow = yaml.safe_load((root / ".github/workflows/harness-conformance.yml").read_text())
    assert main["harness_conformance"]["needs"] == ["pre-flight"]
    assert "harness_conformance" not in main["Nemo_CICD_Test"]["needs"]
    assert "harness_conformance" not in main["notify-failure"]["needs"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["defaults"]["run"]["shell"] == "bash"
    assert not workflow["jobs"]["probe"].get("continue-on-error", False)
    assert workflow["jobs"]["probe"]["strategy"]["fail-fast"] is False
    assert "permissions" not in workflow["jobs"]["probe"]
    assert (
        workflow["jobs"]["report"]["permissions"]
        == main["harness_conformance"]["permissions"]
        == {"contents": "read", "actions": "read", "checks": "write"}
    )
    assert workflow["jobs"]["report"]["if"] == "always()"
    assert set(workflow["jobs"]["report"]["needs"]) == {"select", "probe"}
