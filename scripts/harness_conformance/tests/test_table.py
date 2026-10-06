# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication gates use synthetic probe results, never preset harness verdicts."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from scripts.harness_conformance import table


COMMIT = "a" * 40


@pytest.fixture
def generation(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    page = root / table.PAGE
    page.parent.mkdir(parents=True)
    page.write_text(f"Introduction\n{table.BEGIN}\nold table\n{table.END}\nDeprecated C-table\n")
    calls = []
    monkeypatch.setattr(table, "verify_revision", lambda *args: calls.append("revision"))

    def tests(root, output, *, name, path):
        calls.append(name)
        return {"path": path, "passed": 2, "skipped": 0}

    monkeypatch.setattr(table, "run_tests", tests)

    def probes(*, harnesses, scenarios, output, timeout):
        calls.append("probes")
        assert scenarios == table.SCENARIOS
        for harness in harnesses:
            for scenario in scenarios:
                directory = output / harness / scenario.name
                directory.mkdir(parents=True)
                (directory / "runtime.json").write_text(json.dumps({"version": "synthetic-runtime-v1"}))
        summary = {
            "runner_status": "completed",
            "full_suite": True,
            "suite_sha256": "suite-hash",
            "limits": [],
            "harnesses": {
                harness: {
                    "verdict": "not_fulfilled",
                    "evidence": {key: {"passed": 0, "observed": 1, "required": 7} for key in table.NAMES},
                }
                for harness in harnesses
            },
        }
        (output / "conformance_summary.json").write_text(json.dumps(summary))
        return summary, 1

    monkeypatch.setattr(table, "run_suite", probes)
    return (
        root,
        page,
        calls,
        {"root": root, "commit": COMMIT, "output": tmp_path / "out", "harnesses": ["pi"], "timeout": 1},
    )


@pytest.mark.parametrize("failed_gate", ["runner", "pi", "checker"])
def test_failed_unit_gate_preserves_table_and_never_runs_probes(generation, monkeypatch, failed_gate):
    root, page, calls, kwargs = generation
    original = page.read_bytes()
    run_tests = table.run_tests

    def fail(root, output, *, name, path):
        if name == failed_gate:
            raise ValueError("test failure")
        return run_tests(root, output, name=name, path=path)

    monkeypatch.setattr(table, "run_tests", fail)
    with pytest.raises(ValueError, match="test failure"):
        table.regenerate(**kwargs)
    assert "probes" not in calls
    assert page.read_bytes() == original
    assert not (root / table.ASSETS).exists()


def test_evidence_failures_publish_commit_runtime_and_test_provenance(generation):
    root, page, calls, kwargs = generation
    table.regenerate(**kwargs)
    assert calls == ["revision", "runner", "pi", "checker", "revision", "probes", "revision"]
    text = page.read_text()
    assert text.startswith("Introduction\n") and text.endswith("\nDeprecated C-table\n")
    assert COMMIT in text and "synthetic-runtime-v1" in text
    assert "| `pi` | FAIL 0/1/7" in text
    assert "| `hermes` | Not run" in text
    (asset,) = (root / table.ASSETS).glob("*.json")
    assert asset.name in text
    report = json.loads(asset.read_text())
    assert report["gym_commit"] == COMMIT
    assert set(report["tests"]) == {"runner", "pi", "checker"}


@pytest.mark.parametrize("problem", ["execution_error", "incomplete", "source_changed", "runtime_changed"])
def test_failed_generation_preserves_previous_table(generation, monkeypatch, problem):
    root, page, _, kwargs = generation
    original = page.read_bytes()
    run_suite = table.run_suite

    def fail(**args):
        summary, code = run_suite(**args)
        if problem == "execution_error":
            code = 2
        elif problem == "incomplete":
            summary["full_suite"] = False
        elif problem == "runtime_changed":
            runtime = args["output"] / "pi" / table.SCENARIOS[-1].name / "runtime.json"
            runtime.write_text('{"version": "different"}')
        else:

            def changed(*args):
                raise ValueError("source changed")

            monkeypatch.setattr(table, "verify_revision", changed)
        return summary, code

    monkeypatch.setattr(table, "run_suite", fail)
    with pytest.raises(ValueError):
        table.regenerate(**kwargs)
    assert page.read_bytes() == original
    assert not (root / table.ASSETS).exists()


@pytest.mark.parametrize(
    "code,xml",
    [
        (1, "<testsuites/>"),
        (5, "<testsuites/>"),
        (0, "<testsuites/>"),
        (0, "<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>"),
        (0, "<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>"),
    ],
)
def test_pytest_gate_rejects_failures_and_vacuous_success(tmp_path, monkeypatch, code, xml):
    def run(command, **kwargs):
        Path(command[-1]).write_text(xml)
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(table.subprocess, "run", run)
    with pytest.raises(ValueError):
        table.run_tests(tmp_path, tmp_path, name="checker", path="tests")


def test_pytest_gate_runs_whole_directory_and_reports_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nonexistent")

    def run(command, **kwargs):
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert kwargs["env"]["NEMO_GYM_EXTRA_ROOTS"] == str(tmp_path)
        assert kwargs["cwd"] == tmp_path
        assert command[3] == "tests/unit_tests/harness_capabilities"
        Path(command[-1]).write_text(
            "<testsuites><testsuite><testcase/><testcase><skipped/></testcase></testsuite></testsuites>"
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(table.subprocess, "run", run)
    result = table.run_tests(tmp_path, tmp_path, name="checker", path="tests/unit_tests/harness_capabilities")
    assert result["passed"] == 1 and result["skipped"] == 1


@pytest.mark.skipif(shutil.which("jj") is None, reason="jj is not installed")
@pytest.mark.parametrize("vcs", ["jj", "git"])
def test_revision_rejects_source_changes_and_untracked_files(tmp_path, monkeypatch, vcs):
    def jj(*args):
        return subprocess.check_output(["jj", "--no-pager", *args], cwd=tmp_path, text=True).strip()

    jj("git", "init", "--colocate")
    source = tmp_path / "source.py"
    source.write_text("original\n")
    commit = jj("log", "--no-graph", "-r", "@", "-T", "commit_id")
    if vcs == "git":
        # Exercise read-only Git verification against jj's colocated object store.
        exists = Path.exists
        monkeypatch.setattr(Path, "exists", lambda path: False if path == tmp_path / ".jj" else exists(path))
    table.verify_revision(tmp_path, commit)
    source.write_text("edited\n")
    with pytest.raises(ValueError, match="differs"):
        table.verify_revision(tmp_path, commit)
    source.write_text("original\n")
    (tmp_path / "extra.py").write_text("untracked\n")
    with pytest.raises(ValueError, match="differs"):
        table.verify_revision(tmp_path, commit)


@pytest.mark.parametrize("commit", ["HEAD", "main", "a" * 7, "a" * 39 + ";", "A" * 40])
def test_revision_requires_full_commit_id(tmp_path, commit):
    with pytest.raises(ValueError, match="full lowercase"):
        table.verify_revision(tmp_path, commit)


def test_cli_refuses_modules_from_another_checkout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(table.runner, "__file__", str(tmp_path / "other/runner.py"))
    assert table.main(["--commit", COMMIT, "--output", str(tmp_path / "out")]) == 2
    assert "another checkout" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
