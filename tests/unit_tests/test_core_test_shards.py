# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from scripts.ci.combine_core_coverage import combine
from scripts.ci.core_test_shards import DURATIONS_PATH, partition_files


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/unit-tests.yml"


def test_partition_is_complete_disjoint_balanced_and_order_independent():
    durations = {"migration.py": 8, "benchmarks.py": 7, "other.py": 6, "small.py": 5}
    files = [*durations, "new.py", "migration.py"]
    assignments = partition_files(files, durations, 4)
    assert assignments == partition_files(list(reversed(files)), durations, 4)
    assert set(assignments) == set(files)
    assert set(assignments.values()) == set(range(4))
    assert len({assignments[file] for file in ("migration.py", "benchmarks.py", "other.py", "new.py")}) == 4
    assert partition_files(files, {}, 1) == dict.fromkeys(files, 0)
    with pytest.raises(ValueError, match="positive"):
        partition_files(files, durations, 0)


def test_recorded_weights_spread_the_expensive_files():
    durations = json.loads(DURATIONS_PATH.read_text())["files"]
    assert durations and all(value > 0 for value in durations.values())
    assignments = partition_files(list(durations), durations, 4)
    slowest = sorted(durations, key=durations.get, reverse=True)[:4]
    assert len({assignments[file] for file in slowest}) == 4
    loads = [sum(durations[file] for file, shard in assignments.items() if shard == index) for index in range(4)]
    assert max(loads) - min(loads) < 1


def test_real_collection_keeps_each_file_and_parameter_exactly_once(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    for index in range(8):
        (tmp_path / f"test_{index}.py").write_text(
            "import pytest\n@pytest.mark.parametrize('n', [1, 2])\ndef test_example(n):\n    assert n\n"
        )
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_ADDOPTS="")

    def collect(*options):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "scripts.ci.core_test_shards", "--collect-only", "-q", *options],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return {line for line in result.stdout.splitlines() if line.startswith("test_") and "::" in line}

    complete = collect()
    shards = [collect(f"--core-shard-index={index}", "--core-num-shards=4") for index in range(4)]
    assert len(complete) == 16
    assert Counter(node for shard in shards for node in shard) == Counter(dict.fromkeys(complete, 1))
    for index in range(8):
        assert sum(any(node.startswith(f"test_{index}.py::") for node in shard) for shard in shards) == 1


@pytest.mark.parametrize("args", [["0"], ["-1", "4"], ["4", "4"], ["0", "0"], ["x", "4"], ["0", "4", "x"]])
def test_script_rejects_invalid_shards_before_setup(args):
    result = subprocess.run([str(ROOT / "scripts/ci/core_unit_tests.sh"), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Usage:" in result.stderr or "Expected 0 <=" in result.stderr


@pytest.mark.parametrize(
    "options", [["--core-shard-index=0"], ["--core-num-shards=4"], ["--core-shard-index=4", "--core-num-shards=4"]]
)
def test_plugin_rejects_invalid_shards(tmp_path, options):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "scripts.ci.core_test_shards", *options],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTEST_ADDOPTS=""),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "core sharding requires" in result.stderr


@pytest.mark.parametrize("missing", [*range(5)])
def test_combine_rejects_each_missing_input_before_reporting(tmp_path, monkeypatch, missing):
    names = [*(f".coverage.core-{index}" for index in range(4)), ".coverage.sandbox"]
    for index, name in enumerate(names):
        if index != missing:
            (tmp_path / name).write_bytes(b"data")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(FileNotFoundError, match=names[missing]):
        combine(tmp_path)
    assert not calls


def test_combine_enforces_the_configured_gate_across_checkout_roots(tmp_path, monkeypatch):
    from coverage import Coverage, CoverageData

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COVERAGE_FILE", str(tmp_path / ".coverage"))
    (tmp_path / "target.py").write_text("a = 1\nb = 2\n")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.coverage.run]\nrelative_files = true\nsource = ["target"]\n'
        "[tool.coverage.report]\nfail_under = 95.0\nprecision = 2\n"
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index, name in enumerate([*(f".coverage.core-{i}" for i in range(4)), ".coverage.sandbox"]):
        checkout = tmp_path / f"runner-{index}"
        checkout.mkdir()
        (checkout / "target.py").write_text((tmp_path / "target.py").read_text())
        monkeypatch.chdir(checkout)
        data = CoverageData(basename=str(inputs / name))
        data.add_lines({"target.py": [1]})
        data.write()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        combine(inputs)
    data = CoverageData(basename=str(inputs / ".coverage.sandbox"))
    data.add_lines({"target.py": [1, 2]})
    data.write()
    combine(inputs)
    coverage = Coverage()
    coverage.load()
    assert coverage.get_data().measured_files() == {"target.py"}


def test_workflow_requires_all_shards_sandbox_and_coverage_and_preserves_routing():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    core = jobs["core-tests"]
    assert core["strategy"] == {"fail-fast": False, "matrix": {"shard": [0, 1, 2, 3]}}
    assert any('core_unit_tests.sh "${{ matrix.shard }}" 4' in step.get("run", "") for step in core["steps"])
    for job in ("core-tests", "sandbox-tests"):
        upload = next(step for step in jobs[job]["steps"] if "upload-artifact@" in step.get("uses", ""))
        assert upload["with"]["include-hidden-files"] is True
        assert upload["with"]["if-no-files-found"] == "error"
    assert jobs["core-coverage"]["needs"] == ["core-tests", "sandbox-tests"]
    assert set(jobs["test"]["needs"]) == {
        "detect",
        "core-tests",
        "sandbox-tests",
        "core-coverage",
        "changed-server-tests",
    }
    assert jobs["test"]["name"] == "Test"
    assert jobs["test"]["if"] == "always()"
    assert jobs["changed-server-tests"]["if"] == "needs.detect.outputs.run_servers == 'true'"
    assert jobs["server-suite"]["strategy"]["matrix"]["shard"] == list(range(8))


@pytest.mark.parametrize(
    "failed", ["DETECT_RESULT", "CORE_RESULT", "SANDBOX_RESULT", "COVERAGE_RESULT", "SERVERS_RESULT", None]
)
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_existing_test_gate_fails_closed(failed, result):
    command = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test"]["steps"][0]["run"]
    env = {
        name: "success"
        for name in ("DETECT_RESULT", "CORE_RESULT", "SANDBOX_RESULT", "COVERAGE_RESULT", "SERVERS_RESULT")
    }
    env.update(RUN_FULL="true", RUN_SERVERS="true")
    if failed:
        env[failed] = result
    completed = subprocess.run(["bash", "-c", command], env=env)
    assert completed.returncode == (1 if failed else 0)


def test_docs_only_gate_needs_successful_detection_but_no_tests():
    command = yaml.safe_load(WORKFLOW.read_text())["jobs"]["test"]["steps"][0]["run"]
    completed = subprocess.run(
        ["bash", "-c", command], env={"DETECT_RESULT": "success", "RUN_FULL": "false", "RUN_SERVERS": "false"}
    )
    assert completed.returncode == 0
